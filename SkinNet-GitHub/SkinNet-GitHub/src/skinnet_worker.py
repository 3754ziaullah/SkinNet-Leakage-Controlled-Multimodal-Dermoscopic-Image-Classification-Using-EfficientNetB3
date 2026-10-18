import os
os.environ.setdefault("MPLBACKEND", "Agg")
def display(*a, **k):
    for x in a:
        print(x)
# [RQ-12][RQ-10] Environment, GPU detection, precision policy
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys, gc, json, time, math, glob, shutil, random, hashlib, platform, warnings, subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, replace
import numpy as np
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TensorFlow C++ console noise incl. XLA autotuning alarms (Python errors still raise)
import tensorflow as tf
tf.get_logger().setLevel("ERROR")                  # hide retracing warnings (logging only)
os.environ.setdefault("TQDM_DISABLE", "1")         # hide LIME progress bars (logging only)
import keras
from keras import layers

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60)

SMOKE_TEST = False                       # <- True: quick end-to-end check on a small real subset; False: research run
if os.environ.get("SKINNET_SMOKE_TEST") is not None:
    SMOKE_TEST = os.environ["SKINNET_SMOKE_TEST"] == "1"
SYNTHETIC = os.environ.get("SKINNET_SYNTHETIC", "0") == "1"      # sandbox code check without Kaggle data (random images)
WORKER = os.environ.get("SKINNET_WORKER", "0") == "1"            # background training process (one per GPU), Section 16
if SYNTHETIC:
    SMOKE_TEST = True

GPUS = tf.config.list_physical_devices("GPU")
for _g in GPUS:
    try:
        tf.config.experimental.set_memory_growth(_g, True)       # do not grab all 16 GB at start
    except RuntimeError as e:
        print("Memory growth could not be set (GPU already initialised):", e)

def get_hardware_info():
    info = {"python": sys.version.split()[0], "tensorflow": tf.__version__, "keras": keras.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "n_gpus": len(GPUS),
            "gpu_name": None, "compute_capability": None, "nvidia_smi": None,
            "cpu_count": os.cpu_count(), "cpu_model": None, "ram_gb": None}
    if GPUS:
        d = tf.config.experimental.get_device_details(GPUS[0])
        info["gpu_name"] = d.get("device_name")
        cc = d.get("compute_capability")
        info["compute_capability"] = f"{cc[0]}.{cc[1]}" if cc else None
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip(); break
        with open("/proc/meminfo") as f:
            info["ram_gb"] = round(int(f.readline().split()[1]) / 1024**2, 1)
    except Exception:
        pass
    return info

HW = get_hardware_info()
USE_FP16 = bool(GPUS) and os.environ.get("SKINNET_FORCE_FP32", "0") != "1"
keras.mixed_precision.set_global_policy("mixed_float16" if USE_FP16 else "float32")

def clear_keras_session():
    """keras.backend.clear_session() also resets the global dtype policy to float32 in Keras 3.
    This wrapper keeps the active policy (mixed_float16 for training) after clearing."""
    pol = keras.mixed_precision.global_policy().name
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy(pol)

print("=" * 64)
print(f" TensorFlow {HW['tensorflow']} | Keras {HW['keras']} | Python {HW['python']}")
print(f" GPU            : {HW['gpu_name'] or 'NO GPU DETECTED'}"
      + (f" (CC {HW['compute_capability']})" if HW['compute_capability'] else ""))
print(f" nvidia-smi     : {HW['nvidia_smi']}")
print(f" CPU / RAM      : {HW['cpu_count']} cores, {HW['cpu_model']}, {HW['ram_gb']} GB")
print(f" Precision      : {keras.mixed_precision.global_policy().name}  (BF16 never used)")
print(f" SMOKE_TEST     : {SMOKE_TEST}")
print("=" * 64)
if not GPUS and not SMOKE_TEST:
    print("WARNING: no GPU found. Enable the GPU accelerator (T4 x 2) before running the experiments.")
print(f" Mode           : {'SYNTHETIC code check' if SYNTHETIC else ('SMOKE TEST (real subset)' if SMOKE_TEST else 'RESEARCH RUN')}"
      + (" | background worker" if WORKER else ""))

# [RQ-12][RQ-01] Global configuration (single source of truth)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
N_CLASSES = len(CLASS_NAMES)
MEL_IDX = CLASS_NAMES.index("mel")

CFG = dict(
    SEED=42, IMG_SIZE=300, BATCH_SIZE=16, N_FOLDS=5, TEST_SIZE=0.15,
    # Two-phase schedule (values actually used in the original training cells)
    EPOCHS_P1=15, EPOCHS_P2=20, WARMUP_EPOCHS=3, LR_P1=1e-4, LR_P2=5e-5,
    MIN_LR_P1=1e-7, MIN_LR_P2=1e-8, PATIENCE_P1=6, PATIENCE_P2=5, STEPS_PER_EPOCH=800, GRAD_CLIP=1.0, SGD_LR_MULT=10.0,
    # Architecture
    DENSE_1=256, DENSE_2=128, DROPOUT_1=0.4, DROPOUT_2=0.2, L2=1e-4, CBAM_RATIO=8,
    REFINE_CH=384, META_DENSE=32, LOC_EMB_DIM=8,
    # Loss
    FOCAL_GAMMA=2.0, FOCAL_ALPHA=0.25, LABEL_SMOOTHING=0.1,
    # Imbalance strategy (original values)
    TARGET_PER_CLASS=3000, MAX_OVERSAMPLE_X=5, MINORITY_IDX=[0, 1, 3, 6],  # akiec, bcc, df, vasc
    # Batch mixing (original: CutMix p=0.3 then MixUp on every Phase-2 batch)
    CUTMIX_PROB=0.3, CUTMIX_ALPHA=1.0, MIXUP_PROB=1.0, MIXUP_ALPHA=0.2,
    # ---- Decision points (fix before running; report in manuscript) ----
    LOC_ENCODING="onehot",            # "onehot" (default) | "embedding" (original)
    CLASS_WEIGHT_MODE="per_sample",   # "per_sample" (corrected) | "legacy_batch_scalar" (original behaviour)
    MIXUP_LAMBDA_MODE="beta",         # "beta": Beta(a,a), max(l,1-l) | "legacy": max(u1, 1-u2)
    TTA_MODE="deterministic",         # "deterministic" (id, hflip, vflip, rot90, rot270) | "legacy_random"
    TTA_VIEWS=5,
    SELECTION_METRIC="val_macro_auc", # checkpoint/early-stopping metric on the validation fold (maximised)
    TEST_SPLIT_MODE="stratified_group",  # [R4-4] "stratified_group" (default) | "original" (GroupShuffleSplit, not stratified)
    RUN_PROFILE="reviewer",           # "reviewer": core + benchmarks | "core": ablation only | "full": + all supplementary arms
    REUSE_P1=True,                    # F reuses Phase 1 of E, H reuses Phase 1 of G (identical Phase 1 by construction)
    # External validation (BCN20000) [R7-2]
    EXT_CLASS_MAP={"NV": "nv", "MEL": "mel", "BCC": "bcc", "BKL": "bkl", "AK": "akiec", "DF": "df", "VASC": "vasc", "SCC": "SCC_OOD"},
    EXT_SITE_MAP={"anterior torso": "trunk", "posterior torso": "back", "lateral torso": "trunk", "head/neck": "face",
                  "upper extremity": "upper extremity", "lower extremity": "lower extremity", "palms/soles": "acral",
                  "oral/genital": "genital"},
    EXT_EVAL_EXPS=["M0", "A", "B", "C", "D", "H", "H_NM", "BM_R50", "BM_D121", "BM_CNXT", "BM_V2S", "BM_VIT"],
    VIT_PRESET="vit_base_patch16_224_imagenet", VIT_SIZE=224,
    # Explainability [R1-4]
    LIME_SAMPLES=1000, LIME_SEGMENTS=50, N_FAITH_CASES=60, FAITH_STEPS=10, N_PERM_REPEATS=10, SHAP_BACKGROUND=100,
    # Engineering
    USE_IMAGE_CACHE=True, BACKBONE_WEIGHTS="imagenet", BACKBONE_WEIGHTS_PATH=None,
    TTA_FOR_ALL_EXPERIMENTS=False, KEEP_P1_WEIGHTS=False, SESSION_BUDGET_H=9.0,       # training stops starting new units after 9.5 h so analysis + saving finish before Kaggle's 12-h limit
    MAX_HOST_RAM_FRAC=0.8,
    PARALLEL_GPU_WORKERS=True,        # one background training process per GPU when >= 2 GPUs are visible
    ANALYSIS_WHEN_COMPLETE=True,      # intermediate sessions train only; analysis runs when every fold-run is done
    FORCE_ANALYSIS=False,             # True: run the analysis now on whatever is finished (costs ~1.5-2 h)
    CLEAN_AT_END=True,                # end of session: delete image caches (rebuilt in ~5 min) and unneeded weights (saves disk)
    KEEP_WEIGHTS_FOR=["B", "D", "H"],  # the only arms whose weights later sections reload (SHAP, Grad-CAM, LIME, FP32 check, release)
    DETERMINISTIC_OPS=False, N_BOOT=2000, ECE_BINS=15, XAI_FOLD=1,
    # Expected split sizes from the ORIGINAL notebook's saved outputs (used only to verify reproduction)
    EXPECTED_TEST_COUNTS=[48, 66, 172, 10, 186, 1016, 29],
    EXPECTED_VAL_SIZES=[1697, 1698, 1698, 1698, 1697],
)

if CFG["TEST_SPLIT_MODE"] != "original":
    CFG.update(EXPECTED_TEST_COUNTS=None, EXPECTED_VAL_SIZES=None)   # counts of the original split no longer apply
if SMOKE_TEST:   # quick end-to-end check on a small real subset: produces NO scientific result
    CFG.update(IMG_SIZE=160, N_FOLDS=3, EPOCHS_P1=2, EPOCHS_P2=2, WARMUP_EPOCHS=1, PATIENCE_P1=2, PATIENCE_P2=2,
               TARGET_PER_CLASS=60, N_BOOT=100, EXPECTED_TEST_COUNTS=None, EXPECTED_VAL_SIZES=None, SESSION_BUDGET_H=3,
               LIME_SAMPLES=80, LIME_SEGMENTS=25, N_FAITH_CASES=6, FAITH_STEPS=4, N_PERM_REPEATS=2, SHAP_BACKGROUND=20,
               SMOKE_HAM_PER_CLASS=50, SMOKE_BCN_PER_CLASS=15)
    CFG["MAX_STEPS_OVERRIDE"] = 12
if SYNTHETIC:    # offline code check: random images, random weights, tiny images
    CFG.update(IMG_SIZE=64, BATCH_SIZE=8, TARGET_PER_CLASS=40, BACKBONE_WEIGHTS=None, N_BOOT=50, VIT_SIZE=64,
               LIME_SAMPLES=30, N_FAITH_CASES=4, FAITH_STEPS=3, SHAP_BACKGROUND=10)
    CFG["MAX_STEPS_OVERRIDE"] = 3
IMG_SIZE, BATCH_SIZE, SEED = CFG["IMG_SIZE"], CFG["BATCH_SIZE"], CFG["SEED"]

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    keras.utils.set_random_seed(seed)          # python, numpy and TensorFlow seeds
if CFG["DETERMINISTIC_OPS"]:
    tf.config.experimental.enable_op_determinism()   # slower; some GPU ops may be unsupported
set_seed(SEED)

# ---- Paths ---------------------------------------------------------------
# ---- Kaggle paths (edit here if your datasets are mounted elsewhere) ----
HAM_DIR = Path("/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000")   # HAM10000 (training / internal test)
BCN_DIR = Path("/kaggle/input/datasets/pasutchien/bcn20000")                 # BCN20000 (external validation only)
VIT_DIR = Path("/kaggle/input/models/keras/vit/keras/vit_base_patch16_224_imagenet/3")  # ViT-B/16 KerasHub preset (Kaggle Model input)
# Outputs of earlier sessions (Add Input -> Your Work -> Notebooks). List every attached one; nothing is overwritten.
PREV_OUTPUTS = []
WORK = Path("/kaggle/working")                                                # all outputs, checkpoints and caches
if not WORK.exists():                                                         # outside Kaggle (offline code check)
    WORK = Path.cwd() / "working"
if not SYNTHETIC:
    for _n, _p in [("HAM10000", HAM_DIR), ("BCN20000", BCN_DIR), ("ViT-B/16", VIT_DIR)] + [("Previous", p) for p in PREV_OUTPUTS]:
        print(f"{_n:9s}: {_p} {'(found)' if _p.exists() else '(NOT FOUND - automatic search under /kaggle/input will be used)'}")
if SYNTHETIC:
    WORK = Path(os.environ.get("SKINNET_SMOKE_DIR", "/tmp/skinnet_smoke")) / "working"
OUT = WORK / ("skinnet_smoke" if SMOKE_TEST else "skinnet_revision")      # smoke results never mix with research results
DIRS = {k: OUT / k for k in ["config", "data", "runs", "results", "figures", "logs"]}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
if WORKER:   # background workers share data/ and runs/ but write their own copies of everything else
    _wdir = OUT / "workers" / f"gpu{os.environ.get('SKINNET_WORKER_ID', '0')}"
    for k in ["config", "results", "figures"]:
        DIRS[k] = _wdir / k; DIRS[k].mkdir(parents=True, exist_ok=True)

def import_previous_outputs():
    """Bring in the results of earlier sessions without filling the disk: image caches are linked (not copied);
    model weights are copied only where later sections or Phase-1 reuse need them; everything else is copied."""
    copied = linked = skipped = 0
    roots = set()
    for p in PREV_OUTPUTS:
        if (p / "skinnet_revision").exists():
            roots.add(str(p / "skinnet_revision"))
        elif (p / "runs").exists():
            roots.add(str(p))
    for pat in ["/kaggle/input/*/skinnet_revision", "/kaggle/input/*/*/skinnet_revision", "/kaggle/input/*/*/*/skinnet_revision",
                "/kaggle/input/*/*/*/*/skinnet_revision"]:
        roots.update(glob.glob(pat))
    print("Previous outputs found:", sorted(roots) if roots else "NONE (training starts from zero!)")
    keep_p2, keep_p1 = set(CFG["KEEP_WEIGHTS_FOR"]), {"E", "G"}
    for src_root in sorted(roots):
        for src in Path(src_root).rglob("*"):
            if not src.is_file() or src.name.endswith(".lock"):
                continue
            rel = src.relative_to(src_root); dst = OUT / rel
            if dst.exists() or dst.is_symlink():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix == ".npy" and "cache" in src.name:
                os.symlink(src, dst); linked += 1; continue
            if src.name.endswith(".weights.h5") and len(rel.parts) >= 3 and rel.parts[0] == "runs":
                arm = rel.parts[1]
                if (src.name == "best_p2.weights.h5" and arm not in keep_p2) or (src.name == "best_p1.weights.h5" and arm not in keep_p1):
                    skipped += 1; continue
            shutil.copy2(src, dst); copied += 1
    print(f"Import: {copied} files copied, {linked} caches linked, {skipped} unneeded weight files skipped | "
          f"disk free {shutil.disk_usage(WORK).free / 2**30:.1f} GB")
    return copied
n_imported = 0 if (SMOKE_TEST or WORKER) else import_previous_outputs()

if not WORKER:
    with open(DIRS["config"] / "cfg.json", "w") as f:
        json.dump(CFG, f, indent=2)
    with open(DIRS["config"] / "environment.json", "w") as f:
        json.dump(HW, f, indent=2)
print(f"Output root         : {OUT}")
print(f"Imported from inputs: {n_imported} files")
print("Decision points     :", {k: CFG[k] for k in ["LOC_ENCODING", "CLASS_WEIGHT_MODE", "MIXUP_LAMBDA_MODE", "TTA_MODE", "SELECTION_METRIC", "TEST_SPLIT_MODE", "RUN_PROFILE"]})

# [RQ-11] Journal figure style + saver
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
CLASS_COLORS = dict(zip(CLASS_NAMES, ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00", "#0072B2", "#999999"]))
FIG_W1, FIG_W2 = 3.5, 7.2          # single / double column width (inches)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 600, "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", "#9ECAE1", "#08519C"])

def save_fig(fig, name, formats=("png", "pdf"), show=True):
    if WORKER:                        # background training processes never write figures
        plt.close(fig); return
    paths = []
    for ext in formats:
        p = DIRS["figures"] / f"{name}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None)
        paths.append(p.name)
    if show:
        plt.show()
    plt.close(fig)
    print("  saved figure:", ", ".join(paths))

# [RQ-05][RQ-04] Load metadata and map images
def make_synthetic_ham(root, n_per_class=(30, 36, 50, 21, 50, 110, 21), size=96, seed=0):
    """Synthetic stand-in used ONLY by SMOKE_TEST to exercise the code path."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    root = Path(root); (root / "HAM10000_images_part_1").mkdir(parents=True, exist_ok=True)
    locs = ["back", "lower extremity", "trunk", "upper extremity", "abdomen", "face", "chest", "unknown"]
    rows, img_counter, lesion_counter = [], 0, 0
    for c, n in enumerate(n_per_class):
        while n > 0:
            k = int(min(n, rng.integers(1, 4))); lesion_counter += 1
            for _ in range(k):
                img_counter += 1
                arr = (rng.random((size, size, 3)) * 80 + np.array([40 + 25 * c, 90, 150 - 15 * c])).clip(0, 255)
                iid = f"ISIC_{img_counter:07d}"
                Image.fromarray(arr.astype(np.uint8)).save(root / "HAM10000_images_part_1" / f"{iid}.jpg")
                rows.append(dict(lesion_id=f"HAM_{lesion_counter:07d}", image_id=iid, dx=CLASS_NAMES[c],
                                 dx_type="histo", age=float(rng.choice([np.nan] + list(range(5, 90, 5)))),
                                 sex=str(rng.choice(["male", "female", "unknown"])), localization=str(rng.choice(locs))))
            n -= k
    pd.DataFrame(rows).to_csv(root / "HAM10000_metadata.csv", index=False)
    return root

def find_ham_root():
    if SYNTHETIC:
        root = WORK.parent / "synthetic_ham"
        return root if (root / "HAM10000_metadata.csv").exists() else make_synthetic_ham(root)
    if (HAM_DIR / "HAM10000_metadata.csv").exists():
        return HAM_DIR
    hits = sorted(glob.glob("/kaggle/input/**/HAM10000_metadata.csv", recursive=True))
    if not hits:
        raise FileNotFoundError("HAM10000_metadata.csv not found under /kaggle/input. Attach kmader/skin-cancer-mnist-ham10000.")
    return Path(hits[0]).parent

HAM_ROOT = find_ham_root()
preferred = [HAM_ROOT / "HAM10000_images_part_1", HAM_ROOT / "HAM10000_images_part_2"]
img_dirs = [d for d in preferred if d.exists()] or sorted({Path(p).parent for p in glob.glob(str(HAM_ROOT / "**/*.jpg"), recursive=True)})
path_lookup = {}
for folder in img_dirs:
    for fname in os.listdir(folder):
        if fname.endswith(".jpg"):
            path_lookup.setdefault(fname[:-4], str(Path(folder) / fname))

df = pd.read_csv(HAM_ROOT / "HAM10000_metadata.csv")
df["filepath"] = df["image_id"].map(path_lookup)
n_missing_files = int(df["filepath"].isna().sum())
df = df.dropna(subset=["filepath"]).reset_index(drop=True)
if SMOKE_TEST and not SYNTHETIC:          # small real subset, sampled by whole lesions (no lesion split across subsets)
    _rng, _keep = np.random.default_rng(SEED), []
    for _c, _g in df.groupby("dx"):
        _les = _g.lesion_id.drop_duplicates().values.copy(); _rng.shuffle(_les)
        _cnt = _g.lesion_id.value_counts()
        _n = 0
        for _l in _les:
            _keep.append(_l); _n += int(_cnt[_l])
            if _n >= CFG["SMOKE_HAM_PER_CLASS"]:
                break
    df = df[df.lesion_id.isin(_keep)].reset_index(drop=True)
    print(f"SMOKE TEST: using a real subset of {len(df)} images / {df.lesion_id.nunique()} lesions")
df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASS_NAMES)}).astype(int)
df["row_idx"] = np.arange(len(df))           # stable index into the image cache

raw_audit = {
    "n_images": len(df), "n_lesions": int(df.lesion_id.nunique()), "missing_image_files": n_missing_files,
    "age_missing": int(df.age.isna().sum()), "age_equal_0": int((df.age == 0).sum()),
    "age_min": float(df.age.min()), "age_max": float(df.age.max()),
    "sex_values": df.sex.fillna("NaN").value_counts().to_dict(),
    "localization_values": df.localization.fillna("NaN").value_counts().to_dict(),
    "dx_type_values": df.dx_type.value_counts().to_dict(),
    "images_per_lesion": df.groupby("lesion_id").size().value_counts().sort_index().to_dict(),
    "patient_id_available": False,
}
with open(DIRS["results"] / "raw_metadata_audit.json", "w") as f:
    json.dump(raw_audit, f, indent=2, default=str)
if not WORKER:
    df.to_csv(DIRS["data"] / "df_raw.csv", index=False)

print(f"Images: {raw_audit['n_images']} | lesions: {raw_audit['n_lesions']} | missing files: {n_missing_files}")
print("Class counts:", df.label.value_counts().sort_index().rename(dict(enumerate(CLASS_NAMES))).to_dict())
print(f"Age: missing={raw_audit['age_missing']}, age==0 -> {raw_audit['age_equal_0']} "
      "(age 0 kept as recorded; whether it encodes 'missing' requires verification)")
print("Sex values:", raw_audit["sex_values"])
print("Localization values:", len(raw_audit["localization_values"]), raw_audit["localization_values"])
print("Images per lesion:", raw_audit["images_per_lesion"])
print("NOTE: HAM10000 provides lesion_id but no patient identifier -> grouping is lesion-level only.")

# [RQ-04][RQ-01] Splits
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

if CFG["TEST_SPLIT_MODE"] == "original":
    gss = GroupShuffleSplit(n_splits=1, test_size=CFG["TEST_SIZE"], random_state=SEED)
    trainval_idx, test_idx = next(gss.split(df, df["label"], groups=df["lesion_id"]))
else:   # [R4-4] stratified AND lesion-grouped hold-out: first fold of StratifiedGroupKFold(round(1/TEST_SIZE))
    _k = max(2, int(round(1 / CFG["TEST_SIZE"])))
    _sg = StratifiedGroupKFold(n_splits=_k, shuffle=True, random_state=SEED)
    trainval_idx, test_idx = next(_sg.split(df, df["label"], groups=df["lesion_id"]))
trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

sgkf = StratifiedGroupKFold(n_splits=CFG["N_FOLDS"])
fold_of = np.zeros(len(trainval_df), dtype=int)
for k, (_, va) in enumerate(sgkf.split(trainval_df, trainval_df["label"], groups=trainval_df["lesion_id"]), start=1):
    fold_of[va] = k
trainval_df["cv_fold"] = fold_of
test_df["cv_fold"] = 0

splits = pd.concat([trainval_df.assign(split="trainval"), test_df.assign(split="test")], ignore_index=True)
splits = splits.sort_values("row_idx").reset_index(drop=True)
if not WORKER:
    splits[["row_idx", "image_id", "lesion_id", "label", "dx", "split", "cv_fold"]].to_csv(DIRS["data"] / "splits.csv", index=False)

# ---- Verify reproduction of the original splits -----------------------------
test_counts = test_df.label.value_counts().reindex(range(N_CLASSES), fill_value=0).tolist()
val_sizes = [int((trainval_df.cv_fold == k).sum()) for k in range(1, CFG["N_FOLDS"] + 1)]
print("Test counts:", dict(zip(CLASS_NAMES, test_counts)), "| val sizes:", val_sizes)
if CFG["EXPECTED_TEST_COUNTS"] is not None:
    assert test_counts == CFG["EXPECTED_TEST_COUNTS"], "Test split differs from the original notebook!"
    assert val_sizes == CFG["EXPECTED_VAL_SIZES"], "CV folds differ from the original notebook!"
    print("Splits reproduce the original notebook exactly.")

# ---- Stratification report ---------------------------------------------------
def class_pct(frame):
    return frame.label.value_counts(normalize=True).reindex(range(N_CLASSES), fill_value=0).values * 100
strat = pd.DataFrame({"All": class_pct(df), "Test": class_pct(test_df)}, index=CLASS_NAMES)
for k in range(1, CFG["N_FOLDS"] + 1):
    strat[f"Val F{k}"] = class_pct(trainval_df[trainval_df.cv_fold == k])
strat["Test n"] = test_counts
strat.round(2).to_csv(DIRS["results"] / "split_stratification.csv")
display(strat.round(2))
print("Largest absolute deviation of test class % from full data: "
      f"{(strat['Test'] - strat['All']).abs().max():.2f} percentage points (split mode: {CFG['TEST_SPLIT_MODE']}).")

fig, ax = plt.subplots(figsize=(FIG_W2, 2.4))
cols = ["All", "Test"] + [f"Val F{k}" for k in range(1, CFG["N_FOLDS"] + 1)]
left = np.zeros(len(cols))
for c in CLASS_NAMES:
    v = strat.loc[c, cols].values
    ax.barh(cols, v, left=left, color=CLASS_COLORS[c], label=c, edgecolor="white", linewidth=0.3)
    left += v
ax.invert_yaxis(); ax.set_xlabel("Class composition (%)"); ax.set_xlim(0, 100)
ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.25))
save_fig(fig, "fig_split_composition")

def get_fold_frames(fold):
    """Return (train, val, test) frames for CV fold `fold` (1-based). Metadata untouched here."""
    tr = trainval_df[trainval_df.cv_fold != fold].reset_index(drop=True)
    va = trainval_df[trainval_df.cv_fold == fold].reset_index(drop=True)
    return tr, va, test_df.copy()

# [RQ-12] Build / open the deterministic resized-image cache
CACHE_PATH = DIRS["data"] / f"image_cache_{IMG_SIZE}.npy"
CACHE_META = DIRS["data"] / f"image_cache_{IMG_SIZE}.json"

def _decode_resize(path):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])                 # same op as original parse_image
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

def build_image_cache(frame=None, cache_path=None, meta_path=None):
    frame = df if frame is None else frame
    cache_path, meta_path = cache_path or CACHE_PATH, meta_path or CACHE_META
    if WORKER:
        raise RuntimeError("Image cache missing in a background worker; it must be built by the main notebook first.")
    n = len(frame)
    arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8, shape=(n, IMG_SIZE, IMG_SIZE, 3))
    ds = tf.data.Dataset.from_tensor_slices(frame["filepath"].values).map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE).batch(256)
    i, t0 = 0, time.time()
    for b in ds:
        b = b.numpy(); arr[i:i + len(b)] = b; i += len(b)
    arr.flush(); del arr
    with open(meta_path, "w") as f:
        json.dump({"n": n, "img_size": IMG_SIZE, "image_ids_md5": hashlib.md5("".join(frame.image_id).encode()).hexdigest()}, f)
    print(f"Cache built: {n} images in {time.time() - t0:.0f}s -> {cache_path}")

IMAGE_CACHE = None
if CFG["USE_IMAGE_CACHE"]:
    ok = False
    if CACHE_PATH.exists() and CACHE_META.exists():
        meta_c = json.load(open(CACHE_META))
        ok = meta_c.get("n") == len(df) and meta_c.get("image_ids_md5") == hashlib.md5("".join(df.image_id).encode()).hexdigest()
    if not ok:
        build_image_cache()
    IMAGE_CACHE = np.load(CACHE_PATH, mmap_mode="r")
    print("Image cache:", IMAGE_CACHE.shape, IMAGE_CACHE.dtype, f"{CACHE_PATH.stat().st_size / 1e9:.2f} GB on disk")

# [R7-2][R1-8][R1-2] BCN20000 external dataset (workers load it too, for predictions right after training)
def make_synthetic_bcn(root, per_class=12, size=96, seed=1):
    from PIL import Image
    rng = np.random.default_rng(seed); root = Path(root); (root / "BCN_20k_train" / "img").mkdir(parents=True, exist_ok=True)
    rows, k = [], 0
    for c, dx in enumerate(["NV", "MEL", "BCC", "BKL", "AK", "DF", "VASC", "SCC"]):
        for i in range(per_class):
            k += 1; fn = f"BCN_{k:07d}.jpg"
            arr = (rng.random((size, size, 3)) * 80 + np.array([30 + 20 * c, 100, 140 - 10 * c])).clip(0, 255)
            Image.fromarray(arr.astype(np.uint8)).save(root / "BCN_20k_train" / "img" / fn)
            rows.append(dict(bcn_filename=fn, age_approx=float(rng.choice([np.nan, 35, 50, 65, 80])),
                             anatom_site_general=str(rng.choice(["anterior torso", "head/neck", "lower extremity", "palms/soles", "nan"])),
                             diagnosis=dx, lesion_id=f"BCN_L{k // 2:06d}", capture_date="2012-01-01",
                             sex=str(rng.choice(["male", "female", "nan"])), split="train"))
    pd.DataFrame(rows).to_csv(root / "bcn_20k_train.csv", index=False)
    return root / "bcn_20k_train.csv"

def find_bcn_csv():
    if SYNTHETIC:
        p = WORK.parent / "synthetic_bcn" / "bcn_20k_train.csv"
        return p if p.exists() else make_synthetic_bcn(p.parent)
    hits = sorted(glob.glob(str(BCN_DIR / "**" / "bcn_20k_train.csv"), recursive=True)) if BCN_DIR.exists() else []
    if hits:
        return Path(hits[0])
    for pat in ["/kaggle/input/*/bcn_20k_train.csv", "/kaggle/input/*/*/bcn_20k_train.csv",
                "/kaggle/input/*/*/*/bcn_20k_train.csv", "/kaggle/input/*/*/*/*/bcn_20k_train.csv"]:
        hits = sorted(glob.glob(pat))
        if hits:
            return Path(hits[0])
    return None

BCN_CSV = find_bcn_csv()
BCN_AVAILABLE = BCN_CSV is not None
ext_audit = {"available": BCN_AVAILABLE, "csv": str(BCN_CSV)}
bcn_df, BCN_CACHE = None, None
if not BCN_AVAILABLE:
    print("BCN20000 not attached: external validation (Section 22A) will be skipped. Attach pasutchien/bcn20000.")
else:
    raw = pd.read_csv(BCN_CSV)
    root = BCN_CSV.parent
    lookup = {}
    for pth in glob.glob(str(root / "**" / "*.*"), recursive=True):
        if pth.lower().endswith((".jpg", ".jpeg", ".png")):
            lookup.setdefault(Path(pth).stem, pth)
    b = pd.DataFrame({"image_id": raw.bcn_filename.astype(str).map(lambda s: Path(s).stem)})
    b["filepath"] = b.image_id.map(lookup)
    b["diagnosis_raw"] = raw.diagnosis.astype(str).str.upper().str.strip()
    b["dx"] = b.diagnosis_raw.map(CFG["EXT_CLASS_MAP"])
    b["site_raw"] = raw.anatom_site_general.astype(str).str.lower().str.strip()
    b["localization"] = b.site_raw.map(CFG["EXT_SITE_MAP"]).fillna("unknown")
    b["age"] = pd.to_numeric(raw.age_approx, errors="coerce")
    b["sex"] = raw.sex.astype(str).str.lower().str.strip()
    b["lesion_id"] = raw.lesion_id.astype(str).where(raw.lesion_id.notna(), b.image_id)
    ext_audit.update(n_rows_csv=len(raw), missing_image_files=int(b.filepath.isna().sum()),
                     diagnosis_values=b.diagnosis_raw.value_counts().to_dict(),
                     unmapped_diagnoses=b.loc[b.dx.isna(), "diagnosis_raw"].value_counts().to_dict(),
                     site_values=b.site_raw.value_counts().to_dict(),
                     sites_mapped_to_unknown=b.loc[b.localization == "unknown", "site_raw"].value_counts().to_dict(),
                     sex_values=b.sex.value_counts().to_dict(), age_missing=int(b.age.isna().sum()))
    b = b.dropna(subset=["filepath", "dx"])
    ham_overlap = b.image_id.isin(set(df.image_id))
    ext_audit["excluded_ids_shared_with_ham10000"] = int(ham_overlap.sum())
    b = b[~ham_overlap]
    if SMOKE_TEST and not SYNTHETIC:
        _rng, _keep = np.random.default_rng(SEED), []
        for _c, _g in b.groupby("dx"):
            _les = _g.lesion_id.drop_duplicates().values.copy(); _rng.shuffle(_les)
            _keep += list(_les[:CFG["SMOKE_BCN_PER_CLASS"]])
        b = b[b.lesion_id.isin(_keep)]
    b = b.reset_index(drop=True)
    b["label"] = b.dx.map({c: i for i, c in enumerate(CLASS_NAMES)}).fillna(-1).astype(int)   # SCC_OOD -> -1
    b["row_idx"] = np.arange(len(b))
    bcn_df = b
    ext_audit.update(n_used=len(b), n_7class=int((b.label >= 0).sum()), n_scc_ood=int((b.label < 0).sum()),
                     n_lesions=int(b.lesion_id.nunique()), class_counts=b.dx.value_counts().to_dict())
    bcn_df.to_csv(DIRS["data"] / "bcn_external.csv", index=False)

    if not WORKER:
        desc_bcn = describe_dataset(b.assign(sex=b.sex.where(b.sex.isin(["male", "female"]), "unknown")), conf_col=None,
                                    classes=CLASS_NAMES + ["SCC_OOD"])
        desc_bcn.round(1).to_csv(DIRS["results"] / "dataset_description_bcn20000.csv", index=False)
        display(desc_bcn.round(1))

    # ---- image cache (same deterministic resize as HAM10000) ----
    BCN_CACHE_PATH = DIRS["data"] / f"bcn_cache_{IMG_SIZE}.npy"; BCN_CACHE_META = DIRS["data"] / f"bcn_cache_{IMG_SIZE}.json"
    ok = False
    if BCN_CACHE_PATH.exists() and BCN_CACHE_META.exists():
        mc = json.load(open(BCN_CACHE_META))
        ok = mc.get("n") == len(b) and mc.get("image_ids_md5") == hashlib.md5("".join(b.image_id).encode()).hexdigest()
    if not ok:
        build_image_cache(b, BCN_CACHE_PATH, BCN_CACHE_META)
    BCN_CACHE = np.load(BCN_CACHE_PATH, mmap_mode="r")

    if not WORKER:
      # ---- cross-dataset near-duplicate screen (dHash, Hamming <= 2) ----
      def dhash_array(arr, bs=512):
          out = np.zeros(len(arr), dtype=np.uint64)
          for s in range(0, len(arr), bs):
              g = tf.image.resize(tf.image.rgb_to_grayscale(tf.cast(np.asarray(arr[s:s + bs]), tf.float32)), [8, 9], antialias=True).numpy()[..., 0]
              out[s:s + len(g)] = np.packbits((g[:, :, 1:] > g[:, :, :-1]).reshape(len(g), 64), axis=1).view(">u8").ravel().astype(np.uint64)
          return out
      POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
      h_bcn = dhash_array(BCN_CACHE)
      h_ham = HASHES if "HASHES" in globals() else dhash_array(IMAGE_CACHE)
      near = []
      for i, h in enumerate(h_bcn):
          dist = POP8[np.bitwise_xor(h_ham, h).view(np.uint8).reshape(-1, 8)].sum(1)
          for j in np.where(dist <= 2)[0]:
              near.append((b.image_id.iat[i], df.image_id.iat[j], int(dist[j]), b.dx.iat[i], df.dx.iat[j]))
      near = pd.DataFrame(near, columns=["bcn_image", "ham_image", "hamming", "bcn_dx", "ham_dx"])
      if len(near):
          _bi = {v: i for i, v in enumerate(b.image_id)}; _hi = {v: i for i, v in enumerate(df.image_id)}
          ia_, ib_ = near.bcn_image.map(_bi).values, near.ham_image.map(_hi).values
          near["pixel_corr"] = pair_pixel_corr(BCN_CACHE, ia_, IMAGE_CACHE, ib_); near["likely_duplicate"] = near.pixel_corr >= CORR_DUP
          o_ = np.argsort(-near.pixel_corr.values)[:8]
          pair_montage(BCN_CACHE, ia_[o_], IMAGE_CACHE, ib_[o_], [f"r={near.pixel_corr.iat[k]:.2f}" for k in o_],
                       "fig_bcn_ham_duplicate_candidates", "Most similar BCN20000 (top) vs HAM10000 (bottom) pairs — for manual review")
      ext_audit["bcn_ham_pairs_pixel_corr_ge_0.95"] = int(near.likely_duplicate.sum()) if len(near) else 0
      near.to_csv(DIRS["results"] / "bcn_ham_near_duplicates.csv", index=False)
      ext_audit["near_duplicates_bcn_vs_ham_hamming_le2"] = len(near)
      print(f"BCN20000 used: {len(b)} images ({ext_audit['n_7class']} in the 7 HAM classes, {ext_audit['n_scc_ood']} SCC), "
            f"{ext_audit['n_lesions']} lesions | IDs shared with HAM: {ext_audit['excluded_ids_shared_with_ham10000']} | "
            f"dHash candidates: {len(near)}, confirmed (r >= {CORR_DUP}): {ext_audit['bcn_ham_pairs_pixel_corr_ge_0.95']}")
      print("Unmapped diagnoses:", ext_audit["unmapped_diagnoses"], "| sites mapped to 'unknown':", ext_audit["sites_mapped_to_unknown"])
with open(DIRS["results"] / "external_audit.json", "w") as f:
    json.dump(ext_audit, f, indent=2, default=str)

# [RQ-05][RQ-04] Fold-wise metadata preprocessing
from sklearn.preprocessing import MinMaxScaler

SEX_CATS = ["male", "female", "unknown"]

def _clean_sex(s):
    s = s.fillna("unknown").astype(str).str.lower().str.strip()
    return s.where(s.isin(["male", "female"]), "unknown")

def _clean_loc(s):
    return s.fillna("unknown").astype(str).str.lower().str.strip()

META_FEATURES = ("age", "sex", "site")

class MetaPreprocessor:
    """Fitted on a training fold only; transform() is applied to train, val, test and external data.
    `features` selects a subset of (age, sex, site) for the leave-one-feature-out arms [R7-3]."""
    def __init__(self, loc_encoding="onehot", features=META_FEATURES):
        self.loc_encoding = loc_encoding
        self.features = tuple(features)

    def fit(self, train):
        self.age_median = float(train.age.median())
        self.scaler = MinMaxScaler().fit(train.age.fillna(self.age_median).values.reshape(-1, 1))
        seen = sorted(set(_clean_loc(train.localization)) - {"unknown"})
        self.loc_vocab = seen + ["unknown"]
        self.loc_index = {v: i for i, v in enumerate(self.loc_vocab)}
        self.n_fit_rows = len(train)
        return self

    @property
    def dense_dim(self):   # dimension of the float metadata vector fed to the network
        return len(self.feature_names())

    def feature_groups(self):
        """Column indices of each feature group in the dense vector (used by SHAP / permutation importance)."""
        names, groups = self.feature_names(), {}
        for i, n in enumerate(names):
            g = "age" if n.startswith("age") else ("sex" if n.startswith("sex=") else "site")
            groups.setdefault(g, []).append(i)
        return groups

    def to_json(self):
        return dict(loc_encoding=self.loc_encoding, features=list(self.features), age_median=self.age_median,
                    age_min=float(self.scaler.data_min_[0]), age_max=float(self.scaler.data_max_[0]),
                    sex_categories=SEX_CATS, loc_vocab=self.loc_vocab, feature_order=self.feature_names())

    def transform(self, frame):
        age = self.scaler.transform(frame.age.fillna(self.age_median).values.reshape(-1, 1)).astype(np.float32)
        sex = _clean_sex(frame.sex)
        sex_oh = np.stack([(sex == c).values for c in SEX_CATS], 1).astype(np.float32)
        loc = _clean_loc(frame.localization).map(lambda v: self.loc_index.get(v, self.loc_index["unknown"])).values.astype(np.int32)
        parts = []
        if "age" in self.features: parts.append(age)
        if "sex" in self.features: parts.append(sex_oh)
        if "site" in self.features and self.loc_encoding == "onehot":
            parts.append(np.eye(len(self.loc_vocab), dtype=np.float32)[loc])
        dense = np.concatenate(parts, 1) if parts else np.zeros((len(frame), 0), np.float32)
        return dense.astype(np.float32), loc.reshape(-1, 1)

    def feature_names(self):
        names = []
        if "age" in self.features: names.append("age_minmax")
        if "sex" in self.features: names += [f"sex={c}" for c in SEX_CATS]
        if "site" in self.features and self.loc_encoding == "onehot": names += [f"loc={v}" for v in self.loc_vocab]
        return names

spec_rows = []
for k in range(1, CFG["N_FOLDS"] + 1):
    tr, va, te = get_fold_frames(k)
    mp = MetaPreprocessor(CFG["LOC_ENCODING"]).fit(tr)
    # leakage assertions: statistics equal those of the training fold alone
    assert np.isclose(mp.age_median, tr.age.median())
    imputed = tr.age.fillna(tr.age.median())
    assert np.isclose(mp.scaler.data_min_[0], imputed.min()) and np.isclose(mp.scaler.data_max_[0], imputed.max())
    assert mp.n_fit_rows == len(tr)
    d_va, _ = mp.transform(va); d_te, _ = mp.transform(te)
    spec_rows.append(dict(fold=k, age_median=mp.age_median, age_min=mp.scaler.data_min_[0], age_max=mp.scaler.data_max_[0],
                          n_loc_vocab=len(mp.loc_vocab), meta_dim=mp.dense_dim,
                          val_age_outside_0_1=int(((d_va[:, 0] < 0) | (d_va[:, 0] > 1)).sum()),
                          test_age_outside_0_1=int(((d_te[:, 0] < 0) | (d_te[:, 0] > 1)).sum()),
                          test_unseen_loc=int((~_clean_loc(te.localization).isin(mp.loc_vocab)).sum())))
meta_spec = pd.DataFrame(spec_rows)
meta_spec.to_csv(DIRS["results"] / "metadata_spec.csv", index=False)
display(meta_spec)
print("Metadata feature order:", MetaPreprocessor(CFG["LOC_ENCODING"]).fit(get_fold_frames(1)[0]).feature_names())

# [RQ-07] Imbalance components
from sklearn.utils.class_weight import compute_class_weight

def smart_oversample(train_fold, target, max_x, random_state=42):
    """Original SkinNet balancing (unchanged): cap = min(target, max_x * n); majority downsampled."""
    parts, reports = [], []
    for c in range(N_CLASSES):
        cls_df = train_fold[train_fold.label == c]
        n = len(cls_df)
        if n == 0:
            continue
        eff = min(target, n * max_x)
        if n >= target:
            out, action = cls_df.sample(n=target, replace=False, random_state=random_state), "downsample"
        elif n < eff:
            extra = cls_df.sample(n=eff - n, replace=True, random_state=random_state)
            out, action = pd.concat([cls_df, extra], ignore_index=True), f"oversample {eff / n:.1f}x"
        else:
            out, action = cls_df, "keep"
        parts.append(out)
        reports.append(dict(cls=CLASS_NAMES[c], original=n, effective_target=eff, final=len(out), action=action))
    out_df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out_df, pd.DataFrame(reports)

def fold_class_weights(train_fold):
    return compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=train_fold.label.values).astype(np.float32)

rows = []
for k in range(1, CFG["N_FOLDS"] + 1):
    tr, _, _ = get_fold_frames(k)
    rs, rep = smart_oversample(tr, CFG["TARGET_PER_CLASS"], CFG["MAX_OVERSAMPLE_X"], SEED)
    assert set(rs.image_id) <= set(tr.image_id), "Resampled rows must come from the training fold only"
    cw = fold_class_weights(tr)
    rep["fold"] = k; rep["class_weight"] = cw[[CLASS_NAMES.index(c) for c in rep.cls]]
    rep["effective_mass_resample_plus_cw"] = rep.final * rep.class_weight
    rep["total_after_balancing"] = len(rs)
    rows.append(rep)
balance = pd.concat(rows, ignore_index=True)
balance.to_csv(DIRS["results"] / "imbalance_report.csv", index=False)
display(balance[balance.fold == 1].round(3))
print("Total after balancing per fold:", balance.groupby("fold").total_after_balancing.first().tolist(),
      "(original notebook: [12785, 12775, 12780, 12780, 12780])")

b1 = balance[balance.fold == 1].set_index("cls").reindex(CLASS_NAMES)
fig, axs = plt.subplots(1, 3, figsize=(FIG_W2, 2.3))
for ax, col, title in zip(axs, ["original", "final", "effective_mass_resample_plus_cw"],
                          ["(a) Training fold (original)", "(b) After capped resampling", "(c) Resampled × class weight"]):
    ax.bar(CLASS_NAMES, b1[col], color=[CLASS_COLORS[c] for c in CLASS_NAMES], edgecolor="black", linewidth=0.3)
    ax.set_title(title); ax.tick_params(axis="x", rotation=45)
axs[0].set_ylabel("Images / effective mass")
axs[1].axhline(CFG["TARGET_PER_CLASS"], ls="--", lw=0.8, color="black")
axs[1].text(6.4, CFG["TARGET_PER_CLASS"] * 1.02, "target", ha="right", fontsize=6)
fig.tight_layout()
save_fig(fig, "fig_imbalance_strategy")

# [RQ-06][RQ-07][RQ-08][RQ-12] Augmentation, mixing and datasets
AUTOTUNE = tf.data.AUTOTUNE
MINORITY = tf.constant(CFG["MINORITY_IDX"], dtype=tf.int32)
S = IMG_SIZE

def load_image(src):
    if IMAGE_CACHE is not None:
        img = tf.numpy_function(lambda i: np.asarray(IMAGE_CACHE[int(i)]), [src], tf.uint8)
        img.set_shape([S, S, 3])
        return tf.cast(img, tf.float32)
    img = tf.image.decode_jpeg(tf.io.read_file(src), channels=3)
    return tf.cast(tf.image.resize(img, [S, S]), tf.float32)      # original parse_image, range [0,255]

def cutout(img, pad_size, prob=0.5):
    def apply():
        cy = tf.random.uniform([], 0, S, dtype=tf.int32); cx = tf.random.uniform([], 0, S, dtype=tf.int32)
        y1, y2 = tf.maximum(0, cy - pad_size), tf.minimum(S, cy + pad_size)
        x1, x2 = tf.maximum(0, cx - pad_size), tf.minimum(S, cx + pad_size)
        mask = 1.0 - tf.pad(tf.ones([y2 - y1, x2 - x1, 3], img.dtype), [[y1, S - y2], [x1, S - x2], [0, 0]])
        return img * mask
    return tf.cond(tf.random.uniform([]) < prob, apply, lambda: img)

def standard_aug(img):
    img = tf.image.random_flip_left_right(img); img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, 25.5); img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2); img = tf.image.random_hue(img, 0.05)
    crop = tf.cast(tf.random.uniform([], 0.85, 1.0) * S, tf.int32)
    img = tf.image.resize(tf.image.random_crop(img, [crop, crop, 3]), [S, S])
    img = cutout(img, pad_size=max(1, int(round(40 * S / 300))), prob=0.5)
    return tf.clip_by_value(img, 0.0, 255.0)

def strong_aug(img):
    img = standard_aug(img)
    img = tf.image.random_contrast(img, 0.7, 1.4); img = tf.image.random_brightness(img, 40.0)
    img = cutout(img, pad_size=max(1, int(round(20 * S / 300))), prob=0.4)
    return tf.clip_by_value(img, 0.0, 255.0)

def sample_beta(a, b, shape):
    x = tf.random.gamma(shape, a); y = tf.random.gamma(shape, b)
    return x / (x + y + 1e-12)

def cutmix_batch(img, y, meta, loc, prob, alpha):
    def apply():
        bs = tf.shape(img)[0]
        u = sample_beta(alpha, alpha, [])                         # alpha=1 -> U(0,1) (original)
        cut = tf.cast(S * tf.sqrt(1.0 - u), tf.int32)
        cx = tf.random.uniform([], 0, S, dtype=tf.int32); cy = tf.random.uniform([], 0, S, dtype=tf.int32)
        x1 = tf.clip_by_value(cx - cut // 2, 0, S); x2 = tf.clip_by_value(cx + cut // 2, 0, S)
        y1 = tf.clip_by_value(cy - cut // 2, 0, S); y2 = tf.clip_by_value(cy + cut // 2, 0, S)
        mask = tf.pad(tf.ones([y2 - y1, x2 - x1, 1], img.dtype), [[y1, S - y2], [x1, S - x2], [0, 0]])
        perm = tf.random.shuffle(tf.range(bs))
        lam = 1.0 - tf.cast((y2 - y1) * (x2 - x1), tf.float32) / float(S * S)   # exact area ratio
        img_m = img * (1.0 - mask) + tf.gather(img, perm) * mask
        y_m = lam * y + (1.0 - lam) * tf.gather(y, perm)
        meta_m = lam * meta + (1.0 - lam) * tf.gather(meta, perm)
        loc_m = tf.where(lam >= 0.5, loc, tf.gather(loc, perm))
        return img_m, y_m, meta_m, loc_m
    return tf.cond(tf.random.uniform([]) < prob, apply, lambda: (img, y, meta, loc))

def mixup_batch(img, y, meta, loc, prob, alpha, mode):
    def apply():
        bs = tf.shape(img)[0]
        if mode == "beta":
            lam = sample_beta(alpha, alpha, [bs]); lam = tf.maximum(lam, 1.0 - lam)
        else:   # legacy rule of the original notebook
            lam = tf.maximum(tf.random.uniform([bs]), 1.0 - tf.random.uniform([bs]))
        perm = tf.random.shuffle(tf.range(bs))
        li, lv = tf.reshape(lam, [-1, 1, 1, 1]), tf.reshape(lam, [-1, 1])
        return (li * img + (1 - li) * tf.gather(img, perm), lv * y + (1 - lv) * tf.gather(y, perm),
                lv * meta + (1 - lv) * tf.gather(meta, perm), loc)          # lam>=0.5: own site index dominates
    return tf.cond(tf.random.uniform([]) < prob, apply, lambda: (img, y, meta, loc))

def _to_inputs(exp):
    def f(img, y, meta, loc):
        x = {}
        if exp.use_image:
            x["image_input"] = img
        if exp.use_meta:
            x["meta_input"] = meta
            if CFG["LOC_ENCODING"] == "embedding":
                x["loc_index_input"] = loc
        return x, y
    return f

def _source(frame):
    return frame.row_idx.values.astype(np.int64) if IMAGE_CACHE is not None else frame.filepath.values

def make_train_dataset(frame, meta_dense, loc_idx, exp, apply_mix, seed):
    y = np.eye(N_CLASSES, dtype=np.float32)[frame.label.values]
    ds = tf.data.Dataset.from_tensor_slices((_source(frame), y, meta_dense, loc_idx))
    ds = ds.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True).repeat()
    minority_aug = exp.minority_aug
    def load_aug(src, yy, m, l):
        if not exp.use_image:
            return tf.zeros([1, 1, 3]), yy, m, l
        img = load_image(src)
        if minority_aug:
            is_min = tf.reduce_any(tf.equal(tf.cast(tf.argmax(yy), tf.int32), MINORITY))
            img = tf.cond(is_min, lambda: strong_aug(img), lambda: standard_aug(img))
        else:
            img = standard_aug(img)
        return img, yy, m, l
    ds = ds.map(load_aug, num_parallel_calls=AUTOTUNE).batch(getattr(exp, "batch_size", 0) or BATCH_SIZE, drop_remainder=True)
    mm = getattr(exp, "mix_mode", "both")          # "both" (default, all original arms) | "cutmix" | "mixup" [R7-4]
    if apply_mix and exp.use_image and mm in ("both", "cutmix"):
        ds = ds.map(lambda i, yy, m, l: cutmix_batch(i, yy, m, l, CFG["CUTMIX_PROB"], CFG["CUTMIX_ALPHA"]), num_parallel_calls=AUTOTUNE)
    if apply_mix and exp.use_image and mm in ("both", "mixup"):
        ds = ds.map(lambda i, yy, m, l: mixup_batch(i, yy, m, l, CFG["MIXUP_PROB"], CFG["MIXUP_ALPHA"], CFG["MIXUP_LAMBDA_MODE"]), num_parallel_calls=AUTOTUNE)
    return ds.map(_to_inputs(exp), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

def tta_transform(img, view):
    if CFG["TTA_MODE"] == "deterministic":
        return [lambda x: x, tf.image.flip_left_right, tf.image.flip_up_down,
                lambda x: tf.image.rot90(x, 1), lambda x: tf.image.rot90(x, 3)][view](img)
    if view == 0:      # legacy: clean pass + random photometric/geometric passes
        return img
    img = tf.image.random_flip_left_right(img); img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, 20.0); img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 255.0)

def make_eval_dataset(frame, meta_dense, loc_idx, exp, tta_view=None, cache=None):
    """Validation/test/external pipeline: no shuffling, resampling, augmentation or mixing. TTA only here.
    `cache`: another uint8 image array (e.g. BCN20000) indexed by frame.row_idx."""
    y = np.eye(N_CLASSES, dtype=np.float32)[np.clip(frame.label.values.astype(int), 0, None)]   # external OOD rows: label -1
    if cache is None:
        src_arr, loader = _source(frame), load_image
    else:
        src_arr = frame.row_idx.values.astype(np.int64)
        def loader(src):
            img = tf.numpy_function(lambda i: np.asarray(cache[int(i)]), [src], tf.uint8)
            img.set_shape([S, S, 3])
            return tf.cast(img, tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((src_arr, y, meta_dense, loc_idx))
    def load(src, yy, m, l):
        if not exp.use_image:
            return tf.zeros([1, 1, 3]), yy, m, l
        img = loader(src)
        if tta_view is not None:
            img = tta_transform(img, tta_view)
        return img, yy, m, l
    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE * 2)
    return ds.map(_to_inputs(exp), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ---------------- Unit tests: alignment of image, label and metadata ----------------
def _alignment_test(n=16, trials=20):
    """Encode the sample identity k in all three modalities; after mixing they must still agree."""
    for t in range(trials):
        k = np.arange(n) % N_CLASSES
        y = np.eye(N_CLASSES, dtype=np.float32)[k]
        meta = np.concatenate([np.zeros((n, 4), np.float32), y], 1)            # "site one-hot" == label
        img = np.broadcast_to(k[:, None, None, None].astype(np.float32), (n, S, S, 3)).copy()
        loc = k.reshape(-1, 1).astype(np.int32)
        a = [tf.constant(v) for v in (img, y, meta, loc)]
        a = cutmix_batch(*a, prob=1.0, alpha=1.0)
        a = mixup_batch(*a, prob=1.0, alpha=0.2, mode="beta")
        im, yy, mm, _ = [v.numpy() for v in a]
        assert np.allclose(mm[:, 4:], yy, atol=1e-5), "metadata not aligned with labels"
        assert np.allclose(im.mean(axis=(1, 2, 3)), yy @ np.arange(N_CLASSES), atol=1e-3), "image not aligned with labels"
        assert np.allclose(yy.sum(1), 1.0, atol=1e-5)
    return True
print("MixUp/CutMix alignment test:", "PASS" if _alignment_test() else "FAIL")

lam_beta = np.maximum(*(lambda b: (b, 1 - b))(np.random.default_rng(0).beta(CFG["MIXUP_ALPHA"], CFG["MIXUP_ALPHA"], 100000)))
lam_legacy = np.maximum(np.random.default_rng(1).random(100000), 1 - np.random.default_rng(2).random(100000))
print(f"MixUp lambda mean: beta mode = {lam_beta.mean():.3f}, legacy rule = {lam_legacy.mean():.3f}")

# [RQ-07][RQ-06] Unified loss + LR schedule
@keras.saving.register_keras_serializable(package="skinnet")
class SkinNetLoss(keras.losses.Loss):
    def __init__(self, gamma=0.0, alpha=1.0, label_smoothing=0.0, class_weights=None,
                 weight_mode="per_sample", name="skinnet_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma, self.alpha, self.label_smoothing = float(gamma), float(alpha), float(label_smoothing)
        self.class_weights = None if class_weights is None else np.asarray(class_weights, np.float32)
        self.weight_mode = weight_mode

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
        k = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_s = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / k
        p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        term = -y_s * tf.math.log(p)
        if self.gamma > 0:
            term = term * tf.pow(1.0 - p, self.gamma)
        per_sample = self.alpha * tf.reduce_sum(term, axis=-1)
        if self.class_weights is not None:
            w = tf.reduce_sum(y_true * tf.constant(self.class_weights), axis=-1)   # mixed targets -> mixed weights
            if self.weight_mode == "legacy_batch_scalar":
                return tf.reduce_mean(per_sample) * tf.reduce_mean(w)
            per_sample = per_sample * w
        return per_sample

    def get_config(self):
        cfg = super().get_config()
        cfg.update(gamma=self.gamma, alpha=self.alpha, label_smoothing=self.label_smoothing, weight_mode=self.weight_mode,
                   class_weights=None if self.class_weights is None else self.class_weights.tolist())
        return cfg

LOSS_SETTINGS = {
    "ce":       dict(gamma=0.0, alpha=1.0, label_smoothing=0.0),
    "focal_ls": dict(gamma=CFG["FOCAL_GAMMA"], alpha=CFG["FOCAL_ALPHA"], label_smoothing=CFG["LABEL_SMOOTHING"]),
    "focal":    dict(gamma=CFG["FOCAL_GAMMA"], alpha=CFG["FOCAL_ALPHA"], label_smoothing=0.0),
    "ce_ls":    dict(gamma=0.0, alpha=1.0, label_smoothing=CFG["LABEL_SMOOTHING"]),
}

@keras.saving.register_keras_serializable(package="skinnet")
class CosineWarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr=1e-7):
        self.peak_lr, self.warmup_steps = float(peak_lr), float(warmup_steps)
        self.total_steps, self.min_lr = float(total_steps), float(min_lr)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.peak_lr * step / tf.maximum(self.warmup_steps, 1.0)
        prog = tf.clip_by_value((step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0), 0.0, 1.0)
        cos = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + tf.cos(math.pi * prog))
        return tf.where(step < self.warmup_steps, warm, cos)
    def get_config(self):
        return dict(peak_lr=self.peak_lr, warmup_steps=self.warmup_steps, total_steps=self.total_steps, min_lr=self.min_lr)

# ---------------- unit tests ----------------
_rng = np.random.default_rng(0)
_y = np.eye(N_CLASSES, dtype=np.float32)[_rng.integers(0, N_CLASSES, 32)]
_p = tf.nn.softmax(tf.constant(_rng.normal(size=(32, N_CLASSES)), tf.float32)).numpy()
_cw = np.linspace(0.2, 5, N_CLASSES).astype(np.float32)
ce_ref = keras.losses.CategoricalCrossentropy()(_y, _p).numpy()
assert np.isclose(SkinNetLoss()(_y, _p).numpy(), ce_ref, atol=1e-5), "CE reproduction failed"
manual = np.mean((_y @ _cw) * -np.log(np.clip((_y * _p).sum(1), 1e-7, 1)))
assert np.isclose(SkinNetLoss(class_weights=_cw)(_y, _p).numpy(), manual, atol=1e-4), "per-sample weighting failed"
legacy = SkinNetLoss(class_weights=_cw, weight_mode="legacy_batch_scalar")(_y, _p).numpy()
lam = 0.7; _y2 = _y[::-1]
fl = SkinNetLoss(**LOSS_SETTINGS["focal_ls"])
lin = fl(lam * _y + (1 - lam) * _y2, _p).numpy() - (lam * fl(_y, _p).numpy() + (1 - lam) * fl(_y2, _p).numpy())
assert abs(lin) < 1e-5, "loss not linear in target"
print(f"Loss tests PASS | CE={ce_ref:.4f} | per-sample weighted={manual:.4f} vs legacy batch-scalar={legacy:.4f}")

# [RQ-01][RQ-05][RQ-10] Model builder
@keras.saving.register_keras_serializable(package="skinnet")
class ChannelAttention(layers.Layer):
    def __init__(self, channels, ratio=8, **kw):
        super().__init__(**kw)
        self.channels, self.ratio = channels, ratio
        self.d1 = layers.Dense(max(1, channels // ratio), activation="relu", use_bias=False)
        self.d2 = layers.Dense(channels, use_bias=False)
        self.gap, self.gmp = layers.GlobalAveragePooling2D(), layers.GlobalMaxPooling2D()
    def call(self, x):
        attn = keras.ops.sigmoid(self.d2(self.d1(self.gap(x))) + self.d2(self.d1(self.gmp(x))))
        return x * keras.ops.reshape(keras.ops.cast(attn, x.dtype), (-1, 1, 1, self.channels))
    def get_config(self):
        return {**super().get_config(), "channels": self.channels, "ratio": self.ratio}

@keras.saving.register_keras_serializable(package="skinnet")
class SpatialAttention(layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.conv = layers.Conv2D(1, 7, padding="same", activation="sigmoid", use_bias=False, dtype="float32")
    def attention_map(self, x):
        pooled = keras.ops.concatenate([keras.ops.cast(keras.ops.mean(x, -1, keepdims=True), "float32"),
                                        keras.ops.cast(keras.ops.max(x, -1, keepdims=True), "float32")], -1)
        return self.conv(pooled)
    def call(self, x):
        return keras.ops.cast(keras.ops.cast(x, "float32") * self.attention_map(x), x.dtype)

def refinement_block(x, out_ch):
    out = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="refine_dw")(x)
    out = layers.Activation("relu", name="refine_relu1")(layers.BatchNormalization(name="refine_bn1")(out))
    out = layers.Conv2D(out_ch, 1, padding="same", use_bias=False, name="refine_pw")(out)
    out = layers.Activation("relu", name="refine_relu2")(layers.BatchNormalization(name="refine_bn2")(out))
    if x.shape[-1] != out_ch:
        x = layers.BatchNormalization(name="refine_proj_bn")(layers.Conv2D(out_ch, 1, padding="same", use_bias=False, name="refine_proj")(x))
    return layers.Add(name="refine_out")([out, x])

def get_backbone(weights):
    try:
        return keras.applications.EfficientNetB3(include_top=False, weights=weights, input_shape=(S, S, 3))
    except Exception as e:
        path = CFG.get("BACKBONE_WEIGHTS_PATH")
        if weights == "imagenet" and path:
            bb = keras.applications.EfficientNetB3(include_top=False, weights=None, input_shape=(S, S, 3))
            bb.load_weights(path); return bb
        raise RuntimeError("Could not obtain ImageNet weights. Turn Kaggle Internet ON or set BACKBONE_WEIGHTS_PATH.") from e

@keras.saving.register_keras_serializable(package="skinnet")
class InputNormalize(layers.Layer):
    """Backbone-specific input normalisation from RGB [0, 255] (float32 arithmetic)."""
    def __init__(self, mode="torch", scale=None, offset=None, **kw):
        kw.setdefault("dtype", "float32")
        super().__init__(**kw)
        self.mode, self.scale, self.offset = mode, scale, offset
    def call(self, x):
        x = keras.ops.cast(x, "float32")
        if self.mode == "caffe":                     # ResNet50: RGB->BGR, ImageNet mean subtraction
            x = keras.ops.flip(x, axis=-1) - keras.ops.convert_to_tensor([103.939, 116.779, 123.68], "float32")
        elif self.mode == "torch":                   # DenseNet121: [0,1] then ImageNet mean/std
            x = (x / 255.0 - keras.ops.convert_to_tensor([0.485, 0.456, 0.406], "float32")) / keras.ops.convert_to_tensor([0.229, 0.224, 0.225], "float32")
        elif self.mode == "scale_offset":            # ViT: preset image-converter scale/offset
            x = x * keras.ops.convert_to_tensor(np.asarray(self.scale, "float32")) + keras.ops.convert_to_tensor(np.asarray(self.offset, "float32"))
        return x
    def get_config(self):
        return {**super().get_config(), "mode": self.mode, "scale": self.scale, "offset": self.offset}

@keras.saving.register_keras_serializable(package="skinnet")
class ClassToken(layers.Layer):
    def call(self, x):
        return x[:, 0, :]

BACKBONES = {   # name: (constructor, input normalisation, output kind, display name)
    "effb3": (None, None, "map", "EfficientNetB3"),
    "resnet50": (lambda **k: keras.applications.ResNet50(**k), "caffe", "map", "ResNet50"),
    "densenet121": (lambda **k: keras.applications.DenseNet121(**k), "torch", "map", "DenseNet121"),
    "convnext_tiny": (lambda **k: keras.applications.ConvNeXtTiny(include_preprocessing=True, **k), None, "map", "ConvNeXt-Tiny"),
    "effv2s": (lambda **k: keras.applications.EfficientNetV2S(include_preprocessing=True, **k), None, "map", "EfficientNetV2-S"),
    "vit_b16": (None, "scale_offset", "vec", "ViT-B/16"),
}
VIT_NORM = {}
_VIT_SRC = []

def vit_preset_source():
    """Local copy of the KerasHub ViT preset attached as a Kaggle *Model* input (Add Input -> Models -> keras/vit ->
    vit_base_patch16_224_imagenet). Kaggle refuses to attach models during a committed (non-interactive) run, so the preset
    must be attached beforehand; the directory is found automatically under /kaggle/input."""
    if _VIT_SRC:
        return _VIT_SRC[0]
    if (VIT_DIR / "config.json").exists():               # path set in Section 2
        _VIT_SRC.append(str(VIT_DIR)); return _VIT_SRC[0]
    hits = []
    for pat in ["/kaggle/input/*/*/*/*/config.json", "/kaggle/input/*/*/*/*/*/config.json", "/kaggle/input/*/*/*/config.json",
                "/kaggle/input/*/*/*/*/*/*/config.json"]:
        hits += [Path(h).parent for h in glob.glob(pat) if CFG["VIT_PRESET"] in h]
    if hits:
        _VIT_SRC.append(str(sorted(hits)[-1]))          # highest version if several are attached
    else:
        _VIT_SRC.append(CFG["VIT_PRESET"])               # online preset: works only in interactive sessions
    return _VIT_SRC[0]

def get_comparator_backbone(name, weights):
    """Returns (backbone model named 'backbone', normalisation layer or None, output kind)."""
    ctor, norm, kind, _ = BACKBONES[name]
    if name == "vit_b16":
        import keras_hub
        vs = CFG["VIT_SIZE"]
        if not SYNTHETIC:                           # preset architecture; weights loaded only for training
            core = keras_hub.models.ViTBackbone.from_preset(vit_preset_source(), load_weights=(weights == "imagenet"))
            if not VIT_NORM:
                try:
                    conv = keras_hub.layers.ViTImageConverter.from_preset(vit_preset_source())
                    VIT_NORM.update(scale=np.asarray(conv.scale, "float32").tolist(), offset=np.asarray(conv.offset, "float32").tolist(), source="preset image converter")
                except Exception as e:
                    VIT_NORM.update(scale=1 / 127.5, offset=-1.0,
                                    source=f"[-1, 1], as in the google/vit-base-patch16-224 preprocessing config (converter unavailable: {type(e).__name__})")
                print("ViT input normalisation:", VIT_NORM)
        else:
            core = keras_hub.models.ViTBackbone(image_shape=(vs, vs, 3), patch_size=16 if vs % 16 == 0 else 8, num_layers=2,
                                               num_heads=2, hidden_dim=32, mlp_dim=64)
            VIT_NORM.update(scale=1 / 127.5, offset=-1.0, source="random-init check")
        inp = keras.Input((S, S, 3), dtype="float32")
        x = layers.Resizing(vs, vs, dtype="float32")(inp)
        x = InputNormalize("scale_offset", VIT_NORM["scale"], VIT_NORM["offset"])(x)
        return keras.Model(inp, ClassToken()(core(x)), name="backbone"), None, "vec"
    bb = ctor(include_top=False, weights=weights, input_shape=(S, S, 3))     # keep default names (EfficientNetV2 uses `name` internally)
    return bb, (InputNormalize(norm, name="input_normalize") if norm else None), kind

def build_skinnet(exp, meta_dim, n_loc=None, weights="cfg"):
    """weights: "cfg" -> CFG['BACKBONE_WEIGHTS'] (training); None -> random init (only for loading trained weights)."""
    weights = CFG["BACKBONE_WEIGHTS"] if weights == "cfg" else weights
    reg = keras.regularizers.l2(CFG["L2"])
    inputs, branches = {}, []
    if exp.use_image:
        img_in = keras.Input((S, S, 3), name="image_input", dtype="float32"); inputs["image_input"] = img_in
        if exp.backbone == "effb3":
            backbone = get_backbone(weights)
            backbone.trainable = False
            x = backbone(img_in, training=False)
            x = refinement_block(x, CFG["REFINE_CH"])
            if exp.use_cbam:
                x = ChannelAttention(x.shape[-1], CFG["CBAM_RATIO"], name="channel_attention")(x)
                x = SpatialAttention(name="spatial_attention")(x)
            feat = layers.GlobalAveragePooling2D(name="gap")(x)
        else:   # [R1-6] benchmark: comparator backbone + the same metadata fusion head and training recipe
            backbone, norm, kind = get_comparator_backbone(exp.backbone, weights)
            backbone.trainable = False
            x = norm(img_in) if norm is not None else img_in
            x = backbone(x, training=False)
            feat = layers.GlobalAveragePooling2D(name="gap")(x) if kind == "map" else x
        branches.append(layers.Activation("linear", dtype="float32", name="img_features")(feat))
    if exp.use_meta:
        meta_in = keras.Input((meta_dim,), name="meta_input", dtype="float32"); inputs["meta_input"] = meta_in
        m = meta_in
        if CFG["LOC_ENCODING"] == "embedding":
            loc_in = keras.Input((1,), name="loc_index_input", dtype="int32"); inputs["loc_index_input"] = loc_in
            e = layers.Flatten()(layers.Embedding(n_loc, CFG["LOC_EMB_DIM"], name="loc_embedding")(loc_in))
            m = layers.Concatenate(name="meta_concat")([meta_in, e])
        branches.append(layers.Dense(CFG["META_DENSE"], activation="relu", kernel_regularizer=reg, dtype="float32", name="meta_dense")(m))
    z = layers.Concatenate(name="fusion", dtype="float32")(branches) if len(branches) > 1 else branches[0]
    h = layers.Dense(CFG["DENSE_1"], kernel_regularizer=reg, name="dense_1")(z)
    h = layers.Dropout(CFG["DROPOUT_1"], name="drop_1")(layers.Activation("relu")(layers.BatchNormalization(name="bn_1")(h)))
    h = layers.Dense(CFG["DENSE_2"], kernel_regularizer=reg, name="dense_2")(h)
    h = layers.Dropout(CFG["DROPOUT_2"], name="drop_2")(layers.Activation("relu")(layers.BatchNormalization(name="bn_2")(h)))
    out = layers.Activation("softmax", dtype="float32", name="output")(layers.Dense(N_CLASSES, kernel_regularizer=reg, name="logits")(h))
    return keras.Model(inputs, out, name=f"SkinNet_{exp.code}")

def set_phase(model, phase):
    """phase 1: frozen backbone; phase 2: backbone trainable except BatchNorm layers."""
    bb = next((l for l in model.layers if isinstance(l, keras.Model)), None)   # the (only) nested backbone model
    if bb is None:
        return                                      # metadata-only model
    if phase == 1:
        bb.trainable = False
    else:
        bb.trainable = True
        stack = list(bb.layers)
        while stack:                                # also nested sub-models (e.g. ViT inside its wrapper)
            l = stack.pop()
            if isinstance(l, layers.BatchNormalization):
                l.trainable = False
            stack += list(getattr(l, "layers", []))

def split_image_head(model):
    """(feature_model: image -> img_features, head_model: [img_features, metadata] -> probabilities).
    Used by SHAP and permutation importance so that the expensive image path runs once per image [R7-3, R1-4]."""
    feat_model = keras.Model(model.get_layer("image_input").output, model.get_layer("img_features").output)
    head_inputs = [model.get_layer("img_features").output, model.get_layer("meta_input").output]
    if CFG["LOC_ENCODING"] == "embedding":
        head_inputs.append(model.get_layer("loc_index_input").output)
    return feat_model, keras.Model(head_inputs, model.output)

def count_params(model):
    tr = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    nt = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    return tr, nt

# [RQ-01] Experiment registry
@dataclass(frozen=True)
class Exp:
    code: str
    name: str
    use_image: bool = True
    use_meta: bool = True
    use_cbam: bool = True
    resample: bool = False
    class_weight: bool = False
    minority_aug: bool = False
    mix: bool = False
    loss: str = "ce"
    two_phase: bool = True
    tta_eval: bool = False
    single_lr: float = 0.0          # peak LR if two_phase=False (0 -> CFG LR_P2)
    group: str = "core"
    priority: int = 1
    backbone: str = "effb3"         # "effb3" (SkinNet) or a benchmark backbone (Section 13)
    meta_features: tuple = ("age", "sex", "site")
    batch_size: int = 0             # 0 -> CFG BATCH_SIZE (batch-size sensitivity arms only)
    folds: tuple = ()               # () -> all folds; e.g. (1,) for single-fold checks
    p1_from: str = ""               # reuse the Phase-1 weights of this arm (identical Phase 1)
    rep: int = 0                    # repeated run index (reproducibility check)
    mix_mode: str = "both"          # "both" | "cutmix" | "mixup" (individual augmentation arms) [R7-4]
    optimizer: str = "adam"         # "adam" | "sgd" (optimiser ablation) [R1-5]
    precision: str = "mixed"        # "mixed" (FP16 compute) | "fp32" (mixed-precision ablation) [R2-M1]

SKN = dict(resample=True, class_weight=True, minority_aug=True)          # original imbalance strategy (A-J chain)
SKN_FINAL = dict(resample=True, class_weight=False, minority_aug=False)  # final SkinNet: resampling only (validation evidence, see Markdown)
EXPERIMENTS = {e.code: e for e in [
    Exp("A", "EfficientNetB3 only (image-only baseline)", use_meta=False, use_cbam=False),
    Exp("B", "EfficientNetB3 + Metadata", use_cbam=False),
    Exp("C", "EfficientNetB3 + CBAM", use_meta=False),
    Exp("D", "EfficientNetB3 + CBAM + Metadata"),
    Exp("E", "D + Imbalance strategy", **SKN),
    Exp("F", "E + MixUp/CutMix", mix=True, p1_from="E", **SKN),
    Exp("G", "E + Focal loss + Label smoothing", loss="focal_ls", **SKN),
    Exp("H_CW", "Full SkinNet, original (E + MixUp/CutMix + Focal/LS, with class weights)", mix=True, loss="focal_ls", p1_from="G", **SKN),
    Exp("H", "Full SkinNet (final: resampling, no class weights)", mix=True, loss="focal_ls", tta_eval=True, **SKN_FINAL),
    Exp("H_NM", "Final SkinNet without MixUp/CutMix", mix=False, loss="focal_ls", group="supp", priority=2, **SKN_FINAL),
    Exp("S1_1P", "Full SkinNet, single-phase fine-tuning", mix=True, loss="focal_ls", two_phase=False, group="supp", priority=2, **SKN_FINAL),
    Exp("E_OS", "D + resampling + minority aug (no class weights)", resample=True, minority_aug=True, group="supp", priority=2),
    Exp("E_CW", "D + class weights only", class_weight=True, group="supp", priority=2),
    Exp("E_RS", "D + resampling only", resample=True, group="supp", priority=3),
    Exp("G_F", "E + Focal loss only", loss="focal", group="supp", priority=3, **SKN),
    Exp("G_LS", "E + Label smoothing only (CE)", loss="ce_ls", group="supp", priority=3, **SKN),
    Exp("F_CM", "E + CutMix only", mix=True, mix_mode="cutmix", p1_from="E", group="supp", priority=2, **SKN),
    Exp("F_MU", "E + MixUp only", mix=True, mix_mode="mixup", p1_from="E", group="supp", priority=2, **SKN),
    Exp("H_SGD", "Full SkinNet, SGD + Nesterov instead of Adam", mix=True, loss="focal_ls", optimizer="sgd", group="supp", priority=2, **SKN_FINAL),
    Exp("H_FP32", "Full SkinNet, FP32 training (no mixed precision)", mix=True, loss="focal_ls", precision="fp32", folds=(1,),
        group="repro", priority=2, **SKN_FINAL),
    Exp("M0", "Metadata-only baseline (CE)", use_image=False, two_phase=False, single_lr=CFG["LR_P1"], group="meta", priority=1),
    Exp("M1", "Metadata-only + imbalance + focal/LS", use_image=False, two_phase=False, single_lr=CFG["LR_P1"],
        resample=True, class_weight=True, loss="focal_ls", group="meta", priority=2),
    # ---- reproducibility and batch-size checks (fold 1 only) [R4-9, R3 minor, R6] ----
    Exp("H_REP", "Full SkinNet, repeated run (same seed)", mix=True, loss="focal_ls", folds=(1,), rep=1, group="repro", priority=2, **SKN_FINAL),
    Exp("H_BS8", "Full SkinNet, batch size 8", mix=True, loss="focal_ls", folds=(1,), batch_size=8, group="bs", priority=3, **SKN_FINAL),
    Exp("H_BS32", "Full SkinNet, batch size 32", mix=True, loss="focal_ls", folds=(1,), batch_size=32, group="bs", priority=3, **SKN_FINAL),
    # ---- metadata leave-one-feature-out (retrained) [R7-3] ----
    Exp("H_noAge", "Full SkinNet without age", mix=True, loss="focal_ls", meta_features=("sex", "site"), group="metafeat", priority=3, **SKN_FINAL),
    Exp("H_noSex", "Full SkinNet without sex", mix=True, loss="focal_ls", meta_features=("age", "site"), group="metafeat", priority=3, **SKN_FINAL),
    Exp("H_noSite", "Full SkinNet without anatomical site", mix=True, loss="focal_ls", meta_features=("age", "sex"), group="metafeat", priority=3, **SKN_FINAL),
    # ---- benchmarks retrained under the same protocol [R1-6, R3-5, R4-10] ----
    *[Exp(code, f"{BACKBONES[bb][3]} + metadata (same training protocol)", use_cbam=False, mix=True, loss="focal_ls",
          backbone=bb, group="bench", priority=2, **SKN_FINAL)
      for code, bb in [("BM_R50", "resnet50"), ("BM_D121", "densenet121"), ("BM_V2S", "effv2s"), ("BM_VIT", "vit_b16"), ("BM_CNXT", "convnext_tiny")]],
]}
PROFILES = {"core": ["M0", "A", "B", "C", "D", "E", "F", "G", "H_CW", "H", "S1_1P", "H_REP"]}
# "reviewer": every experiment a reviewer asked for (component decompositions R2-M1/R7-4, optimiser R1-5,
#             mixed precision R2-M1, benchmarks R1-6/R3-5 = classic CNN, modern CNN, transformer)
PROFILES["reviewer"] = ["M0", "A", "B", "C", "D", "E", "F",
                        "E_CW", "E_RS", "E_OS",                                   # imbalance decomposition
                        "G", "H_CW", "H", "S1_1P",                                # original full model, final SkinNet, two-phase
                        "BM_R50", "BM_VIT", "BM_CNXT",                            # benchmarks (final recipe)
                        "H_NM",                                                   # final SkinNet without MixUp/CutMix (variant)
                        "H_REP", "H_FP32",                                        # reproducibility, mixed precision
                        "G_F", "G_LS", "F_CM", "F_MU", "H_SGD"]                   # single-technique and optimiser arms
assert sorted(PROFILES["reviewer"]) == sorted(PROFILES["core"] + ["E_OS", "E_CW", "E_RS", "G_F", "G_LS", "F_CM", "F_MU", "H_SGD",
                                                                  "H_FP32", "BM_R50", "BM_VIT", "BM_CNXT", "H_NM"])
PROFILES["full"] = PROFILES["reviewer"] + ["M1", "BM_D121", "BM_V2S", "H_noAge", "H_noSex", "H_noSite", "H_BS8", "H_BS32"]
try:
    import keras_hub  # noqa: F401
    KERAS_HUB_OK = True
except Exception as _e:
    KERAS_HUB_OK = False
    print("KerasHub not available: BM_VIT cannot be trained in this session.", _e)
VIT_READY = KERAS_HUB_OK and (SYNTHETIC or vit_preset_source() != CFG["VIT_PRESET"])
if not SYNTHETIC:
    print("ViT-B/16 preset:", vit_preset_source() if VIT_READY else
          "NOT ATTACHED -> BM_VIT is skipped. Attach it: Add Input -> Models -> keras/vit -> Keras -> vit_base_patch16_224_imagenet")
def exp_folds(exp):
    return [f for f in range(1, CFG["N_FOLDS"] + 1) if not exp.folds or f in exp.folds]
# Evaluation-only configurations derived from H
EVAL_ALIASES = {"I": ("H", "test", "Full SkinNet without TTA"), "J": ("H", "test_tta", "Full SkinNet with TTA")}
TABLE_ORDER = list("ABCDEFG") + ["H_CW", "H", "I", "J", "H_NM"] + ["S1_1P", "E_OS", "E_CW", "E_RS", "G_F", "G_LS", "F_CM", "F_MU", "H_SGD", "M0", "M1",
                                    "H_noAge", "H_noSex", "H_noSite", "BM_R50", "BM_D121", "BM_V2S", "BM_VIT", "BM_CNXT"]
SINGLE_FOLD_CHECKS = ["H_REP", "H_BS8", "H_BS32", "H_FP32"]     # reported in Sections 21A/21B, not in the ablation table

def exp_hash(exp):
    keys = ["IMG_SIZE", "BATCH_SIZE", "EPOCHS_P1", "EPOCHS_P2", "WARMUP_EPOCHS", "LR_P1", "LR_P2", "PATIENCE_P1", "PATIENCE_P2",
            "MIN_STEPS", "TARGET_PER_CLASS", "MAX_OVERSAMPLE_X", "CUTMIX_PROB", "MIXUP_PROB", "MIXUP_ALPHA", "LOC_ENCODING",
            "CLASS_WEIGHT_MODE", "MIXUP_LAMBDA_MODE", "SELECTION_METRIC", "FOCAL_GAMMA", "FOCAL_ALPHA", "LABEL_SMOOTHING", "SEED",
            "TEST_SPLIT_MODE", "N_FOLDS", "STEPS_PER_EPOCH", "BACKBONE_WEIGHTS"] + (["VIT_PRESET", "VIT_SIZE"] if exp.backbone == "vit_b16" else [])
    blob = json.dumps({"exp": asdict(exp), "cfg": {k: CFG.get(k) for k in keys}}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]

registry_df = pd.DataFrame([{**asdict(e), "hash": exp_hash(e)} for e in EXPERIMENTS.values()])
with open(DIRS["config"] / "experiments.json", "w") as f:
    json.dump({"experiments": registry_df.to_dict("records"), "eval_aliases": EVAL_ALIASES}, f, indent=2)
display(registry_df[["code", "name", "backbone", "use_meta", "use_cbam", "resample", "class_weight", "minority_aug", "mix", "loss", "two_phase", "meta_features", "group", "priority"]])
print("Run profiles:", {k: len(v) for k, v in PROFILES.items()}, "| selected:", CFG["RUN_PROFILE"])

# ---- augmented training examples (Full SkinNet pipeline, fold 1, phase-2 mixing) ----
_tr, _, _ = get_fold_frames(1)
_mp = MetaPreprocessor(CFG["LOC_ENCODING"]).fit(_tr)
_rs, _ = smart_oversample(_tr, CFG["TARGET_PER_CLASS"], CFG["MAX_OVERSAMPLE_X"], SEED)
_d, _l = _mp.transform(_rs)
_x, _y = next(iter(make_train_dataset(_rs, _d, _l, EXPERIMENTS["H"], apply_mix=True, seed=SEED)))
fig, axs = plt.subplots(2, 4, figsize=(FIG_W2, 3.9))
for i, ax in enumerate(axs.flat):
    ax.imshow(np.clip(_x["image_input"][i].numpy() / 255.0, 0, 1)); ax.axis("off")
    top = np.argsort(-_y[i].numpy())[:2]
    ax.set_title(" / ".join(f"{CLASS_NAMES[t]} {_y[i][t]:.2f}" for t in top if _y[i][t] > 0.005), fontsize=6.5)
fig.suptitle("Training batch after augmentation, CutMix and MixUp (soft labels shown)", fontsize=8)
fig.tight_layout()
save_fig(fig, "fig_augmentation_examples")
del _x, _y

# ---- [RQ-09] TTA pipeline verification: each view is the stated transform of the clean image ----
_te = test_df.iloc[:8]; _dte, _lte = _mp.transform(_te)
_views = [next(iter(make_eval_dataset(_te, _dte, _lte, EXPERIMENTS["H"], tta_view=v)))[0]["image_input"].numpy()
          for v in range(CFG["TTA_VIEWS"])]
if CFG["TTA_MODE"] == "deterministic":
    _ref = [lambda a: a, lambda a: a[:, :, ::-1], lambda a: a[:, ::-1], lambda a: np.rot90(a, 1, axes=(1, 2)), lambda a: np.rot90(a, 3, axes=(1, 2))]
    assert all(np.allclose(_views[v], _ref[v](_views[0])) for v in range(CFG["TTA_VIEWS"])), "TTA views are not the stated transforms"
    assert all(not np.allclose(_views[v], _views[0]) for v in range(1, CFG["TTA_VIEWS"])), "a TTA view equals the clean image"
    print("TTA verification: PASS (views = identity, h-flip, v-flip, rot90, rot270 of the clean test image; eval pipeline only)")
del _views

# [MONITOR] Sessions, events and progress display (reporting only; no influence on training)
MON_LEVEL = "epoch"            # "epoch": every epoch; "fold": fold/phase level only
STATUS_EVERY_MIN = 10          # a text status board is printed at this interval (the live board updates continuously)
SESSION_PLAN = [
    ("BASELINES AND ARCHITECTURE ABLATIONS", ["M0", "M1", "A", "B", "C", "D"]),
    ("IMBALANCE, AUGMENTATION, LOSS AND TRAINING-STRATEGY ABLATIONS",
     ["E", "E_OS", "E_CW", "E_RS", "F", "F_CM", "F_MU", "G", "G_F", "G_LS", "H_CW", "H", "H_NM", "S1_1P", "H_SGD", "H_REP", "H_FP32",
      "H_BS8", "H_BS32", "H_noAge", "H_noSex", "H_noSite"]),
    ("BENCHMARK MODELS (SAME PROTOCOL)", ["BM_R50", "BM_D121", "BM_V2S", "BM_VIT", "BM_CNXT"]),
]
GPU_LABEL = os.environ.get("SKINNET_WORKER_ID", "0" if GPUS else "CPU")
LIVE_DIR = DIRS["runs"] / "_live"; LIVE_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_PATH = LIVE_DIR / "events.jsonl"
MONITOR = None                 # set by the run controller in the main notebook
LINE = "─" * 64

def session_of(code):
    for i, (_, codes) in enumerate(SESSION_PLAN):
        if code in codes:
            return i
    return len(SESSION_PLAN) - 1

def hms(sec):
    if sec is None or not np.isfinite(sec):
        return "--:--:--"
    sec = int(round(sec)); return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"

def pbar(frac, width=24):
    frac = min(max(frac, 0.0), 1.0); n = int(round(frac * width))
    return "[" + "█" * n + "░" * (width - n) + f"] {frac * 100:3.0f}%"

def phase_name(code, phase):
    e = EXPERIMENTS[code]
    if not e.two_phase:
        return "Training (metadata network)" if not e.use_image else "Single-phase fine-tuning"
    return "Head training (frozen backbone)" if phase == 1 else "Fine-tuning (backbone unfrozen, BatchNorm frozen)"

def phase_counter(code, phase):
    return f"{phase}/2" if EXPERIMENTS[code].two_phase else "1/1"

def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v

def emit(kind, **kw):
    """Record a progress event. Reporting only: never alters training state."""
    rec = {"t": time.time(), "gpu": GPU_LABEL, "kind": kind, **{k: _jsonable(v) for k, v in kw.items()}}
    try:
        with open(EVENTS_PATH, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass
    mon = MONITOR if MONITOR is not None else _WORKER_MON
    if mon is not None:
        mon.handle(rec)

def unit_is_done(code, fold):
    st_ = read_status(DIRS["runs"] / code / f"fold_{fold}")
    return bool(st_.get("done")) and st_.get("hash") == exp_hash(EXPERIMENTS[code])

def best_val_metrics(code, fold):
    """Validation metrics of the saved (best) checkpoint of the final phase, read from the stored history."""
    d = DIRS["runs"] / code / f"fold_{fold}"; st_ = read_status(d)
    ph = 2
    hp = d / f"history_p{ph}.csv"
    if not hp.exists() or not st_.get(f"p{ph}_best_epoch"):
        return None
    h_ = pd.read_csv(hp); r_ = h_[h_.epoch == st_[f"p{ph}_best_epoch"]]
    if r_.empty:
        return None
    r_ = r_.iloc[-1]
    return dict(val_macro_auc=float(r_.val_macro_auc), val_bal_acc=float(r_.val_bal_acc), val_macro_f1=float(r_.val_macro_f1),
                best_epoch=int(st_[f"p{ph}_best_epoch"]))

class Monitor:
    """Renders progress events as labelled blocks and keeps overall / per-GPU status and ETA."""
    def __init__(self, queue, n_workers=1, t0=None, live_board=True, echo=True):
        self.queue = list(queue); self.total = len(self.queue)
        self.n_workers = max(1, n_workers); self.t0 = t0 or time.time(); self.echo = echo
        self.running, self.durations = {}, {}          # gpu -> state; code -> [seconds of fold-runs finished in this run]
        self.sessions_started, self.sessions_done, self.models_reported = set(), set(), set()
        self.session_t0 = {}
        self.refresh()
        self.done_at_start = set(self.done)
        self.models_reported = {c for c in self.codes() if all(u in self.done for u in self.units_of(c))}
        self.last_board = time.time(); self.handle_board = None
        if live_board:
            try:
                from IPython.display import display as _d, Pretty
                self._Pretty = Pretty; self.handle_board = _d(Pretty(self.board()), display_id=True)
            except Exception:
                self.handle_board = None

    # ---------- bookkeeping ----------
    def codes(self):
        return list(dict.fromkeys(c for c, _ in self.queue))
    def units_of(self, code):
        return [(c, f) for c, f in self.queue if c == code]
    def refresh(self):
        self.done = {u for u in self.queue if unit_is_done(*u)}
    def session_units(self, s):
        return [u for u in self.queue if session_of(u[0]) == s]
    def active_sessions(self):
        return sorted({session_of(c) for c in self.codes()})
    def fold_label(self, code, fold):
        folds = exp_folds(EXPERIMENTS[code])
        return f"{fold}/{len(folds)}" if len(folds) > 1 else f"{fold} (single-fold check)"
    def est_unit_seconds(self, code):
        if self.durations.get(code):
            return float(np.mean(self.durations[code]))
        prev = [read_status(DIRS["runs"] / c / f"fold_{f}").get("unit_seconds") for c, f in self.units_of(code) if (c, f) in self.done]
        prev = [p for p in prev if p]
        if prev:
            return float(np.mean(prev))
        allv = [x for v in self.durations.values() for x in v]
        return float(np.mean(allv)) if allv else None
    def eta(self):
        pending = [u for u in self.queue if u not in self.done]
        if not pending:
            return 0.0
        tot = 0.0
        for c, f in pending:
            e = self.est_unit_seconds(c)
            if e is None:
                return None
            r = next((s for s in self.running.values() if s.get("code") == c and s.get("fold") == f), None)
            tot += max(e - (time.time() - r["t_unit"]), 0.0) if r else e
        return tot / self.n_workers
    def avg_fold(self):
        allv = [x for v in self.durations.values() for x in v]
        return float(np.mean(allv)) if allv else None
    def current_session(self):
        run_s = [session_of(s["code"]) for s in self.running.values() if s.get("code")]
        if run_s:
            return min(run_s)
        pend = [session_of(c) for c, f in self.queue if (c, f) not in self.done]
        return min(pend) if pend else (self.active_sessions()[-1] if self.active_sessions() else 0)

    # ---------- output ----------
    def out(self, text):
        if self.echo:
            print(text, flush=True)
    def plan(self):
        sess = self.active_sessions(); lines = ["=" * 64, "SKINNET TRAINING PLAN", "=" * 64]
        for k, s in enumerate(sess):
            us = self.session_units(s); nd = sum(u in self.done for u in us)
            lines.append(f"SESSION {k + 1}/{len(sess)} — {SESSION_PLAN[s][0]}   ({nd}/{len(us)} fold-runs done)")
            for c in [c for c in self.codes() if session_of(c) == s]:
                marks = " ".join("✓" if u in self.done else "○" for u in self.units_of(c))
                n_d = sum(u in self.done for u in self.units_of(c))
                lines.append(f"   {c:<9} {EXPERIMENTS[c].name[:44]:<44} {marks}  ({n_d}/{len(self.units_of(c))})")
        lines += ["", f"Total fold-runs: {self.total} | already completed: {len(self.done)} | to run: {self.total - len(self.done)}",
                  f"GPU workers: {self.n_workers}      Legend: ✓ completed (skipped, resumable)   ○ pending", "=" * 64]
        self.out("\n".join(lines))
    def session_header(self, s):
        sess = self.active_sessions(); us = self.session_units(s); codes = [c for c in self.codes() if session_of(c) == s]
        self.session_t0[s] = time.time()
        self.out("\n" + "=" * 64 + f"\nSESSION {sess.index(s) + 1}/{len(sess)} — {SESSION_PLAN[s][0]}\n" + "=" * 64 +
                 f"\nModels          : {len(codes)}  ({', '.join(codes)})\nFold-runs       : {len(us)}  "
                 f"(already completed: {sum(u in self.done for u in us)})\nStatus          : RUNNING\n" + "=" * 64)
    def board(self):
        n_d = len(self.done); s = self.current_session(); sess = self.active_sessions()
        lines = ["=" * 64, "SKINNET TRAINING STATUS", "=" * 64,
                 f"Overall   : {n_d} / {self.total} fold-runs   {pbar(n_d / max(self.total, 1))}",
                 f"Session   : {(sess.index(s) + 1) if s in sess else '-'} / {len(sess)} — {SESSION_PLAN[s][0] if sess else ''}",
                 f"Completed : {n_d}      Remaining : {self.total - n_d}",
                 f"Elapsed   : {hms(time.time() - self.t0)}      ETA : {hms(self.eta()) if self.eta() is not None else 'estimating after the first fold-run'}",
                 LINE]
        for g in sorted(self.running, key=str):
            r = self.running[g]
            if not r.get("code"):
                lines.append(f"GPU {g}  idle"); continue
            ep = f"Epoch {r.get('epoch', 0)}/{r.get('n_epochs', '?')}" if r.get("phase") else "starting"
            best = f"best val mAUC {r['best']:.4f}" if r.get("best") is not None else ""
            lines.append(f"GPU {g}  └── {r['code']} — Fold {self.fold_label(r['code'], r['fold'])} — "
                         f"Phase {phase_counter(r['code'], r.get('phase') or 1)} — {ep}  {best}")
            lines.append(f"        {EXPERIMENTS[r['code']].name[:52]}  | fold time {hms(time.time() - r['t_unit'])}")
        lines.append("=" * 64)
        return "\n".join(lines)
    def update_board(self, force_print=False):
        if self.handle_board is not None:
            try: self.handle_board.update(self._Pretty(self.board()))
            except Exception: pass
        if force_print or (time.time() - self.last_board) / 60 >= STATUS_EVERY_MIN:
            self.out("\n" + self.board()); self.last_board = time.time()

    # ---------- events ----------
    def handle(self, r):
        k, g = r["kind"], r["gpu"]
        if k == "unit_start":
            self.refresh(); c, f = r["code"], r["fold"]; s = session_of(c)
            if s not in self.sessions_started:
                self.sessions_started.add(s); self.session_header(s)
            self.running[g] = dict(code=c, fold=f, t_unit=r["t"], phase=None, best=None)
            mdone = sum(u in self.done for u in self.units_of(c))
            self.out(f"\n{LINE}\nMODEL    : {c} — {EXPERIMENTS[c].name}\nFOLD     : {self.fold_label(c, f)}\n"
                     f"PROGRESS : model {mdone}/{len(self.units_of(c))} folds completed | overall {len(self.done)}/{self.total} fold-runs completed\n"
                     f"GPU      : {g}\n{LINE}\n[CHECKPOINT] → Fold {self.fold_label(c, f)} started")
        elif k == "phase_start":
            rr = self.running.setdefault(g, {}); rr.update(phase=r["phase"], epoch=0, n_epochs=r["epochs"], best=None)
            self.out(f"  GPU {g} │ {r['code']} │ Fold {self.fold_label(r['code'], r['fold'])} │ precision {r.get('policy', '?')}\n"
                     f"  PHASE {phase_counter(r['code'], r['phase'])} — {phase_name(r['code'], r['phase'])}\n"
                     f"  ├─ Max epochs : {r['epochs']}   Steps/epoch : {r['steps']}\n"
                     f"  ├─ LR (peak)  : {r['lr']:g}   MixUp/CutMix : {'on' if r['mix'] else 'off'}\n"
                     f"  └─ Trainable  : {r['trainable']:,} parameters")
        elif k == "phase_reused":
            self.out(f"  GPU {g} │ {r['code']} │ Fold {self.fold_label(r['code'], r['fold'])}\n"
                     f"  PHASE 1/2 — reused from {r['source']} (identical Phase 1)   [CHECKPOINT] ✓ Phase 1 completed")
        elif k == "epoch":
            rr = self.running.setdefault(g, {}); rr.update(epoch=r["epoch"], n_epochs=r["n_epochs"])
            if r["saved"]:
                rr["best"] = r["val_macro_auc"] if CFG["SELECTION_METRIC"] == "val_macro_auc" else rr.get("best")
            if MON_LEVEL == "epoch":
                ck = "✓ SAVED (new best)" if r["saved"] else f"— not improved ({r['wait']}/{r['patience']} patience)"
                self.out(f"  GPU {g} │ {r['code']} │ Fold {self.fold_label(r['code'], r['fold'])} │ Phase {phase_counter(r['code'], r['phase'])} │ "
                         f"Epoch {r['epoch']}/{r['n_epochs']} │ Steps {r['steps']}/{r['steps']} │ {r['seconds']:.0f} s\n"
                         f"  ├─ Train Loss : {r['loss']:.4f}\n  ├─ Val mAUC   : {r['val_macro_auc']:.4f}\n"
                         f"  ├─ Val BalAcc : {r['val_bal_acc']:.4f}\n  ├─ Val mF1    : {r['val_macro_f1']:.4f}\n  └─ Checkpoint : {ck}")
        elif k == "phase_end":
            stop = " (early stopping)" if r["epochs_run"] < r["epochs"] else ""
            self.out(f"  [CHECKPOINT] ✓ Phase {phase_counter(r['code'], r['phase'])} completed — {r['epochs_run']} epochs{stop} — "
                     f"best epoch {r['best_epoch']} (val mAUC {r['best']:.4f}) — {hms(r['seconds'])}")
        elif k == "unit_end":
            c, f = r["code"], r["fold"]; self.durations.setdefault(c, []).append(r["seconds"]); self.running[g] = {}
            self.refresh(); mdone = sum(u in self.done for u in self.units_of(c)); eta = self.eta()
            self.out(f"  [CHECKPOINT] ✓ Fold {self.fold_label(c, f)} completed — best checkpoint saved (best_p2.weights.h5)\n"
                     f"  TIME     : elapsed {hms(time.time() - self.t0)} | this fold {hms(r['seconds'])} | "
                     f"average fold {hms(self.avg_fold())} | ETA {hms(eta) if eta is not None else 'estimating'}\n"
                     f"  OVERALL  : {pbar(len(self.done) / max(self.total, 1))}  {len(self.done)}/{self.total} fold-runs | "
                     f"model {c}: {mdone}/{len(self.units_of(c))}")
            if c not in self.models_reported and all(u in self.done for u in self.units_of(c)):
                self.models_reported.add(c); self.model_summary(c)
            s = session_of(c)
            if s not in self.sessions_done and all(u in self.done for u in self.session_units(s)):
                self.sessions_done.add(s); sess = self.active_sessions()
                self.out(f"\n{'=' * 64}\n✓ SESSION {sess.index(s) + 1}/{len(sess)} COMPLETE — {SESSION_PLAN[s][0]}   "
                         f"(session time {hms(time.time() - self.session_t0.get(s, self.t0))})\n{'=' * 64}")
            self.update_board(force_print=False)
        elif k == "unit_fail":
            self.running[g] = {}
            self.out(f"  ⚠ FAILED  GPU {g} │ {r['code']} │ Fold {self.fold_label(r['code'], r['fold'])} │ {r['error']}")
        elif k in ("stop", "warn"):
            self.out(f"  ⚠ GPU {g}: {r['message']}")
        if k in ("phase_start", "epoch", "unit_start"):
            self.update_board()
    def model_summary(self, c):
        rows = [(f, best_val_metrics(c, f)) for _, f in self.units_of(c)]
        rows = [(f, m) for f, m in rows if m]
        lines = [f"\n{'=' * 64}", f"MODEL COMPLETE: {c}", "=" * 64, EXPERIMENTS[c].name, "",
                 f"Folds completed : {len(self.units_of(c))} / {len(self.units_of(c))}"]
        if rows:
            lines += [f"Mean Val mAUC   : {np.mean([m['val_macro_auc'] for _, m in rows]):.4f}",
                      f"Mean Val BalAcc : {np.mean([m['val_bal_acc'] for _, m in rows]):.4f}",
                      f"Mean Val mF1    : {np.mean([m['val_macro_f1'] for _, m in rows]):.4f}"]
            fb, mb = max(rows, key=lambda t: t[1]["val_macro_auc"])
            lines += [f"Best fold       : {fb}  (val mAUC {mb['val_macro_auc']:.4f})",
                      "Per fold (val, saved checkpoint): " + "  ".join(f"f{f}={m['val_macro_auc']:.4f}" for f, m in rows),
                      "(validation values of the selected checkpoints; test metrics are computed in Section 18)"]
        lines += ["", "Status: ✓ COMPLETE", "=" * 64]
        self.out("\n".join(lines))
    def final_summary(self):
        self.refresh(); sess = self.active_sessions(); codes = self.codes()
        s_done = sum(all(u in self.done for u in self.session_units(s)) for s in sess)
        m_done = sum(all(u in self.done for u in self.units_of(c)) for c in codes)
        complete = len(self.done) == self.total
        lines = ["\n" + "=" * 64, "SKINNET TRAINING COMPLETE" if complete else ("SKINNET TRAINING FINISHED WITH FAILURES" if any((DIRS["runs"] / c / f"fold_{f}" / "failed.json").exists() for c, f in self.queue) else "SKINNET TRAINING STOPPED (resume in the next session)"), "=" * 64,
                 f"Sessions completed : {s_done} / {len(sess)}", f"Models completed   : {m_done} / {len(codes)}",
                 f"Fold-runs          : {len(self.done)} / {self.total}   (this run: {len(self.done - self.done_at_start)})",
                 f"Total time         : {hms(time.time() - self.t0)}", "", "MODEL STATUS"]
        failed = [(c, f) for c, f in self.queue if (DIRS["runs"] / c / f"fold_{f}" / "failed.json").exists() and (c, f) not in self.done]
        for c in codes:
            n_d = sum(u in self.done for u in self.units_of(c)); n = len(self.units_of(c)); n_f = sum(u in failed for u in self.units_of(c))
            mark = "✓" if n_d == n else ("✗" if n_f else ("◐" if n_d else "○"))
            lines.append(f"{mark} {c:<9} — {n_d}/{n} folds" + (f", {n_f} FAILED" if n_f else "") + f"   {EXPERIMENTS[c].name[:40]}")
        if failed:
            lines += ["", f"⚠ FAILED fold-runs ({len(failed)}) — they are retried in the next session:"]
            for c, f in failed:
                err = json.loads((DIRS["runs"] / c / f"fold_{f}" / "failed.json").read_text()).get("error", "")
                lines.append(f"   ✗ {c} fold {f}: {err[:160]}")
        lines += ["", "Results saved to:", f"  {DIRS['runs']}  (checkpoints, predictions, histories)", f"  {DIRS['results']}  (tables)",
                  f"Disk free: {shutil.disk_usage(WORK).free / 2**30:.1f} GB", "=" * 64]
        self.out("\n".join(lines))
        self.update_board()

# a worker renders its own events into its log file (overall counts are read from the shared status files)
_WORKER_MON = None

# [RQ-12][RQ-04] Training engine
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, log_loss
import ctypes

def host_rss_gb():
    """Resident host memory of this kernel (GB)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024**2
    except Exception:
        pass
    return float("nan")

def release_host_memory():
    """Free Python/TF objects and return freed heap pages to the OS (limits RAM growth across units)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def macro_auc_safe(y, probs):
    aucs = []
    for c in range(N_CLASSES):
        yc = (y == c).astype(int)
        if 0 < yc.sum() < len(yc):
            aucs.append(roc_auc_score(yc, probs[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")

def _brier_fn(y_true, y_pred):
    return keras.ops.sum(keras.ops.square(keras.ops.cast(y_pred, "float32") - keras.ops.cast(y_true, "float32")), axis=-1)

@keras.saving.register_keras_serializable(package="skinnet")
class BrierScore(keras.metrics.MeanMetricWrapper):
    """Multi-class Brier score, sum_c (p_c - y_c)^2, on the (possibly mixed) training targets [R6]."""
    def __init__(self, name="brier", dtype=None):
        super().__init__(fn=_brier_fn, name=name, dtype=dtype)
    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}

def predict_probs(model, ds):
    return model.predict(ds, verbose=0).astype(np.float32)

class ValMonitor(keras.callbacks.Callback):
    def __init__(self, val_ds, y_val, patience, best_path, hist_path, phase):
        super().__init__()
        self.val_ds, self.y_val, self.patience = val_ds, y_val, patience
        self.best_path, self.hist_path, self.phase = best_path, hist_path, phase
        self.best, self.best_epoch, self.wait = -np.inf, 0, 0
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pr = predict_probs(self.model, self.val_ds)
        yp = pr.argmax(1)
        row = dict(phase=self.phase, epoch=epoch + 1, train_loss=float(logs.get("loss", np.nan)),
                   train_acc=float(logs.get("accuracy", np.nan)), train_brier=float(logs.get("brier", np.nan)),
                   val_brier=float(np.mean(np.sum((pr - np.eye(N_CLASSES)[self.y_val]) ** 2, 1))),
                   val_macro_auc=macro_auc_safe(self.y_val, pr), val_bal_acc=balanced_accuracy_score(self.y_val, yp),
                   val_macro_f1=f1_score(self.y_val, yp, average="macro", zero_division=0),
                   val_ce=log_loss(self.y_val, np.clip(pr, 1e-7, 1), labels=list(range(N_CLASSES))),
                   lr=float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)),
                   epoch_seconds=time.time() - self.t0)
        score = row[CFG["SELECTION_METRIC"]]
        improved = np.isfinite(score) and score > self.best + 1e-6
        if improved:
            self.best, self.best_epoch, self.wait = score, epoch + 1, 0
            self.model.save_weights(self.best_path)
        else:
            self.wait += 1
        row["improved"] = bool(improved)
        pd.DataFrame([row]).to_csv(self.hist_path, mode="a", header=not Path(self.hist_path).exists(), index=False)
        emit("epoch", code=getattr(self, "code", "?"), fold=getattr(self, "fold", 0), phase=self.phase, epoch=epoch + 1,
             n_epochs=self.params.get("epochs"), steps=self.params.get("steps"), loss=row["train_loss"],
             val_macro_auc=row["val_macro_auc"], val_bal_acc=row["val_bal_acc"], val_macro_f1=row["val_macro_f1"],
             saved=bool(improved), wait=self.wait, patience=self.patience, seconds=row["epoch_seconds"])
        if self.wait >= self.patience:
            self.model.stop_training = True

def unit_dir(code, fold, create=True):
    d = DIRS["runs"] / code / f"fold_{fold}"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d

def read_status(d):
    p = d / "status.json"
    if not p.exists():
        return {}
    try:
        return json.load(open(p))
    except Exception:                       # truncated file (e.g. disk full in an earlier session): treat the unit as not done
        return {"corrupt_status": True}

def write_status(d, st):
    with open(d / "status.json", "w") as f:
        json.dump(st, f, indent=2, default=float)

def save_preds(path, frame, probs, views=None):
    np.savez_compressed(path, image_id=np.array(frame.image_id.astype(str).tolist(), dtype="U"), lesion_id=np.array(frame.lesion_id.astype(str).tolist(), dtype="U"),
                        y_true=frame.label.values.astype(np.int16), probs=probs.astype(np.float32),
                        **({"view_probs": views.astype(np.float32)} if views is not None else {}))

TRAIN_PRED_EXPS = {"D", "E", "E_OS", "E_CW", "E_RS", "F", "F_CM", "F_MU", "G", "H_CW", "H", "H_NM"}   # overfitting / synthetic-pattern check [R7-4]

def compile_phase(model, exp, phase, steps, class_weights, epochs, peak_lr, min_lr):
    set_phase(model, phase)
    if getattr(exp, "optimizer", "adam") == "sgd":            # [R1-5] optimiser ablation: SGD needs a larger step size
        peak_lr, min_lr = peak_lr * CFG["SGD_LR_MULT"], min_lr * CFG["SGD_LR_MULT"]
    sched = CosineWarmupSchedule(peak_lr, CFG["WARMUP_EPOCHS"] * steps, epochs * steps, min_lr)
    loss = SkinNetLoss(**LOSS_SETTINGS[exp.loss], class_weights=class_weights, weight_mode=CFG["CLASS_WEIGHT_MODE"])
    opt = (keras.optimizers.SGD(learning_rate=sched, momentum=0.9, nesterov=True, clipnorm=CFG["GRAD_CLIP"])
           if getattr(exp, "optimizer", "adam") == "sgd" else keras.optimizers.Adam(learning_rate=sched, clipnorm=CFG["GRAD_CLIP"]))
    model.compile(optimizer=opt,
                  loss=loss, metrics=[keras.metrics.CategoricalAccuracy(name="accuracy"), BrierScore()])

def train_unit(exp, fold, verbose=True):
    """Precision wrapper [R2-M1]: FP32 arms train under a float32 policy; all other arms under the default policy."""
    if getattr(exp, "precision", "mixed") == "fp32":
        old = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float32")
        try:
            return _train_unit(exp, fold, verbose)
        finally:
            keras.mixed_precision.set_global_policy(old)
    return _train_unit(exp, fold, verbose)

def _train_unit(exp, fold, verbose=True):
    d = unit_dir(exp.code, fold)
    st = read_status(d)
    h = exp_hash(exp)
    if st.get("done"):
        if st.get("hash") != h:
            emit("warn", message=f"{exp.code} fold {fold} finished under a different configuration hash; not reused. Delete {d} to retrain.")
        return st
    if st and st.get("hash") != h:
        raise RuntimeError(f"{d} holds partial results from a different configuration. Delete it to retrain.")
    st.update(exp=exp.code, fold=fold, hash=h, started=st.get("started", time.strftime("%Y-%m-%d %H:%M:%S")))
    t_unit = time.time()
    set_seed(SEED + 10 * fold)                                    # identical seed per fold across arms (paired design)
    tr, va, te = get_fold_frames(fold)
    mp = MetaPreprocessor(CFG["LOC_ENCODING"], features=exp.meta_features).fit(tr)   # fitted on the raw training fold only
    cw = fold_class_weights(tr) if exp.class_weight else None
    tr_used = smart_oversample(tr, CFG["TARGET_PER_CLASS"], CFG["MAX_OVERSAMPLE_X"], SEED)[0] if exp.resample else tr
    bs = exp.batch_size or BATCH_SIZE
    steps = CFG.get("MAX_STEPS_OVERRIDE") or max(1, CFG["STEPS_PER_EPOCH"] * BATCH_SIZE // bs)   # same images per epoch for all arms
    d_tr, l_tr = mp.transform(tr_used); d_va, l_va = mp.transform(va); d_te, l_te = mp.transform(te)
    val_ds = make_eval_dataset(va, d_va, l_va, exp)
    st["dtype_policy"] = keras.mixed_precision.global_policy().name       # recorded for the audit (mixed_float16 expected)
    st.update(n_train_raw=len(tr), n_train_used=len(tr_used), n_val=len(va), n_test=len(te), steps_per_epoch=steps, batch_size=bs,
              gpu=HW.get("gpu_name"), meta_features=list(exp.meta_features), meta_feature_order=mp.feature_names(),
              class_weights=None if cw is None else cw.tolist(), meta_dim=mp.dense_dim, loc_vocab=mp.loc_vocab)

    clear_keras_session(); gc.collect()
    if GPUS:
        try: tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception: pass
    model = build_skinnet(exp, mp.dense_dim, n_loc=len(mp.loc_vocab))
    if exp.two_phase:
        phases = [(1, CFG["EPOCHS_P1"], CFG["LR_P1"], CFG["MIN_LR_P1"], CFG["PATIENCE_P1"], False),
                  (2, CFG["EPOCHS_P2"], CFG["LR_P2"], CFG["MIN_LR_P2"], CFG["PATIENCE_P2"], exp.mix)]
    else:
        lr = exp.single_lr or CFG["LR_P2"]
        phases = [(2, CFG["EPOCHS_P1"] + CFG["EPOCHS_P2"], lr, CFG["MIN_LR_P2"], CFG["PATIENCE_P2"], exp.mix)]
    for i, (ph, epochs, lr, min_lr, patience, mix) in enumerate(phases):
        best_path, hist_path = d / f"best_p{ph}.weights.h5", d / f"history_p{ph}.csv"
        if st.get(f"p{ph}_done") and best_path.exists():
            model.load_weights(best_path); continue
        if i > 0:                                                 # start next phase from BEST previous weights
            model.load_weights(d / f"best_p{phases[i - 1][0]}.weights.h5")
        if ph == 1 and CFG["REUSE_P1"] and exp.p1_from:           # identical Phase 1 already trained by the source arm
            src = EXPERIMENTS[exp.p1_from]; sd = unit_dir(src.code, fold, create=False); sst = read_status(sd)
            if sst.get("p1_done") and sst.get("hash") == exp_hash(src) and (sd / "best_p1.weights.h5").exists():
                shutil.copy2(sd / "best_p1.weights.h5", best_path)
                if (sd / "history_p1.csv").exists():
                    shutil.copy2(sd / "history_p1.csv", hist_path)
                model.load_weights(best_path)
                st.update(p1_done=True, p1_source=src.code, p1_best_epoch=sst.get("p1_best_epoch"),
                          **{f"p1_best_{CFG['SELECTION_METRIC']}": sst.get(f"p1_best_{CFG['SELECTION_METRIC']}")},
                          p1_seconds=0.0, p1_trainable_params=sst.get("p1_trainable_params"), p1_nontrainable_params=sst.get("p1_nontrainable_params"))
                write_status(d, st)
                if verbose:
                    emit("phase_reused", code=exp.code, fold=fold, source=src.code)
                continue
        if hist_path.exists():
            hist_path.unlink()                                    # a phase is always re-run from its start
        compile_phase(model, exp, ph, steps, cw, epochs, lr, min_lr)
        tr_n, nt_n = count_params(model)
        if verbose:
            emit("phase_start", code=exp.code, fold=fold, phase=ph, epochs=epochs, steps=steps, lr=lr, mix=bool(mix), trainable=int(tr_n),
                 policy=keras.mixed_precision.global_policy().name)
        set_seed(SEED + 10 * fold + ph)                           # phase-specific seed: reuse and retraining are equivalent
        ds = make_train_dataset(tr_used, d_tr, l_tr, exp, apply_mix=mix, seed=SEED + 10 * fold + ph)
        mon = ValMonitor(val_ds, va.label.values, patience, best_path, hist_path, ph)
        mon.code, mon.fold = exp.code, fold                       # labels for the progress display only
        t0 = time.time()
        model.fit(ds, epochs=epochs, steps_per_epoch=steps, callbacks=[mon], verbose=0)
        model.load_weights(best_path)
        if verbose:
            emit("phase_end", code=exp.code, fold=fold, phase=ph, epochs=epochs, epochs_run=len(pd.read_csv(hist_path)),
                 best_epoch=mon.best_epoch, best=float(mon.best), seconds=time.time() - t0)
        st.update({f"p{ph}_done": True, f"p{ph}_best_epoch": mon.best_epoch, f"p{ph}_best_{CFG['SELECTION_METRIC']}": mon.best,
                   f"p{ph}_seconds": time.time() - t0, f"p{ph}_trainable_params": tr_n, f"p{ph}_nontrainable_params": nt_n})
        write_status(d, st)
    # ---------------- inference on the final (saved) weights ----------------
    t0 = time.time()
    save_preds(d / "preds_val.npz", va, predict_probs(model, val_ds))
    save_preds(d / "preds_test.npz", te, predict_probs(model, make_eval_dataset(te, d_te, l_te, exp)))
    st["inference_seconds_clean"] = time.time() - t0
    if exp.code in TRAIN_PRED_EXPS:                              # [R7-4] predictions on the raw (non-resampled) training fold
        d_trr, l_trr = mp.transform(tr)
        save_preds(d / "preds_train.npz", tr, predict_probs(model, make_eval_dataset(tr, d_trr, l_trr, exp)))
    if (exp.tta_eval or CFG["TTA_FOR_ALL_EXPERIMENTS"]) and exp.use_image:
        t0 = time.time()
        views = np.stack([predict_probs(model, make_eval_dataset(te, d_te, l_te, exp, tta_view=v)) for v in range(CFG["TTA_VIEWS"])])
        save_preds(d / "preds_test_tta.npz", te, views.mean(0), views=views)
        st["inference_seconds_tta"] = time.time() - t0
    # ---- external predictions (BCN20000) with the model already in memory [R7-2] ----
    if globals().get("BCN_CACHE") is not None and exp.code in set(CFG["EXT_EVAL_EXPS"]) | {"H"}:
        t0 = time.time()
        for kind, frame in [("ext", bcn_df), ("ext_nosite", bcn_df.assign(localization="unknown"))][: (2 if exp.code == "H" else 1)]:
            dx_, lx_ = mp.transform(frame)
            save_preds(d / f"preds_{kind}.npz", frame, predict_probs(model, make_eval_dataset(frame, dx_, lx_, exp, cache=BCN_CACHE)))
        if exp.code == "H":
            dx_, lx_ = mp.transform(bcn_df)
            views = np.stack([predict_probs(model, make_eval_dataset(bcn_df, dx_, lx_, exp, tta_view=v, cache=BCN_CACHE)) for v in range(CFG["TTA_VIEWS"])])
            save_preds(d / "preds_ext_tta.npz", bcn_df, views.mean(0), views=views)
        st["inference_seconds_external"] = time.time() - t0
    if GPUS:
        try: st["peak_gpu_mb"] = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
        except Exception: pass
    p1_needed = any(e.p1_from == exp.code for e in EXPERIMENTS.values()) and CFG["REUSE_P1"]
    if not CFG["KEEP_P1_WEIGHTS"] and not p1_needed and exp.two_phase and (d / "best_p1.weights.h5").exists():
        (d / "best_p1.weights.h5").unlink()
    st.update(done=True, unit_seconds=st.get("unit_seconds", 0) + time.time() - t_unit, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
              disk_free_gb=round(shutil.disk_usage(WORK).free / 2**30, 1))
    write_status(d, st)
    del model; clear_keras_session(); release_host_memory()
    return st

# [RQ-01][RQ-12][R1-11] Run controller
RUN_TRAINING = True
RUN_EXPERIMENTS = list(PROFILES[CFG["RUN_PROFILE"]])
RUN_FOLDS = []                                     # [] = all folds of each arm
SMOKE_ONE_FOLD = set()                             # smoke test only: arms run on fold 1 (code-path check)
if SMOKE_TEST:                                     # every training code path of the reviewer profile once
    RUN_EXPERIMENTS = ["M0", "A", "D", "E", "G", "H", "S1_1P", "H_REP", "E_OS", "E_CW", "E_RS", "G_F", "G_LS", "F_CM", "F_MU",
                       "H_SGD", "H_FP32", "H_BS8", "H_noSite", "BM_R50", "BM_VIT", "BM_CNXT"]
    SMOKE_ONE_FOLD = {"E", "E_OS", "E_CW", "E_RS", "G_F", "G_LS", "F_CM", "F_MU", "H_SGD", "H_noSite"}
    if SYNTHETIC:
        RUN_EXPERIMENTS = ["M0", "A", "D", "E", "G", "H", "S1_1P", "H_REP", "F_CM", "H_SGD", "H_FP32", "BM_R50", "BM_VIT"]
if os.environ.get("SKINNET_RUN_EXPS"):
    RUN_EXPERIMENTS = os.environ["SKINNET_RUN_EXPS"].split(",")
if os.environ.get("SKINNET_RUN_FOLDS"):
    RUN_FOLDS = [int(x) for x in os.environ["SKINNET_RUN_FOLDS"].split(",")]
_skipped_vit = [c for c in RUN_EXPERIMENTS if c in EXPERIMENTS and EXPERIMENTS[c].backbone == "vit_b16" and not VIT_READY]
if _skipped_vit and not WORKER:
    print("⚠ ViT-B/16 preset not attached -> skipped:", _skipped_vit,
          "| Kaggle: Add Input -> Models -> keras/vit -> Keras -> vit_base_patch16_224_imagenet, then re-run.")
RUN_EXPERIMENTS = [c for c in RUN_EXPERIMENTS if c in EXPERIMENTS and c not in _skipped_vit]
QUEUE = [(c, f) for c in RUN_EXPERIMENTS for f in exp_folds(EXPERIMENTS[c])[: (1 if c in SMOKE_ONE_FOLD else None)]
         if not RUN_FOLDS or f in RUN_FOLDS]
SESSION_T0 = float(os.environ.get("SKINNET_SESSION_T0", globals().get("SESSION_T0", time.time())))

def done_unit_hours(code=None):
    hrs = []
    for p_ in DIRS["runs"].glob("*/fold_*/status.json"):
        s_ = read_status(p_.parent)
        if s_.get("done") and (code is None or s_.get("exp") == code) and EXPERIMENTS.get(s_.get("exp"), Exp("x", "x")).use_image:
            hrs.append(s_.get("unit_seconds", 0) / 3600)
    return float(np.mean(hrs)) if hrs else (done_unit_hours() if code is not None else 1.2)

def claim_unit(code, fold):
    lock = unit_dir(code, fold) / "claim.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, f"{os.getpid()} {time.time():.0f}".encode()); os.close(fd)
        return True
    except FileExistsError:
        return False

def reconcile_previous_units():
    """Units imported from earlier sessions whose configuration no longer matches the registry are never left in place
    (they would be skipped without retraining). The original full model trained as "H" in earlier sessions has exactly
    the configuration of today's H_CW, so it is moved there; every other mismatch is archived in runs/_superseded/."""
    legacy_H = Exp("H", "Full SkinNet", mix=True, loss="focal_ls", tta_eval=True, p1_from="G", **SKN)
    h_old, moved, migrated = exp_hash(legacy_H), [], []
    sup = DIRS["runs"] / "_superseded"
    for code, exp in EXPERIMENTS.items():
        for fd in sorted((DIRS["runs"] / code).glob("fold_*")):
            st_ = read_status(fd)
            if not st_ or st_.get("hash") == exp_hash(exp):
                continue
            if code == "H" and st_.get("done") and st_.get("hash") == h_old and "H_CW" in EXPERIMENTS:
                tgt = DIRS["runs"] / "H_CW" / fd.name
                if not read_status(tgt).get("done"):
                    if tgt.exists():
                        shutil.rmtree(tgt)
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(fd), str(tgt))
                    st2 = read_status(tgt); st2.update(exp="H_CW", hash=exp_hash(EXPERIMENTS["H_CW"]),
                                                       migrated_from="H of an earlier session (identical training configuration)")
                    write_status(tgt, st2); migrated.append(f"H/{fd.name}")
                    continue
            dst = sup / code / fd.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(fd), str(dst)); moved.append(f"{code}/{fd.name}")
    if migrated or moved:
        print(f"Reconciled earlier results: {len(migrated)} moved H -> H_CW {migrated} | {len(moved)} archived (will be retrained): {moved}")

ONE_UNIT_PER_PROCESS = os.environ.get("SKINNET_ONE_UNIT", "0") == "1"   # workers: a fresh process per unit (frees all memory)

def failed_this_session(code, fold):
    return (unit_dir(code, fold, create=False) / "failed.json").exists()

def run_queue(queue, tag):
    """Returns 'unit' (one unit attempted and ONE_UNIT_PER_PROCESS), 'stop' (budget/RAM) or 'empty' (nothing left to claim)."""
    for code, fold in queue:
        exp = EXPERIMENTS[code]
        st = read_status(unit_dir(code, fold, create=False))
        if (st.get("done") and st.get("hash") == exp_hash(exp)) or failed_this_session(code, fold):
            continue
        rss = host_rss_gb()
        if HW.get("ram_gb") and rss > CFG["MAX_HOST_RAM_FRAC"] * HW["ram_gb"]:
            emit("stop", message=f"stopping: host RAM {rss:.1f} GB > {CFG['MAX_HOST_RAM_FRAC']:.0%} of {HW['ram_gb']} GB. Finished units are kept."); return "stop"
        elapsed = (time.time() - SESSION_T0) / 3600
        need = done_unit_hours(code) if exp.use_image else 0.05
        if elapsed + need > CFG["SESSION_BUDGET_H"]:
            emit("stop", message=f"stopping: {elapsed:.2f} h used; next unit needs ~{need:.2f} h (budget {CFG['SESSION_BUDGET_H']} h). Resume in a new session."); return "stop"
        if not claim_unit(code, fold):
            continue                                            # another GPU worker is training this unit
        emit("unit_start", code=code, fold=fold)
        t_u = time.time()
        try:
            st = train_unit(exp, fold)
            emit("unit_end", code=code, fold=fold, seconds=time.time() - t_u)
            pd.DataFrame([dict(exp=code, fold=fold, status="done", worker=tag, hours=st.get("unit_seconds", 0) / 3600, host_rss_gb=host_rss_gb(),
                               when=time.strftime("%Y-%m-%d %H:%M:%S"))]).to_csv(
                DIRS["runs"] / f"run_log_{tag}.csv", mode="a", header=not (DIRS["runs"] / f"run_log_{tag}.csv").exists(), index=False)
        except tf.errors.ResourceExhaustedError as e:
            emit("unit_fail", code=code, fold=fold, error=f"out of GPU memory: {str(e)[:200]}")
            clear_keras_session(); gc.collect(); raise
        except Exception as e:
            emit("unit_fail", code=code, fold=fold, error=f"{type(e).__name__}: {str(e)[:300]}")
            import traceback; traceback.print_exc()
            (unit_dir(code, fold) / "failed.json").write_text(json.dumps(dict(error=f"{type(e).__name__}: {str(e)[:1000]}", when=time.time())))
            clear_keras_session(); release_host_memory()
            if not SMOKE_TEST and not WORKER:
                raise
        finally:
            (unit_dir(code, fold) / "claim.lock").unlink(missing_ok=True)
        if ONE_UNIT_PER_PROCESS:
            return "unit"
    return "empty"

N_WORKERS = len(GPUS) if (CFG["PARALLEL_GPU_WORKERS"] and len(GPUS) >= 2 and not WORKER) else 0
if not WORKER and os.environ.get("SKINNET_FORCE_WORKERS"):          # testing only
    N_WORKERS = int(os.environ["SKINNET_FORCE_WORKERS"])
if WORKER:
    _WORKER_MON = Monitor(QUEUE, n_workers=int(os.environ.get("SKINNET_N_WORKERS", "1")), t0=SESSION_T0, live_board=False)
    _res = run_queue(QUEUE, f"gpu{os.environ.get('SKINNET_WORKER_ID', '0')}")
    sys.exit({"unit": 0, "stop": 3, "empty": 4}[_res])            # tells the main notebook whether to start a fresh worker
elif RUN_TRAINING and QUEUE:
    for lk in list(DIRS["runs"].glob("*/fold_*/claim.lock")) + list(DIRS["runs"].glob("*/fold_*/failed.json")):
        lk.unlink()                                             # stale claims / failure marks from an earlier session
    reconcile_previous_units()
    MONITOR = Monitor(QUEUE, n_workers=max(N_WORKERS, 1), t0=time.time())
    MONITOR.plan()
    _ev_offset = EVENTS_PATH.stat().st_size if EVENTS_PATH.exists() else 0     # ignore events of earlier sessions
    if N_WORKERS:
        import base64
        script = WORK / "skinnet_worker.py"
        script.write_text(base64.b64decode(WORKER_SRC_B64).decode("utf-8"))
        procs, procs_env = [], []
        for i in range(N_WORKERS):
            env = dict(os.environ, SKINNET_WORKER="1", SKINNET_WORKER_ID=str(i), CUDA_VISIBLE_DEVICES=str(i) if GPUS else "",
                       SKINNET_SMOKE_TEST="1" if SMOKE_TEST else "0", SKINNET_SYNTHETIC="1" if SYNTHETIC else "0",
                       SKINNET_SESSION_T0=str(SESSION_T0), SKINNET_RUN_EXPS=",".join(RUN_EXPERIMENTS),
                       SKINNET_RUN_FOLDS=",".join(map(str, RUN_FOLDS)), MPLBACKEND="Agg", TF_CPP_MIN_LOG_LEVEL="3",
                       SKINNET_N_WORKERS=str(N_WORKERS), SKINNET_ONE_UNIT="1")
            if not RUN_FOLDS:
                env.pop("SKINNET_RUN_FOLDS")
            procs_env.append(env)
            logf = open(DIRS["logs"] / f"worker_gpu{i}.log", "a")
            procs.append((subprocess.Popen([sys.executable, str(script)], env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=str(WORK)), logf))
            time.sleep(20)                                      # stagger start-up (cache opening, weight download)
        print(f"Started {N_WORKERS} GPU workers (one per GPU). Full per-GPU logs: {DIRS['logs']}/worker_gpu*.log", flush=True)
        seen, restarts = [0] * N_WORKERS, [0] * N_WORKERS
        def _pending():
            return [(c, f) for c, f in QUEUE if not (read_status(unit_dir(c, f, create=False)).get("done")
                                                    and read_status(unit_dir(c, f, create=False)).get("hash") == exp_hash(EXPERIMENTS[c]))]
        def _claimable():
            return [(c, f) for c, f in _pending() if not failed_this_session(c, f) and not (unit_dir(c, f, create=False) / "claim.lock").exists()]
        n_fresh = [0] * N_WORKERS
        while True:
            for i, (pr, lf) in enumerate(procs):                # fresh process for the next unit (exit code 0 = one unit finished)
                if pr.poll() == 0 and _claimable():
                    n_fresh[i] += 1
                    procs[i] = (subprocess.Popen([sys.executable, str(script)], env=dict(procs_env[i]), stdout=lf,
                                                 stderr=subprocess.STDOUT, cwd=str(WORK)), lf)
            for i, (pr, lf) in enumerate(procs):                # restart a crashed worker (e.g. killed for memory) up to twice
                if pr.poll() is not None and pr.returncode not in (0, 3, 4) and restarts[i] < 2 and _claimable() \
                        and (time.time() - SESSION_T0) / 3600 < CFG["SESSION_BUDGET_H"] - 0.5:
                    for lk in DIRS["runs"].glob("*/fold_*/claim.lock"):
                        if lk.read_text().split()[0] == str(pr.pid):
                            lk.unlink()
                    restarts[i] += 1
                    MONITOR.handle(dict(t=time.time(), gpu=str(i), kind="warn", message=f"worker exited with code {pr.returncode}; restarting ({restarts[i]}/2)"))
                    MONITOR.running[str(i)] = {}
                    env_i = dict(procs_env[i]); procs[i] = (subprocess.Popen([sys.executable, str(script)], env=env_i, stdout=lf,
                                                                             stderr=subprocess.STDOUT, cwd=str(WORK)), lf)
            if EVENTS_PATH.exists():                            # render the workers' events as complete blocks
                with open(EVENTS_PATH) as f_:
                    f_.seek(_ev_offset); chunk = f_.read()
                cut = chunk.rfind("\n") + 1                    # only complete lines
                _ev_offset += len(chunk[:cut].encode())
                for ln in chunk[:cut].splitlines():
                    try:
                        MONITOR.handle(json.loads(ln))
                    except Exception:
                        pass
            MONITOR.update_board()
            if all(pr.poll() is not None for pr, _ in procs) and not (any(pr.poll() == 0 for pr, _ in procs) and _claimable()):
                break
            time.sleep(10)
        for i, (pr, lf) in enumerate(procs):
            lf.close()
            print(f"GPU {i} worker finished ({n_fresh[i] + 1} unit processes; last exit code {pr.returncode})"
                  + ("" if pr.returncode in (0, 3, 4) else " ⚠ see its log (last lines below)"))
            if pr.returncode not in (0, 3, 4):
                print("\n".join(open(DIRS["logs"] / f"worker_gpu{i}.log", errors="ignore").read().splitlines()[-30:]))
    else:
        run_queue(QUEUE, "main")

# ---- analysis only when training is complete (intermediate sessions: train only, saves ~1.5 h each) ----
ens, FOLDS = {}, list(range(1, CFG["N_FOLDS"] + 1))                 # defaults used by the audit if analysis is skipped
_pending_all = [(c, f) for c, f in QUEUE if not unit_is_done(c, f)] if not WORKER else []
SKIP_ANALYSIS = bool(_pending_all) and CFG.get("ANALYSIS_WHEN_COMPLETE", True) and not CFG.get("FORCE_ANALYSIS", False) and not SMOKE_TEST
if not WORKER:
    print(f"Analysis sections: {'SKIPPED in this session (' + str(len(_pending_all)) + ' fold-runs still pending; they run automatically in the final session)' if SKIP_ANALYSIS else 'RUN'}")
if not WORKER:
    logs = [pd.read_csv(p_) for p_ in DIRS["runs"].glob("run_log_*.csv")]
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(DIRS["results"] / "run_log.csv", index=False)
    progress = pd.DataFrame([[("done" if read_status(unit_dir(c, f, create=False)).get("done") else ("n/a" if f not in exp_folds(EXPERIMENTS[c]) else "-"))
                              for f in range(1, CFG["N_FOLDS"] + 1)] for c in EXPERIMENTS],
                            index=list(EXPERIMENTS), columns=[f"fold{f}" for f in range(1, CFG["N_FOLDS"] + 1)])
    progress.to_csv(DIRS["results"] / "training_progress.csv")          # full fold table (all arms) is saved, not printed
    if MONITOR is None:
        MONITOR = Monitor(QUEUE, n_workers=max(N_WORKERS, 1), t0=time.time(), live_board=False)
    MONITOR.final_summary()

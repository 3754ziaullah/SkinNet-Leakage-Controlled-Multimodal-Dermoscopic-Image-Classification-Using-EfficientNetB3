"""SkinNet architecture for loading the trained fold checkpoints."""
import glob
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import keras
from keras import layers

CFG = json.loads('{"SEED": 42, "IMG_SIZE": 300, "BATCH_SIZE": 16, "N_FOLDS": 5, "TEST_SIZE": 0.15, "EPOCHS_P1": 15, "EPOCHS_P2": 20, "WARMUP_EPOCHS": 3, "LR_P1": 0.0001, "LR_P2": 5e-05, "MIN_LR_P1": 1e-07, "MIN_LR_P2": 1e-08, "PATIENCE_P1": 6, "PATIENCE_P2": 5, "STEPS_PER_EPOCH": 800, "GRAD_CLIP": 1.0, "SGD_LR_MULT": 10.0, "DENSE_1": 256, "DENSE_2": 128, "DROPOUT_1": 0.4, "DROPOUT_2": 0.2, "L2": 0.0001, "CBAM_RATIO": 8, "REFINE_CH": 384, "META_DENSE": 32, "LOC_EMB_DIM": 8, "FOCAL_GAMMA": 2.0, "FOCAL_ALPHA": 0.25, "LABEL_SMOOTHING": 0.1, "TARGET_PER_CLASS": 3000, "MAX_OVERSAMPLE_X": 5, "MINORITY_IDX": [0, 1, 3, 6], "CUTMIX_PROB": 0.3, "CUTMIX_ALPHA": 1.0, "MIXUP_PROB": 1.0, "MIXUP_ALPHA": 0.2, "LOC_ENCODING": "onehot", "CLASS_WEIGHT_MODE": "per_sample", "MIXUP_LAMBDA_MODE": "beta", "TTA_MODE": "deterministic", "TTA_VIEWS": 5, "SELECTION_METRIC": "val_macro_auc", "TEST_SPLIT_MODE": "stratified_group", "RUN_PROFILE": "reviewer", "REUSE_P1": true, "EXT_CLASS_MAP": {"NV": "nv", "MEL": "mel", "BCC": "bcc", "BKL": "bkl", "AK": "akiec", "DF": "df", "VASC": "vasc", "SCC": "SCC_OOD"}, "EXT_SITE_MAP": {"anterior torso": "trunk", "posterior torso": "back", "lateral torso": "trunk", "head/neck": "face", "upper extremity": "upper extremity", "lower extremity": "lower extremity", "palms/soles": "acral", "oral/genital": "genital"}, "EXT_EVAL_EXPS": ["M0", "A", "B", "C", "D", "H", "H_NM", "BM_R50", "BM_D121", "BM_CNXT", "BM_V2S", "BM_VIT"], "VIT_PRESET": "vit_base_patch16_224_imagenet", "VIT_SIZE": 224, "LIME_SAMPLES": 1000, "LIME_SEGMENTS": 50, "N_FAITH_CASES": 60, "FAITH_STEPS": 10, "N_PERM_REPEATS": 10, "SHAP_BACKGROUND": 100, "USE_IMAGE_CACHE": true, "BACKBONE_WEIGHTS": "imagenet", "BACKBONE_WEIGHTS_PATH": null, "TTA_FOR_ALL_EXPERIMENTS": false, "KEEP_P1_WEIGHTS": false, "SESSION_BUDGET_H": 9.0, "MAX_HOST_RAM_FRAC": 0.8, "PARALLEL_GPU_WORKERS": true, "ANALYSIS_WHEN_COMPLETE": true, "FORCE_ANALYSIS": false, "CLEAN_AT_END": true, "KEEP_WEIGHTS_FOR": ["B", "D", "H"], "DETERMINISTIC_OPS": false, "N_BOOT": 2000, "ECE_BINS": 15, "XAI_FOLD": 1, "EXPECTED_TEST_COUNTS": [48, 66, 172, 10, 186, 1016, 29], "EXPECTED_VAL_SIZES": [1697, 1698, 1698, 1698, 1697]}')
S = CFG["IMG_SIZE"]
N_CLASSES = 7
SYNTHETIC = False
VIT_DIR = Path("/kaggle/input/models/keras/vit/keras/vit_base_patch16_224_imagenet/3")
H_EXP = SimpleNamespace(code="H", use_image=True, use_meta=True,
                        use_cbam=True, backbone="effb3")

class MetaVector:
    """Encode age, sex and anatomical site with a saved fold preprocessor."""
    def __init__(self, spec):
        self.spec = spec
        self.dense_dim = len(spec["feature_order"])

    def vector(self, age=None, sex="unknown", site="unknown"):
        spec = self.spec
        age_value = spec["age_median"] if age is None else float(age)
        age_range = max(spec["age_max"] - spec["age_min"], 1e-9)
        vector = [(age_value - spec["age_min"]) / age_range]
        sex = str(sex).lower().strip()
        if sex not in spec["sex_categories"]:
            sex = "unknown"
        vector.extend(float(sex == item) for item in spec["sex_categories"])
        site = str(site).lower().strip()
        if site not in spec["loc_vocab"]:
            site = "unknown"
        vector.extend(float(site == item) for item in spec["loc_vocab"])
        result = np.asarray(vector, dtype=np.float32)
        if result.size != self.dense_dim:
            raise ValueError("Metadata features do not match the saved fold specification.")
        return result

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

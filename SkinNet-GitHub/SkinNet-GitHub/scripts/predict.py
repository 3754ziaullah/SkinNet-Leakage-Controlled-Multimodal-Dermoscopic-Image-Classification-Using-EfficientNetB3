"""SkinNet H/J prediction using exported fold weights and metadata preprocessors."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skinnet_model import CFG, H_EXP, MetaVector, build_skinnet, set_phase

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load_image(path):
    image = tf.io.read_file(str(path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    return tf.cast(tf.image.resize(image, [CFG["IMG_SIZE"], CFG["IMG_SIZE"]]), tf.float32)


def image_views(image, use_tta):
    if not use_tta:
        return [image]
    return [image, tf.image.flip_left_right(image), tf.image.flip_up_down(image),
            tf.image.rot90(image, k=1), tf.image.rot90(image, k=3)]


def predict_fold(release, fold, views, age, sex, site):
    spec_file = release / "preprocessing" / f"metadata_fold{fold}.json"
    weights_file = release / "weights" / f"H_fold{fold}.weights.h5"
    if not spec_file.is_file() or not weights_file.is_file():
        raise FileNotFoundError(
            f"Fold {fold} is incomplete. Required files: {spec_file} and {weights_file}. "
            "Export the trained H fold checkpoints using the notebook release cell."
        )
    with spec_file.open(encoding="utf-8") as file:
        metadata = MetaVector(json.load(file))
    model = build_skinnet(H_EXP, metadata.dense_dim, weights=None)
    set_phase(model, phase=2)
    model.load_weights(str(weights_file))
    meta = metadata.vector(age, sex, site)[None]
    probabilities = []
    for view in views:
        pred = model({"image_input": view[None], "meta_input": meta}, training=False)
        probabilities.append(np.asarray(pred[0], dtype=np.float64))
    tf.keras.backend.clear_session()
    return np.mean(probabilities, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Predict seven HAM10000 lesion classes with exported SkinNet weights.")
    parser.add_argument("image", type=Path, help="Dermoscopic RGB image")
    parser.add_argument("--release", type=Path, default=ROOT / "skinnet_release",
                        help="Folder exported by the notebook (default: ./skinnet_release)")
    parser.add_argument("--age", type=float, help="Patient age in years; omit if unavailable")
    parser.add_argument("--sex", choices=["male", "female", "unknown"], default="unknown")
    parser.add_argument("--site", default="unknown", help="Anatomical site as used in HAM10000")
    parser.add_argument("--fold", type=int, choices=range(1, 6), help="Use one fold (default: all five)")
    parser.add_argument("--no-tta", action="store_true", help="Disable five-view test-time augmentation")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.release.is_dir():
        parser.error(f"Release directory not found: {args.release}")

    image = load_image(args.image)
    views = image_views(image, use_tta=not args.no_tta)
    folds = [args.fold] if args.fold else list(range(1, 6))
    preds = [predict_fold(args.release, fold, views, args.age, args.sex, args.site) for fold in folds]
    mean = np.mean(preds, axis=0)
    winner = int(np.argmax(mean))
    print(json.dumps({"predicted_class": CLASSES[winner],
                      "probabilities": {name: round(float(p), 6) for name, p in zip(CLASSES, mean)},
                      "folds": folds, "tta_views": len(views)}, indent=2))
    print("Research model only. These probabilities are not calibrated for clinical use.", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Check the contents of the five-fold SkinNet model release."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check exported SkinNet H checkpoints and metadata preprocessors.")
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    root = args.release
    errors = []
    for fold in range(1, 6):
        weights = root / "weights" / f"H_fold{fold}.weights.h5"
        metadata = root / "preprocessing" / f"metadata_fold{fold}.json"
        if not weights.is_file() or weights.stat().st_size == 0:
            errors.append(f"Missing checkpoint: {weights}")
        if not metadata.is_file():
            errors.append(f"Missing metadata preprocessor: {metadata}")
            continue
        with metadata.open(encoding="utf-8") as file:
            spec = json.load(file)
        if len(spec.get("feature_order", [])) != 19:
            errors.append(f"Unexpected metadata dimension in {metadata}")
        if len(spec.get("loc_vocab", [])) != 15:
            errors.append(f"Unexpected site encoding in {metadata}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("Release check passed: five nonempty H checkpoints and five 19-feature metadata preprocessors.")


if __name__ == "__main__":
    main()

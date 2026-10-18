# SkinNet

Code accompanying **SkinNet: Leakage-Controlled Multimodal Dermoscopic Image Classification Using EfficientNetB3, CBAM Attention, and Clinical Metadata**.

The experiments use HAM10000 for training and internal testing and BCN20000 for frozen external evaluation. The design includes a lesion-grouped test split, five lesion-grouped training/validation folds, component ablations, same-protocol benchmarks and analysis of calibration, melanoma operating points and image/metadata explanations. The original executed notebook records 117 completed fold-runs; those outputs have been removed from the GitHub copy.

This is research software, not a clinical screening tool.

## Files

- `notebooks/SkinNet.ipynb` — complete experiment notebook (source code only; saved cell outputs removed for a smaller GitHub file).
- `skinnet_experiments.py` — the same experiment code as an executable Python script.
- `src/skinnet_worker.py` — background GPU worker used for the per-fold training jobs. The public code is readable rather than stored as an encoded string.
- `src/skinnet_model.py` — SkinNet architecture and metadata encoder for loading saved checkpoints.
- `scripts/predict.py` — image classification with a selected H fold or the five-fold ensemble, with optional five-view TTA.
- `scripts/check_release.py` — verifies that all H checkpoints and per-fold metadata preprocessors are present.

## Environment

The completed experiment used Python 3.12.13, TensorFlow 2.20.0, Keras 3.13.2 and two NVIDIA Tesla T4 GPUs. Other package versions are recorded by the notebook's release cell in `requirements_freeze.txt` when the experiment is exported.

```bash
python -m pip install -r requirements.txt
```

The published measurements depend on GPU execution, pretrained backbone weights and the original dataset partitions. Re-running GPU training with the same seed does not guarantee bitwise-identical weights.

## Datasets

The notebook expects three inputs on Kaggle:

| Input | Kaggle source | Purpose |
| --- | --- | --- |
| HAM10000 | `kmader/skin-cancer-mnist-ham10000` | Training, five-fold validation and internal hold-out testing |
| BCN20000 | `pasutchien/bcn20000` | External evaluation only |
| ViT-B/16 preset | Keras model `keras/vit`, preset `vit_base_patch16_224_imagenet` | Transformer benchmark |

Attach those inputs to the Kaggle session before running. The notebook searches the Kaggle input mounts when an exact directory differs. Keep ImageNet pretrained weights available through Kaggle internet access or the configured local weight cache. Original image datasets are not included in this repository.

## Re-running the analysis

The experiment settings are in the first two code cells of the notebook, notably `SMOKE_TEST`, `CFG` and `CFG["RUN_PROFILE"]`. The default `reviewer` profile is the set of configurations used for the reported study. All models use the same saved splits and the same validation checkpoint rule. Predictions, statistics, audit reports and figures are generated from completed training units.

Run the notebook in order in a GPU-enabled Kaggle session, or run the equivalent script from the repository root:

```bash
python skinnet_experiments.py
```

For a short pipeline check on a subset of the real data:

```bash
SKINNET_SMOKE_TEST=1 python skinnet_experiments.py
```

Smoke-test metrics must not be cited as experimental findings. Each complete run writes to `/kaggle/working/skinnet_revision/`; the notebook can resume completed `(configuration, fold)` units by importing output files from earlier sessions. The notebook export cell writes `/kaggle/working/skinnet_release/` with the final H weights, per-fold metadata preprocessing specifications, and environment/configuration records.

## Inference from exported checkpoints

The model exporter is part of the notebook. Checkpoints and metadata preprocessing files are separate artifacts: the uploaded notebook does not contain those files. Copy the completed Kaggle `skinnet_release` folder alongside this README (or pass its full location to `--release`).

To check the release on Kaggle:

```bash
python scripts/check_release.py /kaggle/working/skinnet_release
```

To examine the inference arguments:

```bash
python scripts/predict.py --help
```

To run a prediction on a JPEG from the attached HAM10000 dataset using the exported H ensemble:

```bash
IMAGE=$(find /kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000 -type f -iname '*.jpg' -print -quit)
python scripts/predict.py "$IMAGE" --release /kaggle/working/skinnet_release
```

Missing metadata are encoded as the training-fold age median and `unknown` sex/site. Predictions without the original patient's metadata are not equivalent to the study's full multimodal evaluation. The `--no-tta` flag runs configuration I; the default five-view inference corresponds to configuration J. No checkpoint is downloaded or silently substituted when weights are missing.

## Evaluation limits

HAM10000 provides lesion identifiers but not unique patient identifiers; the split audit establishes lesion separation, not patient-level independence. The independent BCN20000 evaluation exhibits a large performance drop. The model has not been prospectively validated, and the code must not be used for clinical screening or diagnosis.

## Data and code availability

Data are obtained from their original providers. The repository package contains the executable code but not raw images, trained weights or result CSVs from the separate Kaggle output directory. The notebook writes a release directory and results bundle after evaluation; upload those completed artifacts separately if the publication requires public models or per-fold predictions.

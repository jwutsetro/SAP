# SAP (Skeletal Anatomical Pretraining)

This repo contains a clean, minimal reference for:
1. Running inference with the SAP skeletal model (formerly CATL) on whole-body images.
2. Fine-tuning SAP for downstream tasks using the IB-sampling patch-based loader.

The accepted paper PDF is in `1-s2.0-S2212137426000072-main.pdf`.

## Quick Start

### 1. Environment

Install core deps (PyTorch/MONAI should match your CUDA setup):

```bash
pip install -r requirements.txt
```

### 2. SAP Inference

Expected input layout (one folder per case):

```
input_dir/
  CASE_001/
    T1.nii.gz
    b1000.nii.gz
  CASE_002/
    T1.nii.gz
    b1000.nii.gz
```

Inference is driven by a JSON config. Edit `sap/infer_config.json` and run:

```bash
python sap/infer.py sap/infer_config.json
```

If no path is provided, `sap/infer.py` defaults to `sap/infer_config.json`.

Set `model_type` in the config to control the output classes:
1. `\"skeletal\"` → `out_channels=10`
2. `\"lesion\"` → `out_channels=2` (background vs lesion)

Optional: save a probability map for a specific class by setting
`"save_probs": true` and `"prob_class": <idx>` in the config.

```bash
python sap/infer.py \
  --input_dir /path/to/input_dir \
  --output_dir /path/to/output_dir \
  --weights /path/to/sap_checkpoint.pt \
  --save_probs --prob_class 1
```

### 3. SAP Fine-Tuning

The fine-tuning script uses the IB-sampling patch layout. A helper script to
generate patches from whole-body volumes is included at
`utils/ib_sampling/prepare_dataset.py`.

```
train_dir/
  lesion_patches/
    PATIENTID_T1_positive_0001.nii.gz
    PATIENTID_b1000_positive_0001.nii.gz
    PATIENTID_label_positive_0001.nii.gz
  background_patches/
    PATIENTID_T1_negative_0001.nii.gz
    PATIENTID_b1000_negative_0001.nii.gz
    PATIENTID_label_negative_0001.nii.gz
```

Fine-tuning is driven by a JSON config. Edit `sap/finetune_config.json` and run:

```bash
python sap/finetune.py sap/finetune_config.json
```

If no path is provided, `sap/finetune.py` defaults to `sap/finetune_config.json`.

Other initialization modes:

```bash
# Scratch
# Set \"training_mode\": \"Scratch\" in the config.

# SSL (self-supervised backbone)
# Set \"training_mode\": \"SSL\" and provide \"ssl_checkpoint\" in the config.
```

## Dataloader (IB-sampling)

The patch-based loader is vendored from the IB-sampling repository and lives in
`utils/ib_sampling/`. It comes from:

- https://github.com/jwutsetro/IB-sampling

The accompanying paper is:

- Instance-Balanced Patch Sampling for Whole-Body Lesion Segmentation (MICCAI 2025)
  https://doi.org/10.1007/978-3-032-08009-7_18

## Notes

- `SAP` replaces the older name `CATL`.
- The scripts are kept minimal on purpose and avoid hard-coded paths.
- If your checkpoints come from `torch.nn.DataParallel`, the scripts automatically strip the `module.` prefix.

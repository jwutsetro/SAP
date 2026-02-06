import json
from pathlib import Path
from types import SimpleNamespace

DEFAULT_INFER = {
    "model_type": "skeletal",
    "input_dir": None,
    "case_list": None,
    "modalities": ["T1.nii.gz", "b1000.nii.gz"],
    "output_dir": None,
    "weights": None,
    "roi_x": 96,
    "roi_y": 96,
    "roi_z": 96,
    "sw_batch_size": 4,
    "infer_overlap": 0.5,
    "feature_size": 48,
    "in_channels": 2,
    "out_channels": None,
    "use_checkpoint": False,
    "save_probs": False,
    "prob_class": 1,
    "no_amp": False,
}

DEFAULT_FINETUNE = {
    "train_dir": None,
    "val_dir": None,
    "data_dir": None,
    "output_dir": "./runs",
    "run_name": "sap_finetune",
    "training_mode": "SAP",
    "sap_checkpoint": None,
    "ssl_checkpoint": None,
    "max_epochs": 400,
    "batch_size": 2,
    "val_every": 5,
    "roi_x": 96,
    "roi_y": 96,
    "roi_z": 96,
    "sw_batch_size": 4,
    "infer_overlap": 0.5,
    "in_channels": 2,
    "out_channels": 2,
    "feature_size": 48,
    "use_checkpoint": False,
    "optim_lr": 1e-5,
    "reg_weight": 1e-5,
    "encoder_lr_sap": 2e-6,
    "decoder_lr_sap": 1e-5,
    "encoder_lr_ssl": 5e-6,
    "decoder_lr_ssl": 1e-5,
    "lrschedule": "warmup_cosine",
    "warmup_epochs": 10,
    "squared_dice": False,
    "smooth_nr": 1e-5,
    "smooth_dr": 1e-5,
    "ratio": 1,
    "modalities": None,
    "num_workers": 10,
    "seed": 42,
    "no_amp": False,
}


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object.")
    return data


def load_config(path: Path, defaults: dict) -> SimpleNamespace:
    data = _load_json(path)
    merged = defaults.copy()
    merged.update(data)
    return SimpleNamespace(**merged)

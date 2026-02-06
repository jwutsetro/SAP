import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import Compose, EnsureChannelFirstD, LoadImageD, ScaleIntensityRangePercentilesD

from sap.config import DEFAULT_INFER, load_config

DEFAULT_CONFIG = Path(__file__).with_name("infer_config.json")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run SAP inference.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Path to the inference JSON config.",
    )
    parser.add_argument(
        "--majority-vote",
        action="store_true",
        help="Apply connected-component majority vote postprocessing.",
    )
    return parser.parse_args()


def _strip_prefix(state_dict, prefix):
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _normalize_state_dict(state_dict):
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "backbone.")
    # Normalize potential Swin naming from SSL pretraining.
    renamed = {}
    for k, v in state_dict.items():
        if "swin_vit" in k:
            k = k.replace("swin_vit", "swinViT")
        renamed[k] = v
    return renamed


def _load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return _normalize_state_dict(ckpt)


def _build_cases(input_dir: str, case_list: str, modalities: List[str]) -> List[Tuple[str, str]]:
    cases = []
    if case_list:
        with open(case_list, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line_path = Path(line)
                if line_path.is_dir():
                    case_id = line_path.name
                    cases.append((case_id, str(line_path)))
                else:
                    case_id = line
                    cases.append((case_id, str(Path(input_dir) / case_id)))
    else:
        for entry in sorted(Path(input_dir).iterdir()):
            if entry.is_dir():
                cases.append((entry.name, str(entry)))
    if not cases:
        raise ValueError("No cases found. Check --input_dir or --case_list.")

    # Basic validation of modalities for the first case.
    sample_case_dir = Path(cases[0][1])
    missing = [m for m in modalities if not (sample_case_dir / m).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing modalities in sample case {cases[0][0]}: {missing}. "
            "Check --modalities and input directory layout."
        )
    return cases


def _majority_vote_components(pred: np.ndarray) -> np.ndarray:
    mask = pred > 0
    if not np.any(mask):
        return pred
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labeled, num = ndimage.label(mask, structure=structure)
    if num == 0:
        return pred
    out = pred.copy()
    for cid in range(1, num + 1):
        component = labeled == cid
        labels = pred[component]
        if labels.size == 0:
            continue
        counts = np.bincount(labels)
        majority_label = int(np.argmax(counts))
        out[component] = majority_label
    return out


def main():
    parsed = _parse_args()
    config_path = Path(parsed.config)
    args = load_config(config_path, DEFAULT_INFER)
    if parsed.majority_vote:
        args.majority_vote = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not getattr(args, "no_amp", False)) and device.type == "cuda"

    if not args.input_dir or not args.output_dir or not args.weights:
        raise ValueError("Config must define input_dir, output_dir, and weights.")
    if args.out_channels is None:
        if getattr(args, "model_type", "skeletal") == "lesion":
            args.out_channels = 2
        else:
            args.out_channels = 10
    if isinstance(args.modalities, str):
        args.modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    cases = _build_cases(args.input_dir, args.case_list, args.modalities)
    os.makedirs(args.output_dir, exist_ok=True)

    transforms = Compose(
        [
            LoadImageD(keys=["image"], ensure_channel_first=True),
            EnsureChannelFirstD(keys="image"),
            ScaleIntensityRangePercentilesD(
                keys="image", lower=1, upper=99, b_min=0.0, b_max=1.0, channel_wise=True
            ),
        ]
    )

    dataset = Dataset(
        data=[
            {
                "image": [os.path.join(case_dir, m) for m in args.modalities],
                "case_id": case_id,
            }
            for case_id, case_dir in cases
        ],
        transform=transforms,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        dropout_path_rate=0.1,
        use_checkpoint=args.use_checkpoint,
    )
    state_dict = _load_checkpoint(args.weights)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: missing keys={len(missing)}, unexpected keys={len(unexpected)}")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            case_ids = batch["case_id"]
            with torch.cuda.amp.autocast(enabled=amp):
                logits = sliding_window_inference(
                    images,
                    (args.roi_x, args.roi_y, args.roi_z),
                    args.sw_batch_size,
                    model,
                    overlap=args.infer_overlap,
                )
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

            for i, case_id in enumerate(case_ids):
                affine = batch["image_meta_dict"]["affine"][i].cpu().numpy()
                pred_np = preds[i].cpu().numpy().astype(np.uint8)
                if args.majority_vote:
                    pred_np = _majority_vote_components(pred_np)
                out_path = os.path.join(args.output_dir, f"{case_id}_sap_seg.nii.gz")
                nib.save(nib.Nifti1Image(pred_np, affine), out_path)

                if args.save_probs:
                    cls = int(args.prob_class)
                    prob_np = probs[i, cls].cpu().numpy().astype(np.float32)
                    prob_path = os.path.join(args.output_dir, f"{case_id}_sap_prob_c{cls}.nii.gz")
                    nib.save(nib.Nifti1Image(prob_np, affine), prob_path)

                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

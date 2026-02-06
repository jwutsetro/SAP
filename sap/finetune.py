import os
import time
import sys
from pathlib import Path
import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.ib_sampling.loader import get_loader

from sap.config import DEFAULT_FINETUNE, load_config

DEFAULT_CONFIG = Path(__file__).with_name("finetune_config.json")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _strip_prefix(state_dict, prefix):
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _normalize_state_dict(state_dict):
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "backbone.")
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


def separate_encoder_decoder_params(model):
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "swin_vit" in name or "swinViT" in name or "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    return encoder_params, decoder_params


def reinit_final_layer(model):
    if hasattr(model, "out") and hasattr(model.out, "weight"):
        torch.nn.init.xavier_uniform_(model.out.weight)
        if model.out.bias is not None:
            torch.nn.init.zeros_(model.out.bias)


def filter_encoder_weights(state_dict, remove_patterns):
    return {k: v for k, v in state_dict.items() if not any(p in k for p in remove_patterns)}


def build_model(args):
    return SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        dropout_path_rate=0.1,
        use_checkpoint=args.use_checkpoint,
    )


def train_one_epoch(model, loader, optimizer, scaler, loss_func, device, amp):
    model.train()
    running = []
    for batch in loader:
        data = batch["image"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running.append(loss.item())
    return float(np.mean(running)) if running else 0.0


def validate_one_epoch(model, loader, acc_func, post_label, post_pred, device, amp, inferer=None):
    model.eval()
    acc_func.reset()
    with torch.no_grad():
        for batch in loader:
            data = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = inferer(data) if inferer is not None else model(data)
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(x) for x in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(x) for x in val_outputs_list]
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
        acc, not_nans = acc_func.aggregate()
    if torch.is_tensor(acc):
        acc = acc.cpu().numpy()
    if torch.is_tensor(not_nans):
        not_nans = not_nans.cpu().numpy()
    if np.isscalar(acc):
        return float(acc)
    return float(np.mean(acc))


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    args = load_config(config_path, DEFAULT_FINETUNE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    set_seed(args.seed)
    args.rank = 0
    args.world_size = 1
    args.distributed = False

    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.train_dir is None:
        if args.data_dir:
            args.train_dir = args.data_dir
        else:
            raise ValueError("--train_dir is required (or use --data_dir).")

    loader = get_loader(args)
    train_loader, val_loader = loader

    model = build_model(args)

    if args.training_mode == "SAP":
        if not args.sap_checkpoint:
            raise ValueError("--sap_checkpoint is required when training_mode=SAP")
        sap_state = _load_checkpoint(args.sap_checkpoint)
        sap_state = filter_encoder_weights(sap_state, remove_patterns=["out"])
        missing, unexpected = model.load_state_dict(sap_state, strict=False)
        if missing or unexpected:
            print(f"SAP load: missing={len(missing)} unexpected={len(unexpected)}")
        reinit_final_layer(model)
        encoder_params, decoder_params = separate_encoder_decoder_params(model)
        optimizer = torch.optim.Adam(
            [
                {"params": encoder_params, "lr": args.encoder_lr_sap, "weight_decay": args.reg_weight},
                {"params": decoder_params, "lr": args.decoder_lr_sap, "weight_decay": args.reg_weight},
            ]
        )
    elif args.training_mode == "SSL":
        if not args.ssl_checkpoint:
            raise ValueError("--ssl_checkpoint is required when training_mode=SSL")
        ssl_state = _load_checkpoint(args.ssl_checkpoint)
        ssl_state = filter_encoder_weights(ssl_state, remove_patterns=["out", "decoder"])
        missing, unexpected = model.load_state_dict(ssl_state, strict=False)
        if missing or unexpected:
            print(f"SSL load: missing={len(missing)} unexpected={len(unexpected)}")
        reinit_final_layer(model)
        encoder_params, decoder_params = separate_encoder_decoder_params(model)
        optimizer = torch.optim.Adam(
            [
                {"params": encoder_params, "lr": args.encoder_lr_ssl, "weight_decay": args.reg_weight},
                {"params": decoder_params, "lr": args.decoder_lr_ssl, "weight_decay": args.reg_weight},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)

    if args.squared_dice:
        loss_func = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        loss_func = DiceCELoss(
            include_background=False, to_onehot_y=True, softmax=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )

    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    acc_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

    inferer = lambda x: sliding_window_inference(
        x,
        (args.roi_x, args.roi_y, args.roi_z),
        args.sw_batch_size,
        model,
        overlap=args.infer_overlap,
    )

    model.to(device)

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_acc = -1.0

    for epoch in range(args.max_epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_func, device, amp)
        print(f"Epoch {epoch + 1}/{args.max_epochs} - loss: {train_loss:.4f} - time: {time.time() - start_time:.2f}s")

        if (epoch + 1) % args.val_every == 0 and val_loader is not None:
            val_acc = validate_one_epoch(model, val_loader, acc_func, post_label, post_pred, device, amp, inferer)
            print(f"Val {epoch + 1}/{args.max_epochs} - dice: {val_acc:.4f}")
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, os.path.join(run_dir, "model_final.pt"))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, os.path.join(run_dir, "model_best.pt"))

        if scheduler is not None:
            scheduler.step()

    print(f"Training finished. Best dice: {best_acc:.4f}")


if __name__ == "__main__":
    main()

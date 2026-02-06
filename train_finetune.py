import argparse
import os
from functools import partial
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
import wandb

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils_patches import get_loader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete
from collections import OrderedDict

import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to an available port
        return s.getsockname()[1]



def separate_encoder_decoder_params(model):
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'swin_vit' in name or 'swinViT' in name or 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    return encoder_params, decoder_params

def reinit_final_layer(model):
    if hasattr(model, 'out') and hasattr(model.out, 'weight'):
        torch.nn.init.xavier_uniform_(model.out.weight)
        if model.out.bias is not None:
            torch.nn.init.zeros_(model.out.bias)

def remove_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return new_state_dict
    return state_dict

def filter_encoder_weights(state_dict, remove_patterns):
    filtered = {}
    for k, v in state_dict.items():
        if not any(p in k for p in remove_patterns):
            filtered[k] = v
    return filtered

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default='/FARM/jwuts/Train_MICCAI/Training_paper_suprem/Training_paper')
parser.add_argument("--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory")
parser.add_argument("--val_WB", action="store_true", help="use WB_validation")
parser.add_argument("--WB_dir", default='/FARM/jwuts/data/metastatic_data_RAS', type=str, help="directory to load the whole-body data")
parser.add_argument("--data_dir", default='/FARM/jwuts/data/metastatic_data_RAS_patches', type=str, help="directory to load the data")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=400, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=2, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--num_workers", default=10, type=int, help="number of workers")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--smooth_dr", default=1e-5, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=1e-5, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--training_mode", default="Scratch", choices=["Scratch", "CATL", "SSL"], help="Specify the training mode for fine-tuning")
parser.add_argument("--split", default=1, type=int, help="the cross-validation split")
parser.add_argument("--output_save_dir", default='/FARM/jwuts/Train_MICCAI/Training_after_feedback/output', type=str)
parser.add_argument("--seed", default=42, type=int, help="random seed for reproducibility")
parser.add_argument("--ratio", default=1, type=float, help="ratio of positives to negatives")

def main():
    args = parser.parse_args()
    args.distributed ==True
    # Initialize wandb
    wandb.init(project='Your_Project_Name')
    # Override args with wandb config if present
    if hasattr(wandb.config, 'split'):
        args.split = wandb.config.split
    if hasattr(wandb.config, 'training_mode'):
        args.training_mode = wandb.config.training_mode

    os.environ["WANDB_API_KEY"] = 'e48ad9985da342b54e99acab9b58eea42a966039'
    args.amp = True

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

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

    if args.training_mode == "Scratch":
        # From scratch
        args.optim_lr = 1e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)

    elif args.training_mode == "SSL":
        ssl_ckpt = torch.load("/FARM/jwuts/Train_MICCAI/Pretrain/runs/crimson-flower-23/model_last.pt", map_location='cpu')
        ssl_state = ssl_ckpt["state_dict"] if "state_dict" in ssl_ckpt else ssl_ckpt
        ssl_state = remove_module_prefix(ssl_state)
        ssl_state = filter_encoder_weights(ssl_state, remove_patterns=["out", "decoder"])
        model.load_state_dict(ssl_state, strict=False)
        reinit_final_layer(model)
        encoder_params, decoder_params = separate_encoder_decoder_params(model)
        encoder_lr = 5e-6
        decoder_lr = 1e-5
        optimizer = torch.optim.Adam([
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": args.reg_weight},
            {"params": decoder_params, "lr": decoder_lr, "weight_decay": args.reg_weight}
        ])

    elif args.training_mode == "CATL":
        related_ckpt = torch.load("/FARM/jwuts/Train_MICCAI/Training_suprem_bone_Pretrain/Training_paper/cross_validation/1/from_scratch/model_final.pt", map_location='cpu')
        related_state = related_ckpt["state_dict"] if "state_dict" in related_ckpt else related_ckpt
        related_state = remove_module_prefix(related_state)
        related_state = filter_encoder_weights(related_state, remove_patterns=["out"])
        model.load_state_dict(related_state, strict=False)
        reinit_final_layer(model)
        encoder_params, decoder_params = separate_encoder_decoder_params(model)
        encoder_lr = 2e-6
        decoder_lr = 1e-5
        optimizer = torch.optim.Adam([
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": args.reg_weight},
            {"params": decoder_params, "lr": decoder_lr, "weight_decay": args.reg_weight}
        ])

    else:
        args.optim_lr = 1e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)

    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceCELoss(
            include_background=False,to_onehot_y=True, softmax=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )

    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    acc_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    start_epoch = 0
    best_acc = 0

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        scheduler = None

    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=acc_func,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy

if __name__ == "__main__":
    parser.set_defaults(dist_url=f"tcp://127.0.0.1:{find_free_port()}")
    main()

import argparse
import os
from functools import partial

import numpy as np
import torch
import nibabel as nib
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils_wb_finetune_inference import get_loader

from monai.inferers import sliding_window_inference
import os
import shutil
import time
import wandb
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
import wandb

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="/rhea/scratch/brussel/vo/000/bvo00025/vsc10468/models/SWIN_UNETR/Training_paper/cross_validation/", type=str, help="pretrained checkpoint directory"
)
#parser.add_argument("--data_dir", default="/rhea/scratch/brussel/vo/000/bvo00025/vsc10468/Data/metastatic_dataset_RAS", type=str, help="dataset directory")
parser.add_argument("--data_dir", default="/rhea/scratch/brussel/vo/000/bvo00025/vsc10468/Data/PICRIB_data_complete_37_aligned", type=str, help="dataset directory")
parser.add_argument(
    "--pretrained_model_name",
    default="model.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")

parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=2, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=10, type=int, help="number of output channels")
parser.add_argument("--num_workers", default=10, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")


parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--training_mode", default="frozen_encoder", choices=["from_scratch", "frozen_encoder", "lowered_encoder"], help="Specify the training mode for fine-tuning")
parser.add_argument("--split", default=1, type=int, help="the cross validation split")


def main(args):
    loader = get_loader(args)[1]
    acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    args.amp = not args.noamp
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
    
    #model_name=args.pretrained_dir+'/'+str(args.split)+'/'+args.training_mode+'/model.pt'
    model_name="/FARM/jwuts/Train_MICCAI/Training_suprem_bone_Pretrain/Training_paper/cross_validation/1/from_scratch/model_final.pt"
    model_dict = torch.load(model_name)["state_dict"]
    model.load_state_dict(model_dict)


    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    dice_acc = DiceMetric(include_background=False, reduction='mean_batch', get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    model.cuda()
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):

            data = batch_data["image"]
            original_affine = batch_data["image_meta_dict"]["affine"][0].numpy()
            img_name = batch_data["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            print(img_name)
            data = data.cuda()
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                    logits = F.softmax(logits, dim=1)

                    #logits = torch.argmax(logits, dim=1)
                    print(logits.shape)

                    

            #val_outputs_arg = np.argmax(logits.cpu().detach().numpy(), axis=1).astype(np.uint8)[0]
            val_outputs_arg = logits.cpu().detach().numpy()[0,1,:,:,:]
            current_output_directory='/rhea/scratch/brussel/vo/000/bvo00025/vsc10468/Data/output/'+args.training_mode+'/'+img_name+'_output.nii.gz'
            nib.save(
                nib.Nifti1Image(val_outputs_arg,original_affine),current_output_directory
            )    
            
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
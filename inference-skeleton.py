import argparse
import glob
import os
from functools import partial
import os
import numpy as np
import torch
import nibabel as nib
import argparse
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, ScaleIntensityRangePercentilesD,AsDiscrete
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import compute_generalized_dice
from monai.networks.nets import SwinUNETR
from utils.utils import AverageMeter, distributed_all_gather
from monai.data import decollate_batch, list_data_collate
from monai.metrics import DiceMetric
import numpy as np
import torch
import nibabel as nib
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training

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
parser.add_argument("--data_dir", default="/FARM/jwuts/data/PICRIB_data_complete_37_aligned", type=str, help="dataset directory")
parser.add_argument(
    "--pretrained_model_name",
    default="model.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=32, type=int, help="number of sliding window batch size")
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
def remove_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return new_state_dict
    return state_dict

def main(args):
     # Define preprocessing pipeline
    transforms = Compose([
        LoadImageD(keys=["image"],ensure_channel_first=True),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityRangePercentilesD(
            keys="image",
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            channel_wise=True
        ),
    ])

    modalities=['T1.nii.gz','b1000.nii.gz']
    # Prepare dataset and DataLoader
    #val_patients = glob.glob(args.data_dir+'/*')
    #val_patients = [pat.split('/')[-1] for pat in val_patients]
    val_patients=['PICRIB_VUB_4B', 'PICRIB_VUB_6B', 'PICRIB_Erasme_1', 'PICRIB_VUB_2', 'PICRIB_Erasme_2B', 'PICRIB_St_Luc_6A', 'PICRIB_Erasme_6B', 'PICRIB_Erasme_5A', 'PICRIB_St_Luc_1B']
    dataset = Dataset(
        data=[
            {
                "image": [os.path.join(args.data_dir, patient, m) for m in modalities],
                "patient_id": patient
            }
            for patient in val_patients
        ],
        transform=transforms
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
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
    #ssl_ckpt = torch.load("/FARM/jwuts/Train_MICCAI/Pretrain/runs/crimson-flower-23/model_last.pt", map_location='cpu')
    model_dict = torch.load("/FARM/jwuts/Train_MICCAI/Pretrain/runs/crimson-flower-23/model_last.pt")["state_dict"]
    related_ckpt = torch.load("/FARM/jwuts/Train_MICCAI/Training_suprem_bone_Pretrain/Training_paper/cross_validation/1/from_scratch/model_final.pt", map_location='cpu')
    related_state = related_ckpt["state_dict"] if "state_dict" in related_ckpt else related_ckpt
    related_state = remove_module_prefix(related_state)
    model.load_state_dict(related_state)
    # model_path='/FARM/jwuts/Train_MICCAI/Training_paper_suprem/Training_paper/cross_validation_large/'+str(1)+'/'+'CATL'+'/model_final.pt'
    # state_dict = torch.load(model_path, map_location='cpu')
    # if "state_dict" in state_dict:
    #     state_dict = state_dict["state_dict"]
    # state_dict = remove_module_prefix(state_dict)
    # model.load_state_dict(state_dict)
    #model.load_state_dict(related_state)


    post_pred = AsDiscrete(argmax=True, to_onehot=10)
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
        for idx, batch_data in enumerate(dataloader):

            data = batch_data["image"]
            original_affine = batch_data["image_meta_dict"]["affine"][0].numpy()
            img_name = batch_data["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            print(img_name)
            data = data.cuda()
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                    logits = F.softmax(logits, dim=1)

                    binary_logits =  logits[:, 1:].sum(dim=1)

                    binary_logits = binary_logits > 0.1 
                    #binary_logits = torch.argmax(logits, dim=1)
                    #print(binary_logits.shape)

                    

            #val_outputs_arg = np.argmax(logits.cpu().detach().numpy(), axis=1).astype(np.uint8)[0]
            val_outputs_arg = binary_logits.cpu().detach().numpy()[0,:,:,:].astype(float)
            
            current_output_directory='skeleton_seg_split1/'+img_name+'_output_lesion_bin.nii.gz'
            nib.save(
                nib.Nifti1Image(val_outputs_arg,original_affine),current_output_directory
            )    
            
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
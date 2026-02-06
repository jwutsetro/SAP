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
def remove_module_prefix(state_dict):
    """Removes 'module.' prefix from state_dict keys if present."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def get_validation_patients(patients, split):
    """
    Splits the patients list into 7 folds and returns the validation patients for the given split.

    Args:
        patients (list of str): List of patient IDs.
        split (int): The split index (1-based).

    Returns:
        list of str: Validation patients for the given split.
    """
    np.random.seed(42)
    np.random.shuffle(patients)

    split=int(split)
    num_patients = len(patients)
    num_val_patients = max(1, num_patients // 7)

    folds = [patients[i:i + num_val_patients] for i in range(0, num_patients, num_val_patients)]
    if len(folds) > 7:
        folds[-2].extend(folds[-1])
        folds.pop()

    val_fold_index = (split - 1) % len(folds)
    val_patients = folds[val_fold_index]

    return val_patients
    
def validate_model(data_dir, output_dir, training_mode, split, modalities, roi_size=(96, 96, 96), sw_batch_size=42, batch_size=1, args=None):
    """
    Validates a model on whole-body images and saves the predictions.

    Args:
        data_dir (str): Path to dataset directory.
        output_dir (str): Path to save predictions.
        training_mode (str): Training mode (e.g., "finetune").
        split (str): Data split (e.g., "split1").
        modalities (list of str): List of modality filenames.
        roi_size (tuple): Sliding window ROI size.
        sw_batch_size (int): Sliding window batch size.
        batch_size (int): Batch size for DataLoader.
        args (Namespace): Parsed arguments containing model parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct model path based on training mode and split
    model_path = os.path.join("models", training_mode, f"model_{str(split)}.pth")

    # Load model
    model = build_model(args)
    model_path='/FARM/jwuts/Train_MICCAI/Training_paper_suprem/Training_paper/cross_validation_large/'+str(split)+'/'+training_mode+'/model_final.pt'
    state_dict = torch.load(model_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    all_patients = sorted(os.listdir(data_dir))
    val_patients = get_validation_patients(all_patients, str(split))

    # Create output directory
    mode_output_dir = os.path.join(output_dir, training_mode, str(split))
    os.makedirs(mode_output_dir, exist_ok=True)

    # Define preprocessing pipeline
    transforms = Compose([
        LoadImageD(keys=["image", "gt"],ensure_channel_first=True),
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

    # Prepare dataset and DataLoader
    dataset = Dataset(
        data=[
            {
                "image": [os.path.join(data_dir, patient, m) for m in modalities],
                "gt": os.path.join(data_dir, patient, "GT.nii.gz"),
                "patient_id": patient
            }
            for patient in val_patients
        ],
        transform=transforms
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            gts = batch["gt"].to(device)
            gts = gts == 1
            patient_ids = batch["patient_id"]
    
            run_acc = AverageMeter()
            # Perform sliding window inference
            predictions = sliding_window_inference(images, roi_size, sw_batch_size, model, overlap=0.5)
            predictions_softmax = torch.softmax(predictions, dim=1)
            # Extract the probability map for class 1 (channel 1)
            prob_map_class_1 = predictions_softmax[:, 1, ...]  # Assuming the second channel corresponds to class 1
    
            for i, patient_id in enumerate(patient_ids):
                # Save probability map for class 1
                original_affine = batch["image_meta_dict"]["affine"][i].numpy()
                probability_map = prob_map_class_1[i].cpu().numpy()  # Extract the probability map for the current patient
                patient_name = os.path.join(mode_output_dir, f"{patient_id}_prob_class1.nii.gz")
                
                # Save the probability map as a NIfTI file
                nib.save(
                    nib.Nifti1Image(probability_map.astype(np.float32), original_affine),
                    patient_name
                )
    
            print(f"Saved probability maps for patients: {', '.join(patient_ids)}")


    # with torch.no_grad():
    #     for batch in dataloader:
    #         images = batch["image"].to(device)
    #         gts = batch["gt"].to(device)
    #         gts=gts==1
    #         patient_ids = batch["patient_id"]

    #         run_acc = AverageMeter()
    #         # Perform sliding window inference
    #         predictions = sliding_window_inference(images, roi_size, sw_batch_size, model,overlap=0.5)
    #         #predictions = torch.argmax(predictions, dim=1)

    #         for i, patient_id in enumerate(patient_ids):
    #             # Save prediction using MONAI NiftiSaver
    #             original_affine = batch["gt_meta_dict"]["affine"][i].numpy()

    #             post_label = AsDiscrete(to_onehot=2)
    #             post_pred = AsDiscrete(argmax=True, to_onehot=2)
    #             acc_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
                
    #             val_labels_list = decollate_batch(gts)
    #             val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
    #             val_outputs_list = decollate_batch(predictions)
    #             val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
    
    #             acc_func.reset()
    #             acc_func(y_pred=val_output_convert, y=val_labels_convert)
    #             acc, not_nans = acc_func.aggregate()
    #             run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
    #             dice_score = np.mean(run_acc.avg)

    #             print(f"{patient_id}: {dice_score.item():.4f}")
    #             predictions = torch.argmax(predictions, dim=1)[0]
    #             predictions = predictions.cpu().numpy().astype(np.uint8)      
    #             patient_name=mode_output_dir+'/'+patient_id+'.nii.gz'
    #             nib.save(
    #             nib.Nifti1Image(predictions,original_affine),patient_name
    #         )  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model on whole-body images.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save predictions.")
    parser.add_argument("--training_mode", type=str, required=True, help="Training mode (e.g., 'finetune').")
    parser.add_argument("--split", type=int, required=True, help="Data split (e.g., 'split1').")
    parser.add_argument("--modalities", nargs='+', required=True, help="List of modality filenames.")
    parser.add_argument("--roi_x", type=int, default=96, help="ROI size in x-dimension.")
    parser.add_argument("--roi_y", type=int, default=96, help="ROI size in y-dimension.")
    parser.add_argument("--roi_z", type=int, default=96, help="ROI size in z-dimension.")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels.")
    parser.add_argument("--out_channels", type=int, default=2, help="Number of output channels.")
    parser.add_argument("--feature_size", type=int, default=48, help="Feature size for the model.")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use checkpointing to save memory.")

    args = parser.parse_args()

    validate_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        training_mode=args.training_mode,
        split=args.split,
        modalities=args.modalities,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        args=args
    )
#python validation.py --data_dir=/FARM/jwuts/data/metastatic_data_RAS --output_dir=/FARM/jwuts/data/output_large --training_mode=CATL --split=1 --modalities T1.nii.gz b1000.nii.gz --roi_x=96 --roi_y=96 --roi_z=96 --in_channels=2 --out_channels=2 --feature_size=48 

# import os
# import glob
# import numpy as np
# import nibabel as nib
# import torch
# from torch.utils.data import Dataset, DataLoader, Sampler
# import random
# import time
# from monai.data import list_data_collate
# from monai.transforms import (
#     Compose, RandFlipd, RandRotate90d, ToTensord, RandSpatialCropd, SpatialPadd, RandScaleIntensityd, RandShiftIntensityd
# )
# from monai.transforms import MapTransform


# def train_validate_dicts(data_dir, args):
#     lesion_patch_dir = os.path.join(data_dir, 'lesion_patches')
#     background_patch_dir = os.path.join(data_dir, 'background_patches')

#     lesion_files = glob.glob(os.path.join(lesion_patch_dir, '*.nii*'))
#     background_files = glob.glob(os.path.join(background_patch_dir, '*.nii*'))

#     all_files = lesion_files + background_files

#     patient_ids = set()
#     for file_path in all_files:
#         filename = os.path.basename(file_path)
#         if 'label' in filename:
#             continue

#         filename_no_ext = filename.replace('.nii.gz', '').replace('.nii', '')
#         parts = filename_no_ext.split('_')

#         if len(parts) < 4:
#             continue

#         patient_id = '_'.join(parts[:-3])
#         patient_ids.add(patient_id)

#     patient_ids = sorted(list(patient_ids))
#     print(f"All patient IDs: {patient_ids}")

#     np.random.seed(42)
#     np.random.shuffle(patient_ids)

#     num_patients = len(patient_ids)
#     num_val_patients = max(1, num_patients // 7)

#     folds = [patient_ids[i:i + num_val_patients] for i in range(0, num_patients, num_val_patients)]
#     if len(folds) > 7:
#         folds[-2].extend(folds[-1])
#         folds.pop()

#     val_fold_index = (args.split - 1) % len(folds)
#     val_patient_ids = folds[val_fold_index]
#     train_patient_ids = [p for i, fold in enumerate(folds) if i != val_fold_index for p in fold]

#     print(f"Training patient IDs: {train_patient_ids}")
#     print(f"Validation patient IDs: {val_patient_ids}")

#     return train_patient_ids, val_patient_ids


# class MedicalPatchDataset(Dataset):
#     def __init__(self, root_dir, patch_size, transform=None, rank=0, world_size=1, patient_ids=None, is_training=True):
#         self.root_dir = root_dir
#         self.patch_size = patch_size
#         self.transform = transform
#         self.rank = rank
#         self.world_size = world_size
#         self.is_training = is_training

#         # Initialize shuffle and epoch attributes
#         self.shuffle = True
#         self.epoch = 0

#         percentile_dict_path = os.path.join(root_dir, 'percentile_dict.npy')
#         if not os.path.exists(percentile_dict_path):
#             raise FileNotFoundError(f"Percentile dictionary not found at {percentile_dict_path}")
#         self.percentile_dict = np.load(percentile_dict_path, allow_pickle=True).item()

#         self.samples = []
#         self.modalities = ['T1', 'b1000']

#         lesion_patch_dir = os.path.join(root_dir, 'lesion_patches')
#         background_patch_dir = os.path.join(root_dir, 'background_patches')
#         self._load_samples(lesion_patch_dir, 'positive', patient_ids)

#         if self.is_training:
#             self._load_samples(background_patch_dir, 'negative', patient_ids)

#         self.positive_indices = [i for i, s in enumerate(self.samples) if s['type'] == 'positive']
#         self.negative_indices = [i for i, s in enumerate(self.samples) if s['type'] == 'negative']

#         if self.is_training:
#             self.split_positive_indices()
#             self.positive_cache = {}
#             self.negative_cache = {}
#             self._preload_positive_samples()
#         else:
#             self.cache = {}
#             self._preload_all_samples()

#     def split_positive_indices(self):
#         total_positives = len(self.positive_indices)
#         indices_per_gpu = total_positives // self.world_size
#         extra = total_positives % self.world_size
#         indices = self.positive_indices.copy()
#         if self.shuffle:
#             random.seed(self.rank)
#             random.shuffle(indices)
#         splits = []
#         start = 0
#         for i in range(self.world_size):
#             end = start + indices_per_gpu
#             if i < extra:
#                 end += 1
#             splits.append(indices[start:end])
#             start = end
#         self.positive_indices = splits[self.rank]
#         self.num_positives = len(self.positive_indices)

#     def _preload_positive_samples(self):
#         print(f"GPU {self.rank}: Preloading positive samples into cache...")
#         start_time = time.time()
#         for idx in self.positive_indices:
#             sample = self._load_sample(idx, apply_random_crop=False)
#             self.positive_cache[idx] = sample
#         end_time = time.time()
#         print(f"GPU {self.rank}: Preloaded {len(self.positive_cache)} positive samples in {end_time - start_time:.2f} seconds.")

#     def _preload_negative_samples(self, negative_indices):
#         print(f"GPU {self.rank}: Preloading negative samples into cache...")
#         start_time = time.time()
#         self.negative_cache.clear()
#         for idx in negative_indices:
#             sample = self._load_sample(idx, apply_random_crop=False)
#             self.negative_cache[idx] = sample
#         end_time = time.time()
#         print(f"GPU {self.rank}: Preloaded {len(self.negative_cache)} negative samples in {end_time - start_time:.2f} seconds.")

#     def _preload_all_samples(self):
#         print(f"GPU {self.rank}: Preloading all validation samples into cache...")
#         start_time = time.time()
#         for idx in range(len(self.samples)):
#             sample = self._load_sample(idx, apply_random_crop=False)
#             self.cache[idx] = sample
#         end_time = time.time()
#         print(f"GPU {self.rank}: Preloaded {len(self.cache)} validation samples in {end_time - start_time:.2f} seconds.")

#     def _load_sample(self, idx, apply_random_crop=False):
#         sample_info = self.samples[idx]
#         modalities = sample_info['modalities']
#         label_path = sample_info['label_path']
#         patient_id = sample_info['patient_id']

#         modality_images = []
#         for modality in self.modalities:
#             image_path = modalities[modality]
#             image = nib.load(image_path).get_fdata(dtype=np.float32)

#             key = (patient_id, modality)
#             if key in self.percentile_dict:
#                 p1, p99 = self.percentile_dict[key]
#                 image = np.clip(image, p1, p99)
#                 image = (image - p1) / (p99 - p1)
#             else:
#                 image = (image - image.min()) / (image.max() - image.min())

#             modality_images.append(image)

#         image = np.stack(modality_images, axis=0)
#         label = nib.load(label_path).get_fdata(dtype=np.float32)
#         label = np.expand_dims(label, axis=0)

#         sample = {'image': image, 'label': label}

#         pad_transform = SpatialPadd(
#             keys=['image', 'label'],
#             spatial_size=self.patch_size,
#             method='end'
#         )
#         sample = pad_transform(sample)

#         return sample

#     def __len__(self):
#         if self.is_training:
#             return len(self.positive_cache) + len(self.negative_cache)
#         else:
#             return len(self.samples)

#     def __getitem__(self, idx):
#         if self.is_training:
#             if idx in self.positive_cache:
#                 sample = self.positive_cache[idx]
#                 sample_type = 'positive'
#             elif idx in self.negative_cache:
#                 sample = self.negative_cache[idx]
#                 sample_type = 'negative'
#             else:
#                 sample = self._load_sample(idx, apply_random_crop=False)
#                 sample_type = self.samples[idx]['type']
#                 if sample_type == 'positive':
#                     self.positive_cache[idx] = sample
#                 elif sample_type == 'negative':
#                     self.negative_cache[idx] = sample
#                 else:
#                     raise ValueError(f"Unknown sample type {sample_type} for index {idx}")
#             sample = self._apply_transforms(sample, sample_type=sample_type)
#             return sample
#         else:
#             if idx in self.cache:
#                 sample = self.cache[idx]
#             else:
#                 sample = self._load_sample(idx, apply_random_crop=False)
#                 self.cache[idx] = sample
#             if self.transform:
#                 sample = self.transform(sample)
#             return sample

#     def _apply_transforms(self, sample, sample_type):
#         if sample_type == 'positive':
#             crop = RandSpatialCropd(keys=['image', 'label'], roi_size=self.patch_size, random_size=False)
#             sample = crop(sample)

#         if self.transform:
#             sample = self.transform(sample)
#         return sample

#     def _load_samples(self, patch_dir, sample_type, patient_ids=None):
#         image_files = glob.glob(os.path.join(patch_dir, '*.nii*'))
#         sample_dict = {}
#         for image_path in image_files:
#             filename = os.path.basename(image_path)
#             if 'label' in filename:
#                 continue
#             filename_no_ext = filename.replace('.nii.gz', '').replace('.nii', '')
#             parts = filename_no_ext.split('_')
#             if len(parts) < 4:
#                 continue
#             patch_id = parts[-1]
#             patch_type = parts[-2]
#             modality = parts[-3]
#             pid = '_'.join(parts[:-3])

#             if patient_ids is not None and pid not in patient_ids:
#                 continue

#             patch_key = f"{pid}_{patch_type}_{patch_id}"
#             if patch_key not in sample_dict:
#                 sample_dict[patch_key] = {
#                     'patient_id': pid,
#                     'patch_type': patch_type,
#                     'patch_id': patch_id,
#                     'modalities': {},
#                     'label_path': '',
#                     'type': sample_type
#                 }
#             sample_dict[patch_key]['modalities'][modality] = image_path

#         for patch_key, sample_info in sample_dict.items():
#             label_filename = f"{sample_info['patient_id']}_label_{sample_info['patch_type']}_{sample_info['patch_id']}.nii.gz"
#             label_path = os.path.join(patch_dir, label_filename)
#             if not os.path.exists(label_path):
#                 label_filename = label_filename.replace('.nii.gz', '.nii')
#                 label_path = os.path.join(patch_dir, label_filename)
#                 if not os.path.exists(label_path):
#                     continue
#             sample_info['label_path'] = label_path
#             if set(self.modalities).issubset(sample_info['modalities'].keys()):
#                 self.samples.append(sample_info)

#     def set_epoch(self, epoch):
#         self.epoch = epoch


# class BalancedBatchSampler(Sampler):
#     def __init__(self, dataset, ratio=1, num_replicas=None, rank=None, shuffle=True):
#         self.dataset = dataset
#         self.ratio = ratio
#         self.shuffle = shuffle
#         self.epoch = 0

#         if num_replicas is None:
#             self.num_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
#         else:
#             self.num_replicas = num_replicas

#         if rank is None:
#             self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
#         else:
#             self.rank = rank

#         self.positive_indices = self.dataset.positive_indices
#         self.num_positives = len(self.positive_indices)
#         self.num_negatives_per_epoch = int(self.num_positives * self.ratio)

#     def __iter__(self):
#         g = torch.Generator()
#         g.manual_seed(self.epoch + self.rank)
#         self.positive_indices = self.dataset.positive_indices
#         self.num_positives = len(self.positive_indices)
#         self.num_negatives_per_epoch = int(self.num_positives * self.ratio)

#         negative_indices_all = self.dataset.negative_indices
#         num_negatives = self.num_negatives_per_epoch
#         negative_indices = []
#         if len(negative_indices_all) >= num_negatives:
#             indices = torch.randperm(len(negative_indices_all), generator=g)[:num_negatives].tolist()
#             negative_indices = [negative_indices_all[i] for i in indices]
#         else:
#             negative_indices = negative_indices_all.copy()
#             extra_needed = num_negatives - len(negative_indices)
#             if extra_needed > 0:
#                 indices = torch.randint(len(negative_indices_all), size=(extra_needed,), generator=g).tolist()
#                 negative_indices += [negative_indices_all[i] for i in indices]

#         all_indices = self.positive_indices + negative_indices

#         if self.shuffle:
#             indices = torch.randperm(len(all_indices), generator=g).tolist()
#             all_indices = [all_indices[i] for i in indices]

#         self.dataset._preload_negative_samples(negative_indices)
#         self.num_samples = len(all_indices)

#         print(f"GPU {self.rank}: Number of positives: {len(self.positive_indices)}, Number of negatives: {len(negative_indices)}")

#         return iter(all_indices)

#     def __len__(self):
#         return self.num_samples

#     def set_epoch(self, epoch):
#         self.epoch = epoch


# class LabelMapper(MapTransform):
#     def __init__(self, keys):
#         super().__init__(keys)
#         self.mapping = {
#             0: 0,
#             **{i: 1 for i in range(1, 8)},
#             **{i: 2 for i in range(8, 20)},
#             **{i: 3 for i in range(20, 26)},
#             **{i: 4 for i in range(26, 28)},
#             **{i: 5 for i in range(28, 30)},
#             **{i: 6 for i in range(30, 32)},
#             **{i: 7 for i in range(32, 34)},
#             **{i: 8 for i in range(34, 36)},
#             36: 9
#         }

#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             d[key] = np.vectorize(self.mapping.get)(d[key])
#         return d


# def get_loader(args):
#     root_dir = args.data_dir
#     patch_size = (args.roi_x, args.roi_y, args.roi_z)
#     batch_size = args.batch_size
#     ratio = args.ratio

#     random.seed(args.seed + args.rank)
#     torch.manual_seed(args.seed + args.rank)
#     np.random.seed(args.seed + args.rank)

#     train_transforms = Compose([
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#         RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
#         RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
#         RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
#         ToTensord(keys=["image", "label"]),
#     ])

#     val_transforms = Compose([
#         RandSpatialCropd(keys=['image', 'label'], roi_size=patch_size, random_size=False),
#         ToTensord(keys=["image", "label"]),
#     ])

#     train_patient_ids, val_patient_ids = train_validate_dicts(root_dir, args)

#     from torch.utils.data.distributed import DistributedSampler

#     train_dataset = MedicalPatchDataset(
#         root_dir,
#         patch_size,
#         transform=train_transforms,
#         rank=args.rank,
#         world_size=args.world_size,
#         patient_ids=train_patient_ids,
#         is_training=True
#     )

#     if args.distributed:
#         # BalancedBatchSampler is used for training
#         train_sampler = BalancedBatchSampler(
#             train_dataset,
#             ratio=ratio,
#             num_replicas=args.world_size,
#             rank=args.rank,
#             shuffle=True
#         )
#     else:
#         train_sampler = BalancedBatchSampler(
#             train_dataset,
#             ratio=ratio,
#             num_replicas=1,
#             rank=0,
#             shuffle=True
#         )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         sampler=train_sampler,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         collate_fn=list_data_collate,
#     )

#     # Now create val_dataset on all ranks
#     val_dataset = MedicalPatchDataset(
#         root_dir,
#         patch_size,
#         transform=val_transforms,
#         rank=args.rank,
#         world_size=args.world_size,
#         patient_ids=val_patient_ids,
#         is_training=False
#     )

#     # No distributed sampler for val
#     # Each rank loads the entire validation dataset
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#         collate_fn=list_data_collate,
#     )

#     return train_loader, val_loader

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import time
from monai.data import list_data_collate
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, ToTensord, RandSpatialCropd, SpatialPadd, RandScaleIntensityd, RandShiftIntensityd
)
from monai.transforms import MapTransform


def train_validate_dicts(data_dir, args):
    lesion_patch_dir = os.path.join(data_dir, 'lesion_patches')
    background_patch_dir = os.path.join(data_dir, 'background_patches')

    lesion_files = glob.glob(os.path.join(lesion_patch_dir, '*.nii*'))
    background_files = glob.glob(os.path.join(background_patch_dir, '*.nii*'))

    all_files = lesion_files + background_files

    patient_ids = set()
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if 'label' in filename:
            continue

        filename_no_ext = filename.replace('.nii.gz', '').replace('.nii', '')
        parts = filename_no_ext.split('_')

        if len(parts) < 4:
            continue

        patient_id = '_'.join(parts[:-3])
        patient_ids.add(patient_id)

    patient_ids = sorted(list(patient_ids))
    print(f"All patient IDs: {patient_ids}")

    np.random.seed(42)
    np.random.shuffle(patient_ids)

    num_patients = len(patient_ids)
    num_val_patients = max(1, num_patients // 7)

    folds = [patient_ids[i:i + num_val_patients] for i in range(0, num_patients, num_val_patients)]
    if len(folds) > 7:
        folds[-2].extend(folds[-1])
        folds.pop()

    val_fold_index = (args.split - 1) % len(folds)
    val_patient_ids = folds[val_fold_index]
    train_patient_ids = [p for i, fold in enumerate(folds) if i != val_fold_index for p in fold]

    print(f"Training patient IDs: {train_patient_ids}")
    print(f"Validation patient IDs: {val_patient_ids}")

    return train_patient_ids, val_patient_ids


class MedicalPatchDataset(Dataset):
    def __init__(self, root_dir, patch_size, transform=None, rank=0, world_size=1, patient_ids=None, is_training=True):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.is_training = is_training

        # Initialize shuffle and epoch attributes
        self.shuffle = True
        self.epoch = 0

        percentile_dict_path = os.path.join(root_dir, 'percentile_dict.npy')
        if not os.path.exists(percentile_dict_path):
            raise FileNotFoundError(f"Percentile dictionary not found at {percentile_dict_path}")
        self.percentile_dict = np.load(percentile_dict_path, allow_pickle=True).item()

        self.samples = []
        self.modalities = ['T1', 'b1000']

        lesion_patch_dir = os.path.join(root_dir, 'lesion_patches')
        background_patch_dir = os.path.join(root_dir, 'background_patches')
        self._load_samples(lesion_patch_dir, 'positive', patient_ids)

        if self.is_training:
            self._load_samples(background_patch_dir, 'negative', patient_ids)

        self.positive_indices = [i for i, s in enumerate(self.samples) if s['type'] == 'positive']
        self.negative_indices = [i for i, s in enumerate(self.samples) if s['type'] == 'negative']

        if self.is_training:
            self.split_positive_indices()
            self.positive_cache = {}
            self.negative_cache = {}
            self._preload_positive_samples()
        else:
            self.cache = {}
            self._preload_all_samples()

    def split_positive_indices(self):
        total_positives = len(self.positive_indices)
        # Compute how many samples each GPU will get, ignoring any remainder
        indices_per_gpu = total_positives // self.world_size

        indices = self.positive_indices.copy()
        if self.shuffle:
            random.seed(self.rank)
            random.shuffle(indices)

        # Discard remainder so all GPUs get the same amount
        indices = indices[:indices_per_gpu * self.world_size]
        # Split equally
        start = self.rank * indices_per_gpu
        end = start + indices_per_gpu
        self.positive_indices = indices[start:end]
        self.num_positives = len(self.positive_indices)

    def _preload_positive_samples(self):
        print(f"GPU {self.rank}: Preloading positive samples into cache...")
        start_time = time.time()
        for idx in self.positive_indices:
            sample = self._load_sample(idx, apply_random_crop=False)
            self.positive_cache[idx] = sample
        end_time = time.time()
        print(f"GPU {self.rank}: Preloaded {len(self.positive_cache)} positive samples in {end_time - start_time:.2f} seconds.")

    def _preload_negative_samples(self, negative_indices):
        print(f"GPU {self.rank}: Preloading negative samples into cache...")
        start_time = time.time()
        self.negative_cache.clear()
        for idx in negative_indices:
            sample = self._load_sample(idx, apply_random_crop=False)
            self.negative_cache[idx] = sample
        end_time = time.time()
        print(f"GPU {self.rank}: Preloaded {len(self.negative_cache)} negative samples in {end_time - start_time:.2f} seconds.")

    def _preload_all_samples(self):
        print(f"GPU {self.rank}: Preloading all validation samples into cache...")
        start_time = time.time()
        for idx in range(len(self.samples)):
            sample = self._load_sample(idx, apply_random_crop=False)
            self.cache[idx] = sample
        end_time = time.time()
        print(f"GPU {self.rank}: Preloaded {len(self.cache)} validation samples in {end_time - start_time:.2f} seconds.")

    def _load_sample(self, idx, apply_random_crop=False):
        sample_info = self.samples[idx]
        modalities = sample_info['modalities']
        label_path = sample_info['label_path']
        patient_id = sample_info['patient_id']

        modality_images = []
        for modality in self.modalities:
            image_path = modalities[modality]
            image = nib.load(image_path).get_fdata(dtype=np.float32)

            key = (patient_id, modality)
            if key in self.percentile_dict:
                print('are we reading the percentile dict') 
                p1, p99 = self.percentile_dict[key]
                image = np.clip(image, p1, p99)
                image = (image - p1) / (p99 - p1)
            else:
                print('or are we doing soething stupid')
                image = (image - image.min()) / (image.max() - image.min())

            modality_images.append(image)

        image = np.stack(modality_images, axis=0)
        label = nib.load(label_path).get_fdata(dtype=np.float32)
        label = np.expand_dims(label, axis=0)

        sample = {'image': image, 'label': label}

        pad_transform = SpatialPadd(
            keys=['image', 'label'],
            spatial_size=self.patch_size,
            method='end'
        )
        sample = pad_transform(sample)

        return sample

    def __len__(self):
        if self.is_training:
            return len(self.positive_cache) + len(self.negative_cache)
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        if self.is_training:
            if idx in self.positive_cache:
                sample = self.positive_cache[idx]
                sample_type = 'positive'
            elif idx in self.negative_cache:
                sample = self.negative_cache[idx]
                sample_type = 'negative'
            else:
                sample = self._load_sample(idx, apply_random_crop=False)
                sample_type = self.samples[idx]['type']
                if sample_type == 'positive':
                    self.positive_cache[idx] = sample
                elif sample_type == 'negative':
                    self.negative_cache[idx] = sample
                else:
                    raise ValueError(f"Unknown sample type {sample_type} for index {idx}")
            sample = self._apply_transforms(sample, sample_type=sample_type)
            return sample
        else:
            if idx in self.cache:
                sample = self.cache[idx]
            else:
                sample = self._load_sample(idx, apply_random_crop=False)
                self.cache[idx] = sample
            if self.transform:
                sample = self.transform(sample)
            return sample

    def _apply_transforms(self, sample, sample_type):
        if sample_type == 'positive':
            crop = RandSpatialCropd(keys=['image', 'label'], roi_size=self.patch_size, random_size=False)
            sample = crop(sample)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def _load_samples(self, patch_dir, sample_type, patient_ids=None):
        image_files = glob.glob(os.path.join(patch_dir, '*.nii*'))
        sample_dict = {}
        for image_path in image_files:
            filename = os.path.basename(image_path)
            if 'label' in filename:
                continue
            filename_no_ext = filename.replace('.nii.gz', '').replace('.nii', '')
            parts = filename_no_ext.split('_')
            if len(parts) < 4:
                continue
            patch_id = parts[-1]
            patch_type = parts[-2]
            modality = parts[-3]
            pid = '_'.join(parts[:-3])

            if patient_ids is not None and pid not in patient_ids:
                continue

            patch_key = f"{pid}_{patch_type}_{patch_id}"
            if patch_key not in sample_dict:
                sample_dict[patch_key] = {
                    'patient_id': pid,
                    'patch_type': patch_type,
                    'patch_id': patch_id,
                    'modalities': {},
                    'label_path': '',
                    'type': sample_type
                }
            sample_dict[patch_key]['modalities'][modality] = image_path

        for patch_key, sample_info in sample_dict.items():
            label_filename = f"{sample_info['patient_id']}_label_{sample_info['patch_type']}_{sample_info['patch_id']}.nii.gz"
            label_path = os.path.join(patch_dir, label_filename)
            if not os.path.exists(label_path):
                label_filename = label_filename.replace('.nii.gz', '.nii')
                label_path = os.path.join(patch_dir, label_filename)
                if not os.path.exists(label_path):
                    continue
            sample_info['label_path'] = label_path
            if set(self.modalities).issubset(sample_info['modalities'].keys()):
                self.samples.append(sample_info)

    def set_epoch(self, epoch):
        self.epoch = epoch


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, ratio=1, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.ratio = ratio
        self.shuffle = shuffle
        self.epoch = 0

        if num_replicas is None:
            self.num_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        else:
            self.num_replicas = num_replicas

        if rank is None:
            self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        else:
            self.rank = rank

        self.positive_indices = self.dataset.positive_indices
        self.num_positives = len(self.positive_indices)
        self.num_negatives_per_epoch = int(self.num_positives * self.ratio)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank)
        self.positive_indices = self.dataset.positive_indices
        self.num_positives = len(self.positive_indices)
        self.num_negatives_per_epoch = int(self.num_positives * self.ratio)

        negative_indices_all = self.dataset.negative_indices
        num_negatives = self.num_negatives_per_epoch
        negative_indices = []
        if len(negative_indices_all) >= num_negatives:
            indices = torch.randperm(len(negative_indices_all), generator=g)[:num_negatives].tolist()
            negative_indices = [negative_indices_all[i] for i in indices]
        else:
            negative_indices = negative_indices_all.copy()
            extra_needed = num_negatives - len(negative_indices)
            if extra_needed > 0:
                indices = torch.randint(len(negative_indices_all), size=(extra_needed,), generator=g).tolist()
                negative_indices += [negative_indices_all[i] for i in indices]

        all_indices = self.positive_indices + negative_indices

        if self.shuffle:
            indices = torch.randperm(len(all_indices), generator=g).tolist()
            all_indices = [all_indices[i] for i in indices]

        self.dataset._preload_negative_samples(negative_indices)
        self.num_samples = len(all_indices)

        print(f"GPU {self.rank}: Number of positives: {len(self.positive_indices)}, Number of negatives: {len(negative_indices)}")

        return iter(all_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class LabelMapper(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.mapping = {
            0: 0,
            **{i: 1 for i in range(1, 8)},
            **{i: 2 for i in range(8, 20)},
            **{i: 3 for i in range(20, 26)},
            **{i: 4 for i in range(26, 28)},
            **{i: 5 for i in range(28, 30)},
            **{i: 6 for i in range(30, 32)},
            **{i: 7 for i in range(32, 34)},
            **{i: 8 for i in range(34, 36)},
            36: 9
        }

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.vectorize(self.mapping.get)(d[key])
        return d


def get_loader(args):
    root_dir = args.data_dir
    patch_size = (args.roi_x, args.roi_y, args.roi_z)
    batch_size = args.batch_size
    ratio = args.ratio

    random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)

    train_transforms = Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        RandSpatialCropd(keys=['image', 'label'], roi_size=patch_size, random_size=False),
        ToTensord(keys=["image", "label"]),
    ])

    train_patient_ids, val_patient_ids = train_validate_dicts(root_dir, args)

    from torch.utils.data.distributed import DistributedSampler

    train_dataset = MedicalPatchDataset(
        root_dir,
        patch_size,
        transform=train_transforms,
        rank=args.rank,
        world_size=args.world_size,
        patient_ids=train_patient_ids,
        is_training=True
    )

    if args.distributed:
        # BalancedBatchSampler is used for training
        train_sampler = BalancedBatchSampler(
            train_dataset,
            ratio=ratio,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )
    else:
        train_sampler = BalancedBatchSampler(
            train_dataset,
            ratio=ratio,
            num_replicas=1,
            rank=0,
            shuffle=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )

    # Now create val_dataset on all ranks
    val_dataset = MedicalPatchDataset(
        root_dir,
        patch_size,
        transform=val_transforms,
        rank=args.rank,
        world_size=args.world_size,
        patient_ids=val_patient_ids,
        is_training=False
    )

    # No distributed sampler for val
    # Each rank loads the entire validation dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=list_data_collate,
    )

    return train_loader, val_loader


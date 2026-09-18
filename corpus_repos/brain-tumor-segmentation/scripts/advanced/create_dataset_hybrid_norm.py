"""
BraTS Dataset with Hybrid Normalization
========================================
Key insight from experiment results:
- Percentile norm FIXED T1N/T2W (NCR detection: 83 failures → 9)
- Percentile norm HURT T1C (ET dice: 0.7878 → 0.6551)

Solution: Per-modality normalization strategy
- T1N  → percentile (p1-p99): dark NCR vs background
- T1C  → z-score within brain mask: preserves ET bright signal
- T2W  → percentile (p1-p99): edema detection
- T2F  → percentile (p1-p99): general tumor boundary
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import nibabel as nib
from pathlib import Path
import random


class BraTSDatasetHybridNorm(Dataset):

    def __init__(self, data_dir, split_file, split='train',
                 crop_size=(128, 128, 128), augment=False):

        self.data_dir   = Path(data_dir)
        self.crop_size  = crop_size
        self.augment    = augment
        self.split      = split

        with open(split_file) as f:
            splits = json.load(f)
        self.patient_ids = splits[split]

        # Modality order: T1N, T1C, T2W, T2F
        self.modality_suffixes = ['t1n', 't1c', 't2w', 't2f']

        # Normalization strategy per modality
        # 'percentile' fixed NCR failures, 'zscore' preserves ET signal
        self.norm_strategy = {
            't1n': 'percentile',   # dark NCR → needs percentile
            't1c': 'zscore',       # bright ET → needs zscore
            't2w': 'percentile',   # edema → percentile
            't2f': 'percentile',   # general → percentile
        }

        print(f"[HybridNorm Dataset] {split}: {len(self.patient_ids)} patients")
        print(f"[HybridNorm Dataset] T1N=percentile, T1C=zscore, T2W=percentile, T2F=percentile")

    def __len__(self):
        return len(self.patient_ids)

    def find_patient_dir(self, patient_id):
        exact = self.data_dir / patient_id
        if exact.exists():
            return exact
        for d in self.data_dir.iterdir():
            if patient_id in d.name:
                return d
        return None

    def load_modality(self, patient_dir, suffix):
        patient_dir = Path(patient_dir)
        for pattern in [f"*{suffix}*.nii.gz", f"*{suffix}*.nii",
                        f"*{suffix.upper()}*.nii.gz"]:
            files = list(patient_dir.glob(pattern))
            if files:
                return nib.load(files[0]).get_fdata().astype(np.float32)
        return None

    def load_segmentation(self, patient_dir):
        patient_dir = Path(patient_dir)
        for pattern in ["*seg*.nii.gz", "*seg*.nii", "*mask*.nii.gz"]:
            files = list(patient_dir.glob(pattern))
            if files:
                seg = nib.load(files[0]).get_fdata().astype(np.int32)
                remapped = np.zeros_like(seg)
                remapped[seg == 1] = 1  # NCR
                remapped[seg == 2] = 2  # ED
                remapped[seg == 4] = 3  # ET
                return remapped
        return None

    def percentile_normalize(self, volume):
        """Clip to p1-p99 and normalize to [0,1]. Best for NCR/ED detection."""
        brain = volume[volume > 0]
        if len(brain) == 0:
            return volume
        p1  = np.percentile(brain, 1)
        p99 = np.percentile(brain, 99)
        clipped = np.clip(volume, p1, p99)
        range_val = p99 - p1
        if range_val > 0:
            brain_mask = volume > 0
            out = np.zeros_like(clipped)
            out[brain_mask] = (clipped[brain_mask] - p1) / range_val
        else:
            out = clipped
        return out.astype(np.float32)

    def zscore_normalize(self, volume):
        """
        Z-score normalization within brain mask.
        Better for ET detection because it preserves the relative
        brightness of enhancing tumor on T1C without clipping bright values.
        """
        brain_mask = volume > 0
        brain = volume[brain_mask]
        if len(brain) == 0:
            return volume
        mean = brain.mean()
        std  = brain.std()
        if std > 0:
            out = np.zeros_like(volume)
            out[brain_mask] = (volume[brain_mask] - mean) / std
            # Clip to [-5, 5] to handle outliers without losing ET signal
            out = np.clip(out, -5, 5)
            # Rescale to [0, 1] for consistency
            out = (out + 5) / 10.0
        else:
            out = volume
        return out.astype(np.float32)

    def normalize(self, volume, modality_suffix):
        """Apply the right normalization for each modality."""
        strategy = self.norm_strategy.get(modality_suffix, 'percentile')
        if strategy == 'zscore':
            return self.zscore_normalize(volume)
        else:
            return self.percentile_normalize(volume)

    def random_crop(self, images, label):
        D, H, W = images[0].shape
        cd, ch, cw = self.crop_size
        tumor_voxels = np.argwhere(label > 0)
        if len(tumor_voxels) > 0 and random.random() < 0.8:
            center = tumor_voxels[len(tumor_voxels)//2]
            offset = [random.randint(-20, 20) for _ in range(3)]
            d0 = int(np.clip(center[0]+offset[0]-cd//2, 0, max(0, D-cd)))
            h0 = int(np.clip(center[1]+offset[1]-ch//2, 0, max(0, H-ch)))
            w0 = int(np.clip(center[2]+offset[2]-cw//2, 0, max(0, W-cw)))
        else:
            d0 = random.randint(0, max(0, D-cd))
            h0 = random.randint(0, max(0, H-ch))
            w0 = random.randint(0, max(0, W-cw))
        images = [img[d0:d0+cd, h0:h0+ch, w0:w0+cw] for img in images]
        label  = label[d0:d0+cd, h0:h0+ch, w0:w0+cw]
        return self.pad_if_needed(images, label)

    def center_crop(self, images, label):
        D, H, W = images[0].shape
        cd, ch, cw = self.crop_size
        d0 = max(0, (D-cd)//2)
        h0 = max(0, (H-ch)//2)
        w0 = max(0, (W-cw)//2)
        images = [img[d0:d0+cd, h0:h0+ch, w0:w0+cw] for img in images]
        label  = label[d0:d0+cd, h0:h0+ch, w0:w0+cw]
        return self.pad_if_needed(images, label)

    def pad_if_needed(self, images, label):
        cd, ch, cw = self.crop_size
        D, H, W = images[0].shape
        pad_d = max(0, cd-D)
        pad_h = max(0, ch-H)
        pad_w = max(0, cw-W)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pw = ((0,pad_d),(0,pad_h),(0,pad_w))
            images = [np.pad(img, pw, mode='constant') for img in images]
            label  = np.pad(label, pw, mode='constant')
        return images, label

    def augment_sample(self, images, label):
        for axis in range(3):
            if random.random() < 0.5:
                images = [np.flip(img, axis=axis).copy() for img in images]
                label  = np.flip(label, axis=axis).copy()
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            images = [np.clip(img * scale, 0, 1) for img in images]
        if random.random() < 0.3:
            shift = random.uniform(-0.05, 0.05)
            images = [np.clip(img + shift, 0, 1) for img in images]
        return images, label

    def __getitem__(self, idx):
        patient_id  = self.patient_ids[idx]
        patient_dir = self.find_patient_dir(patient_id)
        if patient_dir is None:
            raise ValueError(f"Cannot find directory for {patient_id}")

        images = []
        for suffix in self.modality_suffixes:
            vol = self.load_modality(patient_dir, suffix)
            if vol is None:
                raise ValueError(f"Cannot find {suffix} for {patient_id}")
            vol_norm = self.normalize(vol, suffix)
            images.append(vol_norm)

        label = self.load_segmentation(patient_dir)
        if label is None:
            raise ValueError(f"Cannot find segmentation for {patient_id}")

        if self.split == 'train':
            images, label = self.random_crop(images, label)
            if self.augment:
                images, label = self.augment_sample(images, label)
        else:
            images, label = self.center_crop(images, label)

        image_tensor = torch.tensor(np.stack(images, axis=0), dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

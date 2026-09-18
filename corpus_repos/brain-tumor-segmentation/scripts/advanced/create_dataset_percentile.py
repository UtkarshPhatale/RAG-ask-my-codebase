"""
BraTS Dataset with Percentile Normalization
Key change: normalize each modality to [0,1] using 1st-99th percentile
This handles scanner variability that causes the 29-31% T1C intensity difference
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import nibabel as nib
from pathlib import Path
import random


class BraTSDatasetPercentile(Dataset):
    """
    BraTS dataset with percentile-based normalization.
    
    Key difference from original:
    - Original: (x - mean) / std  → sensitive to scanner differences
    - New:      (x - p1) / (p99 - p1)  → robust to outliers and scanner variability
    
    This directly addresses the 29-31% T1C intensity difference between
    failure and success cases identified in intensity analysis.
    """
    
    def __init__(self, data_dir, split_file, split='train', 
                 crop_size=(128, 128, 128), augment=False):
        
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.split = split
        
        # Load splits
        with open(split_file) as f:
            splits = json.load(f)
        self.patient_ids = splits[split]
        
        # BraTS modality suffixes — adjust if yours differ
        self.modality_suffixes = ['t1n', 't1c', 't2w', 't2f']
        
        print(f"[Dataset] {split}: {len(self.patient_ids)} patients")
        print(f"[Dataset] Normalization: PERCENTILE (p1-p99)")
        print(f"[Dataset] Crop size: {crop_size}")
        print(f"[Dataset] Augmentation: {augment}")
    
    def __len__(self):
        return len(self.patient_ids)
    
    def find_patient_dir(self, patient_id):
        """Find patient directory."""
        exact = self.data_dir / patient_id
        if exact.exists():
            return exact
        for d in self.data_dir.iterdir():
            if patient_id in d.name:
                return d
        return None
    
    def load_modality(self, patient_dir, suffix):
        """Load a single modality NIfTI file."""
        patient_dir = Path(patient_dir)
        for pattern in [f"*{suffix}*.nii.gz", f"*{suffix}*.nii",
                        f"*{suffix.upper()}*.nii.gz"]:
            files = list(patient_dir.glob(pattern))
            if files:
                return nib.load(files[0]).get_fdata().astype(np.float32)
        return None
    
    def load_segmentation(self, patient_dir):
        """Load segmentation mask."""
        patient_dir = Path(patient_dir)
        for pattern in ["*seg*.nii.gz", "*seg*.nii", "*mask*.nii.gz"]:
            files = list(patient_dir.glob(pattern))
            if files:
                seg = nib.load(files[0]).get_fdata().astype(np.int32)
                # Remap BraTS labels: 0→0, 1→1, 2→2, 4→3
                remapped = np.zeros_like(seg)
                remapped[seg == 1] = 1  # NCR
                remapped[seg == 2] = 2  # ED
                remapped[seg == 4] = 3  # ET
                return remapped
        return None
    
    def percentile_normalize(self, volume):
        """
        Percentile normalization - the key improvement.
        
        Instead of z-score (sensitive to scanner differences),
        we clip to 1st-99th percentile and normalize to [0,1].
        
        This ensures that regardless of scanner intensity scale,
        the normalized values are comparable across patients.
        """
        # Only use brain voxels (non-zero) for percentile calculation
        brain_voxels = volume[volume > 0]
        
        if len(brain_voxels) == 0:
            return volume
        
        # Calculate robust percentiles
        p1  = np.percentile(brain_voxels, 1)
        p99 = np.percentile(brain_voxels, 99)
        
        # Clip and normalize to [0, 1]
        volume_clipped = np.clip(volume, p1, p99)
        
        # Normalize: 0 (background) stays 0, brain tissue goes to [0,1]
        range_val = p99 - p1
        if range_val > 0:
            # Only normalize non-zero voxels (preserve background as 0)
            brain_mask = volume > 0
            volume_norm = np.zeros_like(volume_clipped)
            volume_norm[brain_mask] = (volume_clipped[brain_mask] - p1) / range_val
        else:
            volume_norm = volume_clipped
        
        return volume_norm.astype(np.float32)
    
    def random_crop(self, images, label):
        """Random crop to target size with tumor centering bias."""
        D, H, W = images[0].shape
        cd, ch, cw = self.crop_size
        
        # Find tumor bounding box for biased cropping
        tumor_voxels = np.argwhere(label > 0)
        
        if len(tumor_voxels) > 0 and random.random() < 0.8:
            # 80% chance: center crop around tumor
            center = tumor_voxels[len(tumor_voxels)//2]
            # Add random offset
            offset = [random.randint(-20, 20) for _ in range(3)]
            d0 = int(np.clip(center[0] + offset[0] - cd//2, 0, max(0, D-cd)))
            h0 = int(np.clip(center[1] + offset[1] - ch//2, 0, max(0, H-ch)))
            w0 = int(np.clip(center[2] + offset[2] - cw//2, 0, max(0, W-cw)))
        else:
            # 20% chance: random crop
            d0 = random.randint(0, max(0, D-cd))
            h0 = random.randint(0, max(0, H-ch))
            w0 = random.randint(0, max(0, W-cw))
        
        # Crop
        images_cropped = [img[d0:d0+cd, h0:h0+ch, w0:w0+cw] for img in images]
        label_cropped = label[d0:d0+cd, h0:h0+ch, w0:w0+cw]
        
        # Pad if needed
        images_cropped, label_cropped = self.pad_if_needed(images_cropped, label_cropped)
        
        return images_cropped, label_cropped
    
    def center_crop(self, images, label):
        """Center crop for validation/test."""
        D, H, W = images[0].shape
        cd, ch, cw = self.crop_size
        
        d0 = max(0, (D - cd) // 2)
        h0 = max(0, (H - ch) // 2)
        w0 = max(0, (W - cw) // 2)
        
        images_cropped = [img[d0:d0+cd, h0:h0+ch, w0:w0+cw] for img in images]
        label_cropped = label[d0:d0+cd, h0:h0+ch, w0:w0+cw]
        
        return self.pad_if_needed(images_cropped, label_cropped)
    
    def pad_if_needed(self, images, label):
        """Pad to crop_size if volume is smaller."""
        cd, ch, cw = self.crop_size
        D, H, W = images[0].shape
        
        pad_d = max(0, cd - D)
        pad_h = max(0, ch - H)
        pad_w = max(0, cw - W)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))
            images = [np.pad(img, pad_width, mode='constant') for img in images]
            label = np.pad(label, pad_width, mode='constant')
        
        return images, label
    
    def augment_sample(self, images, label):
        """Safe medical augmentation."""
        # Random flips (safe, most common augmentation)
        for axis in range(3):
            if random.random() < 0.5:
                images = [np.flip(img, axis=axis).copy() for img in images]
                label = np.flip(label, axis=axis).copy()
        
        # Mild intensity scaling (10% variation)
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            images = [np.clip(img * scale, 0, 1) for img in images]
        
        # Mild intensity shift
        if random.random() < 0.3:
            shift = random.uniform(-0.05, 0.05)
            images = [np.clip(img + shift, 0, 1) for img in images]
        
        return images, label
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_dir = self.find_patient_dir(patient_id)
        
        if patient_dir is None:
            raise ValueError(f"Cannot find patient directory for {patient_id}")
        
        # Load all modalities
        images = []
        for suffix in self.modality_suffixes:
            vol = self.load_modality(patient_dir, suffix)
            if vol is None:
                raise ValueError(f"Cannot find {suffix} for {patient_id}")
            # Apply percentile normalization
            vol_norm = self.percentile_normalize(vol)
            images.append(vol_norm)
        
        # Load segmentation
        label = self.load_segmentation(patient_dir)
        if label is None:
            raise ValueError(f"Cannot find segmentation for {patient_id}")
        
        # Crop
        if self.split == 'train':
            images, label = self.random_crop(images, label)
            if self.augment:
                images, label = self.augment_sample(images, label)
        else:
            images, label = self.center_crop(images, label)
        
        # Stack modalities: [4, D, H, W]
        image_tensor = torch.tensor(np.stack(images, axis=0), dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor

"""
BraTS Dataset with Strong Augmentation + Hybrid Normalization
==============================================================
Builds on hybrid_norm (best normalization found so far).
Adds aggressive augmentation to directly combat overfitting.

Augmentation pipeline:
  - Random flips (50% each axis)
  - Random rotation ±15deg (40%)
  - Random elastic deformation (30%)
  - Random Gaussian noise (40%)
  - Random gamma correction (35%)
  - Random zoom 0.85-1.15 (30%)
  - Intensity shift/scale (20%)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import nibabel as nib
from pathlib import Path
import random
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates


class BraTSDatasetAugmented(Dataset):

    def __init__(self, data_dir, split_file, split='train',
                 crop_size=(128, 128, 128), augment=False):

        self.data_dir  = Path(data_dir)
        self.crop_size = crop_size
        self.augment   = augment
        self.split     = split

        with open(split_file) as f:
            splits = json.load(f)
        self.patient_ids = splits[split]

        self.modality_suffixes = ['t1n', 't1c', 't2w', 't2f']

        # Same hybrid normalization as best experiment
        self.norm_strategy = {
            't1n': 'percentile',
            't1c': 'zscore',
            't2w': 'percentile',
            't2f': 'percentile',
        }

        print(f"[AugDataset] {split}: {len(self.patient_ids)} patients | "
              f"augment={augment}")

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
        for pattern in [f"*{suffix}*.nii.gz", f"*{suffix}*.nii",
                        f"*{suffix.upper()}*.nii.gz"]:
            files = list(Path(patient_dir).glob(pattern))
            if files:
                return nib.load(files[0]).get_fdata().astype(np.float32)
        return None

    def load_segmentation(self, patient_dir):
        for pattern in ["*seg*.nii.gz", "*seg*.nii", "*mask*.nii.gz"]:
            files = list(Path(patient_dir).glob(pattern))
            if files:
                seg = nib.load(files[0]).get_fdata().astype(np.int32)
                out = np.zeros_like(seg)
                out[seg == 1] = 1  # NCR
                out[seg == 2] = 2  # ED
                out[seg == 4] = 3  # ET
                return out
        return None

    # ── Normalization ──────────────────────────────────────────────
    def percentile_normalize(self, volume):
        brain = volume[volume > 0]
        if len(brain) == 0:
            return volume
        p1, p99 = np.percentile(brain, 1), np.percentile(brain, 99)
        clipped = np.clip(volume, p1, p99)
        r = p99 - p1
        if r > 0:
            mask = volume > 0
            out  = np.zeros_like(clipped)
            out[mask] = (clipped[mask] - p1) / r
        else:
            out = clipped
        return out.astype(np.float32)

    def zscore_normalize(self, volume):
        mask  = volume > 0
        brain = volume[mask]
        if len(brain) == 0:
            return volume
        mean, std = brain.mean(), brain.std()
        if std > 0:
            out = np.zeros_like(volume)
            out[mask] = (volume[mask] - mean) / std
            out = np.clip(out, -5, 5)
            out = (out + 5) / 10.0
        else:
            out = volume
        return out.astype(np.float32)

    def normalize(self, volume, suffix):
        if self.norm_strategy.get(suffix, 'percentile') == 'zscore':
            return self.zscore_normalize(volume)
        return self.percentile_normalize(volume)

    # ── Cropping ───────────────────────────────────────────────────
    def random_crop(self, images, label):
        D, H, W = images[0].shape
        cd, ch, cw = self.crop_size
        tumor = np.argwhere(label > 0)
        if len(tumor) > 0 and random.random() < 0.8:
            center = tumor[len(tumor)//2]
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
        pd, ph, pw = max(0,cd-D), max(0,ch-H), max(0,cw-W)
        if pd > 0 or ph > 0 or pw > 0:
            p = ((0,pd),(0,ph),(0,pw))
            images = [np.pad(img, p, mode='constant') for img in images]
            label  = np.pad(label, p, mode='constant')
        return images, label

    # ── Augmentation ops ───────────────────────────────────────────
    def aug_flip(self, images, label):
        for axis in range(3):
            if random.random() < 0.5:
                images = [np.flip(img, axis=axis).copy() for img in images]
                label  = np.flip(label, axis=axis).copy()
        return images, label

    def aug_rotate(self, images, label):
        """Random rotation ±15 degrees in a random plane."""
        angle = random.uniform(-15, 15)
        plane = random.choice([(0,1),(0,2),(1,2)])
        axes  = plane
        images = [rotate(img, angle, axes=axes, reshape=False,
                         order=1, mode='constant', cval=0.0)
                  for img in images]
        label = rotate(label.astype(np.float32), angle, axes=axes,
                       reshape=False, order=0, mode='constant', cval=0.0)
        return images, label.astype(np.int32)

    def aug_elastic(self, images, label):
        """
        Elastic deformation — simulates tissue shape variability.
        Uses random displacement fields smoothed with Gaussian filter.
        """
        shape = images[0].shape
        alpha = random.uniform(200, 400)   # deformation strength
        sigma = random.uniform(10, 15)     # smoothness

        # Random displacement fields per axis
        dx = gaussian_filter(
            np.random.randn(*shape).astype(np.float32) * alpha, sigma)
        dy = gaussian_filter(
            np.random.randn(*shape).astype(np.float32) * alpha, sigma)
        dz = gaussian_filter(
            np.random.randn(*shape).astype(np.float32) * alpha, sigma)

        # Build coordinate grid
        x, y, z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]),
            np.arange(shape[2]), indexing='ij'
        )
        coords = [
            np.clip(x+dx, 0, shape[0]-1).ravel(),
            np.clip(y+dy, 0, shape[1]-1).ravel(),
            np.clip(z+dz, 0, shape[2]-1).ravel(),
        ]

        images = [
            map_coordinates(img, coords, order=1, mode='constant'
                            ).reshape(shape).astype(np.float32)
            for img in images
        ]
        label = map_coordinates(
            label.astype(np.float32), coords, order=0, mode='constant'
        ).reshape(shape).astype(np.int32)

        return images, label

    def aug_noise(self, images):
        """Add Gaussian noise to each modality independently."""
        out = []
        for img in images:
            std = random.uniform(0.01, 0.05)
            noise = np.random.randn(*img.shape).astype(np.float32) * std
            out.append(np.clip(img + noise, 0, 1))
        return out

    def aug_gamma(self, images):
        """
        Random gamma correction per modality.
        Simulates scanner contrast variability.
        gamma < 1 brightens, gamma > 1 darkens.
        """
        out = []
        for img in images:
            gamma = random.uniform(0.7, 1.5)
            # Apply only to brain voxels
            corrected = np.power(np.clip(img, 1e-6, 1.0), gamma)
            out.append(corrected.astype(np.float32))
        return out

    def aug_zoom(self, images, label):
        """Random zoom 0.85-1.15, then re-crop to original size."""
        factor = random.uniform(0.85, 1.15)
        cd, ch, cw = self.crop_size

        zoomed_imgs = []
        for img in images:
            z = zoom(img, factor, order=1, mode='constant', cval=0.0)
            zoomed_imgs.append(z)
        zoomed_lbl = zoom(
            label.astype(np.float32), factor,
            order=0, mode='constant', cval=0.0
        ).astype(np.int32)

        # Re-crop back to target size
        zoomed_imgs, zoomed_lbl = self.pad_if_needed(zoomed_imgs, zoomed_lbl)
        D, H, W = zoomed_imgs[0].shape
        d0 = max(0, (D-cd)//2)
        h0 = max(0, (H-ch)//2)
        w0 = max(0, (W-cw)//2)
        zoomed_imgs = [img[d0:d0+cd, h0:h0+ch, w0:w0+cw] for img in zoomed_imgs]
        zoomed_lbl  = zoomed_lbl[d0:d0+cd, h0:h0+ch, w0:w0+cw]
        zoomed_imgs, zoomed_lbl = self.pad_if_needed(zoomed_imgs, zoomed_lbl)

        return zoomed_imgs, zoomed_lbl

    def aug_intensity(self, images):
        """Small intensity scale and shift per modality."""
        out = []
        for img in images:
            scale = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.05, 0.05)
            out.append(np.clip(img * scale + shift, 0, 1).astype(np.float32))
        return out

    def augment_sample(self, images, label):
        """Apply augmentations with independent probabilities."""
        # Always try flips
        images, label = self.aug_flip(images, label)

        # Geometric (applied to both image and label)
        if random.random() < 0.40:
            images, label = self.aug_rotate(images, label)

        if random.random() < 0.30:
            images, label = self.aug_elastic(images, label)

        if random.random() < 0.30:
            images, label = self.aug_zoom(images, label)

        # Intensity (image only)
        if random.random() < 0.40:
            images = self.aug_noise(images)

        if random.random() < 0.35:
            images = self.aug_gamma(images)

        if random.random() < 0.20:
            images = self.aug_intensity(images)

        return images, label

    # ── Main getter ────────────────────────────────────────────────
    def __getitem__(self, idx):
        patient_id  = self.patient_ids[idx]
        patient_dir = self.find_patient_dir(patient_id)
        if patient_dir is None:
            raise ValueError(f"Cannot find dir for {patient_id}")

        images = []
        for suffix in self.modality_suffixes:
            vol = self.load_modality(patient_dir, suffix)
            if vol is None:
                raise ValueError(f"Cannot find {suffix} for {patient_id}")
            images.append(self.normalize(vol, suffix))

        label = self.load_segmentation(patient_dir)
        if label is None:
            raise ValueError(f"Cannot find segmentation for {patient_id}")

        if self.split == 'train':
            images, label = self.random_crop(images, label)
            if self.augment:
                images, label = self.augment_sample(images, label)
        else:
            images, label = self.center_crop(images, label)

        image_tensor = torch.tensor(
            np.stack(images, axis=0), dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

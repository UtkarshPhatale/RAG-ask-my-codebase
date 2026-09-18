"""
Safe Data Augmentation for Medical Images
Conservative parameters to preserve medical information
"""

import torch
import numpy as np
from scipy.ndimage import rotate
import random


class SafeMedicalAugmentation:
    """Conservative augmentation safe for medical images"""
    
    def __init__(self):
        self.flip_prob = 0.5
        self.rotate_prob = 0.3
        self.intensity_prob = 0.3
    
    def __call__(self, volume, segmentation):
        """Apply safe augmentations"""
        
        volume = volume.numpy()
        segmentation = segmentation.numpy()
        
        # 1. Random flip (safest)
        if random.random() < self.flip_prob:
            volume, segmentation = self.random_flip(volume, segmentation)
        
        # 2. Small rotation (5 degrees max)
        if random.random() < self.rotate_prob:
            volume, segmentation = self.small_rotation(volume, segmentation)
        
        # 3. Mild intensity scaling
        if random.random() < self.intensity_prob:
            volume = self.intensity_scale(volume)
        
        return torch.FloatTensor(volume), torch.LongTensor(segmentation)
    
    def random_flip(self, volume, segmentation):
        """Flip along random axes"""
        
        # Randomly choose axes to flip
        flip_x = random.random() < 0.5
        flip_y = random.random() < 0.5
        
        if flip_x or flip_y:
            axes = []
            if flip_x:
                axes.append(1)  # H dimension
            if flip_y:
                axes.append(2)  # W dimension
            
            # Flip volume (all channels)
            volume = np.flip(volume, axis=axes).copy()
            
            # Flip segmentation (adjust indices)
            seg_axes = [ax - 1 for ax in axes]
            segmentation = np.flip(segmentation, axis=seg_axes).copy()
        
        return volume, segmentation
    
    def small_rotation(self, volume, segmentation):
        """Small rotation (max 5 degrees)"""
        
        angle = random.uniform(-5, 5)
        
        # Rotate each channel
        rotated_volume = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            rotated_volume[i] = rotate(
                volume[i], angle, 
                axes=(1, 2), 
                reshape=False, 
                order=1,
                mode='constant',
                cval=0
            )
        
        # Rotate segmentation (nearest neighbor)
        rotated_seg = rotate(
            segmentation, angle,
            axes=(1, 2),
            reshape=False,
            order=0,
            mode='constant',
            cval=0
        )
        
        return rotated_volume, rotated_seg
    
    def intensity_scale(self, volume):
        """Mild intensity scaling per channel"""
        
        scaled = volume.copy()
        for i in range(volume.shape[0]):
            scale = random.uniform(0.95, 1.05)
            scaled[i] = np.clip(volume[i] * scale, 0, 1)
        
        return scaled


def test_augmentation():
    """Test augmentation"""
    print("🧪 Testing safe augmentation...")
    
    volume = torch.randn(4, 128, 128, 128)
    segmentation = torch.randint(0, 4, (128, 128, 128))
    
    aug = SafeMedicalAugmentation()
    aug_vol, aug_seg = aug(volume, segmentation)
    
    print(f"✅ Input: {volume.shape}, Output: {aug_vol.shape}")
    print(f"✅ Seg input: {segmentation.shape}, Output: {aug_seg.shape}")
    print("🎉 Test PASSED!")


if __name__ == "__main__":
    test_augmentation()

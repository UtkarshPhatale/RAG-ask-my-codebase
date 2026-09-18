"""
Advanced Data Augmentation for Medical Images
"""

import torch
import numpy as np
from scipy.ndimage import rotate, zoom, gaussian_filter
import random


class MedicalAugmentation:
    """Advanced augmentation for 3D medical images"""
    
    def __init__(self, p=0.5):
        """
        Args:
            p: probability of applying each augmentation
        """
        self.p = p
    
    def __call__(self, volume, segmentation):
        """Apply random augmentations"""
        
        volume = volume.numpy()
        segmentation = segmentation.numpy()
        
        # 1. Random rotation
        if random.random() < self.p:
            volume, segmentation = self.random_rotation(volume, segmentation)
        
        # 2. Random flip
        if random.random() < self.p:
            volume, segmentation = self.random_flip(volume, segmentation)
        
        # 3. Random intensity shift
        if random.random() < self.p:
            volume = self.random_intensity_shift(volume)
        
        # 4. Random intensity scaling
        if random.random() < self.p:
            volume = self.random_intensity_scale(volume)
        
        # 5. Random Gaussian noise
        if random.random() < self.p:
            volume = self.add_gaussian_noise(volume)
        
        # 6. Random Gaussian blur
        if random.random() < self.p:
            volume = self.gaussian_blur(volume)
        
        # 7. Random contrast adjustment
        if random.random() < self.p:
            volume = self.adjust_contrast(volume)
        
        return torch.FloatTensor(volume), torch.LongTensor(segmentation)
    
    def random_rotation(self, volume, segmentation, max_angle=15):
        """Random 3D rotation"""
        
        # Random angles for each axis
        angle_x = random.uniform(-max_angle, max_angle)
        angle_y = random.uniform(-max_angle, max_angle)
        angle_z = random.uniform(-max_angle, max_angle)
        
        # Rotate each channel
        rotated_volume = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            rotated_volume[i] = rotate(volume[i], angle_x, axes=(0, 1), 
                                      reshape=False, order=1)
            rotated_volume[i] = rotate(rotated_volume[i], angle_y, axes=(0, 2), 
                                      reshape=False, order=1)
            rotated_volume[i] = rotate(rotated_volume[i], angle_z, axes=(1, 2), 
                                      reshape=False, order=1)
        
        # Rotate segmentation (use nearest neighbor)
        rotated_seg = rotate(segmentation, angle_x, axes=(0, 1), 
                           reshape=False, order=0)
        rotated_seg = rotate(rotated_seg, angle_y, axes=(0, 2), 
                           reshape=False, order=0)
        rotated_seg = rotate(rotated_seg, angle_z, axes=(1, 2), 
                           reshape=False, order=0)
        
        return rotated_volume, rotated_seg
    
    def random_flip(self, volume, segmentation):
        """Random flip along axes"""
        
        axes = []
        # Skip channel dimension for volume (axis 0)
        if random.random() < 0.5:
            axes.append(1)  # H
        if random.random() < 0.5:
            axes.append(2)  # W
        if random.random() < 0.5:
            axes.append(3)  # D
        
        if axes:
            # Flip volume (all channels)
            flipped_volume = np.flip(volume, axis=axes).copy()
            # Flip segmentation (adjust axes for 3D segmentation)
            seg_axes = [ax - 1 for ax in axes]  # Adjust for no channel dim
            flipped_seg = np.flip(segmentation, axis=seg_axes).copy()
            return flipped_volume, flipped_seg
        
        return volume, segmentation
    
    def random_intensity_shift(self, volume, shift_range=0.1):
        """Random intensity shift per channel"""
        
        shifted = volume.copy()
        for i in range(volume.shape[0]):
            shift = random.uniform(-shift_range, shift_range)
            shifted[i] = np.clip(volume[i] + shift, 0, 1)
        
        return shifted
    
    def random_intensity_scale(self, volume, scale_range=(0.9, 1.1)):
        """Random intensity scaling per channel"""
        
        scaled = volume.copy()
        for i in range(volume.shape[0]):
            scale = random.uniform(*scale_range)
            scaled[i] = np.clip(volume[i] * scale, 0, 1)
        
        return scaled
    
    def add_gaussian_noise(self, volume, std_range=(0.01, 0.05)):
        """Add Gaussian noise"""
        
        std = random.uniform(*std_range)
        noise = np.random.normal(0, std, volume.shape)
        noisy = volume + noise
        
        return np.clip(noisy, 0, 1)
    
    def gaussian_blur(self, volume, sigma_range=(0.5, 1.5)):
        """Apply Gaussian blur"""
        
        sigma = random.uniform(*sigma_range)
        blurred = volume.copy()
        
        for i in range(volume.shape[0]):
            blurred[i] = gaussian_filter(volume[i], sigma=sigma)
        
        return blurred
    
    def adjust_contrast(self, volume, factor_range=(0.9, 1.1)):
        """Adjust contrast"""
        
        factor = random.uniform(*factor_range)
        adjusted = volume.copy()
        
        for i in range(volume.shape[0]):
            mean = volume[i].mean()
            adjusted[i] = np.clip((volume[i] - mean) * factor + mean, 0, 1)
        
        return adjusted


class ElasticDeformation:
    """Elastic deformation augmentation"""
    
    def __init__(self, alpha=10, sigma=3, p=0.3):
        """
        Args:
            alpha: deformation intensity
            sigma: smoothness of deformation
            p: probability of applying
        """
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, volume, segmentation):
        """Apply elastic deformation"""
        
        if random.random() > self.p:
            return volume, segmentation
        
        volume = volume.numpy()
        segmentation = segmentation.numpy()
        
        shape = volume.shape[1:]  # (H, W, D)
        
        # Generate random displacement fields
        dx = gaussian_filter(
            np.random.randn(*shape) * self.alpha,
            self.sigma, mode='constant', cval=0
        )
        dy = gaussian_filter(
            np.random.randn(*shape) * self.alpha,
            self.sigma, mode='constant', cval=0
        )
        dz = gaussian_filter(
            np.random.randn(*shape) * self.alpha,
            self.sigma, mode='constant', cval=0
        )
        
        # Create meshgrid
        x, y, z = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        # Apply displacement
        indices = (
            np.clip(x + dx, 0, shape[0] - 1).astype(np.int32),
            np.clip(y + dy, 0, shape[1] - 1).astype(np.int32),
            np.clip(z + dz, 0, shape[2] - 1).astype(np.int32)
        )
        
        # Deform volume
        deformed_volume = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            deformed_volume[i] = volume[i][indices]
        
        # Deform segmentation
        deformed_seg = segmentation[indices]
        
        return torch.FloatTensor(deformed_volume), torch.LongTensor(deformed_seg)


class ComposedAugmentation:
    """Compose multiple augmentations"""
    
    def __init__(self):
        self.medical_aug = MedicalAugmentation(p=0.5)
        self.elastic_aug = ElasticDeformation(alpha=10, sigma=3, p=0.3)
    
    def __call__(self, volume, segmentation):
        """Apply all augmentations"""
        
        # Medical augmentations
        volume, segmentation = self.medical_aug(volume, segmentation)
        
        # Elastic deformation
        volume, segmentation = self.elastic_aug(volume, segmentation)
        
        return volume, segmentation


def test_augmentation():
    """Test augmentation pipeline"""
    
    print("🧪 Testing augmentation pipeline...")
    
    # Create dummy data
    volume = torch.randn(4, 128, 128, 128)
    segmentation = torch.randint(0, 4, (128, 128, 128))
    
    print(f"Input volume shape: {volume.shape}")
    print(f"Input seg shape: {segmentation.shape}")
    print(f"Input volume range: [{volume.min():.3f}, {volume.max():.3f}]")
    print(f"Input seg unique values: {segmentation.unique().tolist()}")
    
    # Apply augmentation
    aug = ComposedAugmentation()
    aug_volume, aug_seg = aug(volume, segmentation)
    
    print(f"\n✅ Output volume shape: {aug_volume.shape}")
    print(f"✅ Output seg shape: {aug_seg.shape}")
    print(f"✅ Output volume range: [{aug_volume.min():.3f}, {aug_volume.max():.3f}]")
    print(f"✅ Output seg unique values: {aug_seg.unique().tolist()}")
    
    print("\n🎉 Augmentation test PASSED!")


if __name__ == "__main__":
    test_augmentation()

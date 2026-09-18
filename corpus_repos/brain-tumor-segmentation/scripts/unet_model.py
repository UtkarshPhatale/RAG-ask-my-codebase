"""
3D U-Net Architecture for Brain Tumor Segmentation
Based on: Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (Conv3D -> BatchNorm -> ReLU) x 2"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        # Transposed convolution for upsampling
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatches
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for brain tumor segmentation"""
    
    def __init__(self, in_channels=4, num_classes=4, base_channels=32):
        """
        Args:
            in_channels: Number of input channels (4 for T1, T1CE, T2, FLAIR)
            num_classes: Number of segmentation classes (4 for BraTS: 0, 1, 2, 4)
            base_channels: Base number of feature channels (32 for memory efficiency)
        """
        super(UNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # Decoder (Expanding Path)
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        # Output layer
        self.outc = nn.Conv3d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def test_model():
    """Test the U-Net model with a sample input"""
    
    print("🧪 Testing 3D U-Net Model...")
    
    # Create model
    model = UNet3D(in_channels=4, num_classes=4, base_channels=32)
    
    # Count parameters
    total, trainable = model.count_parameters()
    print(f"✅ Total parameters: {total:,}")
    print(f"✅ Trainable parameters: {trainable:,}")
    
    # Test forward pass
    batch_size = 1
    sample_input = torch.randn(batch_size, 4, 128, 128, 128)
    
    print(f"\n📊 Input shape: {sample_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Expected shape: torch.Size([{batch_size}, 4, 128, 128, 128])")
    
    # Check output
    if output.shape == torch.Size([batch_size, 4, 128, 128, 128]):
        print("\n🎉 Model test PASSED!")
        return True
    else:
        print("\n❌ Model test FAILED - shape mismatch")
        return False


if __name__ == "__main__":
    test_model()

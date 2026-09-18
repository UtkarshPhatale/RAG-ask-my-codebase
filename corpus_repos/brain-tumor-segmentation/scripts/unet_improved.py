"""
Enhanced 3D U-Net with Improved Architecture
- Larger capacity (base_channels=64)
- Residual connections
- Instance normalization
- Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
        
        self.dropout = nn.Dropout3d(0.1)
    
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class EncoderBlock(nn.Module):
    """Encoder block with residual connection"""
    
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2)
    
    def forward(self, x):
        features = self.res_block(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """Decoder block with skip connection"""
    
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Handle size mismatch
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]
        
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                     diffY // 2, diffY - diffY // 2,
                     diffZ // 2, diffZ - diffZ // 2])
        
        # Concatenate with skip connection
        x = torch.cat([skip, x], dim=1)
        x = self.res_block(x)
        
        return x


class ImprovedUNet3D(nn.Module):
    """
    Enhanced 3D U-Net with:
    - Residual connections
    - Instance normalization
    - Dropout regularization
    - Larger capacity
    """
    
    def __init__(self, in_channels=4, num_classes=4, base_channels=64):
        super(ImprovedUNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Initial convolution
        self.input_block = ResidualBlock(in_channels, base_channels)
        
        # Encoder
        self.encoder1 = EncoderBlock(base_channels, base_channels * 2)
        self.encoder2 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.encoder3 = EncoderBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.decoder3 = DecoderBlock(base_channels * 16, base_channels * 8)
        self.decoder2 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.decoder1 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.decoder0 = DecoderBlock(base_channels * 2, base_channels)
        
        # Output
        self.output_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Input
        x0 = self.input_block(x)
        
        # Encoder
        x1, skip1 = self.encoder1(x0)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        
        # Bottleneck
        x4 = self.bottleneck(x3)
        
        # Decoder
        x = self.decoder3(x4, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.decoder0(x, x0)
        
        # Output
        logits = self.output_conv(x)
        
        return logits
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def test_improved_model():
    """Test the improved model"""
    
    print("🧪 Testing Improved 3D U-Net...")
    
    # Create model
    model = ImprovedUNet3D(in_channels=4, num_classes=4, base_channels=64)
    
    # Count parameters
    total, trainable = model.count_parameters()
    print(f"✅ Total parameters: {total:,}")
    print(f"✅ Trainable parameters: {trainable:,}")
    
    # Test forward pass
    batch_size = 1
    sample_input = torch.randn(batch_size, 4, 128, 128, 128)
    
    print(f"\n📊 Input shape: {sample_input.shape}")
    
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Expected shape: torch.Size([{batch_size}, 4, 128, 128, 128])")
    
    if output.shape == torch.Size([batch_size, 4, 128, 128, 128]):
        print("\n🎉 Improved model test PASSED!")
        return True
    else:
        print("\n❌ Model test FAILED")
        return False


if __name__ == "__main__":
    test_improved_model()

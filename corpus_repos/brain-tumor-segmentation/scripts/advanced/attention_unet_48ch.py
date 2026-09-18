"""
attention_unet_48ch.py
48-Channel 3D Attention U-Net with Spatial Dropout
Save to: scripts/advanced/attention_unet_48ch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.block(x))


class AttentionGate(nn.Module):
    """
    F_g = channels of gating signal (from decoder/bottleneck, FULL channel count)
    F_l = channels of skip connection (from encoder)
    F_int = intermediate channels (typically F_l // 2)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm3d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm3d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class AttentionUNet48Ch(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 48,
        deep_supervision: bool = True,
        dropout_p: float = 0.15,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        b = base_channels  # 48

        # Encoder
        self.enc1 = ConvBlock(in_channels, b)        # out: 48
        self.enc2 = ConvBlock(b,     b * 2)          # out: 96
        self.enc3 = ConvBlock(b * 2, b * 4)          # out: 192
        self.enc4 = ConvBlock(b * 4, b * 8)          # out: 384
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(b * 8, b * 16, dropout_p=dropout_p)  # out: 768

        # Attention Gates
        # F_g = full channel count of gating signal BEFORE the up-conv halves it
        # F_l = skip connection channels from encoder
        # F_int = F_l // 2
        self.att4 = AttentionGate(F_g=b * 16, F_l=b * 8, F_int=b * 4)   # g=768, x=384, int=192
        self.att3 = AttentionGate(F_g=b * 8,  F_l=b * 4, F_int=b * 2)   # g=384, x=192, int=96
        self.att2 = AttentionGate(F_g=b * 4,  F_l=b * 2, F_int=b)       # g=192, x=96,  int=48
        self.att1 = AttentionGate(F_g=b * 2,  F_l=b,     F_int=b // 2)  # g=96,  x=48,  int=24

        # Decoder (UpBlock handles the ConvTranspose internally)
        self.up4 = UpBlock(b * 16, b * 8, b * 8)   # 768 -> 384
        self.up3 = UpBlock(b * 8,  b * 4, b * 4)   # 384 -> 192
        self.up2 = UpBlock(b * 4,  b * 2, b * 2)   # 192 -> 96
        self.up1 = UpBlock(b * 2,  b,     b)        # 96  -> 48

        # Output heads
        self.out_main = nn.Conv3d(b,     num_classes, kernel_size=1)
        self.out_aux1 = nn.Conv3d(b * 2, num_classes, kernel_size=1)
        self.out_aux2 = nn.Conv3d(b * 4, num_classes, kernel_size=1)
        self.out_aux3 = nn.Conv3d(b * 8, num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        bn = self.bottleneck(self.pool(e4))

        # Decoder — pass full bottleneck/decoder output as gating signal g
        # then interpolate to match skip spatial size
        g4 = F.interpolate(bn,  size=e4.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.up4(bn,  self.att4(g=g4, x=e4))

        g3 = F.interpolate(d4, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.up3(d4, self.att3(g=g3, x=e3))

        g2 = F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.up2(d3, self.att2(g=g2, x=e2))

        g1 = F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.up1(d2, self.att1(g=g1, x=e1))

        # Outputs
        out_main = self.out_main(d1)

        if self.deep_supervision and self.training:
            out_aux1 = F.interpolate(self.out_aux1(d2), size=x.shape[2:], mode='trilinear', align_corners=False)
            out_aux2 = F.interpolate(self.out_aux2(d3), size=x.shape[2:], mode='trilinear', align_corners=False)
            out_aux3 = F.interpolate(self.out_aux3(d4), size=x.shape[2:], mode='trilinear', align_corners=False)
            return [out_main, out_aux1, out_aux2, out_aux3]

        return out_main


# ---------------------------------------------------------------------------
# Feasibility Check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = AttentionUNet48Ch(
        in_channels=4, num_classes=4,
        base_channels=48, deep_supervision=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:     {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {total_params / 1e6:.1f}M")

    model.train()
    print("\nTesting batch_size=2 ...")
    try:
        x = torch.randn(2, 4, 128, 128, 128).to(device)
        outputs = model(x)
        main_out = outputs[0] if isinstance(outputs, list) else outputs
        print(f"  Output shape: {main_out.shape}")
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory allocated: {mem:.2f} GB")
        print("  PASS: batch_size=2 fits")
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("  OOM at batch_size=2, trying batch_size=1 ...")
            torch.cuda.empty_cache()
            x = torch.randn(1, 4, 128, 128, 128).to(device)
            outputs = model(x)
            main_out = outputs[0] if isinstance(outputs, list) else outputs
            print(f"  Output shape: {main_out.shape}")
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU memory at bs=1: {mem:.2f} GB")
            print("  PASS: batch_size=1 fits")
        else:
            raise e

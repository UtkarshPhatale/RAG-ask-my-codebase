"""
Attention U-Net with Spatial Dropout
======================================
Identical to ImprovedAttentionUNet3D but adds spatial dropout
at the bottleneck (deepest feature level only).

Spatial dropout drops entire feature map channels rather than
individual units — much more effective for 3D CNNs because
adjacent voxels are highly correlated.

dropout_p=0.1 means 10% of feature channels are zeroed each
forward pass during training. Forces redundant representations
and directly reduces overfitting.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from attention_unet_v2_improved import ImprovedAttentionUNet3D


class SpatialDropout3D(nn.Module):
    """
    Drop entire feature map channels (not individual voxels).
    Far more effective than standard dropout for 3D convolutions
    because neighboring voxels share information.
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        B, C = x.shape[:2]
        mask = torch.bernoulli(
            torch.full((B, C, 1, 1, 1), 1 - self.p, device=x.device)
        )
        return x * mask / (1 - self.p)


class AugmentedAttentionUNet3D(ImprovedAttentionUNet3D):
    """
    Extends ImprovedAttentionUNet3D with spatial dropout at bottleneck.
    All other architecture is identical — same weights, same deep
    supervision, same attention gates. Only difference is dropout
    injected at the deepest encoder output during training.
    """

    def __init__(self, in_channels=4, num_classes=4,
                 base_channels=32, deep_supervision=True,
                 dropout_p=0.1):

        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision
        )
        self.spatial_dropout = SpatialDropout3D(p=dropout_p)
        self.dropout_p = dropout_p

    def _find_bottleneck(self):
        """Find the bottleneck module by checking common attribute names."""
        for name in ['bottleneck', 'encoder4', 'down4', 'center',
                     'encoder_blocks', 'conv_blocks']:
            if hasattr(self, name):
                attr = getattr(self, name)
                # If it's a list/ModuleList, take the last one
                if isinstance(attr, (list, nn.ModuleList)):
                    return attr[-1]
                return attr
        return None

    def forward(self, x):
        # Eval mode: no dropout, use parent directly
        if not self.training:
            return super().forward(x)

        # Train mode: inject spatial dropout at bottleneck via hook
        bottleneck = self._find_bottleneck()

        if bottleneck is None:
            # No bottleneck found — safe fallback, training still works
            return super().forward(x)

        def hook_fn(module, input, output):
            return self.spatial_dropout(output)

        handle = bottleneck.register_forward_hook(hook_fn)
        try:
            out = super().forward(x)
        finally:
            handle.remove()

        return out


def get_augmented_model(in_channels=4, num_classes=4,
                        base_channels=32, deep_supervision=True,
                        dropout_p=0.1):
    model = AugmentedAttentionUNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        dropout_p=dropout_p
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[AugmentedUNet] Total params:     {total/1e6:.1f}M")
    print(f"[AugmentedUNet] Trainable params: {trainable/1e6:.1f}M")
    print(f"[AugmentedUNet] Spatial dropout:  p={dropout_p} at bottleneck")
    return model

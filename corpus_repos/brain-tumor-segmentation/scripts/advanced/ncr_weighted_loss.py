import torch
import torch.nn as nn
import torch.nn.functional as F

class NCRWeightedLoss(nn.Module):
    """
    Loss function that heavily weights NCR predictions
    to prevent the model from ignoring necrotic core
    """
    def __init__(self, ncr_weight=5.0, et_weight=2.0, ed_weight=1.0):
        super().__init__()
        self.ncr_weight = ncr_weight
        self.et_weight = et_weight
        self.ed_weight = ed_weight
        
    def dice_loss(self, pred, target, smooth=1e-5):
        """Compute Dice loss for one class"""
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: (B, 4, D, H, W) - logits
            targets: (B, D, H, W) - ground truth labels (0,1,2,4)
        """
        # Convert outputs to probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Create one-hot encoded targets
        # Class 0: background
        # Class 1: NCR (label 1)
        # Class 2: ED (label 2)  
        # Class 3: ET (label 4)
        
        batch_size = targets.size(0)
        device = targets.device
        
        # One-hot encode targets
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.unsqueeze(1).long(), 1)
        
        # Map BraTS labels to class indices
        # 0 -> 0 (background)
        # 1 -> 1 (NCR)
        # 2 -> 2 (ED)
        # 4 -> 3 (ET)
        targets_mapped = targets.clone()
        targets_mapped[targets == 4] = 3
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets_mapped.unsqueeze(1).long(), 1)
        
        # Compute weighted dice loss for each class
        loss_ncr = self.dice_loss(probs[:, 1], targets_onehot[:, 1]) * self.ncr_weight
        loss_ed = self.dice_loss(probs[:, 2], targets_onehot[:, 2]) * self.ed_weight
        loss_et = self.dice_loss(probs[:, 3], targets_onehot[:, 3]) * self.et_weight
        
        # Combined loss
        total_loss = loss_ncr + loss_ed + loss_et
        
        return total_loss

class CombinedNCRLoss(nn.Module):
    """
    Combines NCR-weighted loss with existing combo loss
    """
    def __init__(self):
        super().__init__()
        self.ncr_loss = NCRWeightedLoss(ncr_weight=5.0, et_weight=2.0, ed_weight=1.0)
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        # Map targets for CE loss
        targets_ce = targets.clone()
        targets_ce[targets == 4] = 3
        
        dice = self.ncr_loss(outputs, targets)
        ce = self.ce_loss(outputs, targets_ce.long())
        
        return 0.7 * dice + 0.3 * ce

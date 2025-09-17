import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dynamic_weighting import DynamicWeightAveraging, UncertaintyWeighting

# -----------------------------
# Focal Loss
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# -----------------------------
# Dice Loss
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255, eps: float = 1e-8):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = F.softmax(inputs, dim=1)

        num_classes = inputs.size(1)
        device = inputs.device

        valid_mask = (targets != self.ignore_index)
        if not torch.any(valid_mask):
            return torch.zeros(1, device=device)

        targets_valid = targets.clone()
        targets_valid[~valid_mask] = 0
        targets_one_hot = F.one_hot(targets_valid, num_classes=num_classes).permute(0, 3, 1, 2).float()

        mask = valid_mask.unsqueeze(1).float()
        inputs = inputs * mask
        targets_one_hot = targets_one_hot * mask

        intersection = torch.sum(inputs * targets_one_hot, dim=(2, 3))
        union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)
        dice_loss = 1 - dice.mean()

        return torch.clamp(dice_loss, min=0.0, max=1.0)


# -----------------------------
# Depth Loss
# -----------------------------
class DepthLoss(nn.Module):
    def __init__(self, l1_weight=1.0, l2_weight=0.5, grad_weight=0.5, mask_invalid=True, eps=1e-6):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.grad_weight = grad_weight
        self.mask_invalid = mask_invalid
        self.eps = eps

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.size(2) <= 1 or pred.size(3) <= 1:
            return torch.zeros(1, device=pred.device)

        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return grad_loss_x + grad_loss_y

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = pred.device

        pred = torch.clamp(pred, min=0.01, max=100.0)
        target = torch.clamp(target, min=0.01, max=100.0)

        if self.mask_invalid:
            valid_mask = (target > 0.01) & (target < 100.0) & torch.isfinite(target)
            if valid_mask.float().mean() < 0.1:
                return {k: torch.zeros(1, device=device) for k in ["depth_loss", "depth_l1", "depth_l2", "depth_grad"]}

            pred = torch.where(valid_mask, pred, torch.zeros_like(pred))
            target = torch.where(valid_mask, target, torch.zeros_like(target))

        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)

        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss + self.grad_weight * grad_loss

        return {
            "depth_loss": total_loss,
            "depth_l1": l1_loss,
            "depth_l2": l2_loss,
            "depth_grad": grad_loss,
        }


# -----------------------------
# Combined Segmentation Loss with Label Smoothing
# -----------------------------
class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_weight=1.0, focal_weight=0.5, dice_weight=0.5, ignore_index=255, label_smoothing=0.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        ce_loss = self.ce_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)

        total_loss = self.ce_weight * ce_loss + self.focal_weight * focal_loss + self.dice_weight * dice_loss

        return {"seg_loss": total_loss, "seg_ce": ce_loss, "seg_focal": focal_loss, "seg_dice": dice_loss}


# -----------------------------
# Enhanced MTAN Loss with Dynamic Weighting
# -----------------------------
class MTANLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Enhanced segmentation loss with label smoothing
        label_smoothing = getattr(config.training, 'label_smoothing', 0.0)
        self.seg_loss_fn = CombinedSegmentationLoss(label_smoothing=label_smoothing)
        self.depth_loss_fn = DepthLoss()
        
        # Task weighting strategy
        self.task_weighting = getattr(config.training, 'task_weighting', 'equal')
        
        if self.task_weighting == 'dwa':
            self.dwa = DynamicWeightAveraging(
                num_tasks=2, 
                temperature=getattr(config.training, 'dwa_temperature', 2.0)
            )
        elif self.task_weighting == 'uncertainty':
            self.uncertainty_weighting = UncertaintyWeighting(num_tasks=2)
        
        # Initial weights (used for equal weighting or DWA initialization)
        self.seg_weight = config.training.seg_loss_weight
        self.depth_weight = config.training.depth_loss_weight
        self.aux_weight = config.training.aux_loss_weight
        self.use_aux = config.model.use_aux_loss
        
        # Track loss history for DWA
        self.loss_history = []
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Calculate individual task losses
        seg_losses = self.seg_loss_fn(outputs["segmentation"], targets["segmentation"])
        losses.update({f"main_{k}": v for k, v in seg_losses.items()})
        
        depth_losses = self.depth_loss_fn(outputs["depth"], targets["depth"])
        losses.update({f"main_{k}": v for k, v in depth_losses.items()})
        
        # Handle auxiliary losses
        if self.use_aux and "aux" in outputs:
            aux_seg_losses = self.seg_loss_fn(outputs["aux"], targets["segmentation"])
            losses.update({f"aux_{k}": v * self.aux_weight for k, v in aux_seg_losses.items()})
        
        # Get main task losses for weighting
        main_seg_loss = losses["main_seg_loss"]
        main_depth_loss = losses["main_depth_loss"]
        
        # Apply dynamic weighting
        if self.task_weighting == 'dwa':
            # Update DWA weights
            current_losses = [main_seg_loss.item(), main_depth_loss.item()]
            weights = self.dwa.update_weights(current_losses)
            seg_weight, depth_weight = weights[0], weights[1]
            
            # Store weights for logging
            losses["dwa_seg_weight"] = torch.tensor(seg_weight)
            losses["dwa_depth_weight"] = torch.tensor(depth_weight)
            
        elif self.task_weighting == 'uncertainty':
            # Use uncertainty weighting
            task_losses = [main_seg_loss, main_depth_loss]
            total_loss = self.uncertainty_weighting.forward(task_losses)
            
            # Get current weights for logging
            uncertainty_weights = self.uncertainty_weighting.get_weights()
            losses["uncertainty_seg_weight"] = torch.tensor(uncertainty_weights[0])
            losses["uncertainty_depth_weight"] = torch.tensor(uncertainty_weights[1])
            losses["total_loss"] = total_loss
            
            return losses
        else:
            # Equal weighting (original approach)
            seg_weight = self.seg_weight
            depth_weight = self.depth_weight
        
        # Calculate weighted total loss
        total_loss = (
            main_seg_loss * seg_weight +
            main_depth_loss * depth_weight
        )
        
        # Add auxiliary losses
        if self.use_aux:
            total_loss += losses.get("aux_seg_loss", 0.0)
        
        losses["total_loss"] = total_loss
        
        # Store current weights for logging
        losses["current_seg_weight"] = torch.tensor(seg_weight)
        losses["current_depth_weight"] = torch.tensor(depth_weight)
        
        return losses
    
    def get_task_weights(self):
        """Get current task weights for logging/analysis"""
        if self.task_weighting == 'dwa' and hasattr(self, 'dwa'):
            return self.dwa.get_current_weights()
        elif self.task_weighting == 'uncertainty' and hasattr(self, 'uncertainty_weighting'):
            return self.uncertainty_weighting.get_weights()
        else:
            return [self.seg_weight, self.depth_weight]
    
    def reset_weighting(self):
        """Reset dynamic weighting (useful for validation)"""
        if self.task_weighting == 'dwa' and hasattr(self, 'dwa'):
            self.dwa.reset()


# -----------------------------
# Uncertainty Weighting
# -----------------------------
class LossScheduler:
    def __init__(self, method="uncertainty"):
        self.method = method
        self.log_vars = nn.Parameter(torch.zeros(2))  # [seg, depth]

    def uncertainty_weighting(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seg_loss = losses["main_seg_loss"]
        depth_loss = losses["main_depth_loss"]

        weighted_seg = torch.exp(-self.log_vars[0]) * seg_loss + self.log_vars[0]
        weighted_depth = torch.exp(-self.log_vars[1]) * depth_loss + self.log_vars[1]

        losses["weighted_seg_loss"] = weighted_seg
        losses["weighted_depth_loss"] = weighted_depth
        losses["total_loss"] = weighted_seg + weighted_depth
        return losses

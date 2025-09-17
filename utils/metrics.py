import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
import torchmetrics

class SegmentationMetrics:
    """Comprehensive segmentation metrics"""
    def __init__(self, num_classes: int, ignore_index: int = 255, device: str = 'cuda'):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        
        # Initialize torchmetrics
        self.iou = torchmetrics.JaccardIndex(
            task='multiclass', 
            num_classes=num_classes, 
            ignore_index=ignore_index,
            average=None
        ).to(device)
        
        self.accuracy = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='macro'
        ).to(device)
        
        self.f1_score = torchmetrics.F1Score(
            task='multiclass',
            num_classes=num_classes,
            ignore_index=ignore_index,
            average=None
        ).to(device)
        
        # For confusion matrix calculation
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task='multiclass',
            num_classes=num_classes,
            ignore_index=ignore_index,
            normalize='true'
        ).to(device)
        
        # Reset metrics
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.iou.reset()
        self.accuracy.reset()
        self.f1_score.reset()
        self.confusion_matrix.reset()
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with new predictions"""
        pred = pred.argmax(dim=1)  # Convert logits to class predictions
        
        # Update all metrics
        self.iou.update(pred, target)
        self.accuracy.update(pred, target)
        self.f1_score.update(pred, target)
        self.confusion_matrix.update(pred, target)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        iou_per_class = self.iou.compute()
        mean_iou = iou_per_class.mean()
        
        accuracy = self.accuracy.compute()
        f1_per_class = self.f1_score.compute()
        mean_f1 = f1_per_class.mean()
        
        conf_matrix = self.confusion_matrix.compute()
        
        return {
            'mIoU': mean_iou.item(),
            'IoU_per_class': iou_per_class.cpu().numpy(),
            'pixel_accuracy': accuracy.item(),
            'mF1': mean_f1.item(),
            'F1_per_class': f1_per_class.cpu().numpy(),
            'confusion_matrix': conf_matrix.cpu().numpy()
        }
    
    def get_class_metrics(self, class_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Get per-class metrics with class names"""
        metrics = self.compute()
        
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            if i < len(metrics['IoU_per_class']):
                class_metrics[class_name] = {
                    'IoU': metrics['IoU_per_class'][i],
                    'F1': metrics['F1_per_class'][i]
                }
        
        return class_metrics

class DepthMetrics:
    """Comprehensive depth estimation metrics"""
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulated values"""
        self.abs_rel_errors = []
        self.sq_rel_errors = []
        self.rmse_errors = []
        self.rmse_log_errors = []
        self.delta1_accuracies = []
        self.delta2_accuracies = []
        self.delta3_accuracies = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Update metrics with new predictions"""
        if mask is None:
            # Create mask for valid depth values
            mask = (target > 0) & (target < 1)  # Assuming normalized depth
        
        pred = pred[mask]
        target = target[mask]
        
        if pred.numel() == 0:  # No valid pixels
            return
        
        # Absolute relative error
        abs_rel = torch.mean(torch.abs(pred - target) / target)
        self.abs_rel_errors.append(abs_rel.item())
        
        # Squared relative error
        sq_rel = torch.mean(((pred - target) ** 2) / target)
        self.sq_rel_errors.append(sq_rel.item())
        
        # RMSE
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        self.rmse_errors.append(rmse.item())
        
        # RMSE log
        rmse_log = torch.sqrt(torch.mean((torch.log(pred + 1e-8) - torch.log(target + 1e-8)) ** 2))
        self.rmse_log_errors.append(rmse_log.item())
        
        # Delta accuracies
        ratio = torch.max(pred / target, target / pred)
        delta1 = torch.mean((ratio < 1.25).float())
        delta2 = torch.mean((ratio < 1.25 ** 2).float())
        delta3 = torch.mean((ratio < 1.25 ** 3).float())
        
        self.delta1_accuracies.append(delta1.item())
        self.delta2_accuracies.append(delta2.item())
        self.delta3_accuracies.append(delta3.item())
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        if not self.abs_rel_errors:
            return {
                'abs_rel': 0.0,
                'sq_rel': 0.0,
                'rmse': 0.0,
                'rmse_log': 0.0,
                'delta1': 0.0,
                'delta2': 0.0,
                'delta3': 0.0
            }
        
        return {
            'abs_rel': np.mean(self.abs_rel_errors),
            'sq_rel': np.mean(self.sq_rel_errors),
            'rmse': np.mean(self.rmse_errors),
            'rmse_log': np.mean(self.rmse_log_errors),
            'delta1': np.mean(self.delta1_accuracies),
            'delta2': np.mean(self.delta2_accuracies),
            'delta3': np.mean(self.delta3_accuracies)
        }

class MultiTaskMetrics:
    """Combined metrics for multi-task learning"""
    def __init__(self, num_classes: int, class_names: List[str], 
                 ignore_index: int = 255, device: str = 'cuda'):
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        
        # Initialize task-specific metrics
        self.seg_metrics = SegmentationMetrics(num_classes, ignore_index, device)
        self.depth_metrics = DepthMetrics(device)
    
    def reset(self):
        """Reset all metrics"""
        self.seg_metrics.reset()
        self.depth_metrics.reset()
    
    def update(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Update metrics with model outputs and targets"""
        # Update segmentation metrics
        self.seg_metrics.update(outputs['segmentation'], targets['segmentation'])
        
        # Update depth metrics
        self.depth_metrics.update(outputs['depth'], targets['depth'])
    
    def compute(self) -> Dict[str, Dict[str, float]]:
        """Compute all metrics"""
        seg_results = self.seg_metrics.compute()
        depth_results = self.depth_metrics.compute()
        
        return {
            'segmentation': seg_results,
            'depth': depth_results
        }
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary metrics for logging"""
        results = self.compute()
        
        return {
            'seg_mIoU': results['segmentation']['mIoU'],
            'seg_pixel_acc': results['segmentation']['pixel_accuracy'],
            'seg_mF1': results['segmentation']['mF1'],
            'depth_abs_rel': results['depth']['abs_rel'],
            'depth_rmse': results['depth']['rmse'],
            'depth_delta1': results['depth']['delta1']
        }
    
    def get_detailed_results(self) -> str:
        """Get detailed results as formatted string"""
        results = self.compute()
        
        # Segmentation results
        seg_text = "Segmentation Results:\n"
        seg_text += f"  mIoU: {results['segmentation']['mIoU']:.4f}\n"
        seg_text += f"  Pixel Accuracy: {results['segmentation']['pixel_accuracy']:.4f}\n"
        seg_text += f"  mF1: {results['segmentation']['mF1']:.4f}\n"
        
        # Per-class IoU
        seg_text += "  Per-class IoU:\n"
        for i, class_name in enumerate(self.class_names):
            if i < len(results['segmentation']['IoU_per_class']):
                iou = results['segmentation']['IoU_per_class'][i]
                seg_text += f"    {class_name}: {iou:.4f}\n"
        
        # Depth results
        depth_text = "\nDepth Estimation Results:\n"
        depth_text += f"  Abs Rel: {results['depth']['abs_rel']:.4f}\n"
        depth_text += f"  Sq Rel: {results['depth']['sq_rel']:.4f}\n"
        depth_text += f"  RMSE: {results['depth']['rmse']:.4f}\n"
        depth_text += f"  RMSE Log: {results['depth']['rmse_log']:.4f}\n"
        depth_text += f"  δ1: {results['depth']['delta1']:.4f}\n"
        depth_text += f"  δ2: {results['depth']['delta2']:.4f}\n"
        depth_text += f"  δ3: {results['depth']['delta3']:.4f}\n"
        
        return seg_text + depth_text

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience: int = 7, min_delta: float = 0, 
                 restore_best_weights: bool = True, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop early"""
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print('Restored best weights')
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

class MetricTracker:
    """Track metrics over training epochs"""
    def __init__(self):
        self.metrics_history = {
            'train': [],
            'val': []
        }
    
    def update(self, metrics: Dict[str, float], split: str):
        """Update metrics for a split"""
        self.metrics_history[split].append(metrics)
    
    def get_best_epoch(self, metric_name: str, split: str = 'val', mode: str = 'max') -> Tuple[int, float]:
        """Get best epoch for a specific metric"""
        if not self.metrics_history[split]:
            return 0, 0.0
        
        values = [m.get(metric_name, 0.0) for m in self.metrics_history[split]]
        
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return best_idx, values[best_idx]
    
    def get_latest(self, split: str = 'val') -> Dict[str, float]:
        """Get latest metrics"""
        if not self.metrics_history[split]:
            return {}
        return self.metrics_history[split][-1]
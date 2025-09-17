import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import torch
import torch.nn.functional as F
import cv2
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import json

from data import CityscapesVisualization
from config import CITYSCAPES_CLASSES, CITYSCAPES_COLORS

class TrainingVisualizer:
    """Visualization tools for training progress and results"""
    
    def __init__(self, output_dir: str = './outputs'):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Setup matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, metric_tracker: Any, save_path: Optional[str] = None):
        """Plot training and validation curves"""
        if not metric_tracker.metrics_history['train']:
            return
        
        train_history = metric_tracker.metrics_history['train']
        val_history = metric_tracker.metrics_history['val']
        
        epochs = range(1, len(train_history) + 1)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(epochs, [m['loss'] for m in train_history], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, [m['loss'] for m in val_history], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Segmentation mIoU
        axes[0, 1].plot(epochs, [m['seg_mIoU'] for m in train_history], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, [m['seg_mIoU'] for m in val_history], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Segmentation mIoU', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mIoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Segmentation Pixel Accuracy
        axes[0, 2].plot(epochs, [m['seg_pixel_acc'] for m in train_history], 'b-', label='Train', linewidth=2)
        axes[0, 2].plot(epochs, [m['seg_pixel_acc'] for m in val_history], 'r-', label='Val', linewidth=2)
        axes[0, 2].set_title('Pixel Accuracy', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Depth Absolute Relative Error
        axes[1, 0].plot(epochs, [m['depth_abs_rel'] for m in train_history], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, [m['depth_abs_rel'] for m in val_history], 'r-', label='Val', linewidth=2)
        axes[1, 0].set_title('Depth Abs Rel Error', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Depth RMSE
        axes[1, 1].plot(epochs, [m['depth_rmse'] for m in train_history], 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, [m['depth_rmse'] for m in val_history], 'r-', label='Val', linewidth=2)
        axes[1, 1].set_title('Depth RMSE', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Depth Delta1 Accuracy
        axes[1, 2].plot(epochs, [m['depth_delta1'] for m in train_history], 'b-', label='Train', linewidth=2)
        axes[1, 2].plot(epochs, [m['depth_delta1'] for m in val_history], 'r-', label='Val', linewidth=2)
        axes[1, 2].set_title('Depth δ1 Accuracy', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'training_curves.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_path}")
    
    def visualize_predictions(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                            config: Any, num_samples: int = 8, save_path: Optional[str] = None):
        """Visualize model predictions"""
        model.eval()
        device = next(model.parameters()).device
        
        # Get random samples
        samples = []
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            samples.append(batch)
        
        # Create visualization grid
        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        with torch.no_grad():
            for idx, batch in enumerate(samples):
                # Move to device
                images = batch['image'].to(device)
                seg_targets = batch['segmentation']
                depth_targets = batch['depth']
                
                # Get predictions
                outputs = model(images)
                seg_pred = outputs['segmentation']
                depth_pred = outputs['depth']
                
                # Convert to numpy and denormalize
                image = images[0].cpu()
                seg_target = seg_targets[0].cpu().numpy()
                depth_target = depth_targets[0, 0].cpu().numpy()
                seg_pred = torch.softmax(seg_pred[0], dim=0).argmax(dim=0).cpu().numpy()
                depth_pred = depth_pred[0, 0].cpu().numpy()
                
                # Denormalize image
                image_np = CityscapesVisualization.denormalize_image(
                    image, config.data.normalize_mean, config.data.normalize_std
                )
                
                # Create visualizations
                seg_target_color = CityscapesVisualization.decode_segmap(seg_target)
                seg_pred_color = CityscapesVisualization.decode_segmap(seg_pred)
                depth_target_color = CityscapesVisualization.visualize_depth(depth_target)
                depth_pred_color = CityscapesVisualization.visualize_depth(depth_pred)
                
                # Plot
                axes[idx, 0].imshow(image_np)
                axes[idx, 0].set_title('Input Image', fontsize=12, fontweight='bold')
                axes[idx, 0].axis('off')
                
                axes[idx, 1].imshow(seg_target_color)
                axes[idx, 1].set_title('GT Segmentation', fontsize=12, fontweight='bold')
                axes[idx, 1].axis('off')
                
                axes[idx, 2].imshow(seg_pred_color)
                axes[idx, 2].set_title('Pred Segmentation', fontsize=12, fontweight='bold')
                axes[idx, 2].axis('off')
                
                axes[idx, 3].imshow(depth_target_color)
                axes[idx, 3].set_title('GT Depth', fontsize=12, fontweight='bold')
                axes[idx, 3].axis('off')
                
                axes[idx, 4].imshow(depth_pred_color)
                axes[idx, 4].set_title('Pred Depth', fontsize=12, fontweight='bold')
                axes[idx, 4].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'predictions.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Predictions visualization saved to {save_path}")
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, class_names: List[str],
                            save_path: Optional[str] = None):
        """Plot confusion matrix for segmentation"""
        plt.figure(figsize=(15, 12))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
        
        plt.title('Segmentation Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
        plt.ylabel('True Class', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_class_metrics(self, class_metrics: Dict[str, Dict[str, float]], 
                          save_path: Optional[str] = None):
        """Plot per-class metrics"""
        class_names = list(class_metrics.keys())
        iou_scores = [class_metrics[name]['IoU'] for name in class_names]
        f1_scores = [class_metrics[name]['F1'] for name in class_names]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # IoU scores
        bars1 = ax1.bar(class_names, iou_scores, color='skyblue', alpha=0.8)
        ax1.set_title('Per-Class IoU Scores', fontsize=16, fontweight='bold')
        ax1.set_ylabel('IoU Score', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, iou_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # F1 scores
        bars2 = ax2.bar(class_names, f1_scores, color='lightcoral', alpha=0.8)
        ax2.set_title('Per-Class F1 Scores', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Class', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels
        for ax in [ax1, ax2]:
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'class_metrics.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class metrics plot saved to {save_path}")
    
    def create_class_legend(self, save_path: Optional[str] = None):
        """Create a legend for Cityscapes classes"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create color patches
        patches = []
        labels = []
        
        for i, (class_name, color) in enumerate(zip(CITYSCAPES_CLASSES, CITYSCAPES_COLORS)):
            color_normalized = [c/255.0 for c in color]
            patch = Rectangle((0, 0), 1, 1, facecolor=color_normalized, edgecolor='black')
            patches.append(patch)
            labels.append(f'{i}: {class_name}')
        
        # Create legend
        ax.legend(patches, labels, loc='center', fontsize=12, 
                 ncol=2, frameon=True, fancybox=True, shadow=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Cityscapes Classes Color Legend', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'class_legend.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class legend saved to {save_path}")
    
    def plot_depth_error_analysis(self, depth_pred: np.ndarray, depth_gt: np.ndarray,
                                save_path: Optional[str] = None):
        """Analyze depth estimation errors"""
        # Calculate errors
        valid_mask = (depth_gt > 0) & (depth_gt < 1)
        pred_valid = depth_pred[valid_mask]
        gt_valid = depth_gt[valid_mask]
        
        abs_error = np.abs(pred_valid - gt_valid)
        rel_error = abs_error / gt_valid
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Error histogram
        axes[0, 0].hist(abs_error, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Absolute Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Relative error histogram
        axes[0, 1].hist(rel_error, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('Relative Error Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Relative Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot: predicted vs ground truth
        axes[1, 0].scatter(gt_valid, pred_valid, alpha=0.5, s=1)
        axes[1, 0].plot([gt_valid.min(), gt_valid.max()], 
                       [gt_valid.min(), gt_valid.max()], 'r--', linewidth=2)
        axes[1, 0].set_title('Predicted vs Ground Truth', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Ground Truth Depth')
        axes[1, 0].set_ylabel('Predicted Depth')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error vs depth
        axes[1, 1].scatter(gt_valid, abs_error, alpha=0.5, s=1)
        axes[1, 1].set_title('Error vs Depth', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Ground Truth Depth')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'depth_error_analysis.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Depth error analysis saved to {save_path}")
    
    def create_model_architecture_diagram(self, model: torch.nn.Module, 
                                        input_size: Tuple[int, int, int, int] = (1, 3, 512, 1024),
                                        save_path: Optional[str] = None):
        """Create a simple model architecture diagram"""
        try:
            from torchviz import make_dot
            
            # Create dummy input
            dummy_input = torch.randn(input_size)
            
            # Forward pass
            model.eval()
            output = model(dummy_input)
            
            # Create computation graph
            dot = make_dot(output['segmentation'], params=dict(model.named_parameters()))
            
            # Save diagram
            if save_path is None:
                save_path = os.path.join(self.viz_dir, 'model_architecture')
            
            dot.render(save_path, format='png')
            print(f"Model architecture diagram saved to {save_path}.png")
            
        except ImportError:
            print("torchviz not available. Install with: pip install torchviz")
        except Exception as e:
            print(f"Could not create architecture diagram: {e}")
    
    def save_training_summary(self, metric_tracker: Any, config: Any, 
                            final_metrics: Dict[str, float], save_path: Optional[str] = None):
        """Save comprehensive training summary"""
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'training_summary.json')
        
        # Get best metrics
        best_miou_epoch, best_miou = metric_tracker.get_best_epoch('seg_mIoU', 'val', 'max')
        best_depth_epoch, best_depth = metric_tracker.get_best_epoch('depth_abs_rel', 'val', 'min')
        
        summary = {
            'config': {
                'model': config.model.__dict__,
                'training': config.training.__dict__,
                'data': config.data.__dict__
            },
            'best_metrics': {
                'segmentation': {
                    'best_epoch': int(best_miou_epoch),
                    'best_mIoU': float(best_miou)
                },
                'depth': {
                    'best_epoch': int(best_depth_epoch),
                    'best_abs_rel': float(best_depth)
                }
            },
            'final_metrics': final_metrics,
            'total_epochs': len(metric_tracker.metrics_history['train'])
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to {save_path}")

def create_comprehensive_report(visualizer: TrainingVisualizer, model: torch.nn.Module,
                              dataloader: torch.utils.data.DataLoader, config: Any,
                              metric_tracker: Any, final_metrics: Dict[str, float]):
    """Create comprehensive training report with all visualizations"""
    print("Creating comprehensive training report...")
    
    # Training curves
    visualizer.plot_training_curves(metric_tracker)
    
    # Model predictions
    visualizer.visualize_predictions(model, dataloader, config)
    
    # Class legend
    visualizer.create_class_legend()
    
    # Training summary
    visualizer.save_training_summary(metric_tracker, config, final_metrics)
    
    print("Comprehensive report created in visualizations directory!")
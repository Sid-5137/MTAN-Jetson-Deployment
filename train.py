#!/usr/bin/env python3
"""
MTAN Training Script for Edge Devices
Multi-Task Attention Network for Segmentation and Depth Estimation
Optimized for Jetson and edge deployment
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config, CITYSCAPES_CLASSES
from models import create_mtan_model
from data import create_dataloaders, CityscapesVisualization
from utils import (MTANLoss, MultiTaskMetrics, EarlyStopping, MetricTracker, 
                   setup_training_components, CutMix, MixUp)

class Trainer:
    """MTAN Trainer class"""
    
    def __init__(self, config: Config, resume_path: Optional[str] = None):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Create directories first before setting up logging
        config.create_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Setup model
        self.model = self._setup_model()
        
        # Setup data loaders
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()
        
        # Setup loss and metrics
        self.criterion = MTANLoss(config).to(self.device)
        self.metrics = MultiTaskMetrics(
            num_classes=config.model.num_classes_seg,
            class_names=CITYSCAPES_CLASSES,
            device=self.device
        )
        
        # Setup training components
        self.training_components = setup_training_components(self.model, config)
        
        # Setup tracking
        self.metric_tracker = MetricTracker()
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            verbose=True
        )
        
        # Setup tensorboard and wandb
        self.writer = SummaryWriter(config.log_dir)
        self._setup_wandb()
        
        # Setup batch augmentations
        self.cutmix = None
        self.mixup = None
        if getattr(config.data, 'use_cutmix', False):
            self.cutmix = CutMix(alpha=1.0, prob=getattr(config.data, 'cutmix_prob', 0.5))
        if getattr(config.data, 'use_mixup', False):
            self.mixup = MixUp(alpha=0.2, prob=getattr(config.data, 'mixup_prob', 0.3))
        
        # Resume training if specified
        self.start_epoch = 0
        self.best_score = 0.0
        if resume_path:
            self._resume_training(resume_path)
        
        self.logger.info(f"Trainer initialized. Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Ensure log directory exists
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_model(self) -> nn.Module:
        """Setup and initialize model with gradient explosion prevention"""
        model = create_mtan_model(self.config)
        
        # Initialize model weights properly to prevent gradient explosion
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                # More conservative initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # More conservative initialization for linear layers
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # Conservative initialization for transposed conv
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)
        model.to(self.device)
        
        # Enable mixed precision and optimization for inference
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Data parallel if multiple GPUs
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        return model
    
    def _setup_data(self):
        """Setup data loaders"""
        self.logger.info("Setting up data loaders...")
        return create_dataloaders(self.config)
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        try:
            wandb.init(
                project="mtan-edge-deployment",
                config=self.config.__dict__,
                name=f"mtan_{int(time.time())}",
                save_code=True
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize wandb: {e}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, 
                        filename: Optional[str] = None):
        """Save training checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        # Get model state dict (handle DataParallel)
        model_state = (self.model.module.state_dict() 
                      if hasattr(self.model, 'module') 
                      else self.model.state_dict())
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.training_components['optimizer'].state_dict(),
            'scheduler_state_dict': (self.training_components['scheduler'].state_dict() 
                                   if self.training_components['scheduler'] else None),
            'best_score': self.best_score,
            'config': self.config,
            'metric_tracker': self.metric_tracker
        }
        
        # Add mixed precision scaler state if available
        if (hasattr(self.training_components['mixed_precision'], 'scaler') and 
            self.training_components['mixed_precision'].scaler is not None):
            checkpoint['scaler_state_dict'] = self.training_components['mixed_precision'].scaler.state_dict()
        
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
        
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def _resume_training(self, resume_path: str):
        """Resume training from checkpoint"""
        self.logger.info(f"Resuming training from {resume_path}")
        
        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.training_components['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if (self.training_components['scheduler'] and 
            checkpoint.get('scheduler_state_dict')):
            self.training_components['scheduler'].load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load mixed precision scaler
        if (hasattr(self.training_components['mixed_precision'], 'scaler') and
            self.training_components['mixed_precision'].scaler is not None and
            checkpoint.get('scaler_state_dict')):
            self.training_components['mixed_precision'].scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_score = checkpoint.get('best_score', 0.0)
        
        if 'metric_tracker' in checkpoint:
            self.metric_tracker = checkpoint['metric_tracker']
        
        self.logger.info(f"Resumed from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        valid_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.training.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            seg_targets = batch['segmentation'].to(self.device, non_blocking=True)
            depth_targets = batch['depth'].to(self.device, non_blocking=True)
            
            # Apply batch-level augmentations (CutMix/MixUp)
            batch_data = {
                'images': images,
                'masks': seg_targets,
                'depths': depth_targets
            }
            
            # Apply CutMix with probability
            if self.cutmix and torch.rand(1).item() < 0.5:
                batch_data = self.cutmix(batch_data)
                images = batch_data['images']
                seg_targets = batch_data['masks']
                depth_targets = batch_data['depths']
            elif self.mixup and torch.rand(1).item() < 0.3:
                batch_data = self.mixup(batch_data)
                images = batch_data['images']
                seg_targets = batch_data['masks']
                depth_targets = batch_data['depths']
            
            # Validate input data
            if torch.isnan(images).any() or torch.isinf(images).any():
                self.logger.warning(f"NaN/Inf detected in input images at batch {batch_idx}, skipping...")
                continue
                
            if torch.isnan(seg_targets).any() or torch.isinf(seg_targets).any():
                self.logger.warning(f"NaN/Inf detected in segmentation targets at batch {batch_idx}, skipping...")
                continue
                
            if torch.isnan(depth_targets).any() or torch.isinf(depth_targets).any():
                self.logger.warning(f"NaN/Inf detected in depth targets at batch {batch_idx}, skipping...")
                continue
            
            targets = {
                'segmentation': seg_targets,
                'depth': depth_targets
            }
            
            # Zero gradients
            self.training_components['optimizer'].zero_grad()
            
            # Forward pass with mixed precision
            try:
                with torch.amp.autocast('cuda', enabled=self.config.training.use_amp):
                    outputs = self.model(images)
                    
                    # Validate model outputs before loss calculation
                    for key, output in outputs.items():
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            self.logger.warning(f"NaN/Inf detected in {key} output at batch {batch_idx}, skipping...")
                            continue
                    
                    losses = self.criterion(outputs, targets)
                    loss = losses['total_loss']
                    
                    # Validate loss for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping...")
                        continue
                        
            except AttributeError:
                # Fallback for older PyTorch versions
                with torch.cuda.amp.autocast(enabled=self.config.training.use_amp):
                    outputs = self.model(images)
                    
                    # Validate model outputs before loss calculation
                    for key, output in outputs.items():
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            self.logger.warning(f"NaN/Inf detected in {key} output at batch {batch_idx}, skipping...")
                            continue
                    
                    losses = self.criterion(outputs, targets)
                    loss = losses['total_loss']
                    
                    # Validate loss for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping...")
                        continue
            
            # Backward pass with improved gradient handling
            self.training_components['mixed_precision'].backward(loss)
            
            # Unscale gradients before clipping for mixed precision
            if self.config.training.use_amp and self.training_components['mixed_precision'].scaler is not None:
                self.training_components['mixed_precision'].unscale_gradients(
                    self.training_components['optimizer']
                )
            
            # Gradient clipping with NaN/Inf detection
            if self.training_components['gradient_clipper']:
                grad_norm = self.training_components['gradient_clipper'].clip_gradients(self.model)
                
                # Enhanced gradient checking
                if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)) or grad_norm > 1000:
                    self.logger.warning(f"NaN/Inf/Large gradients detected at batch {batch_idx} (grad_norm: {grad_norm:.2f}), skipping optimizer step...")
                    self.training_components['optimizer'].zero_grad()
                    
                    # Skip optimizer step but don't update scaler to prevent scaling issues
                    if self.config.training.use_amp and self.training_components['mixed_precision'].scaler is not None:
                        self.training_components['mixed_precision'].scaler.update()
                    continue
            
            # Check individual parameter gradients for debugging
            has_valid_gradients = True
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        self.logger.warning(f"NaN/Inf gradients in parameter: {name}")
                        has_valid_gradients = False
                        break
            
            if not has_valid_gradients:
                self.training_components['optimizer'].zero_grad()
                if self.config.training.use_amp and self.training_components['mixed_precision'].scaler is not None:
                    self.training_components['mixed_precision'].scaler.update()
                continue
            
            # Optimizer step
            self.training_components['mixed_precision'].step_optimizer(
                self.training_components['optimizer']
            )
            
            # Update EMA if available
            if self.training_components['ema']:
                self.training_components['ema'].update(self.model)
            
            # Update metrics
            with torch.no_grad():
                # Validate outputs before updating metrics
                valid_outputs = True
                for key, output in outputs.items():
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        self.logger.warning(f"NaN/Inf detected in {key} output, skipping metrics update...")
                        valid_outputs = False
                        break
                
                if valid_outputs:
                    self.metrics.update(outputs, targets)
            
            # Accumulate loss
            total_loss += loss.item()
            valid_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'seg_loss': f'{losses["main_seg_loss"].item():.4f}',
                'depth_loss': f'{losses["main_depth_loss"].item():.4f}',
                'lr': f'{self.training_components["optimizer"].param_groups[0]["lr"]:.2e}',
                'valid_batches': f'{valid_batches}/{batch_idx+1}'
            })
            
            # Log batch metrics
            if batch_idx % self.config.log_interval == 0:
                self._log_batch_metrics(epoch, batch_idx, num_batches, losses)
        
        # Calculate epoch metrics
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
        else:
            avg_loss = 0.0
            self.logger.warning("No valid batches in this epoch!")
            
        train_metrics = self.metrics.get_summary()
        train_metrics['loss'] = avg_loss
        
        return train_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                images = batch['image'].to(self.device, non_blocking=True)
                seg_targets = batch['segmentation'].to(self.device, non_blocking=True)
                depth_targets = batch['depth'].to(self.device, non_blocking=True)
                
                targets = {
                    'segmentation': seg_targets,
                    'depth': depth_targets
                }
                
                # Forward pass
                try:
                    with torch.amp.autocast('cuda', enabled=self.config.training.use_amp):
                        outputs = self.model(images)
                        losses = self.criterion(outputs, targets)
                        loss = losses['total_loss']
                except AttributeError:
                    # Fallback for older PyTorch versions
                    with torch.cuda.amp.autocast(enabled=self.config.training.use_amp):
                        outputs = self.model(images)
                        losses = self.criterion(outputs, targets)
                        loss = losses['total_loss']
                
                # Update metrics
                self.metrics.update(outputs, targets)
                total_loss += loss.item()
        
        # Calculate validation metrics
        avg_loss = total_loss / num_batches
        val_metrics = self.metrics.get_summary()
        val_metrics['loss'] = avg_loss
        
        return val_metrics
    
    def _log_batch_metrics(self, epoch: int, batch_idx: int, num_batches: int, 
                          losses: Dict[str, torch.Tensor]):
        """Log batch-level metrics"""
        global_step = epoch * num_batches + batch_idx
        
        # Tensorboard logging
        for key, value in losses.items():
            if torch.is_tensor(value):
                self.writer.add_scalar(f'batch/{key}', value.item(), global_step)
        
        # Wandb logging
        try:
            wandb.log({f'batch/{k}': v.item() if torch.is_tensor(v) else v 
                      for k, v in losses.items()}, step=global_step)
        except:
            pass
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float]):
        """Log epoch-level metrics"""
        # Tensorboard logging
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        # Learning rate logging
        if self.training_components['scheduler']:
            self.writer.add_scalar('lr', 
                                 self.training_components['optimizer'].param_groups[0]['lr'], 
                                 epoch)
        
        # Wandb logging
        try:
            log_dict = {}
            log_dict.update({f'train/{k}': v for k, v in train_metrics.items()})
            log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
            log_dict['epoch'] = epoch
            wandb.log(log_dict, step=epoch)
        except:
            pass
        
        # Console logging
        self.logger.info(f'Epoch {epoch}:')
        self.logger.info(f'  Train - Loss: {train_metrics["loss"]:.4f}, '
                        f'mIoU: {train_metrics["seg_mIoU"]:.4f}, '
                        f'Depth AbsRel: {train_metrics["depth_abs_rel"]:.4f}')
        self.logger.info(f'  Val   - Loss: {val_metrics["loss"]:.4f}, '
                        f'mIoU: {val_metrics["seg_mIoU"]:.4f}, '
                        f'Depth AbsRel: {val_metrics["depth_abs_rel"]:.4f}')
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            if self.training_components['scheduler']:
                self.training_components['scheduler'].step()
            
            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Update metric tracker
            self.metric_tracker.update(train_metrics, 'train')
            self.metric_tracker.update(val_metrics, 'val')
            
            # Check if best model
            current_score = val_metrics['seg_mIoU']  # Use mIoU as main metric
            is_best = current_score > self.best_score
            if is_best:
                self.best_score = current_score
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0 or is_best:
                self._save_checkpoint(epoch, is_best)
            
            # Early stopping check
            if self.early_stopping(current_score, self.model):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s')
        
        # Final evaluation
        self._final_evaluation()
        
        # Close writers
        self.writer.close()
        try:
            wandb.finish()
        except:
            pass
        
        self.logger.info("Training completed!")
    
    def _final_evaluation(self):
        """Final evaluation on test set if available"""
        if self.test_loader is None:
            return
        
        self.logger.info("Running final evaluation on test set...")
        
        # Load best model
        best_model_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded best model for final evaluation")
        
        # Evaluate on test set
        test_metrics = self.validate_epoch('test')
        
        # Log test results
        self.logger.info("Test Results:")
        self.logger.info(self.metrics.get_detailed_results())
        
        # Save test results
        test_results_path = os.path.join(self.config.output_dir, 'test_results.txt')
        with open(test_results_path, 'w') as f:
            f.write("Final Test Results:\n")
            f.write(self.metrics.get_detailed_results())
            f.write(f"\nTest Metrics Summary:\n")
            for key, value in test_metrics.items():
                f.write(f"{key}: {value:.4f}\n")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MTAN Training for Edge Devices')
    parser.add_argument('--config', type=str, default='config.py',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ID')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    
    # Create trainer and start training
    trainer = Trainer(config, resume_path=args.resume)
    trainer.train()

if __name__ == '__main__':
    main()
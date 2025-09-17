#!/usr/bin/env python3
"""
Enhanced MTAN Training Script with All Research-Based Improvements
Multi-Task Attention Network for Segmentation and Depth Estimation
Optimized for maximum accuracy and convergence speed based on online research
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
                   setup_training_components, CutMix, MixUp, DynamicWeightAveraging)

def main():
    """Enhanced main training function with all improvements"""
    print("🚀 Starting Enhanced MTAN Training with Research-Based Improvements")
    print("="*80)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Enhanced MTAN Training')
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
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load enhanced configuration
    config = Config()
    
    # Apply research-based optimizations to config
    print("📋 Applying Research-Based Configuration Optimizations:")
    print(f"   • Task Weighting: {config.training.task_weighting}")
    print(f"   • Learning Rate: {config.training.learning_rate}")
    print(f"   • Warmup Epochs: {config.training.warmup_epochs}")
    print(f"   • DWA Temperature: {config.training.dwa_temperature}")
    print(f"   • Gradient Clipping: {config.training.gradient_clip_val}")
    print(f"   • Label Smoothing: {getattr(config.training, 'label_smoothing', 0.0)}")
    print(f"   • Data Augmentations: CutMix={config.data.use_cutmix}, MixUp={config.data.use_mixup}")
    
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
        # Enable deterministic behavior
        torch.backends.cudnn.deterministic = False  # For performance
        torch.backends.cudnn.benchmark = True
    
    print(f"🎯 Training Configuration:")
    print(f"   • Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   • Batch Size: {config.training.batch_size}")
    print(f"   • Learning Rate: {config.training.learning_rate}")
    print(f"   • Epochs: {config.training.num_epochs}")
    print(f"   • Input Size: {config.model.input_size}")
    print(f"   • Optimizer: {config.training.optimizer}")
    print(f"   • Scheduler: {config.training.scheduler}")
    
    # Create and start enhanced trainer
    from train import Trainer
    trainer = Trainer(config, resume_path=args.resume)
    
    print("\n🔥 All research-based improvements have been applied!")
    print("💡 Key Improvements Include:")
    print("   ✅ Dynamic Weight Averaging (DWA) for adaptive task balancing")
    print("   ✅ Enhanced data augmentation with CutMix and advanced transforms")
    print("   ✅ Improved learning rate scheduling with warmup")
    print("   ✅ Label smoothing for better generalization")
    print("   ✅ Conservative gradient clipping and mixed precision")
    print("   ✅ Exponential Moving Average (EMA) for stable training")
    print("   ✅ Differential learning rates for different model components")
    print("   ✅ Improved loss functions and gradient flow")
    
    print(f"\n🚀 Starting training with {sum(p.numel() for p in trainer.model.parameters()):,} parameters...")
    print("="*80)
    
    # Start training
    trainer.train()
    
    print("\n🎉 Training completed successfully!")
    print("📊 Check logs and checkpoints for results.")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
MTAN Test Script for Cityscapes Test Set Evaluation
Evaluates model performance on the Cityscapes test set and saves comprehensive results
"""

import os
import sys
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config, CITYSCAPES_CLASSES
from data import CityscapesDataset
from models import create_mtan_model
from utils.metrics import MultiTaskMetrics
from utils.visualization import TrainingVisualizer

def evaluate_model_on_test_set(model_path: str, config: Config, output_dir: str = './test_results'):
    """
    Evaluate the MTAN model on the Cityscapes test set
    
    Args:
        model_path: Path to the trained model checkpoint
        config: Configuration object
        output_dir: Directory to save test results
    """
    print(f"Starting evaluation on Cityscapes test set...")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    try:
        model = create_mtan_model(config)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create test dataset
    print("Loading test dataset...")
    try:
        test_dataset = CityscapesDataset(config, split='test', transform_type='val')
        print(f"Test dataset loaded with {len(test_dataset)} samples")
    except Exception as e:
        print(f"Warning: Could not load test dataset: {e}")
        print("Falling back to validation dataset for demonstration")
        try:
            test_dataset = CityscapesDataset(config, split='val', transform_type='val')
            print(f"Validation dataset loaded with {len(test_dataset)} samples")
        except Exception as e2:
            print(f"Error loading validation dataset: {e2}")
            return
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one at a time for evaluation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Initialize metrics
    metrics = MultiTaskMetrics(
        num_classes=len(CITYSCAPES_CLASSES),
        class_names=CITYSCAPES_CLASSES,
        device=device
    )
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(output_dir=output_dir)
    
    # Evaluation loop
    print("Starting evaluation...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            try:
                # Move data to device
                images = batch['image'].to(device, non_blocking=True)
                seg_targets = batch['segmentation'].to(device, non_blocking=True)
                depth_targets = batch['depth'].to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(images)
                
                # Update metrics
                metrics.update(outputs, {
                    'segmentation': seg_targets,
                    'depth': depth_targets
                })
                
                # Visualize first few samples
                if batch_idx < 5:
                    # Save individual predictions
                    sample_output_dir = os.path.join(viz_dir, f'sample_{batch_idx}')
                    os.makedirs(sample_output_dir, exist_ok=True)
                    
                    # Get predictions
                    seg_pred = torch.softmax(outputs['segmentation'], dim=1).argmax(dim=1)
                    depth_pred = outputs['depth']
                    
                    # Convert to numpy for visualization
                    image_np = images[0].cpu()
                    seg_target_np = seg_targets[0].cpu().numpy()
                    depth_target_np = depth_targets[0, 0].cpu().numpy()
                    seg_pred_np = seg_pred[0].cpu().numpy()
                    depth_pred_np = depth_pred[0, 0].cpu().numpy()
                    
                    # Denormalize image
                    mean = np.array(config.data.normalize_mean).reshape(3, 1, 1)
                    std = np.array(config.data.normalize_std).reshape(3, 1, 1)
                    image_denorm = image_np * std + mean
                    image_denorm = np.clip(image_denorm, 0, 1)
                    image_vis = (image_denorm * 255).astype(np.uint8)
                    image_vis = np.transpose(image_vis, (1, 2, 0))
                    image_vis = np.ascontiguousarray(image_vis[:, :, ::-1])  # RGB to BGR for OpenCV
                    
                    # Create segmentation visualizations
                    from data import CityscapesVisualization
                    seg_target_vis = CityscapesVisualization.decode_segmap(seg_target_np)
                    seg_pred_vis = CityscapesVisualization.decode_segmap(seg_pred_np)
                    
                    # Create depth visualizations
                    depth_target_vis = CityscapesVisualization.visualize_depth(depth_target_np)
                    depth_pred_vis = CityscapesVisualization.visualize_depth(depth_pred_np)
                    
                    # Save visualizations
                    cv2.imwrite(os.path.join(sample_output_dir, 'input.png'), image_vis)
                    cv2.imwrite(os.path.join(sample_output_dir, 'seg_target.png'), seg_target_vis)
                    cv2.imwrite(os.path.join(sample_output_dir, 'seg_pred.png'), seg_pred_vis)
                    cv2.imwrite(os.path.join(sample_output_dir, 'depth_target.png'), depth_target_vis)
                    cv2.imwrite(os.path.join(sample_output_dir, 'depth_pred.png'), depth_pred_vis)
                    
            except Exception as batch_error:
                print(f"Error processing batch {batch_idx}: {batch_error}")
                continue
                
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Compute final metrics
    print("Computing final metrics...")
    final_metrics = metrics.compute()
    
    # Print summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Segmentation mIoU: {final_metrics['seg_metrics']['mIoU']:.4f}")
    print(f"Segmentation Pixel Accuracy: {final_metrics['seg_metrics']['pixel_accuracy']:.4f}")
    print(f"Depth Abs Rel: {final_metrics['depth_metrics']['abs_rel']:.4f}")
    print(f"Depth RMSE: {final_metrics['depth_metrics']['rmse']:.4f}")
    print(f"Depth δ1: {final_metrics['depth_metrics']['delta1']:.4f}")
    print(f"Depth δ2: {final_metrics['depth_metrics']['delta2']:.4f}")
    print(f"Depth δ3: {final_metrics['depth_metrics']['delta3']:.4f}")
    print("="*50)
    
    # Save metrics to JSON
    metrics_dict = {
        'segmentation': {
            'mIoU': float(final_metrics['seg_metrics']['mIoU']),
            'pixel_accuracy': float(final_metrics['seg_metrics']['pixel_accuracy']),
            'mF1': float(final_metrics['seg_metrics']['mF1'])
        },
        'depth': {
            'abs_rel': float(final_metrics['depth_metrics']['abs_rel']),
            'sq_rel': float(final_metrics['depth_metrics']['sq_rel']),
            'rmse': float(final_metrics['depth_metrics']['rmse']),
            'rmse_log': float(final_metrics['depth_metrics']['rmse_log']),
            'delta1': float(final_metrics['depth_metrics']['delta1']),
            'delta2': float(final_metrics['depth_metrics']['delta2']),
            'delta3': float(final_metrics['depth_metrics']['delta3'])
        },
        'evaluation_info': {
            'samples_processed': len(test_dataset),
            'evaluation_time_seconds': evaluation_time
        }
    }
    
    # Save comprehensive results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Evaluation results saved to {results_path}")
    
    # Save class-wise metrics
    class_metrics = metrics.seg_metrics.get_class_metrics(CITYSCAPES_CLASSES)
    class_analysis_path = os.path.join(output_dir, 'class_analysis.json')
    with open(class_analysis_path, 'w') as f:
        json.dump(class_metrics, f, indent=2)
    print(f"Class-wise analysis saved to {class_analysis_path}")
    
    # Save detailed report
    report_path = os.path.join(output_dir, 'comprehensive_report.txt')
    with open(report_path, 'w') as f:
        f.write("MTAN Model Evaluation Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n")
        f.write(f"Classes: {len(CITYSCAPES_CLASSES)}\n")
        f.write(f"Evaluation Time: {evaluation_time:.2f} seconds\n\n")
        
        f.write("SEGMENTATION METRICS\n")
        f.write("-"*20 + "\n")
        f.write(f"mIoU: {final_metrics['seg_metrics']['mIoU']:.4f}\n")
        f.write(f"Pixel Accuracy: {final_metrics['seg_metrics']['pixel_accuracy']:.4f}\n")
        f.write(f"mF1: {final_metrics['seg_metrics']['mF1']:.4f}\n\n")
        
        f.write("DEPTH ESTIMATION METRICS\n")
        f.write("-"*25 + "\n")
        f.write(f"Abs Rel: {final_metrics['depth_metrics']['abs_rel']:.4f}\n")
        f.write(f"Sq Rel: {final_metrics['depth_metrics']['sq_rel']:.4f}\n")
        f.write(f"RMSE: {final_metrics['depth_metrics']['rmse']:.4f}\n")
        f.write(f"RMSE log: {final_metrics['depth_metrics']['rmse_log']:.4f}\n")
        f.write(f"δ1: {final_metrics['depth_metrics']['delta1']:.4f}\n")
        f.write(f"δ2: {final_metrics['depth_metrics']['delta2']:.4f}\n")
        f.write(f"δ3: {final_metrics['depth_metrics']['delta3']:.4f}\n\n")
        
        f.write("CLASS-WISE IoU\n")
        f.write("-"*15 + "\n")
        for class_name, class_metrics in class_metrics.items():
            f.write(f"{class_name:20s}: {class_metrics['IoU']:.4f}\n")
    
    print(f"Comprehensive report saved to {report_path}")
    
    # Create visualizations
    print("Generating visualizations...")
    
    # Confusion matrix
    try:
        conf_matrix = final_metrics['seg_metrics']['confusion_matrix']
        visualizer.plot_confusion_matrix(conf_matrix, CITYSCAPES_CLASSES, 
                                       os.path.join(viz_dir, 'confusion_matrix.png'))
    except Exception as e:
        print(f"Warning: Could not generate confusion matrix: {e}")
    
    # Class legend
    try:
        visualizer.create_class_legend(os.path.join(viz_dir, 'class_legend.png'))
    except Exception as e:
        print(f"Warning: Could not generate class legend: {e}")
    
    # Sample predictions visualization
    try:
        visualizer.visualize_predictions(model, test_loader, config, num_samples=5,
                                       save_path=os.path.join(viz_dir, 'predictions.png'))
    except Exception as e:
        print(f"Warning: Could not generate predictions visualization: {e}")
    
    print(f"All visualizations saved to {viz_dir}")
    print("Test evaluation completed successfully!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate MTAN model on Cityscapes test set')
    parser.add_argument('--model', type=str, default='./checkpoints_large/best_model.pth',
                       help='Path to the model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                       help='Directory to save test results')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load configuration
    config = Config()
    
    # Run evaluation
    evaluate_model_on_test_set(args.model, config, args.output_dir)

if __name__ == '__main__':
    main()
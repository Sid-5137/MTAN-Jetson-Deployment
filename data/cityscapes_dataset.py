import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.augmentation import MTANAugmentation

class CityscapesDataset(data.Dataset):
    """Cityscapes dataset for multi-task learning (segmentation + depth)"""
    
    # Cityscapes label mapping (trainId to id)
    TRAIN_ID_TO_COLOR = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]
    
    # Original Cityscapes classes to train IDs mapping
    CITYSCAPES_ID_TO_TRAIN_ID = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 
        28: 15, 31: 16, 32: 17, 33: 18
    }
    
    def __init__(self, 
                 config,
                 split: str = 'train',
                 transform_type: str = 'train'):
        """
        Args:
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
            transform_type: Type of transforms ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        self.transform_type = transform_type
        
        self.cityscapes_root = config.data.cityscapes_root
        self.depth_root = config.data.depth_root
        
        # Dataset paths
        self.images_dir = os.path.join(self.cityscapes_root, 'leftImg8bit', split)
        self.labels_dir = os.path.join(self.cityscapes_root, 'gtFine', split)
        self.depth_dir = os.path.join(self.depth_root, split)
        
        # Collect all samples
        self.samples = self._collect_samples()
        
        # Setup transforms
        self.transforms = self._setup_transforms()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _collect_samples(self) -> List[Dict[str, str]]:
        """Collect all valid samples with image, label, and depth"""
        samples = []
        
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        for city in os.listdir(self.images_dir):
            city_img_dir = os.path.join(self.images_dir, city)
            city_label_dir = os.path.join(self.labels_dir, city)
            city_depth_dir = os.path.join(self.depth_dir, city)
            
            if not os.path.isdir(city_img_dir):
                continue
                
            for img_file in os.listdir(city_img_dir):
                if not img_file.endswith('_leftImg8bit.png'):
                    continue
                    
                # Generate corresponding label and depth paths
                base_name = img_file.replace('_leftImg8bit.png', '')
                
                img_path = os.path.join(city_img_dir, img_file)
                label_path = os.path.join(city_label_dir, f"{base_name}_gtFine_labelIds.png")
                depth_path = os.path.join(city_depth_dir, f"{base_name}_crestereo_depth.png")
                
                # Check if all files exist
                if (os.path.exists(img_path) and 
                    os.path.exists(label_path) and 
                    os.path.exists(depth_path)):
                    
                    samples.append({
                        'image': img_path,
                        'label': label_path,
                        'depth': depth_path,
                        'city': city,
                        'base_name': base_name
                    })
        
        return samples
    
    def _setup_transforms(self) -> A.Compose:
        """Setup enhanced data augmentation transforms using MTAN augmentation pipeline"""
        # Use the new MTAN augmentation pipeline
        augmentation_strength = getattr(self.config.data, 'augmentation_strength', 'medium')
        
        if self.transform_type == 'train':
            # Enhanced training augmentations
            self.mtan_aug = MTANAugmentation(
                input_size=(self.config.model.input_size[0], self.config.model.input_size[1]),
                is_training=True,
                augmentation_strength=augmentation_strength
            )
            # Return the MTAN augmentation transform directly
            return self.mtan_aug.transform
        else:
            # Validation/Test transforms
            self.mtan_aug = MTANAugmentation(
                input_size=(self.config.model.input_size[0], self.config.model.input_size[1]),
                is_training=False
            )
            return self.mtan_aug.transform
    
    def _encode_segmap(self, mask: np.ndarray) -> np.ndarray:
        """Convert Cityscapes label IDs to train IDs"""
        label_mask = np.zeros_like(mask)
        for class_id, train_id in self.CITYSCAPES_ID_TO_TRAIN_ID.items():
            label_mask[mask == class_id] = train_id
        return label_mask
    
    def _load_depth(self, depth_path: str) -> np.ndarray:
        """Load and normalize depth map with robust error handling"""
        if depth_path.endswith('.png'):
            # Load as 16-bit PNG
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                # Fallback: create dummy depth with small valid values
                print(f"Warning: Could not load depth {depth_path}, using dummy data")
                depth = np.full((1024, 2048), 0.5, dtype=np.float32)  # Use 0.5 instead of zeros
            else:
                depth = depth.astype(np.float32)
                
                # Check for invalid values and clean them
                if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
                    print(f"Warning: Found NaN/Inf in depth {depth_path}, cleaning...")
                    depth = np.nan_to_num(depth, nan=0.5, posinf=self.config.data.depth_max, neginf=self.config.data.depth_min)
                
                # Ensure all values are finite
                if not np.all(np.isfinite(depth)):
                    print(f"Warning: Non-finite values in depth {depth_path}, replacing with 0.5")
                    depth = np.where(np.isfinite(depth), depth, 0.5)
                
                # Clip and normalize depth values more safely
                depth = np.clip(depth, self.config.data.depth_min, self.config.data.depth_max)
                
                # Improved normalization to prevent values too close to zero
                depth = (depth - self.config.data.depth_min) / (self.config.data.depth_max - self.config.data.depth_min)
                
                # Ensure minimum value to prevent division by zero in loss
                depth = np.clip(depth, 0.01, 1.0)  # Minimum 0.01 instead of 0
                
                # Final check for any remaining invalid values
                if np.any(np.isnan(depth)) or np.any(np.isinf(depth)) or not np.all(np.isfinite(depth)):
                    print(f"Warning: Still found invalid values in depth {depth_path}, using fallback")
                    depth = np.full_like(depth, 0.5, dtype=np.float32)
        else:
            # Handle other formats if needed
            try:
                depth = np.load(depth_path).astype(np.float32)
            except:
                print(f"Warning: Could not load depth {depth_path}, using dummy data")
                depth = np.full((1024, 2048), 0.5, dtype=np.float32)
            
            # Check for invalid values
            if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
                print(f"Warning: Found NaN/Inf in depth {depth_path}, cleaning...")
                depth = np.nan_to_num(depth, nan=0.5, posinf=self.config.data.depth_max, neginf=self.config.data.depth_min)
            
            # Ensure all values are finite
            if not np.all(np.isfinite(depth)):
                print(f"Warning: Non-finite values in depth {depth_path}, replacing with 0.5")
                depth = np.where(np.isfinite(depth), depth, 0.5)
            
            depth = np.clip(depth, self.config.data.depth_min, self.config.data.depth_max)
            depth = (depth - self.config.data.depth_min) / (self.config.data.depth_max - self.config.data.depth_min)
            depth = np.clip(depth, 0.01, 1.0)  # Ensure minimum value
            
            # Final check for any remaining invalid values
            if np.any(np.isnan(depth)) or np.any(np.isinf(depth)) or not np.all(np.isfinite(depth)):
                print(f"Warning: Still found invalid values in depth {depth_path}, using fallback")
                depth = np.full_like(depth, 0.5, dtype=np.float32)
            
        return depth
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            # Load image with error handling
            image = cv2.imread(sample['image'])
            if image is None:
                # Fallback: try next sample if current image is corrupted
                print(f"Warning: Corrupted image {sample['image']}, trying next sample...")
                return self.__getitem__((idx + 1) % len(self.samples))
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load segmentation mask with error handling
            mask = cv2.imread(sample['label'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Corrupted label {sample['label']}, trying next sample...")
                return self.__getitem__((idx + 1) % len(self.samples))
            
            mask = self._encode_segmap(mask)
            
            # Load depth with error handling
            depth = self._load_depth(sample['depth'])
            if depth is None:
                print(f"Warning: Corrupted depth {sample['depth']}, trying next sample...")
                return self.__getitem__((idx + 1) % len(self.samples))
            
        except Exception as e:
            print(f"Warning: Error loading sample {idx}: {e}, trying next sample...")
            return self.__getitem__((idx + 1) % len(self.samples))
        
        # Ensure all have same spatial dimensions
        h, w = image.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        try:
            # Apply enhanced MTAN transforms with error handling
            try:
                if hasattr(self, 'mtan_aug'):
                    # Use MTAN augmentation pipeline
                    transformed_data = self.mtan_aug(image=image, mask=mask, depth=depth)
                    transformed = {
                        'image': transformed_data['image'],
                        'mask': transformed_data['mask'],
                        'depth': transformed_data['depth']
                    }
                else:
                    # Fallback to albumentations
                    transformed = self.transforms(
                        image=image,
                        mask=mask,
                        depth=depth
                    )
            except Exception as aug_error:
                print(f"Warning: Augmentation error for sample {idx}: {aug_error}, using fallback...")
                # Fallback to basic transforms
                basic_transform = A.Compose([
                    A.Resize(height=self.config.model.input_size[0], width=self.config.model.input_size[1]),
                    A.Normalize(mean=self.config.data.normalize_mean, std=self.config.data.normalize_std),
                    ToTensorV2()
                ], additional_targets={'mask': 'mask', 'depth': 'mask'})
                
                transformed = basic_transform(
                    image=image,
                    mask=mask,
                    depth=depth
                )
            
            # Validate transformed data
            image_tensor = transformed['image']
            mask_tensor = transformed['mask']
            depth_tensor = transformed['depth']
            
            # Check for NaN/Inf in transformed data
            if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
                print(f"Warning: NaN/Inf in transformed image for {sample['image']}, using fallback")
                return self.__getitem__((idx + 1) % len(self.samples))
                
            if torch.isnan(mask_tensor).any() or torch.isinf(mask_tensor).any():
                print(f"Warning: NaN/Inf in transformed mask for {sample['label']}, using fallback")
                return self.__getitem__((idx + 1) % len(self.samples))
                
            if torch.isnan(depth_tensor).any() or torch.isinf(depth_tensor).any():
                print(f"Warning: NaN/Inf in transformed depth for {sample['depth']}, using fallback")
                return self.__getitem__((idx + 1) % len(self.samples))
            
            return {
                'image': image_tensor,
                'segmentation': mask_tensor.long(),
                'depth': depth_tensor.unsqueeze(0).float(),  # Add channel dimension
                'sample_info': {
                    'city': sample['city'],
                    'base_name': sample['base_name'],
                    'image_path': sample['image']
                }
            }
            
        except Exception as e:
            print(f"Warning: Transform error for sample {idx}: {e}, trying next sample...")
            return self.__getitem__((idx + 1) % len(self.samples))

def create_dataloaders(config) -> Tuple[data.DataLoader, data.DataLoader, Optional[data.DataLoader]]:
    """Create train, validation, and test dataloaders"""
    
    # Create datasets
    train_dataset = CityscapesDataset(config, split='train', transform_type='train')
    val_dataset = CityscapesDataset(config, split='val', transform_type='val')
    
    # Test dataset (optional)
    test_dataset = None
    if os.path.exists(os.path.join(config.data.cityscapes_root, 'leftImg8bit', 'test')):
        try:
            test_dataset = CityscapesDataset(config, split='test', transform_type='test')
        except:
            print("Test dataset not available or incomplete")
    
    # Create dataloaders with optimization for GPU utilization
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else 2
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else 2
    )
    
    test_loader = None
    if test_dataset:
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=1,  # Process one at a time for testing
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    if test_loader:
        print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader

class CityscapesVisualization:
    """Utilities for visualizing Cityscapes data"""
    
    @staticmethod
    def decode_segmap(label_mask: np.ndarray, num_classes: int = 19) -> np.ndarray:
        """Convert label mask to RGB color image"""
        label_colors = np.array(CityscapesDataset.TRAIN_ID_TO_COLOR)
        
        r = np.zeros_like(label_mask).astype(np.uint8)
        g = np.zeros_like(label_mask).astype(np.uint8)
        b = np.zeros_like(label_mask).astype(np.uint8)
        
        for class_id in range(num_classes):
            idx = label_mask == class_id
            r[idx] = label_colors[class_id, 0]
            g[idx] = label_colors[class_id, 1]
            b[idx] = label_colors[class_id, 2]
            
        rgb = np.stack([r, g, b], axis=2)
        return rgb
    
    @staticmethod
    def visualize_depth(depth: np.ndarray) -> np.ndarray:
        """Convert depth map to colormap visualization"""
        # Normalize depth to 0-255
        depth_norm = (depth * 255).astype(np.uint8)
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        return cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def denormalize_image(image: torch.Tensor, mean: Tuple[float, float, float], 
                         std: Tuple[float, float, float]) -> np.ndarray:
        """Denormalize image tensor to numpy array"""
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        image = image.permute(1, 2, 0).numpy()
        return (image * 255).astype(np.uint8)
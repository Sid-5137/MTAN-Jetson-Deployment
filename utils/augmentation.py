"""
Advanced Data Augmentation for Multi-Task Learning on Cityscapes
Based on best practices from MTAN research and state-of-the-art implementations
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import random
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageFilter, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MTANAugmentation:
    """
    Multi-Task Augmentation pipeline optimized for MTAN training.
    Includes augmentations that preserve spatial relationships for both
    semantic segmentation and depth estimation tasks.
    """
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (256, 512),
                 is_training: bool = True,
                 augmentation_strength: str = "medium"):
        """
        Initialize augmentation pipeline.
        
        Args:
            input_size: Target image size (height, width)
            is_training: Whether to apply training augmentations
            augmentation_strength: "light", "medium", or "strong"
        """
        self.input_size = input_size
        self.is_training = is_training
        self.strength = augmentation_strength
        
        # Define augmentation parameters based on strength
        self.aug_params = self._get_augmentation_params()
        
        # Create augmentation pipeline
        self.transform = self._create_transform()
        
    def _get_augmentation_params(self) -> Dict[str, Any]:
        """Get augmentation parameters based on strength level."""
        if self.strength == "light":
            return {
                "flip_prob": 0.3,
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.1,
                "hue": 0.05,
                "blur_prob": 0.1,
                "noise_prob": 0.1,
                "scale_range": (0.95, 1.05),
                "rotation_limit": 2
            }
        elif self.strength == "medium":
            return {
                "flip_prob": 0.5,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
                "blur_prob": 0.2,
                "noise_prob": 0.15,
                "scale_range": (0.9, 1.1),
                "rotation_limit": 5
            }
        else:  # strong
            return {
                "flip_prob": 0.5,
                "brightness": 0.3,
                "contrast": 0.3,
                "saturation": 0.3,
                "hue": 0.15,
                "blur_prob": 0.3,
                "noise_prob": 0.2,
                "scale_range": (0.85, 1.15),
                "rotation_limit": 10
            }
    
    def _create_transform(self):
        """Create albumentations transform pipeline."""
        if not self.is_training:
            # Validation/test transforms - only resize and normalize
            return A.Compose([
                A.Resize(height=self.input_size[0], width=self.input_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        # Training transforms with augmentations
        transforms_list = [
            # Geometric transformations (preserve spatial relationships)
            A.HorizontalFlip(p=self.aug_params["flip_prob"]),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=[self.aug_params["scale_range"][0] - 1, self.aug_params["scale_range"][1] - 1],
                rotate_limit=self.aug_params["rotation_limit"],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            ),
            
            # Resize to target size
            A.Resize(height=self.input_size[0], width=self.input_size[1]),
            
            # Color augmentations
            A.ColorJitter(
                brightness=self.aug_params["brightness"],
                contrast=self.aug_params["contrast"],
                saturation=self.aug_params["saturation"],
                hue=self.aug_params["hue"],
                p=0.6
            ),
            
            # Weather and lighting effects
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.RandomGamma(p=1.0),
                A.HueSaturationValue(p=1.0),
            ], p=0.4),
            
            # Blur and noise (mild to avoid destroying details)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=self.aug_params["blur_prob"]),
            
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(p=1.0),
            ], p=self.aug_params["noise_prob"]),
            
            # Final normalization and tensor conversion
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        return A.Compose(transforms_list)
    
    def __call__(self, image: np.ndarray, mask: np.ndarray = None, depth: np.ndarray = None):
        """
        Apply augmentations to image and corresponding masks/depth maps.
        
        Args:
            image: Input RGB image (H, W, 3)
            mask: Segmentation mask (H, W) - optional
            depth: Depth map (H, W) - optional
            
        Returns:
            Dictionary with augmented image and masks
        """
        # Prepare targets for albumentations
        targets = {"image": image}
        
        if mask is not None:
            targets["mask"] = mask
            
        if depth is not None:
            targets["depth"] = depth
            
        # Apply transforms
        if mask is not None and depth is not None:
            # Multi-target augmentation
            transform = A.Compose(
                self.transform.transforms,
                additional_targets={"depth": "mask"}
            )
            augmented = transform(image=image, mask=mask, depth=depth)
            return {
                "image": augmented["image"],
                "mask": augmented["mask"],
                "depth": augmented["depth"]
            }
        elif mask is not None:
            augmented = self.transform(image=image, mask=mask)
            return {
                "image": augmented["image"],
                "mask": augmented["mask"]
            }
        elif depth is not None:
            transform = A.Compose(
                self.transform.transforms,
                additional_targets={"depth": "mask"}
            )
            augmented = transform(image=image, depth=depth)
            return {
                "image": augmented["image"],
                "depth": augmented["depth"]
            }
        else:
            augmented = self.transform(image=image)
            return {"image": augmented["image"]}


class CutMix:
    """
    CutMix augmentation for multi-task learning.
    Adapted to work with both segmentation and depth estimation.
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Initialize CutMix augmentation.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying CutMix
        """
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply CutMix to a batch of data.
        
        Args:
            batch_data: Dictionary containing 'images', 'masks', 'depths'
            
        Returns:
            Mixed batch data with lambda parameter
        """
        if random.random() > self.prob:
            return batch_data
            
        batch_size = batch_data["images"].size(0)
        
        # Generate lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate random indices for mixing
        rand_index = torch.randperm(batch_size)
        
        # Calculate bounding box
        W, H = batch_data["images"].size(3), batch_data["images"].size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Calculate box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix to images
        batch_data["images"][:, :, bby1:bby2, bbx1:bbx2] = \
            batch_data["images"][rand_index, :, bby1:bby2, bbx1:bbx2]
        
        # Apply CutMix to masks if present
        if "masks" in batch_data:
            batch_data["masks"][:, bby1:bby2, bbx1:bbx2] = \
                batch_data["masks"][rand_index, bby1:bby2, bbx1:bbx2]
        
        # Apply CutMix to depth if present
        if "depths" in batch_data:
            batch_data["depths"][:, :, bby1:bby2, bbx1:bbx2] = \
                batch_data["depths"][rand_index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        batch_data["cutmix_lambda"] = lam
        batch_data["cutmix_index"] = rand_index
        
        return batch_data


class MixUp:
    """
    MixUp augmentation for multi-task learning.
    """
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying MixUp
        """
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply MixUp to batch data."""
        if random.random() > self.prob:
            return batch_data
            
        batch_size = batch_data["images"].size(0)
        
        # Generate lambda and random indices
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch_size)
        
        # Mix images
        batch_data["images"] = lam * batch_data["images"] + \
                              (1 - lam) * batch_data["images"][rand_index]
        
        # For segmentation masks, use hard mixing (no interpolation)
        if "masks" in batch_data and random.random() < 0.5:
            batch_data["masks"] = batch_data["masks"][rand_index]
        
        # Mix depth maps
        if "depths" in batch_data:
            batch_data["depths"] = lam * batch_data["depths"] + \
                                  (1 - lam) * batch_data["depths"][rand_index]
        
        batch_data["mixup_lambda"] = lam
        batch_data["mixup_index"] = rand_index
        
        return batch_data


def create_augmentation_pipeline(input_size: Tuple[int, int] = (256, 512),
                               augmentation_strength: str = "medium",
                               use_cutmix: bool = True,
                               use_mixup: bool = False) -> Dict[str, Any]:
    """
    Create complete augmentation pipeline for MTAN training.
    
    Args:
        input_size: Target image size
        augmentation_strength: Augmentation intensity level
        use_cutmix: Whether to use CutMix
        use_mixup: Whether to use MixUp
        
    Returns:
        Dictionary containing augmentation transforms
    """
    # Basic augmentations
    train_aug = MTANAugmentation(
        input_size=input_size,
        is_training=True,
        augmentation_strength=augmentation_strength
    )
    
    val_aug = MTANAugmentation(
        input_size=input_size,
        is_training=False
    )
    
    pipeline = {
        "train_transform": train_aug,
        "val_transform": val_aug,
    }
    
    # Add batch-level augmentations
    if use_cutmix:
        pipeline["cutmix"] = CutMix(alpha=1.0, prob=0.5)
        
    if use_mixup:
        pipeline["mixup"] = MixUp(alpha=0.2, prob=0.5)
    
    return pipeline
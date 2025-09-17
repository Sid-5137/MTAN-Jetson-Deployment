import os
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class ModelConfig:
    """Model configuration for MTAN network - Jetson Nano Optimized"""
    backbone: str = "mobilenetv3_large"  # Smaller backbone for Jetson Nano
    encoder_channels: List[int] = field(default_factory=lambda: [24, 40, 80, 160])  # MobileNetV3 Large channels
    decoder_channels: List[int] = field(default_factory=lambda: [128, 64, 32, 16])
    attention_channels: int = 128  # Reduced for Jetson Nano
    num_classes_seg: int = 19  # Cityscapes classes
    input_size: Tuple[int, int] = (256, 512)  # Further reduced for Jetson Nano performance and memory
    dropout: float = 0.05
    use_aux_loss: bool = True  # Default (overridden based on mode)


@dataclass
class DataConfig:
    """Data configuration for Cityscapes dataset - Enhanced with MTAN augmentations"""
    cityscapes_root: str = "D:/Datasets/Vision/Cityscapes"
    depth_root: str = "D:/Datasets/Vision/Cityscapes/crestereo_depth"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    
    # Enhanced data augmentation
    horizontal_flip: bool = True
    color_jitter: bool = True
    random_scale: Tuple[float, float] = (0.75, 1.25)
    random_crop_size: Tuple[int, int] = (256, 512)  # Match input size
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # New augmentation parameters based on MTAN research
    augmentation_strength: str = "medium"  # 'light', 'medium', 'strong'
    use_cutmix: bool = True
    use_mixup: bool = False
    cutmix_prob: float = 0.5
    mixup_prob: float = 0.3
    
    # Depth normalization
    depth_max: float = 100.0
    depth_min: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration - Enhanced with MTAN research findings"""
    batch_size: int = 4  # Reduced from 8 to 4 to avoid OOM
    num_epochs: int = 300
    learning_rate: float = 3e-4  # Further reduced based on research for stable convergence
    weight_decay: float = 5e-5  # Slightly increased for better regularization
    momentum: float = 0.9
    optimizer: str = "adamw"  # adamw, sgd, adam
    scheduler: str = "cosine"  # cosine, poly, step
    warmup_epochs: int = 10  # Increased for better convergence
    
    # Dynamic task weighting based on MTAN research
    task_weighting: str = "dwa"  # 'equal', 'dwa', 'uncertainty'
    dwa_temperature: float = 1.5  # Lower temperature for more aggressive reweighting
    
    # Loss weights (initial values for equal weighting)
    seg_loss_weight: float = 1.0
    depth_loss_weight: float = 1.0  # Changed from 0.1 to 1.0 for DWA
    aux_loss_weight: float = 0.3  # Reduced for better balance
    
    # Enhanced learning rate schedule
    use_cosine_annealing: bool = True
    lr_min: float = 5e-7  # Lower minimum learning rate
    
    # Gradient clipping - more conservative
    gradient_clip_val: float = 0.5  # Reduced for better stability
    
    # Mixed precision - more conservative settings
    use_amp: bool = True
    amp_init_scale: float = 1024.0  # Reduced initial scale
    amp_growth_factor: float = 1.5  # Slower growth
    
    # Early stopping
    patience: int = 30  # Increased patience for better training
    min_delta: float = 5e-5  # More sensitive early stopping
    
    # Label smoothing for better generalization
    label_smoothing: float = 0.1
    
    # Use EMA for more stable training
    use_ema: bool = True
    ema_decay: float = 0.9999


@dataclass
class OptimizationConfig:
    """Optimization for Jetson Nano edge devices"""
    use_quantization: bool = False
    quantization_mode: str = "ptq"  # Post-training quantization
    
    use_tensorrt: bool = True
    tensorrt_precision: str = "fp16"
    
    export_onnx: bool = True
    onnx_opset_version: int = 11
    
    use_pruning: bool = False
    pruning_ratio: float = 0.4


@dataclass
class Config:
    """Main configuration class - Jetson Nano Optimized"""
    mode: str = "train"  # "train" or "inference"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_interval: int = 50
    val_interval: int = 2
    save_interval: int = 20
    
    # Seed
    seed: int = 42
    
    def create_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# Cityscapes metadata
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]

CITYSCAPES_COLORS = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
]

# Create default config instance
config = Config()

# 🔀 Adjust aux usage depending on mode
if config.mode == "train":
    config.model.use_aux_loss = True
    config.training.aux_loss_weight = 0.5
elif config.mode == "inference":
    config.model.use_aux_loss = False
    config.training.aux_loss_weight = 0.0

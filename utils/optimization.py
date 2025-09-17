import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Dict, Any, Optional, List
import numpy as np

class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine Annealing with Warm Restarts"""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_i:
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                   for base_lr in self.base_lrs]
        else:
            return [self.eta_min for _ in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_0) * self.T_mult + self.T_0
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class PolynomialLR(_LRScheduler):
    """Polynomial Learning Rate Scheduler"""
    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < 0 or self.last_epoch > self.max_epochs:
            return [0 for _ in self.base_lrs]
        
        return [base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power 
                for base_lr in self.base_lrs]

class WarmupScheduler(_LRScheduler):
    """Warmup learning rate scheduler"""
    def __init__(self, optimizer, warmup_epochs, base_scheduler=None, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Use base scheduler if available
            if self.base_scheduler:
                return self.base_scheduler.get_lr()
            return self.base_lrs
    
    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epochs and self.base_scheduler:
            self.base_scheduler.step(epoch - self.warmup_epochs if epoch else None)
        super(WarmupScheduler, self).step(epoch)

def create_optimizer(model: nn.Module, config) -> optim.Optimizer:
    """Create enhanced optimizer based on MTAN research findings"""
    
    # Separate parameters for different learning rates (differential learning rates)
    backbone_params = []
    decoder_params = []
    attention_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name or 'backbone' in name:
            backbone_params.append(param)
        elif 'attention' in name or 'task_attention' in name:
            attention_params.append(param)
        elif 'head' in name or 'classifier' in name:
            head_params.append(param)
        else:
            decoder_params.append(param)
    
    # Enhanced parameter groups with different learning rates based on research
    param_groups = [
        {'params': backbone_params, 'lr': config.training.learning_rate * 0.1, 'name': 'backbone'},  # Lower LR for pretrained backbone
        {'params': attention_params, 'lr': config.training.learning_rate * 1.5, 'name': 'attention'},  # Higher LR for attention modules
        {'params': decoder_params, 'lr': config.training.learning_rate, 'name': 'decoder'},
        {'params': head_params, 'lr': config.training.learning_rate * 2.0, 'name': 'heads'}  # Higher LR for prediction heads
    ]
    
    # Filter out empty parameter groups
    param_groups = [group for group in param_groups if len(group['params']) > 0]
    
    if config.training.optimizer == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True  # More stable convergence
        )
    elif config.training.optimizer == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
            nesterov=True
        )
    elif config.training.optimizer == 'adam':
        optimizer = optim.Adam(
            param_groups,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
    
    return optimizer

def create_scheduler(optimizer: optim.Optimizer, config) -> Optional[_LRScheduler]:
    """Create enhanced learning rate scheduler based on MTAN research"""
    
    # Get enhanced learning rate parameters
    eta_min = getattr(config.training, 'lr_min', config.training.learning_rate * 0.001)
    use_cosine_annealing = getattr(config.training, 'use_cosine_annealing', True)
    
    if use_cosine_annealing or config.training.scheduler == 'cosine':
        T_max = max(1, config.training.num_epochs - config.training.warmup_epochs)
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=T_max,
            eta_min=eta_min
        )
    elif config.training.scheduler == 'cosine_restart':
        base_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, config.training.num_epochs // 4),
            T_mult=2,
            eta_min=eta_min
        )
    elif config.training.scheduler == 'poly':
        base_scheduler = PolynomialLR(
            optimizer,
            max_epochs=max(1, config.training.num_epochs - config.training.warmup_epochs),
            power=0.9
        )
    elif config.training.scheduler == 'step':
        # Enhanced step schedule based on research
        milestones = [50, 80, 110]  # Based on MTAN paper recommendations
        base_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
    elif config.training.scheduler == 'exponential':
        base_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.98  # Slightly more aggressive decay
        )
    else:
        return None
    
    # Add warmup if specified (warmup is crucial for MTAN training)
    warmup_epochs = max(config.training.warmup_epochs, 5)  # Minimum 5 epochs warmup
    if warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            base_scheduler=base_scheduler
        )
    else:
        scheduler = base_scheduler
    
    return scheduler

class GradientClipping:
    """Gradient clipping utilities"""
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return total norm"""
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm, 
            norm_type=self.norm_type
        ).item()

class ModelEMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = 'cuda'):
        self.decay = decay
        self.device = device
        self.ema_model = self._create_ema_model(model)
    
    def _create_ema_model(self, model: nn.Module) -> nn.Module:
        """Create EMA model"""
        ema_model = type(model)(model.config).to(self.device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        
        for param in ema_model.parameters():
            param.requires_grad_(False)
        
        return ema_model
    
    def update(self, model: nn.Module):
        """Update EMA model parameters"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data = self.decay * ema_param.data + (1 - self.decay) * param.data
    
    def get_model(self) -> nn.Module:
        """Get EMA model"""
        return self.ema_model

class MixedPrecisionTraining:
    """Mixed precision training utilities with conservative gradient scaling"""
    def __init__(self, enabled: bool = True, init_scale: float = 2048.0, growth_factor: float = 2.0):
        self.enabled = enabled
        if enabled:
            try:
                self.scaler = torch.amp.GradScaler(
                    'cuda',
                    init_scale=init_scale,
                    growth_factor=growth_factor,
                    backoff_factor=0.5,
                    growth_interval=1000  # Conservative growth interval
                )
            except AttributeError:
                # Fallback for older PyTorch versions
                self.scaler = torch.cuda.amp.GradScaler(
                    init_scale=init_scale,
                    growth_factor=growth_factor,
                    backoff_factor=0.5,
                    growth_interval=1000
                )
        else:
            self.scaler = None
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision"""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: optim.Optimizer):
        """Step optimizer with gradient scaling"""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with mixed precision"""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def unscale_gradients(self, optimizer: optim.Optimizer):
        """Unscale gradients before clipping"""
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)

class OptimizerState:
    """Save and load optimizer state"""
    @staticmethod
    def save_state(optimizer: optim.Optimizer, scheduler: Optional[_LRScheduler], 
                   scaler: Optional[torch.cuda.amp.GradScaler], filepath: str):
        """Save optimizer, scheduler, and scaler state"""
        state = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict() if scaler else None
        }
        torch.save(state, filepath)
    
    @staticmethod
    def load_state(optimizer: optim.Optimizer, scheduler: Optional[_LRScheduler],
                   scaler: Optional[torch.cuda.amp.GradScaler], filepath: str):
        """Load optimizer, scheduler, and scaler state"""
        state = torch.load(filepath)
        
        optimizer.load_state_dict(state['optimizer'])
        
        if scheduler and state['scheduler']:
            scheduler.load_state_dict(state['scheduler'])
        
        if scaler and state['scaler']:
            scaler.load_state_dict(state['scaler'])

def setup_training_components(model: nn.Module, config) -> Dict[str, Any]:
    """Setup all training components with improved gradient explosion prevention"""
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Setup mixed precision with conservative settings
    mixed_precision = MixedPrecisionTraining(
        enabled=config.training.use_amp,
        init_scale=getattr(config.training, 'amp_init_scale', 2048.0),
        growth_factor=getattr(config.training, 'amp_growth_factor', 2.0)
    )
    
    # Setup gradient clipping
    gradient_clipper = GradientClipping(
        max_norm=config.training.gradient_clip_val,
        norm_type=2.0
    )
    
    # Setup EMA (optional)
    ema = None
    if hasattr(config.training, 'use_ema') and config.training.use_ema:
        ema = ModelEMA(model, decay=0.9999, device=config.device)
    
    return {
        'optimizer': optimizer,
        'scheduler': scheduler,
        'mixed_precision': mixed_precision,
        'gradient_clipper': gradient_clipper,
        'ema': ema
    }
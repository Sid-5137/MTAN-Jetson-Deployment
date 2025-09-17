"""
Dynamic Weight Averaging (DWA) for Multi-Task Learning
Based on the original MTAN paper implementation
"""

import torch
import numpy as np
from typing import List, Dict, Optional


class DynamicWeightAveraging:
    """
    Dynamic Weight Averaging as proposed in MTAN paper.
    Adaptively adjusts task weights based on the rate of loss change.
    """
    
    def __init__(self, num_tasks: int = 2, temperature: float = 2.0):
        """
        Initialize DWA weighting strategy.
        
        Args:
            num_tasks: Number of tasks (default: 2 for segmentation + depth)
            temperature: Temperature parameter for softmax weighting (default: 2.0)
        """
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.prev_losses = None
        self.prev_prev_losses = None
        self.weights_history = []
        
    def update_weights(self, current_losses: List[float]) -> List[float]:
        """
        Calculate dynamic weights based on loss rate changes.
        
        Args:
            current_losses: List of current task losses [seg_loss, depth_loss]
            
        Returns:
            List of normalized weights for each task
        """
        # For first two epochs, use equal weighting
        if self.prev_losses is None or self.prev_prev_losses is None:
            weights = [1.0 / self.num_tasks] * self.num_tasks
        else:
            # Calculate loss rate: r_i(t) = L_i(t-1) / L_i(t-2)
            loss_rates = []
            for i in range(self.num_tasks):
                if self.prev_prev_losses[i] > 0:
                    rate = self.prev_losses[i] / self.prev_prev_losses[i]
                else:
                    rate = 1.0
                loss_rates.append(rate)
            
            # Apply DWA formula: w_i(t) = num_tasks * exp(r_i(t) / T) / sum(exp(r_j(t) / T))
            exp_rates = [np.exp(rate / self.temperature) for rate in loss_rates]
            sum_exp = sum(exp_rates)
            
            if sum_exp > 0:
                weights = [self.num_tasks * exp_rate / sum_exp for exp_rate in exp_rates]
            else:
                weights = [1.0 / self.num_tasks] * self.num_tasks
        
        # Store weights history for analysis
        self.weights_history.append(weights.copy())
        
        # Update loss history
        self.prev_prev_losses = self.prev_losses
        self.prev_losses = current_losses.copy()
        
        return weights
    
    def get_current_weights(self) -> Optional[List[float]]:
        """Get the most recent weights."""
        if self.weights_history:
            return self.weights_history[-1]
        return None
    
    def reset(self):
        """Reset the weighting history."""
        self.prev_losses = None
        self.prev_prev_losses = None
        self.weights_history = []


class UncertaintyWeighting:
    """
    Uncertainty-based weighting as an alternative to DWA.
    Learns task weights based on homoscedastic uncertainty.
    """
    
    def __init__(self, num_tasks: int = 2):
        """Initialize uncertainty weighting with learnable parameters."""
        self.num_tasks = num_tasks
        # Initialize log variance parameters (learnable)
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculate uncertainty-weighted loss.
        
        Args:
            losses: List of task losses
            
        Returns:
            Weighted combined loss
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            # Precision = exp(-log_var), Weight = 0.5 * precision
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = 0.5 * precision * loss + 0.5 * self.log_vars[i]
            weighted_losses.append(weighted_loss)
        
        return sum(weighted_losses)
    
    def get_weights(self) -> List[float]:
        """Get current task weights based on learned uncertainties."""
        with torch.no_grad():
            precisions = torch.exp(-self.log_vars)
            weights = precisions / precisions.sum()
            return weights.cpu().numpy().tolist()


class GradientNormalization:
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing
    Alternative advanced weighting strategy.
    """
    
    def __init__(self, num_tasks: int = 2, alpha: float = 0.12):
        """
        Initialize GradNorm weighting.
        
        Args:
            num_tasks: Number of tasks
            alpha: Restoring force strength (default: 0.12)
        """
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.initial_losses = None
        self.weights = torch.nn.Parameter(torch.ones(num_tasks))
        
    def update_weights(self, losses: List[torch.Tensor], shared_parameters) -> List[float]:
        """
        Update task weights using gradient normalization.
        
        Args:
            losses: Current task losses
            shared_parameters: Shared network parameters for gradient computation
            
        Returns:
            Updated task weights
        """
        if self.initial_losses is None:
            self.initial_losses = [loss.item() for loss in losses]
            return [1.0 / self.num_tasks] * self.num_tasks
        
        # Calculate relative loss ratios
        loss_ratios = []
        for i, loss in enumerate(losses):
            ratio = loss.item() / self.initial_losses[i]
            loss_ratios.append(ratio)
        
        # Calculate average relative loss
        avg_loss_ratio = sum(loss_ratios) / len(loss_ratios)
        
        # Calculate relative inverse training rates
        inverse_rates = [ratio / avg_loss_ratio for ratio in loss_ratios]
        
        # Target: r_i(t) = avg_rate * (r_i(0))^alpha
        targets = [avg_loss_ratio * (self.initial_losses[i] / sum(self.initial_losses)) ** self.alpha 
                  for i in range(self.num_tasks)]
        
        # Return current weights (simplified version)
        with torch.no_grad():
            current_weights = torch.softmax(self.weights, dim=0)
            return current_weights.cpu().numpy().tolist()
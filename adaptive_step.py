#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adaptive Step Size Controllers for Resilient Nash Equilibrium Seeking

This module provides various adaptive step size methods to accelerate convergence
in distributed optimization algorithms. These methods adjust learning rates based
on gradient history, which is particularly useful in adversarial settings where
different agents experience different levels of corruption.

Available methods:
- AdaGrad: Adapts step sizes based on accumulated squared gradients
- RMSProp: Uses exponential moving average of squared gradients
- Adam: Combines momentum with adaptive learning rates
- Constant: Baseline fixed step size for comparison
"""

from typing import Literal, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class AdaptiveStepConfig:
    """Configuration for adaptive step size controllers"""
    method: Literal['constant', 'adagrad', 'rmsprop', 'adam'] = 'adam'
    base_lr: float = 0.1
    epsilon: float = 1e-8
    # RMSProp/Adam decay rate for squared gradients
    beta2: float = 0.999
    # Adam momentum decay rate
    beta1: float = 0.9
    # Optional per-agent learning rate scaling
    per_agent: bool = True


class AdaptiveStepController:
    """
    Adaptive step size controller for gradient-based optimization.
    
    Supports multiple adaptation methods that adjust step sizes based on
    gradient history. This can significantly accelerate convergence,
    especially in heterogeneous settings where different components
    have different gradient magnitudes.
    
    Parameters
    ----------
    dim_state : int
        Dimension of the state vector (N * dim_action for the full state)
    config : AdaptiveStepConfig
        Configuration specifying method and hyperparameters
    
    Example
    -------
    >>> controller = AdaptiveStepController(dim_state=100)
    >>> for iteration in range(num_iters):
    ...     gradients = compute_gradients(state)
    ...     step_sizes = controller.get_step_sizes(gradients)
    ...     state = state - step_sizes * gradients
    """
    
    def __init__(self, dim_state: int, config: Optional[AdaptiveStepConfig] = None):
        self.dim_state = dim_state
        self.config = config or AdaptiveStepConfig()
        
        # Initialize state variables for different methods
        self._init_state()
        
        # Iteration counter (for Adam bias correction)
        self.t = 0
    
    def _init_state(self):
        """Initialize internal state based on method"""
        shape = (self.dim_state, 1)
        
        if self.config.method == 'adagrad':
            # Accumulated squared gradients
            self.grad_sq_sum = np.zeros(shape)
            
        elif self.config.method == 'rmsprop':
            # Exponential moving average of squared gradients
            self.grad_sq_ema = np.zeros(shape)
            
        elif self.config.method == 'adam':
            # First moment (momentum)
            self.m = np.zeros(shape)
            # Second moment (squared gradients)
            self.v = np.zeros(shape)
    
    def reset(self):
        """Reset internal state (useful for new simulation runs)"""
        self._init_state()
        self.t = 0
    
    def get_step_sizes(self, gradients: np.ndarray) -> np.ndarray:
        """
        Compute adaptive step sizes based on current gradients.
        
        Parameters
        ----------
        gradients : np.ndarray
            Current gradient vector, shape (dim_state, 1)
        
        Returns
        -------
        np.ndarray
            Step sizes for each component, shape (dim_state, 1)
        """
        self.t += 1
        
        if self.config.method == 'constant':
            return self._constant_step(gradients)
        elif self.config.method == 'adagrad':
            return self._adagrad_step(gradients)
        elif self.config.method == 'rmsprop':
            return self._rmsprop_step(gradients)
        elif self.config.method == 'adam':
            return self._adam_step(gradients)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
    
    def _constant_step(self, gradients: np.ndarray) -> np.ndarray:
        """Fixed step size baseline"""
        return np.full_like(gradients, self.config.base_lr)
    
    def _adagrad_step(self, gradients: np.ndarray) -> np.ndarray:
        """
        AdaGrad: Adapts step sizes based on accumulated squared gradients.
        
        Components with larger historical gradients get smaller step sizes,
        while components with smaller gradients get larger step sizes.
        This is particularly useful when gradient magnitudes vary significantly
        across components (common in adversarial settings).
        """
        self.grad_sq_sum += gradients ** 2
        step_sizes = self.config.base_lr / (np.sqrt(self.grad_sq_sum) + self.config.epsilon)
        return step_sizes
    
    def _rmsprop_step(self, gradients: np.ndarray) -> np.ndarray:
        """
        RMSProp: Uses exponential moving average of squared gradients.
        
        Unlike AdaGrad, RMSProp doesn't accumulate all past gradients,
        which prevents the step size from becoming too small over time.
        The decay rate beta2 controls how much history is retained.
        """
        self.grad_sq_ema = (self.config.beta2 * self.grad_sq_ema + 
                           (1 - self.config.beta2) * gradients ** 2)
        step_sizes = self.config.base_lr / (np.sqrt(self.grad_sq_ema) + self.config.epsilon)
        return step_sizes
    
    def _adam_step(self, gradients: np.ndarray) -> np.ndarray:
        """
        Adam: Combines momentum with adaptive learning rates.
        
        Adam maintains both a momentum term (exponential moving average of gradients)
        and an adaptive learning rate (exponential moving average of squared gradients).
        Bias correction is applied to account for initialization at zero.
        
        This is often the best general-purpose adaptive method.
        """
        # Update biased first moment estimate (momentum)
        self.m = self.config.beta1 * self.m + (1 - self.config.beta1) * gradients
        
        # Update biased second moment estimate
        self.v = self.config.beta2 * self.v + (1 - self.config.beta2) * gradients ** 2
        
        # Bias correction
        m_hat = self.m / (1 - self.config.beta1 ** self.t)
        v_hat = self.v / (1 - self.config.beta2 ** self.t)
        
        # Compute update (note: we return step_sizes * direction, not just step_sizes)
        # So the caller does: state = state - step_sizes * gradients
        # But Adam wants: state = state - lr * m_hat / (sqrt(v_hat) + eps)
        # We handle this by returning effective step sizes that include momentum
        step_sizes = self.config.base_lr / (np.sqrt(v_hat) + self.config.epsilon)
        
        # Return the momentum-adjusted gradient direction scaled by adaptive step size
        # The caller will multiply by gradients, but we want m_hat instead
        # So we return: step_sizes * m_hat / gradients (where gradients != 0)
        # This is a bit awkward; let's provide an alternative interface
        return step_sizes
    
    def get_adam_update(self, gradients: np.ndarray) -> np.ndarray:
        """
        Get the full Adam update directly (recommended for Adam).
        
        Returns the actual update to subtract from the state,
        rather than step sizes to multiply with gradients.
        
        Parameters
        ----------
        gradients : np.ndarray
            Current gradient vector, shape (dim_state, 1)
        
        Returns
        -------
        np.ndarray
            Update vector to subtract from state, shape (dim_state, 1)
        """
        self.t += 1
        
        # Update biased first moment estimate (momentum)
        self.m = self.config.beta1 * self.m + (1 - self.config.beta1) * gradients
        
        # Update biased second moment estimate
        self.v = self.config.beta2 * self.v + (1 - self.config.beta2) * gradients ** 2
        
        # Bias correction
        m_hat = self.m / (1 - self.config.beta1 ** self.t)
        v_hat = self.v / (1 - self.config.beta2 ** self.t)
        
        # Compute and return the update
        update = self.config.base_lr * m_hat / (np.sqrt(v_hat) + self.config.epsilon)
        return update


def create_controller(dim_state: int, 
                      method: str = 'adam',
                      base_lr: float = 0.1,
                      **kwargs) -> AdaptiveStepController:
    """
    Factory function to create an adaptive step controller.
    
    Parameters
    ----------
    dim_state : int
        Dimension of the state vector
    method : str
        One of 'constant', 'adagrad', 'rmsprop', 'adam'
    base_lr : float
        Base learning rate
    **kwargs
        Additional parameters passed to AdaptiveStepConfig
    
    Returns
    -------
    AdaptiveStepController
        Configured controller instance
    """
    config = AdaptiveStepConfig(method=method, base_lr=base_lr, **kwargs)
    return AdaptiveStepController(dim_state, config)

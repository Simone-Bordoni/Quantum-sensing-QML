"""
Callback utilities for tracking optimization progress in quantum sensing experiments.

This module provides callback classes for monitoring and saving optimization metrics
during quantum sensing experiments, including loss functions, detection probabilities,
and contrast values.
"""

from typing import Optional, Dict, List, Any
import numpy as np
from pathlib import Path


class OptimizationCallback:
    """
    Callback for tracking optimization progress with detailed metrics.
    
    This callback tracks:
    - Loss function values
    - Detection probabilities (with and without photon)
    - Sensing contrast
    - Parameter values at each epoch
    - Best parameters found
    
    Attributes:
        save_every (int): Save history every N epochs
        save_best (bool): Track best parameters based on loss
        epoch (int): Current epoch number
        history (Dict): Complete optimization history
        best_parameters (Optional[np.ndarray]): Best parameters found
        best_loss (float): Best loss value found
        best_metrics (Optional[Dict]): Metrics at best parameters
    
    Example:
        >>> callback = OptimizationCallback(save_every=5, save_best=True)
        >>> # During optimization loop:
        >>> callback(
        ...     parameters=params,
        ...     loss=loss_value,
        ...     prob_with=0.85,
        ...     prob_without=0.15,
        ...     contrast=0.70
        ... )
        >>> callback.save('optimization_results.npz')
    """
    
    def __init__(self, save_every: int = 1, save_best: bool = True):
        """
        Initialize the optimization callback.
        
        Args:
            save_every: Save metrics every N epochs (default: 1 = every epoch)
            save_best: Whether to track best parameters (default: True)
        """
        self.save_every = save_every
        self.save_best = save_best
        self.epoch = 0
        
        # Initialize history containers
        self.history: Dict[str, List[Any]] = {
            'epochs': [],
            'loss': [],
            'contrast': [],
            'prob_with': [],
            'prob_without': [],
            'parameters': []
        }
        
        # Best tracking
        self.best_parameters: Optional[np.ndarray] = None
        self.best_loss: float = float('inf')
        self.best_metrics: Optional[Dict[str, float]] = None
    
    def __call__(self, 
                 parameters: np.ndarray,
                 loss: float,
                 prob_with: float,
                 prob_without: float,
                 contrast: float) -> None:
        """
        Record metrics from current optimization step.
        
        This method is called after each optimization step to record the current
        state of the optimization, including loss, probabilities, and parameters.
        
        Args:
            parameters: Current parameter values
            loss: Current loss function value
            prob_with: Detection probability with photon interaction
            prob_without: Detection probability without photon interaction
            contrast: Sensing contrast (prob_with - prob_without)
        """
        self.epoch += 1
        
        # Save history every N epochs
        if self.epoch % self.save_every == 0:
            self.history['epochs'].append(self.epoch)
            self.history['loss'].append(float(loss))
            self.history['contrast'].append(float(contrast))
            self.history['prob_with'].append(float(prob_with))
            self.history['prob_without'].append(float(prob_without))
            self.history['parameters'].append(parameters.copy())
        
        # Track best parameters if enabled
        if self.save_best and loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_parameters = parameters.copy()
            self.best_metrics = {
                'epoch': self.epoch,
                'loss': float(loss),
                'contrast': float(contrast),
                'prob_with': float(prob_with),
                'prob_without': float(prob_without)
            }
    
    def get_best_parameters(self) -> Optional[np.ndarray]:
        """
        Get the best parameters found during optimization.
        
        Returns:
            Best parameters array, or None if no parameters recorded yet
        """
        return self.best_parameters
    
    def get_best_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get the metrics at the best parameters.
        
        Returns:
            Dictionary containing epoch, loss, contrast, and probabilities
            at the best parameters, or None if no best parameters found
        """
        return self.best_metrics
    
    def get_history(self) -> Dict[str, List[Any]]:
        """
        Get the complete optimization history.
        
        Returns:
            Dictionary with lists of epochs, loss values, contrasts,
            probabilities, and parameters
        """
        return self.history
    
    def save(self, filepath: str) -> None:
        """
        Save optimization results to an NPZ file.
        
        The saved file contains all history arrays and best parameters
        in a format optimized for easy loading and plotting.
        
        Args:
            filepath: Path to save the NPZ file (e.g., 'results.npz')
        
        Example:
            >>> callback.save('optimization_results.npz')
            >>> # Later, load with:
            >>> data = np.load('optimization_results.npz')
            >>> epochs = data['epochs']
            >>> loss = data['loss']
            >>> contrast = data['contrast']
        """
        filepath = Path(filepath)
        
        # Prepare data for saving
        save_dict = {
            'epochs': np.array(self.history['epochs']),
            'loss': np.array(self.history['loss']),
            'contrast': np.array(self.history['contrast']),
            'prob_with': np.array(self.history['prob_with']),
            'prob_without': np.array(self.history['prob_without']),
            'parameters': np.array(self.history['parameters'])
        }
        
        # Add best parameters if available
        if self.best_parameters is not None:
            save_dict['best_parameters'] = self.best_parameters
            save_dict['best_loss'] = np.array(self.best_loss)
            
            if self.best_metrics is not None:
                save_dict['best_epoch'] = np.array(self.best_metrics['epoch'])
                save_dict['best_contrast'] = np.array(self.best_metrics['contrast'])
                save_dict['best_prob_with'] = np.array(self.best_metrics['prob_with'])
                save_dict['best_prob_without'] = np.array(self.best_metrics['prob_without'])
        
        # Save to NPZ file
        np.savez(filepath, **save_dict)
    
    @staticmethod
    def load(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load optimization results from an NPZ file.
        
        Args:
            filepath: Path to the NPZ file to load
        
        Returns:
            Dictionary containing all saved arrays
        
        Example:
            >>> data = OptimizationCallback.load('results.npz')
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(data['epochs'], data['contrast'])
            >>> plt.xlabel('Epoch')
            >>> plt.ylabel('Contrast')
            >>> plt.show()
        """
        data = np.load(filepath)
        return {key: data[key] for key in data.files}
    
    def reset(self) -> None:
        """
        Reset callback to initial state.
        
        Clears all history and best parameter tracking.
        """
        self.epoch = 0
        self.history = {
            'epochs': [],
            'loss': [],
            'contrast': [],
            'prob_with': [],
            'prob_without': [],
            'parameters': []
        }
        self.best_parameters = None
        self.best_loss = float('inf')
        self.best_metrics = None
    
    def __repr__(self) -> str:
        """String representation of the callback."""
        status = f"OptimizationCallback(epoch={self.epoch}, "
        status += f"history_size={len(self.history['epochs'])}, "
        if self.best_parameters is not None:
            status += f"best_loss={self.best_loss:.6f})"
        else:
            status += "best_loss=None)"
        return status

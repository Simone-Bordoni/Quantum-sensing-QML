"""
Callback utilities for tracking optimization progress in quantum sensing experiments.

This module provides callback classes for monitoring and saving optimization metrics
during quantum sensing experiments, including detection probabilities,
contrast values, and trainable parameters.
"""

from typing import Optional, Dict, List, Any, TYPE_CHECKING
import numpy as np
from pathlib import Path
import copy

if TYPE_CHECKING:
    from qsopt.core.trainable_parameters import TrainableParameters


class OptimizationCallback:
    """
    Callback for tracking optimization progress with detailed metrics.
    
    This callback tracks:
    - Detection probabilities (with and without photon)
    - Sensing contrast (optimization objective)
    - Trainable parameters at each epoch
    - Best parameters found (maximizing contrast)
    
    Attributes:
        save_every (int): Save history every N epochs
        save_best (bool): Track best parameters based on contrast
        epoch (int): Current epoch number
        history (Dict): Complete optimization history
        best_trainable_params (Optional): Best trainable parameters found
        best_metrics (Optional[Dict]): Metrics at best parameters
    """
    
    def __init__(self, save_every: int = 1, save_best: bool = True):
        """
        Initialize the optimization callback.
        
        Args:
            save_every: Save metrics every N epochs (default: 1 = every epoch)
            save_best: Whether to track best parameters based on contrast (default: True)
        """
        self.save_every = save_every
        self.save_best = save_best
        self.epoch = 0
        
        # Initialize history containers
        self.history: Dict[str, List[Any]] = {
            'epochs': [],
            'contrast': [],
            'prob_with': [],
            'prob_without': [],
            'trainable_params': []
        }
        
        # Best tracking (maximize contrast)
        self.best_trainable_params: Optional[Any] = None
        self.best_contrast: float = -float('inf')
        self.best_metrics: Optional[Dict[str, float]] = None
        
        # Optimization completion info
        self.converged: bool = False
        self.final_grad_norm: Optional[float] = None
    
    def __call__(self, 
                 trainable_params: 'TrainableParameters',
                 prob_with: float,
                 prob_without: float,
                 contrast: float) -> None:
        """
        Record metrics from current optimization step.
        
        This method is called after each optimization step to record the current
        state of the optimization, including probabilities, contrast, and parameters.
        
        Args:
            trainable_params: Current trainable parameters object
            prob_with: Detection probability with photon interaction
            prob_without: Detection probability without photon interaction
            contrast: Sensing contrast (prob_with - prob_without)
        """
        self.epoch += 1
        
        # Save history every N epochs
        if self.epoch % self.save_every == 0:
            self.history['epochs'].append(self.epoch)
            self.history['contrast'].append(float(contrast))
            self.history['prob_with'].append(float(prob_with))
            self.history['prob_without'].append(float(prob_without))
            # Save a deep copy of trainable_params to preserve state
            self.history['trainable_params'].append(copy.deepcopy(trainable_params))
        
        # Track best parameters if enabled (maximize contrast)
        if self.save_best and contrast > self.best_contrast:
            self.best_contrast = float(contrast)
            self.best_trainable_params = copy.deepcopy(trainable_params)
            self.best_metrics = {
                'epoch': self.epoch,
                'contrast': float(contrast),
                'prob_with': float(prob_with),
                'prob_without': float(prob_without)
            }
    
    def get_best_trainable_params(self) -> Optional[Any]:
        """
        Get the best trainable parameters found during optimization.
        
        Returns:
            Best TrainableParameters object, or None if no parameters recorded yet
        """
        return self.best_trainable_params
    
    def get_best_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get the metrics at the best parameters.
        
        Returns:
            Dictionary containing epoch, contrast, and probabilities
            at the best parameters, or None if no best parameters found
        """
        return self.best_metrics
    
    def get_history(self) -> Dict[str, List[Any]]:
        """
        Get the complete optimization history.
        
        Returns:
            Dictionary with lists of epochs, contrasts,
            probabilities, and trainable parameters
        """
        return self.history
    
    def set_convergence_info(self, converged: bool, final_grad_norm: float) -> None:
        """
        Set convergence information at the end of optimization.
        
        Args:
            converged: Whether the optimization converged
            final_grad_norm: Final gradient norm value
        """
        self.converged = converged
        self.final_grad_norm = final_grad_norm
    
    def save(self, filepath: str = 'optimization_results.npz') -> None:
        """
        Save optimization results to an NPZ file.
        
        The saved file contains all history arrays and best parameters.
        Note: trainable_params objects are converted to parameter arrays for NPZ storage.
        
        Args:
            filepath: Path to save the NPZ file (e.g., 'results.npz')
        
        Example:
            >>> callback.save('optimization_results.npz')
            >>> # Later, load with:
            >>> data = np.load('optimization_results.npz')
            >>> epochs = data['epochs']
            >>> contrast = data['contrast']
        """
        filepath = Path(filepath)
        
        # Convert trainable_params to parameter arrays for saving
        param_arrays = []
        for tp in self.history['trainable_params']:
            angles = tp.get_rotation_angles()
            param_arrays.append(np.array([angles[name][0] for name in angles.keys()]))
        
        # Prepare data for saving
        save_dict = {
            'epochs': np.array(self.history['epochs']),
            'contrast': np.array(self.history['contrast']),
            'prob_with': np.array(self.history['prob_with']),
            'prob_without': np.array(self.history['prob_without']),
            'parameters': np.array(param_arrays) if param_arrays else np.array([])
        }
        
        # Add best parameters if available
        if self.best_trainable_params is not None:
            best_angles = self.best_trainable_params.get_rotation_angles()
            save_dict['best_parameters'] = np.array([best_angles[name][0] for name in best_angles.keys()])
            save_dict['best_contrast'] = np.array(self.best_contrast)
            
            if self.best_metrics is not None:
                save_dict['best_epoch'] = np.array(self.best_metrics['epoch'])
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
            'contrast': [],
            'prob_with': [],
            'prob_without': [],
            'trainable_params': []
        }
        self.best_trainable_params = None
        self.best_contrast = -float('inf')
        self.best_metrics = None
        self.converged = False
        self.final_grad_norm = None
    
    def __repr__(self) -> str:
        """
        Pretty-print callback results with all key metrics.
        
        For run_simulation (1 epoch): Shows current angles and metrics
        For optimize (>1 epoch): Shows best angles, convergence, and optimization info
        """
        lines = []
        lines.append("=" * 70)
        lines.append("Quantum Sensing Results")
        lines.append("=" * 70)
        
        # Determine if this is from run_simulation (1 epoch) or optimization (>1 epoch)
        is_simulation = (self.epoch == 1 and self.converged is False and 
                        self.final_grad_norm is None)
        
        if is_simulation:
            lines.append("Mode: Single Simulation (run_simulation)")
        else:
            lines.append("Mode: Optimization")
            lines.append(f"  Total iterations: {self.epoch}")
            lines.append(f"  Converged: {self.converged}")
            if self.final_grad_norm is not None:
                lines.append(f"  Final gradient norm: {self.final_grad_norm:.6e}")
        
        lines.append("-" * 70)
        
        # Show angles (best for optimization, current for simulation)
        if self.best_trainable_params is not None:
            angles = self.best_trainable_params.get_rotation_angles()
            angle_names = list(angles.keys())
            
            if is_simulation:
                lines.append("Current Parameters:")
            else:
                lines.append("Best Parameters:")
            
            for i, name in enumerate(angle_names[:2]):  # Show first two rotation angles
                value = angles[name][0]
                lines.append(f"  {name}: {value:.6f} rad ({np.rad2deg(value):.2f}°)")
        
        lines.append("-" * 70)
        
        # Show metrics (best for optimization, current for simulation)
        if self.best_metrics is not None:
            lines.append("Detection Probabilities:")
            lines.append(f"  P(with photon):    {self.best_metrics['prob_with']:.6f}")
            lines.append(f"  P(without photon): {self.best_metrics['prob_without']:.6f}")
            lines.append(f"  Contrast:          {self.best_metrics['contrast']:.6f}")
            
            if not is_simulation:
                lines.append(f"  Best epoch:        {self.best_metrics['epoch']}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)

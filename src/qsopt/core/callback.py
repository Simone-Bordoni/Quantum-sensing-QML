"""
Callback utilities for tracking optimization progress in quantum sensing experiments.

This module provides callback classes for monitoring and saving optimization metrics
during quantum sensing experiments, including detection measures,
metric values, and trainable parameters.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

class OptimizationCallback:
    """
    Callback for tracking optimization progress with detailed metrics.

    This callback tracks:
    - Detection measures (with and without photon)
    - Metric (optimization objective)
    - Trainable parameters at each epoch
    - Best parameters found (maximizing the metric)

    Attributes:
        save_every (int): Save history every N epochs
        save_best (bool): Track best parameters based on the metric
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
            save_best: Whether to track best parameters based on the metric (default: True)
        """
        self.save_every = save_every
        self.save_best = save_best
        self.epoch = 0

        # Initialize history containers
        self.history: Dict[str, List[Any]] = {
            "epochs": [],
            "metric": [],
            "prob_with": [],
            "prob_without": [],
            "trainable_params": [],
        }

        # Best tracking (maximize the metric)
        self.best_trainable_params: Optional[Any] = None
        self.best_metric: float = -float("inf")
        self.best_metrics: Optional[Dict[str, float]] = None

        # Optimization completion info
        self.converged: bool = False
        self.final_grad_norm: Optional[float] = None

    def __call__(
        self,
        trainable_params_initial: Optional[list] = None,
        trainable_params_final: Optional[list] = None,
        prob_with: float = 0.0,
        prob_without: float = 0.0,
        metric: float = 0.0,
        trainable_params: Optional[tuple] = None,  # Backward compatibility
        **kwargs
    ) -> None:
        """
        Record metrics from current optimization step.

        This method is called after each optimization step to record the current
        state of the optimization, including detection measures, the metric, and parameters.

        Args:
            trainable_params_initial: Initial circuit trainable parameters (list of values)
            trainable_params_final: Final circuit trainable parameters (list of values)
            prob_with: Detection measure with photon interaction
            prob_without: Detection measure without photon interaction
            metric: Optimization metric value
            trainable_params: (Deprecated) Tuple of (initial, final) params for backward compatibility
            **kwargs: Additional keyword arguments (for backward compatibility)
        """
        self.epoch += 1
        metric_value = float(metric)

        # Handle backward compatibility: if trainable_params tuple is provided, unpack it
        if trainable_params is not None:
            trainable_params_initial, trainable_params_final = trainable_params

        # Package parameters as tuple for internal storage
        trainable_params = (trainable_params_initial, trainable_params_final)

        # Save history every N epochs
        if self.epoch % self.save_every == 0:
            self.history["epochs"].append(self.epoch)
            self.history["metric"].append(metric_value)
            self.history["prob_with"].append(float(prob_with))
            self.history["prob_without"].append(float(prob_without))
            # Save a deep copy of trainable_params to preserve state
            self.history["trainable_params"].append(copy.deepcopy(trainable_params))

        # Track best parameters if enabled (maximize the metric)
        if self.save_best and metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_trainable_params = copy.deepcopy(trainable_params)
            self.best_metrics = {
                "epoch": self.epoch,
                "metric": metric_value,
                "prob_with": float(prob_with),
                "prob_without": float(prob_without),
            }

    def get_best_trainable_params(self) -> Optional[tuple[list, list]]:
        """
        Get the best trainable parameters found during optimization.

        Returns:
            Tuple of (initial_circuit_params, final_circuit_params),
            or None if no parameters recorded yet
        """
        return self.best_trainable_params

    def get_best_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get the metrics at the best parameters.

        Returns:
            Dictionary containing epoch, metric, and detection measures
            at the best parameters, or None if no best parameters found
        """
        return self.best_metrics

    def get_history(self) -> Dict[str, List[Any]]:
        """
        Get the complete optimization history.

        Returns:
            Dictionary with lists of epochs, metric values,
            detection measures, and trainable parameters
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

    def save(self, filepath: str = "optimization_results.npz") -> None:
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
            >>> metric = data['metric']
        """
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert trainable_params to parameter arrays for saving
        param_arrays = []
        for tp_tuple in self.history["trainable_params"]:
            # tp_tuple is (initial_params, final_params)
            initial_params, final_params = tp_tuple
            # Flatten both lists into a single array
            all_params = [float(p) for p in initial_params] + [float(p) for p in final_params]
            param_arrays.append(np.array(all_params))

        # Prepare data for saving
        save_dict = {
            "epochs": np.array(self.history["epochs"]),
            "metric": np.array(self.history["metric"]),
            "prob_with": np.array(self.history["prob_with"]),
            "prob_without": np.array(self.history["prob_without"]),
            "parameters": np.array(param_arrays) if param_arrays else np.array([]),
        }

        # Add best parameters if available
        if self.best_trainable_params is not None:
            initial_best, final_best = self.best_trainable_params
            all_best = [float(p) for p in initial_best] + [float(p) for p in final_best]
            save_dict["best_parameters"] = np.array(all_best)
            save_dict["best_metric"] = np.array(self.best_metric)

            if self.best_metrics is not None:
                save_dict["best_epoch"] = np.array(self.best_metrics["epoch"])
                save_dict["best_prob_with"] = np.array(self.best_metrics["prob_with"])
                save_dict["best_prob_without"] = np.array(self.best_metrics["prob_without"])

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
            >>> plt.plot(data['epochs'], data['metric'])
            >>> plt.xlabel('Epoch')
            >>> plt.ylabel('Metric')
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
            "epochs": [],
            "metric": [],
            "prob_with": [],
            "prob_without": [],
            "trainable_params": [],
        }
        self.best_trainable_params = None
        self.best_metric = -float("inf")
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

        # Determine if this is from run_simulation (1 epoch) or optimization (>1 epoch)
        is_simulation = self.epoch == 1 and self.converged is False and self.final_grad_norm is None

        if is_simulation:
            lines.append("MODE: Single Simulation")
        else:
            lines.append("MODE: Optimization")
            lines.append(f"     Total iterations: {self.epoch}")
            if self.best_metrics is not None:
                lines.append(f"     Best epoch:        {self.best_metrics['epoch']}")
            lines.append(f"     Converged: {self.converged}")
            if self.final_grad_norm is not None:
                lines.append(f"     Final gradient norm: {self.final_grad_norm:.6e}")

        # Show angles (best for optimization, current for simulation)
        if self.best_trainable_params is not None:
            initial_params, final_params = self.best_trainable_params

            if is_simulation:
                lines.append("  Current Parameters:")
            else:
                lines.append("  Best Parameters:")

            # Show initial circuit parameters
            lines.append("     Initial circuit:")
            for i, value in enumerate(initial_params):
                # Convert to numpy to handle both regular floats and JAX arrays
                val_float = float(np.asarray(value))
                lines.append(f"        param_{i}: {val_float:.6f} rad ({np.rad2deg(val_float):.2f}°)")

            # Show final circuit parameters
            lines.append("     Final circuit:")
            for i, value in enumerate(final_params):
                # Convert to numpy to handle both regular floats and JAX arrays
                val_float = np.asarray(value)
                lines.append(f"        param_{i}: {val_float:.6f} rad ({np.rad2deg(val_float):.2f}°)")

        # Show metrics (best for optimization, current for simulation)
        if self.best_metrics is not None:
            lines.append("  Detection Probabilities:")
            lines.append(f"     P(with photon):    {self.best_metrics['prob_with']:.6f}")
            lines.append(f"     P(without photon): {self.best_metrics['prob_without']:.6f}")
            lines.append(f"     Metric:            {self.best_metrics['metric']:.6f}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation (calls __repr__)."""
        return self.__repr__()

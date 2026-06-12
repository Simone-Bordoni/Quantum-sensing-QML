"""
Callback utilities for tracking optimization progress in quantum sensing experiments.

This module provides callback classes for monitoring and saving optimization metrics
during quantum sensing experiments, including detection measures,
metric values, and trainable parameters.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

class OptimizationCallback:
    """
    Callback for tracking optimization progress with detailed metrics.

    This callback tracks:
    - Detection measures for each configuration (detection_dict)
    - Metric (optimization objective) and validation value
    - Trainable parameters (initial and final circuit trainable parameters)
    - Gradients and optimizer state
    - Best parameters found (maximizing the validation value)
    - Detection state classification and confusion matrix across configurations

    Attributes:
        save_every (int): Save history every N epochs
        save_best (bool): Track best parameters based on the validation value
        epoch (int): Current epoch number
        history (Dict): Complete optimization history
        best_trainable_params (Optional): Best trainable parameters found
        best_validation (float): Best validation value observed so far
        best_metrics (Optional[Dict]): Metrics at best parameters (epoch, metric,
            validation and per-configuration detection_dict)
        state_probabilities (Optional[Dict[str, Dict[str, float]]]): Per-configuration
            measurement-state probabilities, mapping configuration name to a dict of
            binary state strings (e.g. '001', '010') to probabilities.
        detection_states (Dict[str, List[str]]): Mapping of configuration name to the
            list of binary state strings that are classified as that configuration
            (i.e. the configuration that maximises the probability of the state).
        confusion_matrix (Dict[Tuple[str, str], float]): Soft confusion matrix mapping
            (true_configuration, predicted_configuration) to the accumulated probability
            mass. Each true-configuration row sums to 1 when the input probabilities are
            normalised.
    """

    def __init__(self, save_every: int = 1, save_best: bool = True):
        """
        Initialize the optimization callback.

        Args:
            save_every: Save metrics every N epochs (default: 1 = every epoch)
            save_best: Whether to track best parameters based on the validation value (default: True)
        """
        self.save_every = save_every
        self.save_best = save_best
        self.epoch = 0

        # Initialize history containers
        self.history: Dict[str, List[Any]] = self._empty_history_template()

        # Best tracking (maximize the metric)
        self.best_trainable_params: Optional[Any] = None
        self.best_validation: float = -float("inf")
        self.best_metrics: Optional[Dict[str, Union[int, float, Dict[str, float]]]] = None

        # Optional per-run state probabilities populated by experiment.run_simulation
        self.state_probabilities: Optional[Dict[str, Dict[str, float]]] = None

        # Optimization completion info
        self.converged: bool = False
        self.final_grad_norm: Optional[float] = None

        # Detection state tracking from set_measurement_protocol
        self.detection_states: Dict[str, List[str]] = {}
        self.confusion_matrix: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def _empty_history_template() -> Dict[str, List[Any]]:
        """Create an empty history dictionary with the expected keys."""
        return {
            "epochs": [],
            "metric": [],
            "validation": [],
            "detection_dict": [],
            "trainable_params": [],
            "optimizer_state": [],
            "grads": [],
        }

    @staticmethod
    def _to_object_array(values: List[Any]) -> np.ndarray:
        """Convert a Python list to a 1D object array for NPZ serialization."""
        arr = np.empty(len(values), dtype=object)
        arr[:] = values
        return arr

    def __call__(
        self,
        trainable_params_initial: Optional[list] = None,
        trainable_params_final: Optional[list] = None,
        detection_dict: Dict[str, float] = None,
        metric: float = 0.0,
        validation: float = 0.0,
        optimizer_state: Any = None,
        grads: Any = None,
        state_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """
        Record metrics from current optimization step.

        This method is called after each optimization step to record the current
        state of the optimization, including detection measures, the metric, and parameters.

        Args:
            trainable_params_initial: Initial circuit trainable parameters (list of values)
            trainable_params_final: Final circuit trainable parameters (list of values)
            detection_dict: Dictionary mapping configuration names to detection measures
            metric: Optimization metric value
            validation: Optimization validation metric value
            optimizer_state: Optimizer state
            grads: Gradients at current step
            state_probabilities: Optional dict containing per configuration state probabilities
        """
        self.epoch += 1
        metric_value = float(metric)
        validation_value = float(validation)

        # Store trainable parameters as plain Python float lists for robust serialization.
        initial_values = (
            np.asarray(trainable_params_initial, dtype=float).reshape(-1).tolist()
            if trainable_params_initial is not None
            else []
        )
        final_values = (
            np.asarray(trainable_params_final, dtype=float).reshape(-1).tolist()
            if trainable_params_final is not None
            else []
        )

        # Package parameters as tuple for internal storage
        trainable_params = (initial_values, final_values)

        # Save history every N epochs
        if self.epoch % self.save_every == 0:
            self.history["epochs"].append(self.epoch)
            self.history["metric"].append(metric_value)
            self.history["validation"].append(validation_value)
            # Save a deep copy of trainable_params to preserve state
            self.history["detection_dict"].append(copy.deepcopy(detection_dict))
            self.history["trainable_params"].append(copy.deepcopy(trainable_params))
            self.history["optimizer_state"].append(copy.deepcopy(optimizer_state))
            self.history["grads"].append(copy.deepcopy(grads))

        # Track best parameters if enabled (maximize the metric)
        if self.save_best and validation_value > self.best_validation:
            self.best_validation = validation_value
            self.best_trainable_params = copy.deepcopy(trainable_params)
            self.best_metrics = {
                "epoch": self.epoch,
                "metric": metric_value,
                "validation": validation_value,
                "detection_dict": copy.deepcopy(detection_dict),
            }

        if state_probabilities is not None:
            self.set_measurement_protocol(state_probabilities)

    def set_measurement_protocol(self, state_probabilities: Optional[Dict[str, Dict[str, float]]]) -> None:
        """
        Set state probabilities for the current optimization step and classify detection states.

        Classifies each measurement state to the configuration that maximises its
        probability, building:
        - detection_states: mapping each configuration name to the list of states it "wins".
        - confusion_matrix: soft confusion matrix mapping (true, predicted) configuration
          pairs to the accumulated probability mass.

        States are represented as binary strings, e.g., for 3 qubits: '000', '001', ..., '111'
        formatted as format(i, f"0{n_qubits}b") for i in range(2**n_qubits).

        This can be used to update the state probabilities at any point during optimization,
        such as after a final evaluation run. Passing ``None`` or an empty mapping clears the
        currently stored protocol.

        Args:
            state_probabilities: Dict mapping each configuration name to a dict of binary
                state strings to the probability of measuring that state for the configuration.
        """

        if not state_probabilities:
            self.state_probabilities = None
            self.detection_states = {}
            self.confusion_matrix = {}
            return

        self.state_probabilities = copy.deepcopy(state_probabilities)
        config_names = list(state_probabilities.keys())
        
        # dictionary of binary strings where interaction is detected
        self.detection_states = {name: [] for name in config_names}

        # dictionary to hold probabilities for each state (True: name1, Predicted: name2)
        self.confusion_matrix = {(name1, name2): 0.0 for name1 in config_names for name2 in config_names}

        for state in state_probabilities[config_names[0]].keys():
            prob_state = {name: probs.get(state, 0.0) for name, probs in state_probabilities.items()}
            max_config = max(prob_state, key=prob_state.get)
            self.detection_states[max_config].append(state)
            # Confusion matrix: add each config's probability for this state to the
            # (config, argmax_config) cell, so entry (A,B) = P(classified as B | config A).
            for name in config_names:
                self.confusion_matrix[(name, max_config)] += prob_state.get(name, 0.0)
            
    def get_best_trainable_params(self) -> Optional[tuple[list, list]]:
        """
        Get the best trainable parameters found during optimization.

        Returns:
            Tuple of (initial_circuit_params, final_circuit_params),
            or None if no parameters recorded yet
        """
        return self.best_trainable_params

    def get_best_metrics(self) -> Optional[Dict[str, Union[int, float, Dict[str, float]]]]:
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

    def get_params(self, epoch: int = -1) -> Tuple[list, list, int]:
        """
        Get the trainable parameters from the specified epoch.

        Args:
            epoch: Index of the epoch to retrieve parameters from. Default is -1 (last epoch).

        Returns:
            Tuple of (initial_circuit_params, final_circuit_params) from the specified epoch.
        Raises:
            ValueError if no history recorded yet or if the epoch index is out of bounds.
        """
        if self.history["trainable_params"] == []:
            raise ValueError("No trainable parameters saved in history.")
        if epoch < 0:
            epoch = self.history["epochs"][-1] + epoch + 1
        if not (1 <= epoch <= self.history["epochs"][-1]):
            raise ValueError("Epoch index is out of bounds.")
        idx: Optional[int] = None
        selected_epoch: Optional[int] = None
        for idx, ep in enumerate(self.history["epochs"]):
            diff = abs(ep - epoch)
            if 0 <= diff < self.save_every:
                if diff!=0:
                    print(f"Epoch {epoch} requested but not present, returning parameters saved from epoch {ep} (diff={diff})")
                selected_epoch = ep
                break 
        if idx is None or selected_epoch is None:
            raise ValueError("No parameters found for the requested epoch.")
            
        initial_params, final_params = self.history["trainable_params"][idx]
        return copy.deepcopy(initial_params), copy.deepcopy(final_params), selected_epoch

    def get_opt_state(self, epoch: int = -1) -> Tuple[Any, Any]:
        """
        Get optimizer state and gradients from the specified epoch.

        Args:
            epoch: Index of the epoch to retrieve optimizer state from. Default is -1 (last epoch).

        Returns:
            Tuple of (optimizer_state, grads) from the specified epoch.
        Raises:
            ValueError if no history recorded yet or if the epoch index is out of bounds.
        """
        if self.history["optimizer_state"] == []:
            raise ValueError("No optimizer state saved in history.")
        if epoch < 0:
            epoch = self.history["epochs"][-1] + epoch + 1
        if not (1 <= epoch <= self.history["epochs"][-1]):
            raise ValueError("Epoch index is out of bounds.")
        idx: Optional[int] = None
        for idx, ep in enumerate(self.history["epochs"]):
            diff = abs(ep - epoch)
            if 0 <= diff < self.save_every:
                if diff != 0:
                    print(f"Epoch {epoch} requested but not present, returning optimizer state saved from epoch {ep} (diff={diff})")
                break
        if idx is None:
            raise ValueError("No optimizer state found for the requested epoch.")

        optimizer_state = copy.deepcopy(self.history["optimizer_state"][idx])
        grads_history = self.history.get("grads", [])
        grads = copy.deepcopy(grads_history[idx]) if len(grads_history) > idx else None
        return optimizer_state, grads

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
        It also includes serialized optimizer state and trainable-parameter histories
        to support hot-start continuation across Python sessions.

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

        initial_history = [list(params[0]) for params in self.history["trainable_params"]]
        final_history = [list(params[1]) for params in self.history["trainable_params"]]
        final_grad_norm_value = np.nan if self.final_grad_norm is None else float(self.final_grad_norm)

        # Prepare data for saving
        save_dict = {
            "epochs": np.array(self.history["epochs"]),
            "metric": np.array(self.history["metric"]),
            "validation": np.array(self.history["validation"]),
            "detection_dict": np.array(self.history["detection_dict"], dtype=object),
            "trainable_params_initial_history": self._to_object_array(initial_history),
            "trainable_params_final_history": self._to_object_array(final_history),
            "optimizer_state_history": self._to_object_array(self.history["optimizer_state"]),
            "grads_history": self._to_object_array(self.history["grads"]),
            "save_every": np.array(self.save_every),
            "save_best": np.array(self.save_best),
            "epoch": np.array(self.epoch),
            "converged": np.array(self.converged),
            "final_grad_norm": np.array(final_grad_norm_value),
        }

        # Save optional per-run state probabilities if available
        if self.state_probabilities is not None:
            save_dict["state_probabilities"] = np.array(self.state_probabilities, dtype=object)

        # Save detection states and metrics
        save_dict["detection_states"] = np.array(self.detection_states, dtype=object)
        save_dict["confusion_matrix"] = np.array(self.confusion_matrix, dtype=object)

        # Add best parameters if available
        if self.best_trainable_params is not None:
            initial_best, final_best = self.best_trainable_params
            save_dict["best_initial_params"] = np.array(initial_best, dtype=float) 
            save_dict["best_final_params"] = np.array(final_best, dtype=float)
            save_dict["best_validation"] = np.array(self.best_validation)

            if self.best_metrics is not None:
                save_dict["best_epoch"] = np.array(self.best_metrics["epoch"])
                save_dict["best_metric"] = np.array(self.best_metrics["metric"])
                save_dict["best_detection"] = np.array(self.best_metrics["detection_dict"], dtype=object)

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
        data = np.load(filepath, allow_pickle=True)
        return {key: data[key] for key in data.files}

    @classmethod
    def load_callback(cls, filepath: str) -> "OptimizationCallback":
        """
        Reconstruct a callback object from a saved NPZ file.

        Args:
            filepath: Path to the saved NPZ file.

        Returns:
            OptimizationCallback populated with history, best metrics, and optimizer state.
        """
        data = cls.load(filepath)

        save_every = int(np.asarray(data.get("save_every", np.array(1))).item())
        save_best = bool(np.asarray(data.get("save_best", np.array(True))).item())
        callback = cls(save_every=save_every, save_best=save_best)

        callback.history["epochs"] = np.asarray(data.get("epochs", np.array([])), dtype=int).tolist()
        callback.history["metric"] = np.asarray(data.get("metric", np.array([])), dtype=float).tolist()
        callback.history["validation"] = np.asarray(data.get("validation", np.array([])), dtype=float).tolist()
        callback.history["detection_dict"] = np.asarray(
            data.get("detection_dict", np.array([])), dtype=object
        ).tolist()

        if (
            "trainable_params_initial_history" in data
            and "trainable_params_final_history" in data
        ):
            initial_history = data["trainable_params_initial_history"].tolist()
            final_history = data["trainable_params_final_history"].tolist()
            callback.history["trainable_params"] = [
                (
                    np.asarray(initial_params, dtype=float).reshape(-1).tolist(),
                    np.asarray(final_params, dtype=float).reshape(-1).tolist(),
                )
                for initial_params, final_params in zip(initial_history, final_history)
            ]
        else:
            callback.history["trainable_params"] = []

        if "optimizer_state_history" in data:
            callback.history["optimizer_state"] = data["optimizer_state_history"].tolist()
        else:
            callback.history["optimizer_state"] = [None] * len(callback.history["epochs"])

        if "grads_history" in data:
            callback.history["grads"] = data["grads_history"].tolist()
        else:
            callback.history["grads"] = [None] * len(callback.history["epochs"])

        callback.epoch = int(
            np.asarray(data.get("epoch", np.array(len(callback.history["epochs"])))).item()
        )
        callback.converged = bool(np.asarray(data.get("converged", np.array(False))).item())

        final_grad_norm_value = float(np.asarray(data.get("final_grad_norm", np.array(np.nan))).item())
        callback.final_grad_norm = None if np.isnan(final_grad_norm_value) else final_grad_norm_value

        if "best_validation" in data:
            callback.best_validation = float(np.asarray(data["best_validation"]).item())

        if "best_initial_params" in data and "best_final_params" in data:
            callback.best_trainable_params = (
                np.asarray(data["best_initial_params"], dtype=float).reshape(-1).tolist(),
                np.asarray(data["best_final_params"], dtype=float).reshape(-1).tolist(),
            )

        if (
            "best_epoch" in data
            and "best_metric" in data
            and "best_validation" in data
            and "best_detection" in data
        ):
            callback.best_metrics = {
                "epoch": int(np.asarray(data["best_epoch"]).item()),
                "metric": float(np.asarray(data["best_metric"]).item()),
                "validation": float(np.asarray(data["best_validation"]).item()),
                "detection_dict": data["best_detection"].item(),
            }

        # Restore optional per-run state probabilities if present
        if "state_probabilities" in data:
            callback.state_probabilities = data["state_probabilities"].item()

        # Restore detection states and metrics
        if "detection_states" in data:
            callback.detection_states = data["detection_states"].item()
        if "confusion_matrix" in data:
            callback.confusion_matrix = data["confusion_matrix"].item()

        return callback

    def reset(self) -> None:
        """
        Reset callback to initial state.

        Clears all history and best parameter tracking.
        """
        # Clear existing history buffers in-place so any external references obtained
        # through get_history() are also emptied and no stale large objects are kept.
        if isinstance(self.history, dict):
            for value in self.history.values():
                if isinstance(value, list):
                    value.clear()
            self.history.clear()
            self.history.update(self._empty_history_template())
        else:
            self.history = self._empty_history_template()

        # Release possible list references contained in best params before dropping.
        if self.best_trainable_params is not None:
            try:
                for params in self.best_trainable_params:
                    if isinstance(params, list):
                        params.clear()
            except TypeError:
                pass
        if isinstance(self.best_metrics, dict):
            self.best_metrics.clear()

        self.epoch = 0
        self.best_trainable_params = None
        self.best_validation = -float("inf")
        self.best_metrics = None
        self.converged = False
        self.final_grad_norm = None
        self.state_probabilities = None
        self.detection_states = {}
        self.confusion_matrix = {}

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
                # Route through numpy then extract a native scalar to handle
                # plain floats, numpy values and JAX arrays uniformly.
                val_float = np.asarray(value, dtype=float).item()
                lines.append(f"        param_{i}: {val_float:.6f} rad ({np.rad2deg(val_float):.2f}°)")

            # Show final circuit parameters
            lines.append("     Final circuit:")
            for i, value in enumerate(final_params):
                val_float = np.asarray(value, dtype=float).item()
                lines.append(f"        param_{i}: {val_float:.6f} rad ({np.rad2deg(val_float):.2f}°)")

        # Show metrics (best for optimization, current for simulation)
        if self.best_metrics is not None:
            detection_dict = self.best_metrics.get("detection_dict") or {}
            if detection_dict:
                lines.append("  Detection Measures:")
                label_width = max(len(str(name)) for name in detection_dict)
                for name, value in detection_dict.items():
                    lines.append(f"     {str(name):<{label_width}} : {float(value):.6f}")
            lines.append("  Objective:")
            lines.append(f"     Metric:     {self.best_metrics['metric']:.6f}")
            if self.best_metrics.get("validation") is not None:
                lines.append(f"     Validation: {self.best_metrics['validation']:.6f}")

        # Show per-configuration measurement-state probabilities
        if self.state_probabilities:
            lines.append("  State Probabilities:")
            for config_name, probs in self.state_probabilities.items():
                lines.append(f"     {config_name}:")
                for state, prob in probs.items():
                    lines.append(f"        {state}: {float(prob):.6f}")

        # Show detection protocol: which states are classified as which configuration
        if self.detection_states:
            lines.append("  Detection Protocol (states classified per configuration):")
            for config_name, states in self.detection_states.items():
                lines.append(f"     {config_name}: {list(states)}")

        # Show confusion matrix (rows: true configuration, cols: predicted configuration)
        if self.confusion_matrix:
            # Preserve first-seen ordering of configuration names
            config_names: List[str] = []
            for true_name, pred_name in self.confusion_matrix:
                for name in (true_name, pred_name):
                    if name not in config_names:
                        config_names.append(name)

            col_width = max(10, max(len(str(name)) for name in config_names) + 2)
            lines.append("  Confusion Matrix (rows: true, cols: predicted):")
            header = " " * col_width + "".join(f"{str(name):>{col_width}}" for name in config_names)
            lines.append("     " + header)
            for true_name in config_names:
                row = f"{str(true_name):>{col_width}}"
                for pred_name in config_names:
                    val = float(self.confusion_matrix.get((true_name, pred_name), 0.0))
                    row += f"{val:>{col_width}.4f}"
                lines.append("     " + row)

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation (calls __repr__)."""
        return self.__repr__()

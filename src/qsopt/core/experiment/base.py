"""
Abstract Base Class for Quantum Sensing Experiments
===================================================

Defines the interface that all quantum sensing experiment classes must implement.
This provides a common structure for single-qubit, two-qubit, and future multi-qubit
experiments.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import qutip as qt

from qsopt.core.callback import OptimizationCallback
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import TrainableParameters


class Experiment(ABC):
    """
    Abstract base class for quantum sensing experiments.

    This class defines the interface that all experiment implementations must follow.
    Subclasses should implement the abstract methods to provide specific functionality
    for different qubit configurations (single qubit, two qubits, etc.).

    Attributes:
        experimental_params: Physical and measurement parameters for the experiment
        trainable_params: Parameters that can be optimized (rotation angles, etc.)
        operators: Dictionary of quantum operators for the system
        hamiltonians: Dictionary of Hamiltonians (time-dependent and static)
        lindblad_operators: Dictionary of Lindblad operators for noise modeling
        callback: Optimization callback for tracking progress
    """

    def __init__(
        self, experimental_params: ExperimentalParameters, trainable_params: TrainableParameters
    ):
        """
        Initialize the experiment with physical and trainable parameters.

        Args:
            experimental_params: Physical constants, system dimensions, and measurement protocol
            trainable_params: Rotation angles and other parameters to be optimized
        """
        self.experimental_params = experimental_params
        self.trainable_params = trainable_params

        # Storage for operators and Hamiltonians
        self.operators: Optional[Dict[str, qt.Qobj]] = None
        self.hamiltonians: Optional[Dict[str, Union[qt.QobjEvo, qt.Qobj]]] = None
        self.lindblad_operators: Optional[Dict[str, List[Union[qt.Qobj, qt.QobjEvo]]]] = None

        # Optimization callback
        self.callback: OptimizationCallback = OptimizationCallback(save_every=1, save_best=True)

    @abstractmethod
    def _generate_operators(self) -> None:
        """
        Generate the quantum operators for the experiment.

        This method should create all necessary operators (creation, annihilation,
        Pauli operators, etc.) in the appropriate composite Hilbert space.
        Results should be stored in self.operators.
        """
        pass

    @abstractmethod
    def _generate_hamiltonian(self) -> None:
        """
        Generate the Hamiltonian and Lindblad operators for the experiment.

        This method should construct:
        1. Time-dependent Hamiltonian
        2. Static interaction Hamiltonians
        3. Lindblad operators for noise processes

        Results should be stored in self.hamiltonians and self.lindblad_operators.
        """
        pass

    @abstractmethod
    def _initialize_caches(self) -> None:
        """
        Initialize cached objects for performance optimization.

        This method should cache frequently-used objects that don't change during
        optimization (initial states, projectors, solvers, etc.) to improve performance.
        """
        pass

    @abstractmethod
    def get_initial_state(self) -> qt.Qobj:
        """
        Get the initial quantum state for the experiment.

        Returns:
            Initial state as a QuTiP Qobj (density matrix)
        """
        pass

    @abstractmethod
    def simulation(self, *args, **kwargs):
        """
        Run a single simulation with given parameters.

        This method should perform the core quantum simulation:
        1. Apply initial rotation
        2. Evolve under the Hamiltonian
        3. Apply final rotation
        4. Perform measurement

        Returns:
            Measurement outcome (probability or other observable)
        """
        pass

    @abstractmethod
    def run_simulation(self, batch_size: int = 1) -> OptimizationCallback:
        """
        Run the quantum sensing protocol with current parameters.

        Args:
            batch_size: Number of independent runs for averaging over measurement uncertainty

        Returns:
            OptimizationCallback containing simulation results
        """
        pass

    @abstractmethod
    def optimize_rotations(
        self,
        num_steps: int = 100,
        batch_size: int = 1,
        tolerance: float = 1e-6,
        verbose: bool = True,
        verbose_step: int = 10,
        callback: Optional[OptimizationCallback] = None,
        **kwargs,
    ) -> OptimizationCallback:
        """
        Optimize rotation parameters to maximize sensing contrast.

        Args:
            num_steps: Maximum number of optimization iterations
            batch_size: Number of realizations for measurement uncertainty averaging
            tolerance: Convergence threshold for gradient norm
            verbose: Print progress information
            verbose_step: Frequency of progress printing
            callback: Optional callback for tracking optimization
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationCallback with optimization history
        """
        pass

    def optimize_measurement_times(self, *args, **kwargs) -> Dict:
        """
        Optimize measurement timing protocol.

        Default implementation raises NotImplementedError.
        Subclasses can override to provide measurement time optimization.

        Returns:
            Dictionary with optimization results
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement measurement time optimization"
        )

    @property
    def rotation_angles(self) -> Dict[str, float]:
        """Get current rotation angle values."""
        angles = self.trainable_params.get_rotation_angles()
        return {name: float(val[0]) for name, val in angles.items()}

    @rotation_angles.setter
    def rotation_angles(self, angles: Dict[str, float]) -> None:
        """
        Set rotation angle values.

        Args:
            angles: Dictionary mapping parameter names to angle values in radians
        """
        self.trainable_params.set_rotation_angles(angles)

    def save_experiment_report(self, save_path: str = "results/report.json") -> None:
        """
        Save a comprehensive experiment report.

        Default implementation raises NotImplementedError.
        Subclasses should override to provide experiment-specific reporting.

        Args:
            save_path: Path where the report will be saved
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement experiment report saving"
        )

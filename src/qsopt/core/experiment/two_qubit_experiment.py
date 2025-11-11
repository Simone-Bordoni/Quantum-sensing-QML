"""
Two Qubit Quantum Sensing Experiment
====================================

Placeholder for two-qubit quantum sensing experiments.
This class will be implemented with specific two-qubit protocols.
"""

import warnings
from typing import Dict, List, Optional, Union

import qutip as qt
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.core.callback import OptimizationCallback
from .base import Experiment
from .quantum_utils import (
    generate_two_qubit_operators,
    generate_initial_state,
)

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import


class TwoQubitExperiment(Experiment):
    """
    Two-qubit quantum sensing experiment (placeholder).
    
    This class extends the base Experiment interface for two-qubit sensing protocols.
    Implementation is in progress and will include:
    - Two-qubit gate operations
    - Entanglement-based sensing
    - Multi-qubit measurement protocols
    
    Note: This is a placeholder class. Methods raise NotImplementedError until
    the full implementation is completed.
    """
    
    def __init__(
        self,
        experimental_params: ExperimentalParameters,
        trainable_params: TrainableParameters
    ):
        """
        Initialize two-qubit experiment.
        
        Args:
            experimental_params: Physical and measurement parameters
            trainable_params: Rotation angles and other optimizable parameters
        """
        super().__init__(experimental_params, trainable_params)
        
        # Two-qubit specific caches
        self._cached_initial_state: Optional[qt.Qobj] = None
        self._cached_two_qubit_gates: Dict[str, qt.Qobj] = {}
    
    def _generate_operators(self) -> None:
        """
        Generate operators for two-qubit system.
        
        Will create operators in composite Hilbert space:
        input_field ⊗ resonator_cavity ⊗ qubit1 ⊗ qubit2
        
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        # Placeholder: Use utility function when implemented
        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels
        qubit_levels = self.experimental_params.qubit_levels
        
        # Generate operators (this function exists but needs extension for 2-qubit)
        self.operators = generate_two_qubit_operators(
            field_levels,
            cavity_levels,
            qubit_levels
        )
    
    def _generate_hamiltonian(self) -> None:
        """
        Generate Hamiltonian for two-qubit system.
        
        Will create:
        - Two-qubit coupling terms
        - Individual qubit-cavity interactions
        - Collective noise operators
        
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "TwoQubitExperiment._generate_hamiltonian is not yet implemented. "
            "This will include two-qubit coupling and collective effects."
        )
    
    def _initialize_caches(self) -> None:
        """
        Initialize cached objects for two-qubit system.
        
        Will cache:
        - Two-qubit gates (CNOT, CZ, etc.)
        - Initial entangled states
        - Measurement projectors for both qubits
        
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "TwoQubitExperiment._initialize_caches is not yet implemented. "
            "This will cache two-qubit gates and projectors."
        )
    
    def get_initial_state(self) -> qt.Qobj:
        """
        Get initial two-qubit state.
        
        Returns:
            Initial two-qubit state (can be entangled)
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "TwoQubitExperiment.get_initial_state is not yet implemented. "
            "This will support entangled initial states."
        )
    
    def simulation(self, *args, **kwargs):
        """
        Run two-qubit quantum simulation.
        
        Will implement:
        1. Prepare initial (potentially entangled) state
        2. Apply two-qubit gates
        3. Evolve under two-qubit Hamiltonian
        4. Perform joint or individual measurements
        
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "TwoQubitExperiment.simulation is not yet implemented. "
            "This will include two-qubit gates and measurements."
        )
    
    def run_simulation(self, batch_size: int = 1) -> OptimizationCallback:
        """
        Run two-qubit sensing protocol.
        
        Args:
            batch_size: Number of independent runs
            
        Returns:
            OptimizationCallback with results
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "TwoQubitExperiment.run_simulation is not yet implemented. "
            "This will run the complete two-qubit protocol."
        )
    
    def optimize_rotations(
        self,
        num_steps: int = 100,
        batch_size: int = 1,
        tolerance: float = 1e-6,
        verbose: bool = True,
        verbose_step: int = 10,
        callback: Optional[OptimizationCallback] = None,
        **kwargs
    ) -> OptimizationCallback:
        """
        Optimize rotation parameters for two-qubit system.
        
        Will optimize:
        - Individual qubit rotations
        - Two-qubit gate parameters
        - Measurement basis choices
        
        Args:
            num_steps: Maximum optimization iterations
            batch_size: Number of realizations for averaging
            tolerance: Convergence threshold
            verbose: Print progress
            verbose_step: Progress printing frequency
            callback: Optional callback for tracking
            **kwargs: Additional parameters
            
        Returns:
            OptimizationCallback with optimization history
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "TwoQubitExperiment.optimize_rotations is not yet implemented. "
            "This will optimize two-qubit rotation and gate parameters."
        )
    
    def apply_two_qubit_gate(
        self,
        rho: qt.Qobj,
        gate_type: str,
        **params
    ) -> qt.Qobj:
        """
        Apply a two-qubit gate to the state.
        
        Args:
            rho: Current density matrix
            gate_type: Type of gate ('CNOT', 'CZ', 'SWAP', etc.)
            **params: Gate parameters (angles, etc.)
            
        Returns:
            State after gate application
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "TwoQubitExperiment.apply_two_qubit_gate is not yet implemented. "
            "This will apply various two-qubit gates."
        )
    
    def measure_both_qubits(self, rho: qt.Qobj) -> Dict[str, float]:
        """
        Perform joint measurement on both qubits.
        
        Args:
            rho: State to measure
            
        Returns:
            Dictionary with joint measurement probabilities
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "TwoQubitExperiment.measure_both_qubits is not yet implemented. "
            "This will perform joint two-qubit measurements."
        )

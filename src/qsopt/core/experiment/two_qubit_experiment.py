"""
Two Qubit Quantum Sensing Experiment
====================================

Two-qubit quantum sensing experiment implementation.
This class handles quantum sensing protocols with two qubits coupled to a shared cavity.
"""

import warnings
from typing import Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import qutip as qt
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.core.callback import OptimizationCallback
from .base import Experiment
from .quantum_utils import (
    gu,
    generate_two_qubit_operators,
    generate_initial_state,
    build_qubit_noise_operators,
)

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import

# Suppress Diffrax complex dtype warning
warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax is a work in progress*")


class TwoQubitExperiment(Experiment):
    """
    Two-qubit quantum sensing experiment.
    
    This class implements quantum sensing protocols with two qubits coupled dispersively
    to a shared resonator cavity. The composite Hilbert space structure is:
    
        input_field ⊗ resonator_cavity ⊗ qubit1 ⊗ qubit2
    
    Each qubit has its own dispersive coupling χᵢ to the cavity, allowing for
    differential sensing and multi-qubit protocols.
    
    System Hamiltonian:
        H = H_cavity-field + H_dispersive
        
    where:
        H_cavity-field = (i/2)√γ (a_in† a - a_in a†) g(t)
        H_dispersive = -Σᵢ (χᵢ/2) a† a σz_i
        
    The qubits are initialized in equal superposition and can be measured
    individually or jointly.
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
        
        # Verify we have 2 qubits configured
        if experimental_params.n_qubits != 2:
            raise ValueError(
                f"TwoQubitExperiment requires n_qubits=2, got {experimental_params.n_qubits}"
            )
        
        # Two-qubit specific caches
        self._cached_initial_state: Optional[qt.Qobj] = None
        self._cached_projectors: Dict[str, qt.Qobj] = {}
        self._cached_solvers: Dict[str, qt.MESolver] = {}
        
        # Initialize quantum objects
        self.__post_init__()
    
    def __post_init__(self):
        """Post-initialization to set up operators and hamiltonian."""
        self._generate_operators()
        self._generate_hamiltonian()
        self._initialize_caches()
    
    def _generate_operators(self) -> None:
        """
        Generate operators for two-qubit system.
        
        Creates operators in composite Hilbert space:
        input_field ⊗ resonator_cavity ⊗ qubit1 ⊗ qubit2
        
        Operators include:
        - Field and cavity creation/annihilation operators
        - Individual qubit Pauli operators (σx, σy, σz) for each qubit
        - Joint measurement projectors |00⟩, |01⟩, |10⟩, |11⟩
        - Individual qubit projectors
        """
        # Get system dimensions
        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels
        qubit_levels_list = self.experimental_params.qubit_levels
        
        # Generate two-qubit operators using utility function
        self.operators = generate_two_qubit_operators(
            field_levels,
            cavity_levels,
            qubit_levels_list
        )
    
    def _generate_hamiltonian(self) -> None:
        """
        Generate Hamiltonian for two-qubit system.
        
        Creates:
        1. Time-dependent cavity-field coupling: H_cavity = (i/2)√γ (a_in† a - a_in a†) g(t)
        2. Dispersive qubit-cavity interactions: H_dispersive = -Σᵢ (χᵢ/2) a† a σz_i
        3. Lindblad operators for noise processes on each qubit
        
        The Hamiltonian follows Fabio's notebook formulation with individual chi values
        for each qubit, allowing for differential dispersive coupling.
        """
        if self.operators is None:
            raise RuntimeError("Operators must be generated before Hamiltonian")
        
        # Extract coupling constants
        gm = self.experimental_params.photon_cavity_coupling
        chi_list = self.experimental_params.chi  # List of [chi1, chi2]
        sigma = self.experimental_params.inverse_pulse_width
        
        # Extract individual chi values for each qubit
        # Type narrowing: chi is always a list for two-qubit experiments
        if isinstance(chi_list, list):
            chi1 = chi_list[0]
            chi2 = chi_list[1]
        else:
            # Should not reach here due to __init__ validation, but type checker needs this
            chi1 = chi2 = chi_list
        
        # Get operators
        a_in = self.operators['a_in']
        a_in_dag = self.operators['a_in_dag']
        a = self.operators['a']
        a_dag = self.operators['a_dag']
        
        # Qubit 1 operators
        sigma_z1 = self.operators['sigma_z1']
        sigma_x1 = self.operators['sigma_x1']
        sigma_y1 = self.operators['sigma_y1']
        sigma_minus1 = self.operators['sigma_minus1']
        
        # Qubit 2 operators
        sigma_z2 = self.operators['sigma_z2']
        sigma_x2 = self.operators['sigma_x2']
        sigma_y2 = self.operators['sigma_y2']
        sigma_minus2 = self.operators['sigma_minus2']
        
        # Time-dependent coupling function arguments
        args = {'sigma': sigma}
        
        # Time-dependent cavity-field coupling Hamiltonian
        # H_c = (i/2)√γ (a_in† a - a_in a†)
        coupling_coeff = 1j/2 * jnp.sqrt(gm)
        H_coupling = qt.Qobj(coupling_coeff * (a_in_dag * a - a_in * a_dag))  # type: ignore
        
        # Dispersive qubit-resonator interaction Hamiltonians
        # H_q = -Σᵢ (χᵢ/2) a† a σz_i
        H_dispersive1 = qt.Qobj(-chi1/2 * a_dag * a * sigma_z1)  # type: ignore
        H_dispersive2 = qt.Qobj(-chi2/2 * a_dag * a * sigma_z2)  # type: ignore
        H_dispersive = H_dispersive1 + H_dispersive2
        
        # Complete time-dependent Hamiltonian
        # H(t) = H_dispersive + H_coupling * g(t)
        H_total = qt.QobjEvo([H_dispersive, [H_coupling, gu]], args=args)
        
        # Noise configuration
        noise_config = self.experimental_params.noise_config
        
        # Extract noise rates for each qubit
        # Type narrowing: noise rates are always lists for two-qubit experiments
        depolarizing = noise_config.depolarizing
        dephasing = noise_config.dephasing
        relaxation = noise_config.relaxation
        
        if isinstance(depolarizing, list) and isinstance(dephasing, list) and isinstance(relaxation, list):
            depolarizing1, depolarizing2 = depolarizing[0], depolarizing[1]
            dephasing1, dephasing2 = dephasing[0], dephasing[1]
            relaxation1, relaxation2 = relaxation[0], relaxation[1]
        else:
            # Should not reach here due to __init__ validation, but type checker needs this
            depolarizing1 = depolarizing2 = depolarizing if isinstance(depolarizing, float) else 0.0
            dephasing1 = dephasing2 = dephasing if isinstance(dephasing, float) else 0.0
            relaxation1 = relaxation2 = relaxation if isinstance(relaxation, float) else 0.0
        
        # Build Lindblad noise operators for qubit 1 using helper function
        lindblad_noise_q1 = build_qubit_noise_operators(
            sigma_x=sigma_x1,
            sigma_y=sigma_y1,
            sigma_z=sigma_z1,
            sigma_minus=sigma_minus1,
            depolarizing_rate=depolarizing1,
            dephasing_rate=dephasing1,
            relaxation_rate=relaxation1
        )
        
        # Build Lindblad noise operators for qubit 2 using helper function
        lindblad_noise_q2 = build_qubit_noise_operators(
            sigma_x=sigma_x2,
            sigma_y=sigma_y2,
            sigma_z=sigma_z2,
            sigma_minus=sigma_minus2,
            depolarizing_rate=depolarizing2,
            dephasing_rate=dephasing2,
            relaxation_rate=relaxation2
        )
        
        # Combine noise operators for both qubits
        lindblad_noise: List[Union[qt.Qobj, qt.QobjEvo]] = lindblad_noise_q1 + lindblad_noise_q2
        
        # Add custom Lindblad operators if provided
        if noise_config.custom_operators is not None:
            lindblad_noise.extend(noise_config.custom_operators)
        
        # Lindblad interaction operator (same for with/without photon)
        L_int = qt.QobjEvo([a_in, gu], args=args) + np.sqrt(gm) * a
        
        interaction_ops: List[Union[qt.Qobj, qt.QobjEvo]] = [L_int] + lindblad_noise
        no_interaction_ops: List[Union[qt.Qobj, qt.QobjEvo]] = lindblad_noise
        
        # Store Hamiltonians and Lindblad operators
        self.hamiltonians = {
            'total': H_total,
            'dispersive': H_dispersive,
            'dispersive1': H_dispersive1,
            'dispersive2': H_dispersive2,
            'coupling': H_coupling
        }
        
        self.lindblad_operators = {
            'interaction': interaction_ops,
            'no_interaction': no_interaction_ops,
        }
    
    def _initialize_caches(self) -> None:
        """
        Initialize cached values for two-qubit experiment.
        
        Caches:
        - Joint measurement projectors (|00⟩, |01⟩, |10⟩, |11⟩)
        - Individual qubit projectors
        - Initial state with qubits in equal superposition
        """
        if self.operators is None:
            raise RuntimeError("Operators must be generated before initializing caches")
        
        # Cache joint measurement projectors
        self._cached_joint_projectors = {
            '00': self.operators['P00'],
            '01': self.operators['P01'],
            '10': self.operators['P10'],
            '11': self.operators['P11']
        }
        
        # Cache individual qubit projectors
        self._cached_qubit1_projectors = {
            '0': self.operators['P0_q1'],
            '1': self.operators['P1_q1']
        }
        
        self._cached_qubit2_projectors = {
            '0': self.operators['P0_q2'],
            '1': self.operators['P1_q2']
        }
        
        # Generate and cache initial state
        self._cached_initial_state = generate_initial_state(
            initial_config=self.experimental_params.initial_state,
            field_levels=self.experimental_params.field_levels,
            cavity_levels=self.experimental_params.cavity_levels,
            qubit_levels=self.experimental_params.qubit_levels,
            num_qubits=2
        )
    
    def get_initial_state(self) -> qt.Qobj:
        """
        Get initial two-qubit state.
        
        Returns the cached initial state with:
        - Single photon in input field
        - Vacuum in cavity
        - Both qubits in equal superposition (|0⟩ + |1⟩)/√2
        
        Returns:
            Initial two-qubit quantum state
        
        Raises:
            RuntimeError: If initial state has not been cached
        """
        if self._cached_initial_state is None:
            raise RuntimeError(
                "Initial state has not been cached. "
                "Ensure _initialize_caches was called."
            )
        return self._cached_initial_state
    
    def get_joint_projector(self, state: str) -> qt.Qobj:
        """
        Get joint measurement projector for both qubits.
        
        Args:
            state: Joint state to project onto ('00', '01', '10', or '11')
        
        Returns:
            Joint measurement projector |state⟩⟨state|
        
        Raises:
            ValueError: If state string is invalid
            RuntimeError: If projectors have not been cached
        """
        if self._cached_joint_projectors is None:
            raise RuntimeError(
                "Joint projectors have not been cached. "
                "Ensure _initialize_caches was called."
            )
        
        if state not in self._cached_joint_projectors:
            raise ValueError(
                f"Invalid joint state '{state}'. "
                f"Must be one of: {list(self._cached_joint_projectors.keys())}"
            )
        
        return self._cached_joint_projectors[state]
    
    def get_qubit_projector(self, qubit: int, state: str) -> qt.Qobj:
        """
        Get measurement projector for a specific qubit.
        
        Args:
            qubit: Qubit index (1 or 2)
            state: Qubit state to project onto ('0' or '1')
        
        Returns:
            Single-qubit measurement projector
        
        Raises:
            ValueError: If qubit index or state is invalid
            RuntimeError: If projectors have not been cached
        """
        if qubit == 1:
            projectors = self._cached_qubit1_projectors
        elif qubit == 2:
            projectors = self._cached_qubit2_projectors
        else:
            raise ValueError(f"Invalid qubit index {qubit}. Must be 1 or 2.")
        
        if projectors is None:
            raise RuntimeError(
                f"Projectors for qubit {qubit} have not been cached. "
                "Ensure _initialize_caches was called."
            )
        
        if state not in projectors:
            raise ValueError(
                f"Invalid qubit state '{state}'. Must be '0' or '1'."
            )
        
        return projectors[state]
    
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

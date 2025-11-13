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
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InteractionType,
    QubitInteraction,
)
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.core.callback import OptimizationCallback
from qsopt.core.loss_functions import DetectionFromProbabilities
from .base import Experiment
from .quantum_utils import (
    gu,
    generate_two_qubit_operators,
    generate_initial_state,
    build_qubit_noise_operators,
    apply_qubit_rotation,
    measure_qubits_probability,
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
        trainable_params: TrainableParameters,
        detector: Optional[DetectionFromProbabilities] = None
    ):
        """
        Initialize two-qubit experiment.
        
        Args:
            experimental_params: Physical and measurement parameters
            trainable_params: Rotation angles and other optimizable parameters
            detector: Custom detection probability calculator. If None, uses default 1-P(00).
        """
        super().__init__(experimental_params, trainable_params)
        
        # Verify we have 2 qubits configured
        if experimental_params.n_qubits != 2:
            raise ValueError(
                f"TwoQubitExperiment requires n_qubits=2, got {experimental_params.n_qubits}"
            )
        
        # Detection probability calculator
        self.detector = detector if detector is not None else DetectionFromProbabilities()
        
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
    
    def _build_qubit_interaction_hamiltonian(self) -> qt.Qobj:
        """
        Build qubit-qubit interaction Hamiltonian from configured interactions.
        
        Constructs interaction terms like:
        - ZZ: (χ/2) σz ⊗ σz
        - XX: (χ/2) σx ⊗ σx
        - YY: (χ/2) σy ⊗ σy
        
        Returns:
            Hamiltonian operator for qubit-qubit interactions (0 if no interactions)
        """
        from qsopt.core.experimental_parameters import InteractionType
        
        if self.operators is None:
            raise RuntimeError("Operators must be generated before building interaction Hamiltonian")
        
        # Get qubit interactions from experimental parameters
        interactions = self.experimental_params.physical_constants.qubit_interactions
        
        if not interactions:
            # No interactions - return zero operator
            dims = self.operators['a'].dims
            return qt.Qobj(np.zeros((np.prod(dims[0]), np.prod(dims[0]))), dims=dims)
        
        # Start with zero Hamiltonian
        dims = self.operators['a'].dims
        H_interaction = qt.Qobj(np.zeros((np.prod(dims[0]), np.prod(dims[0]))), dims=dims)
        
        # Build each interaction term
        for interaction in interactions:
            idx1, idx2 = interaction.qubit_indices
            chi = interaction.chi
            interaction_type = interaction.interaction_type
            
            # Get appropriate Pauli operators based on interaction type
            if interaction_type == InteractionType.ZZ:
                # σz ⊗ σz interaction
                sigma1 = self.operators[f'sigma_z{idx1+1}']
                sigma2 = self.operators[f'sigma_z{idx2+1}']
            elif interaction_type == InteractionType.XX:
                # σx ⊗ σx interaction
                sigma1 = self.operators[f'sigma_x{idx1+1}']
                sigma2 = self.operators[f'sigma_x{idx2+1}']
            elif interaction_type == InteractionType.YY:
                # σy ⊗ σy interaction
                sigma1 = self.operators[f'sigma_y{idx1+1}']
                sigma2 = self.operators[f'sigma_y{idx2+1}']
            else:
                raise ValueError(f"Unknown interaction type: {interaction_type}")
            
            # Add interaction term: (χ/2) σᵢ ⊗ σⱼ
            H_interaction += qt.Qobj((chi / 2) * sigma1 * sigma2)  # type: ignore
        
        return H_interaction
    
    def _generate_hamiltonian(self) -> None:
        """
        Generate Hamiltonian for two-qubit system.
        
        Creates:
        1. Time-dependent cavity-field coupling: H_cavity = (i/2)√γ (a_in† a - a_in a†) g(t)
        2. Dispersive qubit-cavity interactions: H_dispersive = -Σᵢ (χᵢ/2) a† a σz_i
        3. Lindblad operators for noise processes on each qubit
        
        The Hamiltonian uses individual chi values for each qubit, allowing for 
        differential dispersive coupling strengths between qubits and the cavity.
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
        
        # Qubit-qubit interaction Hamiltonians
        # H_interaction = Σⱼ (χⱼ/2) σᵢ ⊗ σⱼ
        # where σᵢ and σⱼ can be σx, σy, or σz depending on interaction type
        H_qubit_interaction = self._build_qubit_interaction_hamiltonian()
        
        # Complete time-dependent Hamiltonian
        # H(t) = H_dispersive + H_qubit_interaction + H_coupling * g(t)
        H_total = qt.QobjEvo([H_dispersive + H_qubit_interaction, [H_coupling, gu]], args=args)
        
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
    
    def get_solver_with_interaction(self) -> qt.MESolver:
        """Get Lindblad master equation solver WITH input photon interaction (cached)."""
        if 'with_interaction' not in self._cached_solvers:
            self._cached_solvers['with_interaction'] = qt.MESolver(
                self.hamiltonians['total'],
                self.lindblad_operators['interaction'],
                options={'method': 'diffrax', 'progress_bar': False}
            )
        return self._cached_solvers['with_interaction']
    
    def get_solver_no_interaction(self) -> qt.MESolver:
        """Get Lindblad master equation solver WITHOUT input photon interaction (cached)."""
        if 'no_interaction' not in self._cached_solvers:
            self._cached_solvers['no_interaction'] = qt.MESolver(
                self.hamiltonians['dispersive'],
                self.lindblad_operators['no_interaction'],
                options={'method': 'diffrax', 'progress_bar': False}
            )
        return self._cached_solvers['no_interaction']
    
    def apply_rotation(self, rho: qt.Qobj, theta: float, qubit: int, axis: str = 'y') -> qt.Qobj:
        """
        Apply rotation to specified qubit.
        
        Unified rotation method for any qubit in the system.
        
        Args:
            rho: Density matrix in composite space
            theta: Rotation angle in radians
            qubit: Qubit index (0 for qubit 1, 1 for qubit 2)
            axis: Rotation axis ('x', 'y', or 'z'), default 'y'
            
        Returns:
            Rotated density matrix
            
        Example:
            >>> # Rotate qubit 1 by π/4 around Y-axis
            >>> rho_rot = experiment.apply_rotation(rho, np.pi/4, qubit=0)
            >>> # Rotate qubit 2 by π/2 around X-axis
            >>> rho_rot = experiment.apply_rotation(rho, np.pi/2, qubit=1, axis='x')
        """
        if self.operators is None:
            raise RuntimeError("Operators must be generated before applying rotations")
        
        return apply_qubit_rotation(rho, theta, qubit, self.operators, axis=axis)
    
    def prob(self, rho: qt.Qobj, qubits: List[int], state: str = '0') -> float:
        """
        Measure probability for specified qubits.
        
        Unified measurement method supporting:
        - Single qubit: qubits=[0] measures first qubit only
        - Single qubit: qubits=[1] measures second qubit only  
        - Both qubits: qubits=[0, 1] measures both jointly
        
        Args:
            rho: Density matrix in composite space
            qubits: List of qubit indices to measure (0 for q1, 1 for q2)
            state: State to measure:
                   - For single qubit: '0' or '1'
                   - For both qubits: '00', '01', '10', '11'
                   
        Returns:
            Measurement probability ∈ [0,1]
            
        Example:
            >>> # Measure qubit 1 in ground state
            >>> p0_q1 = experiment.prob(rho, qubits=[0], state='0')
            >>> # Measure qubit 2 in excited state
            >>> p1_q2 = experiment.prob(rho, qubits=[1], state='1')
            >>> # Measure both qubits in |00⟩
            >>> p00 = experiment.prob(rho, qubits=[0, 1], state='00')
        """
        if self.operators is None:
            raise RuntimeError("Operators must be generated before measuring")
        
        return measure_qubits_probability(rho, qubits, self.operators, state=state)
    
    def compute_final_probabilities(
        self,
        solver: qt.MESolver,
        rho: qt.Qobj,
        theta1_q1: float,
        theta2_q1: float,
        theta1_q2: float,
        theta2_q2: float,
        t_start: float = -5.0,
        t_end: float = 5.0,
        args: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute final state probabilities after evolution without repeated measurements.
        
        This simulates a single evolution and final measurement, returning all
        joint two-qubit outcome probabilities. Useful for parameter sweeps and
        reproducing experiments from the reference notebook.
        
        Workflow:
        1. Apply first rotations Ry(θ₁) to each qubit
        2. Evolve from t_start to t_end
        3. Apply second rotations Ry(θ₂) to each qubit
        4. Measure all joint probabilities P(00), P(01), P(10), P(11)
        
        Args:
            solver: Configured quantum evolution solver
            rho: Initial density matrix in composite space
            theta1_q1: First Y-rotation angle for qubit 1
            theta2_q1: Second Y-rotation angle for qubit 1
            theta1_q2: First Y-rotation angle for qubit 2
            theta2_q2: Second Y-rotation angle for qubit 2
            t_start: Evolution start time (default: -5.0)
            t_end: Evolution end time (default: 5.0)
            args: System parameters (optional, uses experimental_params if None)
            
        Returns:
            Dictionary with probabilities: {'p00': ..., 'p01': ..., 'p10': ..., 'p11': ...}
            
        Example:
            >>> solver = experiment.get_solver_with_interaction()
            >>> rho0 = experiment.get_initial_state()
            >>> probs = experiment.compute_final_probabilities(
            ...     solver, rho0, theta1_q1=0.0, theta2_q1=np.pi/2,
            ...     theta1_q2=0.0, theta2_q2=np.pi/2
            ... )
            >>> print(f"P(11) = {probs['p11']:.4f}")
        """
        if args is None:
            args = {'sigma': self.experimental_params.inverse_pulse_width}
        
        # Initial rotations on both qubits
        rho_rotated = self.apply_rotation(rho, theta1_q1, qubit=0)
        rho_rotated = self.apply_rotation(rho_rotated, theta1_q2, qubit=1)
        
        # Evolve system
        result = solver.run(rho_rotated, tlist=[t_start, t_end], args=args)
        rho_final = result.states[-1]
        
        # Apply final rotations
        rho_final = self.apply_rotation(rho_final, theta2_q1, qubit=0)
        rho_final = self.apply_rotation(rho_final, theta2_q2, qubit=1)
        
        # Measure all joint probabilities
        probs = self.measure_all_states(rho_final)
        
        # Return with consistent naming for detector
        return {
            'p00': float(probs['00']),
            'p01': float(probs['01']),
            'p10': float(probs['10']),
            'p11': float(probs['11'])
        }
    
    def simulation(
        self,
        solver: qt.MESolver,
        rho: qt.Qobj,
        theta1_q1: float,
        theta2_q1: float,
        theta1_q2: float,
        theta2_q2: float,
        measurements: Union[List[float], np.ndarray],
        args: Optional[Dict] = None,
        measure_qubit: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Complete two-qubit quantum photon detection simulation workflow.
        
        Protocol steps:
        1. Apply first rotations Ry(θ₁) to each qubit for initial preparation
        2. Time evolution under cavity-qubit Hamiltonian H(t)
        3. Apply second rotations Ry(θ₂) to each qubit for measurement optimization  
        4. Sequential projective measurements with conditional state updates
        5. Calculate cumulative detection probability
        
        Args:
            solver: Configured quantum evolution solver
            rho: Initial density matrix in composite space
            theta1_q1: First Y-rotation angle for qubit 1
            theta2_q1: Second Y-rotation angle for qubit 1
            theta1_q2: First Y-rotation angle for qubit 2
            theta2_q2: Second Y-rotation angle for qubit 2
            measurements: List or array of measurement times (sorted)
            args: System parameters (optional, uses experimental_params if None)
            measure_qubit: Which qubit to measure (None=both, 1=qubit1, 2=qubit2)
                
        Returns:
            Probability of detecting at least one excited state P(detection) ∈ [0,1]
        """
        if args is None:
            args = {'sigma': self.experimental_params.inverse_pulse_width}
        
        measurement_array = np.asarray(measurements, dtype=float)
        if measurement_array.ndim != 1 or measurement_array.size < 2:
            raise ValueError("measurements must be a 1D array with at least 2 time points")
        
        # Initial rotations on both qubits (qubit indices: 0=q1, 1=q2)
        rho_current = self.apply_rotation(rho, theta1_q1, qubit=0)
        rho_current = self.apply_rotation(rho_current, theta1_q2, qubit=1)
        
        # Track probability of remaining in ground state across all measurements
        prob_all_ground = jnp.array(1.0)
        
        # Loop over measurement intervals
        for t0, t1 in zip(measurement_array[:-1], measurement_array[1:]):
            # Evolve system from t0 to t1
            result = solver.run(rho_current, tlist=[t0, t1], args=args)
            rho_evolved = result.states[-1]
            
            # Apply final rotations (inverse preparation)
            rho_rotated = self.apply_rotation(rho_evolved, theta2_q1, qubit=0)
            rho_rotated = self.apply_rotation(rho_rotated, theta2_q2, qubit=1)
            
            # Measure qubits using unified prob() method
            if measure_qubit is None:
                # Measure both qubits - probability of ground state |00⟩
                prob_ground = self.prob(rho_rotated, qubits=[0, 1], state='00')
            elif measure_qubit == 1:
                # Measure only qubit 1 (index 0)
                prob_ground = self.prob(rho_rotated, qubits=[0], state='0')
            elif measure_qubit == 2:
                # Measure only qubit 2 (index 1)
                prob_ground = self.prob(rho_rotated, qubits=[1], state='0')
            else:
                raise ValueError(f"measure_qubit must be None, 1, or 2, got {measure_qubit}")
            
            # Project onto ground state and renormalize
            if measure_qubit is None:
                P_ground = self.get_joint_projector('00')
            elif measure_qubit == 1:
                P_ground = self.get_qubit_projector(1, '0')
            else:
                P_ground = self.get_qubit_projector(2, '0')
            
            rho_projected = P_ground * rho_rotated * P_ground.dag()  # type: ignore
            
            # Renormalize (avoid division by zero)
            if prob_ground > 1e-15:
                rho_current = rho_projected / prob_ground
            else:
                # If prob_ground is too small, detection has occurred
                rho_current = rho_projected
            
            # Update cumulative ground state probability
            prob_all_ground = prob_all_ground * prob_ground
            
            # Undo rotations for next evolution interval
            rho_current = self.apply_rotation(rho_current, -theta2_q2, qubit=1)
            rho_current = self.apply_rotation(rho_current, -theta2_q1, qubit=0)
        
        # Detection probability = 1 - P(all measurements in ground state)
        prob_detection = 1 - prob_all_ground
        return prob_detection
    
    def run_simulation(
        self,
        batch_size: int = 1,
        measure_qubit: Optional[int] = None
    ) -> OptimizationCallback:
        """
        Run two-qubit sensing protocol with current parameters.
        
        This method executes the complete two-qubit quantum sensing workflow:
        - Applies rotations to both qubits independently
        - Evolves under two-qubit Hamiltonian
        - Performs measurements (joint or individual)
        - Computes detection probabilities with and without photon interaction
        
        Args:
            batch_size: Number of random realizations to average over for measurement
                       uncertainty (default: 1). Each realization uses a different
                       random shift in measurement times based on initial_time_uncertainty.
            measure_qubit: Which qubit to measure (None=both jointly, 1=qubit1 only, 2=qubit2 only)
        
        Returns:
            OptimizationCallback: Callback containing simulation results with:
                - Single epoch (epoch=1)
                - Current parameter values
                - Detection probabilities (prob_with, prob_without) averaged over batch
                - Sensing contrast averaged over batch
        
        Raises:
            ValueError: If fewer than 4 rotation parameters are defined (2 per qubit)
        """
        # Get rotation parameters - need 4 angles total (2 per qubit)
        rotation_angles = self.trainable_params.get_rotation_angles()
        
        if len(rotation_angles) < 4:
            raise ValueError(
                f"Two-qubit experiment requires at least 4 rotation parameters (2 per qubit), "
                f"got {len(rotation_angles)}"
            )
        
        # Extract rotation angles - order matters!
        # Assuming order: theta1_q1, theta2_q1, theta1_q2, theta2_q2
        param_names = list(rotation_angles.keys())
        theta1_q1 = float(rotation_angles[param_names[0]][0])
        theta2_q1 = float(rotation_angles[param_names[1]][0])
        theta1_q2 = float(rotation_angles[param_names[2]][0])
        theta2_q2 = float(rotation_angles[param_names[3]][0])
        
        # Get initial state and solvers
        rho0 = self.get_initial_state()
        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()
        
        # Prepare measurement time realizations for batch averaging
        measurement_times_batch = self.experimental_params.get_measurement_times_with_uncertainty(batch_size)
        if measurement_times_batch.ndim == 1:
            measurement_sequences = [measurement_times_batch]
        else:
            measurement_sequences = [measurement_times_batch[i, :] for i in range(batch_size)]
        
        # Run simulations with batch averaging over uncertainty realizations
        prob_with_list = []
        prob_without_list = []
        
        for measurement_times in measurement_sequences:
            # Simulation with photon interaction
            prob_with = self.simulation(
                solver=solver_with,
                rho=rho0,
                theta1_q1=theta1_q1,
                theta2_q1=theta2_q1,
                theta1_q2=theta1_q2,
                theta2_q2=theta2_q2,
                measurements=measurement_times,
                measure_qubit=measure_qubit
            )
            prob_with_list.append(prob_with)
            
            # Simulation without photon interaction (reference)
            prob_without = self.simulation(
                solver=solver_without,
                rho=rho0,
                theta1_q1=theta1_q1,
                theta2_q1=theta2_q1,
                theta1_q2=theta1_q2,
                theta2_q2=theta2_q2,
                measurements=measurement_times,
                measure_qubit=measure_qubit
            )
            prob_without_list.append(prob_without)
        
        # Average over batch
        prob_with = jnp.mean(jnp.array(prob_with_list))
        prob_without = jnp.mean(jnp.array(prob_without_list))
        contrast = jnp.abs(prob_with - prob_without)
        
        # Create callback with single epoch for simulation results
        callback = OptimizationCallback(save_every=1, save_best=True)
        callback(
            trainable_params=self.trainable_params,
            prob_with=float(prob_with),
            prob_without=float(prob_without),
            contrast=float(contrast)
        )
        
        return callback
    
    def run_simulation_with_probabilities(
        self,
        t_start: float = -5.0,
        t_end: float = 5.0
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Run simulation and return all final state probabilities and detection metrics.
        
        This method computes final state probabilities after evolution, then uses
        the configured detector to compute detection probabilities and contrast.
        Useful for parameter sweeps and reproducing notebook experiments.
        
        Args:
            t_start: Evolution start time (default: -5.0)
            t_end: Evolution end time (default: 5.0)
            
        Returns:
            Dictionary containing:
                - 'probs_with': Dict with p00, p01, p10, p11 (with photon)
                - 'probs_without': Dict with p00, p01, p10, p11 (without photon)
                - 'detection_with': Detection probability with photon
                - 'detection_without': Detection probability without photon
                - 'contrast': Sensing contrast
                
        Example:
            >>> experiment = TwoQubitExperiment(exp_params, train_params)
            >>> results = experiment.run_simulation_with_probabilities()
            >>> print(f"P(11) with photon: {results['probs_with']['p11']:.4f}")
            >>> print(f"Contrast: {results['contrast']:.4f}")
        """
        # Get rotation parameters
        rotation_angles = self.trainable_params.get_rotation_angles()
        
        if len(rotation_angles) < 4:
            raise ValueError(
                f"Two-qubit experiment requires at least 4 rotation parameters (2 per qubit), "
                f"got {len(rotation_angles)}"
            )
        
        # Extract rotation angles
        param_names = list(rotation_angles.keys())
        theta1_q1 = float(rotation_angles[param_names[0]][0])
        theta2_q1 = float(rotation_angles[param_names[1]][0])
        theta1_q2 = float(rotation_angles[param_names[2]][0])
        theta2_q2 = float(rotation_angles[param_names[3]][0])
        
        # Get initial state and solvers
        rho0 = self.get_initial_state()
        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()
        
        # Compute final probabilities with photon
        probs_with = self.compute_final_probabilities(
            solver_with, rho0,
            theta1_q1, theta2_q1, theta1_q2, theta2_q2,
            t_start, t_end
        )
        
        # Compute final probabilities without photon
        probs_without = self.compute_final_probabilities(
            solver_without, rho0,
            theta1_q1, theta2_q1, theta1_q2, theta2_q2,
            t_start, t_end
        )
        
        # Use detector to compute detection probabilities
        detection_with = float(self.detector(probs_with))
        detection_without = float(self.detector(probs_without))
        
        # Compute contrast using detector's method
        contrast = float(DetectionFromProbabilities.compute_contrast(
            detection_with, detection_without
        ))
        
        return {
            'probs_with': probs_with,
            'probs_without': probs_without,
            'detection_with': detection_with,
            'detection_without': detection_without,
            'contrast': contrast
        }
    
    def time_evolution(
        self,
        t_start: float = -5.0,
        t_end: float = 5.0,
        n_points: int = 200,
        with_interaction: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute time evolution of two-qubit probabilities.
        
        Simulates the quantum system evolution over time and returns probability
        distributions for all two-qubit states (|00⟩, |01⟩, |10⟩, |11⟩).
        The system starts in superposition (after first rotations), evolves under 
        the Hamiltonian, and probabilities are measured after the second rotations.
            
        Args:
            t_start: Start time for evolution (default: -5.0)
            t_end: End time for evolution (default: 5.0)
            n_points: Number of time points to sample (default: 200)
            with_interaction: If True, use Hamiltonian with chi coupling.
                             If False, use Hamiltonian without chi (default: True)
        
        Returns:
            Dictionary containing:
                - 'times': Array of time points, shape (n_points,)
                - 'prob_00': Probability of |00⟩ state, shape (n_points,)
                - 'prob_01': Probability of |01⟩ state, shape (n_points,)
                - 'prob_10': Probability of |10⟩ state, shape (n_points,)
                - 'prob_11': Probability of |11⟩ state, shape (n_points,)
                - 'pulse_shape': Pulse envelope u(t), shape (n_points,)
        
        Example:
            >>> # Get time evolution data
            >>> evolution = experiment.time_evolution(t_start=-5, t_end=5, n_points=200)
            >>> 
            >>> # Plot with matplotlib
            >>> import matplotlib.pyplot as plt
            >>> labels = ['P₀₀', 'P₀₁', 'P₁₀', 'P₁₁']
            >>> linestyles = ['-', '--', '-.', ':']
            >>> for k, state in enumerate(['00', '01', '10', '11']):
            ...     plt.plot(evolution['times'], evolution[f'prob_{state}'], 
            ...              label=labels[k], linestyle=linestyles[k])
            >>> plt.fill_between(evolution['times'], 0, evolution['pulse_shape'], alpha=0.2)
            >>> plt.legend()
            >>> plt.show()
            >>> 
            >>> # Or use the visualization utility
            >>> from qsopt.utils import plot_time_evolution
            >>> fig = plot_time_evolution(evolution, n_qubits=2)
        """
        # Get current rotation angles (need 4 angles for 2 qubits)
        rotation_angles = self.trainable_params.get_rotation_angles()
        if len(rotation_angles) < 4:
            raise ValueError("Need at least 4 rotation angle parameters (2 per qubit)")
        
        param_names = list(rotation_angles.keys())
        theta1_q1 = float(rotation_angles[param_names[0]][0])
        theta2_q1 = float(rotation_angles[param_names[1]][0])
        theta1_q2 = float(rotation_angles[param_names[2]][0])
        theta2_q2 = float(rotation_angles[param_names[3]][0])
        
        # Get initial state and solver
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")
        
        solver = self.get_solver_with_interaction() if with_interaction else self.get_solver_no_interaction()
        
        # Apply first rotations
        rho_rotated = self.apply_rotation(rho0, theta1_q1, qubit=0)
        rho_rotated = self.apply_rotation(rho_rotated, theta1_q2, qubit=1)
        
        # Time evolution
        times = np.linspace(t_start, t_end, n_points)
        args = {'sigma': self.experimental_params.inverse_pulse_width}
        result = solver.run(rho_rotated, tlist=times, args=args)
        
        # Extract probabilities at each time point
        prob_00_list = []
        prob_01_list = []
        prob_10_list = []
        prob_11_list = []
        
        for rho_t in result.states:
            # Apply second rotations
            rho_final = self.apply_rotation(rho_t, theta2_q1, qubit=0)
            rho_final = self.apply_rotation(rho_final, theta2_q2, qubit=1)
            
            # Measure two-qubit probabilities
            p00 = float(self.prob(rho_final, qubits=[0, 1], state='00'))
            p01 = float(self.prob(rho_final, qubits=[0, 1], state='01'))
            p10 = float(self.prob(rho_final, qubits=[0, 1], state='10'))
            p11 = float(self.prob(rho_final, qubits=[0, 1], state='11'))
            
            prob_00_list.append(p00)
            prob_01_list.append(p01)
            prob_10_list.append(p10)
            prob_11_list.append(p11)
        
        # Compute pulse shape u(t) = exp(-t^2)
        pulse_shape = np.exp(-times**2)
        
        return {
            'times': times,
            'prob_00': np.array(prob_00_list),
            'prob_01': np.array(prob_01_list),
            'prob_10': np.array(prob_10_list),
            'prob_11': np.array(prob_11_list),
            'pulse_shape': pulse_shape
        }
    
    def sweep_chi_gamma(
        self,
        chi_interval: list = [0.1, 100.0],
        gamma_interval: list = [1.0, 100.0],
        resolution_chi: int = 20,
        resolution_gamma: int = 20,
        chi_scale: str = 'linear',
        gamma_scale: str = 'linear',
        batch_size: int = 1,
        verbose: bool = True
    ) -> Dict[str, Union[np.ndarray, float, str]]:
        """
        Sweep over chi and gamma parameters for two-qubit system.
        
        This method evaluates sensing contrast and detection probability across
        a 2D grid of chi (dispersive coupling) and gamma (cavity decay rate)
        values. For two-qubit systems, chi is set equal for both qubits.
        
        Args:
            chi_interval: List [min, max] for chi values. Default: [0.1, 100.0].
            gamma_interval: List [min, max] for gamma values. Default: [1.0, 100.0].
            resolution_chi: Number of chi points. Default: 20.
            resolution_gamma: Number of gamma points. Default: 20.
            chi_scale: Scale type for chi: 'linear' or 'log'. Default: 'linear'.
            gamma_scale: Scale type for gamma: 'linear' or 'log'. Default: 'linear'.
            batch_size: Number of random realizations to average over. Default: 1.
            verbose: Print progress information. Default: True.
            
        Returns:
            Dictionary with 'chi_vals', 'gamma_vals', 'contrast_map', 
            'detection_map', 'detection_without_map', 'chi_scale', 'gamma_scale'.
            
        Example:
            >>> results = experiment.sweep_chi_gamma(
            ...     chi_interval=[0.1, 50.0],
            ...     resolution_chi=15,
            ...     resolution_gamma=15,
            ...     chi_scale='log'
            ... )
            >>> max_idx = np.unravel_index(
            ...     np.argmax(results['contrast_map']),
            ...     results['contrast_map'].shape
            ... )
            >>> print(f"Optimal chi: {results['chi_vals'][max_idx[1]]:.3f}")
            
        Note:
            Chi is assumed equal for both qubits (χ₁ = χ₂).
        """
        from qsopt.utils.chi_lambda_sweep import compute_chi_gamma_sweep
        return compute_chi_gamma_sweep(
            self, chi_interval, gamma_interval,
            resolution_chi, resolution_gamma,
            chi_scale, gamma_scale, batch_size, verbose
        )
    
    # Backward compatibility alias
    sweep_chi_lambda = sweep_chi_gamma
    
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
    
    def measure_all_states(self, rho: qt.Qobj) -> Dict[str, float]:
        """
        Measure probabilities for all joint qubit states.
        
        Convenience method to get all joint measurement outcomes at once.
        
        Args:
            rho: State to measure
            
        Returns:
            Dictionary with joint measurement probabilities:
            {'00': p00, '01': p01, '10': p10, '11': p11}
            
        Example:
            >>> probs = experiment.measure_all_states(rho)
            >>> print(f"P(00) = {probs['00']:.4f}")
        """
        return {
            '00': self.prob(rho, qubits=[0, 1], state='00'),
            '01': self.prob(rho, qubits=[0, 1], state='01'),
            '10': self.prob(rho, qubits=[0, 1], state='10'),
            '11': self.prob(rho, qubits=[0, 1], state='11'),
        }

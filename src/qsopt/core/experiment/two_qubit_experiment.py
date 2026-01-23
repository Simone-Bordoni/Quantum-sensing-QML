"""
Two Qubit Quantum Sensing Experiment
====================================

Two-qubit quantum sensing experiment implementation.
This class handles quantum sensing protocols with two qubits coupled to a shared cavity.
"""

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import qutip as qt

from qsopt.core.callback import OptimizationCallback
from qsopt.core.circuit import QuantumCircuit, create_ry_circuit_layer
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InteractionType,
    QubitInteraction,
)
from qsopt.core.loss_functions import DetectionFromProbabilities
from qsopt.core.trainable_parameters import TrainableParameters

if TYPE_CHECKING:
    from qsopt.utils.results import TimeEvolutionResults

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import

from .base import Experiment
from .quantum_utils import (
    apply_qubit_rotation,
    build_qubit_noise_operators,
    generate_initial_state,
    generate_two_qubit_operators,
    gu,
    measure_qubits_probability,
)

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
        initial_circuit: Optional[QuantumCircuit] = None,
        final_circuit: Optional[QuantumCircuit] = None,
        optimizer: Optional = None,
        detector: Optional[DetectionFromProbabilities] = None,
    ):
        """
        Initialize two-qubit experiment.

        Args:
            experimental_params: Physical and measurement parameters
            initial_circuit: QuantumCircuit to apply before evolution. If None, creates
                           default 2-qubit RY circuit with trainable parameters.
            final_circuit: QuantumCircuit to apply after evolution. If None, creates
                          default 2-qubit RY circuit with trainable parameters.
            optimizer: Optax optimizer for trainable parameters. If None, uses SGD(0.1).
            detector: Custom detection probability calculator. If None, uses default 1-P(00).
        """
        import optax
        
        # Create default circuits if not provided (2-qubit RY gates)
        if initial_circuit is None:
            initial_circuit = create_ry_circuit_layer(num_qubits=2, theta_values=[np.pi/2, np.pi/2])
        
        if final_circuit is None:
            final_circuit = create_ry_circuit_layer(num_qubits=2, theta_values=[-np.pi/2, -np.pi/2])
        
        # Store circuits
        self.initial_circuit = initial_circuit
        self.final_circuit = final_circuit
        
        # Create TrainableParameters from circuits
        trainable_params = self._create_trainable_params_from_circuits(optimizer)
        
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
            field_levels, cavity_levels, qubit_levels_list
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
            raise RuntimeError(
                "Operators must be generated before building interaction Hamiltonian"
            )

        # Get qubit interactions from experimental parameters
        interactions = self.experimental_params.physical_constants.qubit_interactions

        if not interactions:
            # No interactions - return zero operator
            dims = self.operators["a"].dims
            return qt.Qobj(np.zeros((np.prod(dims[0]), np.prod(dims[0]))), dims=dims)

        # Start with zero Hamiltonian
        dims = self.operators["a"].dims
        H_interaction = qt.Qobj(np.zeros((np.prod(dims[0]), np.prod(dims[0]))), dims=dims)

        # Build each interaction term
        for interaction in interactions:
            idx1, idx2 = interaction.qubit_indices
            chi = interaction.chi
            interaction_type = interaction.interaction_type

            # Get appropriate Pauli operators based on interaction type
            if interaction_type == InteractionType.ZZ:
                # σz ⊗ σz interaction
                sigma1 = self.operators[f"sigma_z{idx1+1}"]
                sigma2 = self.operators[f"sigma_z{idx2+1}"]
            elif interaction_type == InteractionType.XX:
                # σx ⊗ σx interaction
                sigma1 = self.operators[f"sigma_x{idx1+1}"]
                sigma2 = self.operators[f"sigma_x{idx2+1}"]
            elif interaction_type == InteractionType.YY:
                # σy ⊗ σy interaction
                sigma1 = self.operators[f"sigma_y{idx1+1}"]
                sigma2 = self.operators[f"sigma_y{idx2+1}"]
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
        a_in = self.operators["a_in"]
        a_in_dag = self.operators["a_in_dag"]
        a = self.operators["a"]
        a_dag = self.operators["a_dag"]

        # Qubit 1 operators
        sigma_z1 = self.operators["sigma_z1"]
        sigma_x1 = self.operators["sigma_x1"]
        sigma_y1 = self.operators["sigma_y1"]
        sigma_minus1 = self.operators["sigma_minus1"]

        # Qubit 2 operators
        sigma_z2 = self.operators["sigma_z2"]
        sigma_x2 = self.operators["sigma_x2"]
        sigma_y2 = self.operators["sigma_y2"]
        sigma_minus2 = self.operators["sigma_minus2"]

        # Time-dependent coupling function arguments
        args = {"sigma": sigma}

        # Time-dependent cavity-field coupling Hamiltonian
        # H_c = (i/2)√γ (a_in† a - a_in a†)
        coupling_coeff = 1j / 2 * jnp.sqrt(gm)
        H_coupling = qt.Qobj(coupling_coeff * (a_in_dag * a - a_in * a_dag))  # type: ignore

        # Dispersive qubit-resonator interaction Hamiltonians
        # H_q = -Σᵢ (χᵢ/2) a† a σz_i
        H_dispersive1 = qt.Qobj(-chi1 / 2 * a_dag * a * sigma_z1)  # type: ignore
        H_dispersive2 = qt.Qobj(-chi2 / 2 * a_dag * a * sigma_z2)  # type: ignore
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

        if (
            isinstance(depolarizing, list)
            and isinstance(dephasing, list)
            and isinstance(relaxation, list)
        ):
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
            relaxation_rate=relaxation1,
        )

        # Build Lindblad noise operators for qubit 2 using helper function
        lindblad_noise_q2 = build_qubit_noise_operators(
            sigma_x=sigma_x2,
            sigma_y=sigma_y2,
            sigma_z=sigma_z2,
            sigma_minus=sigma_minus2,
            depolarizing_rate=depolarizing2,
            dephasing_rate=dephasing2,
            relaxation_rate=relaxation2,
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
            "total": H_total,
            "dispersive": H_dispersive,
            "dispersive1": H_dispersive1,
            "dispersive2": H_dispersive2,
            "coupling": H_coupling,
        }

        self.lindblad_operators = {
            "interaction": interaction_ops,
            "no_interaction": no_interaction_ops,
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
            "00": self.operators["P00"],
            "01": self.operators["P01"],
            "10": self.operators["P10"],
            "11": self.operators["P11"],
        }

        # Cache individual qubit projectors
        self._cached_qubit1_projectors = {
            "0": self.operators["P0_q1"],
            "1": self.operators["P1_q1"],
        }

        self._cached_qubit2_projectors = {
            "0": self.operators["P0_q2"],
            "1": self.operators["P1_q2"],
        }

        # Generate and cache initial state
        self._cached_initial_state = generate_initial_state(
            initial_config=self.experimental_params.initial_state,
            field_levels=self.experimental_params.field_levels,
            cavity_levels=self.experimental_params.cavity_levels,
            qubit_levels=self.experimental_params.qubit_levels,
            num_qubits=2,
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
                "Initial state has not been cached. " "Ensure _initialize_caches was called."
            )
        return self._cached_initial_state

    def _create_trainable_params_from_circuits(self, optimizer=None) -> TrainableParameters:
        """
        Extract trainable parameters from quantum circuits.

        For two-qubit system, we expect 4 trainable parameters total:
        - 2 parameters from initial_circuit (one per qubit)
        - 2 parameters from final_circuit (one per qubit)

        Args:
            optimizer: Optax optimizer. If None, uses SGD(0.1).

        Returns:
            TrainableParameters initialized from circuit parameters
        """
        import optax

        if optimizer is None:
            optimizer = optax.sgd(0.1)

        trainable_params = TrainableParameters()

        # Extract parameters from initial circuit as dict
        initial_params = self.initial_circuit.get_trainable_parameters()
        if len(initial_params) != 2:
            raise ValueError(
                f"Expected 2 trainable parameters in initial_circuit, got {len(initial_params)}"
            )
        
        # Extract parameters from final circuit as dict
        final_params = self.final_circuit.get_trainable_parameters()
        if len(final_params) != 2:
            raise ValueError(
                f"Expected 2 trainable parameters in final_circuit, got {len(final_params)}"
            )

        # Convert dicts to lists preserving order (sorted by key)
        initial_values = [v for k, v in sorted(initial_params.items())]
        final_values = [v for k, v in sorted(final_params.items())]

        # Combine into flat list: [theta1_initial, theta2_initial, theta1_final, theta2_final]
        all_param_values = initial_values + final_values
        
        # Create parameter names
        param_names = ["theta1_q1", "theta1_q2", "theta2_q1", "theta2_q2"]

        # Add all parameters to TrainableParameters
        trainable_params.add_rotation_angles(
            names=param_names, 
            initial_values=all_param_values, 
            optimizer=optimizer
        )

        return trainable_params

    def _update_circuits_from_trainable_params(self) -> None:
        """
        Update circuit parameters from TrainableParameters state.

        Synchronizes the QuantumCircuit objects with current optimization values.
        Distributes the 4 parameters as:
        - params[0:2] -> initial_circuit (qubit 1, qubit 2)
        - params[2:4] -> final_circuit (qubit 1, qubit 2)
        """
        current_params = self.trainable_params.get_parameter_vector()

        # Get the parameter keys from circuits (sorted order)
        initial_keys = sorted(self.initial_circuit.get_trainable_parameters().keys())
        final_keys = sorted(self.final_circuit.get_trainable_parameters().keys())

        # Build dicts for setting parameters
        initial_dict = {initial_keys[0]: current_params[0], initial_keys[1]: current_params[1]}
        final_dict = {final_keys[0]: current_params[2], final_keys[1]: current_params[3]}

        # Update circuits
        self.initial_circuit.set_trainable_parameters(initial_dict)
        self.final_circuit.set_trainable_parameters(final_dict)

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
                "Joint projectors have not been cached. " "Ensure _initialize_caches was called."
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
            raise ValueError(f"Invalid qubit state '{state}'. Must be '0' or '1'.")

        return projectors[state]

    def get_solver_with_interaction(self) -> qt.MESolver:
        """Get Lindblad master equation solver WITH input photon interaction (cached)."""
        if "with_interaction" not in self._cached_solvers:
            self._cached_solvers["with_interaction"] = qt.MESolver(
                self.hamiltonians["total"],
                self.lindblad_operators["interaction"],
                options={"method": "diffrax", "progress_bar": False, "normalize_output": False},
            )
        return self._cached_solvers["with_interaction"]

    def get_solver_no_interaction(self) -> qt.MESolver:
        """Get Lindblad master equation solver WITHOUT input photon interaction (cached)."""
        if "no_interaction" not in self._cached_solvers:
            self._cached_solvers["no_interaction"] = qt.MESolver(
                self.hamiltonians["dispersive"],
                self.lindblad_operators["no_interaction"],
                options={"method": "diffrax", "progress_bar": False, "normalize_output": False},
            )
        return self._cached_solvers["no_interaction"]

    def apply_rotation(self, rho: qt.Qobj, theta: float, qubit: int, axis: str = "y") -> qt.Qobj:
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

    def prob(self, rho: qt.Qobj, qubits: List[int], state: str = "0") -> float:
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

    def _embed_circuit_unitary(self, U_circuit: qt.Qobj) -> qt.Qobj:
        """
        Embed a 2-qubit circuit unitary into the full composite Hilbert space.

        The composite space is: input_field ⊗ resonator_cavity ⊗ qubit1 ⊗ qubit2
        The circuit acts only on the qubit subspace (qubit1 ⊗ qubit2).

        Args:
            U_circuit: 4x4 unitary matrix for the 2-qubit circuit (as qt.Qobj)

        Returns:
            Full-space operator: I_field ⊗ I_cavity ⊗ U_circuit
        """
        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels

        # Extract JAX array from circuit unitary
        if hasattr(U_circuit.data, '_jxa'):  # JaxArray backend
            U_jax = U_circuit.data._jxa
        elif hasattr(U_circuit.data, 'to_array'):  # Dense backend
            U_jax = jnp.array(U_circuit.data.to_array())
        else:
            U_jax = jnp.array(U_circuit.full())

        # Build full operator using JAX Kronecker products
        # I_field ⊗ I_cavity ⊗ U_circuit
        I_field = jnp.eye(field_levels, dtype=jnp.complex128)
        I_cavity = jnp.eye(cavity_levels, dtype=jnp.complex128)
        
        # Kronecker product: I_field ⊗ I_cavity ⊗ U_circuit
        U_full_jax = jnp.kron(jnp.kron(I_field, I_cavity), U_jax)

        # Wrap in Qobj with proper dimensions
        return qt.Qobj(
            U_full_jax, 
            dims=[[field_levels, cavity_levels, 2, 2], [field_levels, cavity_levels, 2, 2]]
        )

    def _prepare_rotation_gates(
        self, theta1_q1: float, theta2_q1: float, theta1_q2: float, theta2_q2: float
    ) -> tuple:
        """
        Build rotation unitaries from circuits for two-qubit optimization.

        Updates the quantum circuits with current parameters and returns
        embedded unitary operators. The circuits compute their unitaries using
        JAX-compatible gate.matrix() methods, maintaining gradient flow.

        Args:
            theta1_q1: First rotation parameter for qubit 1 (initial circuit)
            theta2_q1: Second rotation parameter for qubit 1 (final circuit)
            theta1_q2: First rotation parameter for qubit 2 (initial circuit)
            theta2_q2: Second rotation parameter for qubit 2 (final circuit)

        Returns:
            Tuple of (R1_full, R2_full) - embedded unitaries for initial and final circuits
        """
        # Get parameter keys (sorted for consistent ordering)
        initial_keys = sorted(self.initial_circuit.get_trainable_parameters().keys())
        final_keys = sorted(self.final_circuit.get_trainable_parameters().keys())

        # Update circuits with current parameter values
        initial_dict = {initial_keys[0]: theta1_q1, initial_keys[1]: theta1_q2}
        final_dict = {final_keys[0]: theta2_q1, final_keys[1]: theta2_q2}
        
        self.initial_circuit.set_trainable_parameters(initial_dict)
        self.final_circuit.set_trainable_parameters(final_dict)

        # Get circuit unitaries (4x4 matrices acting on qubit1 ⊗ qubit2)
        # These are JAX-compatible through gate.matrix() -> expm()
        U_initial = self.initial_circuit.get_unitary()
        U_final = self.final_circuit.get_unitary()

        # Embed in full composite Hilbert space
        R1_full = self._embed_circuit_unitary(U_initial)
        R2_full = self._embed_circuit_unitary(U_final)

        return R1_full, R2_full

    def _embed_two_qubit_unitary(self, U: jnp.ndarray, qubit_index: int) -> qt.Qobj:
        """
        Embed a 2x2 single-qubit unitary into the full composite Hilbert space.

        The composite space is: input_field ⊗ resonator_cavity ⊗ qubit1 ⊗ qubit2

        Args:
            U: 2x2 unitary matrix (JAX array) for the qubit
            qubit_index: Which qubit this unitary acts on (0 for qubit1, 1 for qubit2)

        Returns:
            Full-space operator that applies U to specified qubit
        """
        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels
        qubit_levels = self.experimental_params.qubit_levels[qubit_index]
        other_qubit_levels = self.experimental_params.qubit_levels[1 - qubit_index]

        # Create identity operators
        I_field = qt.qeye(field_levels)
        I_cavity = qt.qeye(cavity_levels)
        I_qubit1 = qt.qeye(self.experimental_params.qubit_levels[0])
        I_qubit2 = qt.qeye(self.experimental_params.qubit_levels[1])

        # Convert JAX array to QuTiP Qobj
        U_qobj = qt.Qobj(U, dims=[[qubit_levels], [qubit_levels]])

        # Build full operator based on which qubit we're acting on
        if qubit_index == 0:
            # Act on qubit 1: I_field ⊗ I_cavity ⊗ U ⊗ I_qubit2
            R_full = qt.tensor(I_field, I_cavity, U_qobj, I_qubit2)
        else:
            # Act on qubit 2: I_field ⊗ I_cavity ⊗ I_qubit1 ⊗ U
            R_full = qt.tensor(I_field, I_cavity, I_qubit1, U_qobj)

        return R_full

    def _measure_joint_probabilities(self, rho: qt.Qobj) -> Dict[str, jnp.ndarray]:
        """
        Measure all joint two-qubit probabilities (JAX-compatible).

        Args:
            rho: Density matrix

        Returns:
            Dictionary with p00, p01, p10, p11 as JAX arrays
        """
        P00 = self._cached_joint_projectors["00"]
        P01 = self._cached_joint_projectors["01"]
        P10 = self._cached_joint_projectors["10"]
        P11 = self._cached_joint_projectors["11"]

        # Use real part and ensure JAX array
        p00 = jnp.real((P00 * rho * P00).tr())  # type: ignore
        p01 = jnp.real((P01 * rho * P01).tr())  # type: ignore
        p10 = jnp.real((P10 * rho * P10).tr())  # type: ignore
        p11 = jnp.real((P11 * rho * P11).tr())  # type: ignore

        return {"p00": p00, "p01": p01, "p10": p10, "p11": p11}

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
        args: Optional[Dict] = None,
    ) -> Dict[str, jnp.ndarray]:
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
            args = {"sigma": self.experimental_params.inverse_pulse_width}

        # Build rotation gates (JAX-compatible)
        R1_q1, R2_q1, R1_q2, R2_q2 = self._prepare_rotation_gates(
            theta1_q1, theta2_q1, theta1_q2, theta2_q2
        )

        # Initial rotations on both qubits
        rho_rotated = R1_q1 * rho * R1_q1.dag()  # type: ignore
        rho_rotated = R1_q2 * rho_rotated * R1_q2.dag()  # type: ignore

        # Evolve system
        result = solver.run(rho_rotated, tlist=[t_start, t_end], args=args)
        rho_final = result.states[-1]

        # Apply final rotations
        rho_final = R2_q1 * rho_final * R2_q1.dag()  # type: ignore
        rho_final = R2_q2 * rho_final * R2_q2.dag()  # type: ignore

        # Measure all joint probabilities (JAX-compatible)
        return self._measure_joint_probabilities(rho_final)

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
        precomputed_rotations: Optional[tuple] = None,
        loss_function: Optional[DetectionFromProbabilities] = None,
    ) -> jnp.ndarray:
        """
        JAX-compatible simulation for two-qubit system with customizable detection.

        This method implements sequential measurements, projecting onto joint
        measurement states at each time point and computing cumulative detection
        probability using a user-defined or default detection criterion.

        Protocol:
        1. Apply first rotations to both qubits
        2. Sequential time evolution with measurements
        3. At each measurement: apply second rotations, measure final probabilities
        4. Apply detection criterion to get detection probability
        5. Project to non-detected state and continue

        Args:
            solver: Configured quantum evolution solver
            rho: Initial density matrix
            theta1_q1: First Y-rotation for qubit 1
            theta2_q1: Second Y-rotation for qubit 1
            theta1_q2: First Y-rotation for qubit 2
            theta2_q2: Second Y-rotation for qubit 2
            measurements: Array of measurement times (sorted)
            args: System parameters (optional)
            precomputed_rotations: Optional tuple (R1_q1, R2_q1, R1_q2, R2_q2) to avoid recomputation
            loss_function: Optional DetectionFromProbabilities instance. If None, uses 1-P(00).

        Returns:
            Detection probability as JAX array

        Example:
            >>> # Default: 1 - P(00)
            >>> prob = experiment.simulation(solver, rho0, 0.5, -0.5, 0.3, -0.3, times)
            >>>
            >>> # Custom: P(11) only
            >>> detector = DetectionFromProbabilities(lambda p: p['p11'], name="P(11)")
            >>> prob = experiment.simulation(solver, rho0, 0.5, -0.5, 0.3, -0.3, times,
            ...                               loss_function=detector)
        """
        if args is None:
            args = {"sigma": self.experimental_params.inverse_pulse_width}

        # Default to 1-P(00) detection
        if loss_function is None:
            loss_function = DetectionFromProbabilities()

        measurement_array = np.asarray(measurements, dtype=float)
        if measurement_array.ndim != 1 or measurement_array.size < 2:
            raise ValueError("measurements must be a 1D array with at least 2 time points")

        # Get rotation unitaries (precomputed or build new)
        if precomputed_rotations is None:
            R1_full, R2_full = self._prepare_rotation_gates(
                theta1_q1, theta2_q1, theta1_q2, theta2_q2
            )
        else:
            R1_full, R2_full = precomputed_rotations

        # Precompute dagger operators
        R1_full_dag = R1_full.dag()
        R2_full_dag = R2_full.dag()

        # Get projectors
        P00 = self._cached_joint_projectors["00"]

        # Initial state
        rho_current = rho

        # Track cumulative probability of non-detection
        prob_all_no_detect = jnp.array(1.0)

        # Loop over measurement intervals
        for t0, t1 in zip(measurement_array[:-1], measurement_array[1:]):
            # Convert to concrete float values (times should not be differentiated)
            t0_float = float(t0)
            t1_float = float(t1)

            # Apply initial circuit rotations (both qubits simultaneously)
            rho_after_r1 = R1_full * rho_current * R1_full_dag  # type: ignore

            # Evolve
            evolution_result = solver.run(rho_after_r1, [t0_float, t1_float], args=args)
            rho_evolved = evolution_result.states[-1]

            # Apply final circuit rotations (both qubits simultaneously)
            rho_final = R2_full * rho_evolved * R2_full_dag  # type: ignore

            # Measure all joint probabilities
            probs = self._measure_joint_probabilities(rho_final)

            # Apply detection criterion
            prob_detect_this_step = loss_function(probs)
            prob_no_detect_this_step = 1.0 - prob_detect_this_step

            # Update cumulative non-detection probability
            prob_all_no_detect = prob_all_no_detect * prob_no_detect_this_step

            # Project onto |00⟩ (non-detected state) and renormalize
            rho_projected = P00 * rho_final * P00  # type: ignore
            trace_val = rho_projected.tr()
            rho_current = rho_projected if trace_val == 0 else rho_projected / trace_val

            # Undo final circuit rotations for next iteration
            rho_current = R2_full_dag * rho_current * R2_full  # type: ignore

        # Total detection probability = 1 - P(no detection at any step)
        prob_detection = 1 - prob_all_no_detect
        return prob_detection

    def run_simulation(self, batch_size: int = 1) -> OptimizationCallback:
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
        measurement_times_batch = self.experimental_params.get_measurement_times_with_uncertainty(
            batch_size
        )
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
            contrast=float(contrast),
        )

        return callback

    def run_simulation_with_probabilities(
        self, t_start: float = -5.0, t_end: float = 5.0
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
            solver_with, rho0, theta1_q1, theta2_q1, theta1_q2, theta2_q2, t_start, t_end
        )

        # Compute final probabilities without photon
        probs_without = self.compute_final_probabilities(
            solver_without, rho0, theta1_q1, theta2_q1, theta1_q2, theta2_q2, t_start, t_end
        )

        # Use detector to compute detection probabilities
        detection_with = float(self.detector(probs_with))
        detection_without = float(self.detector(probs_without))

        # Compute contrast using detector's method
        contrast = float(
            DetectionFromProbabilities.compute_contrast(detection_with, detection_without)
        )

        return {
            "probs_with": probs_with,
            "probs_without": probs_without,
            "detection_with": detection_with,
            "detection_without": detection_without,
            "contrast": contrast,
        }

    def time_evolution(
        self,
        n_points: int = 200,
        with_interaction: bool = True,
        measurement_protocol: Optional["MeasurementProtocol"] = None,
    ) -> "TimeEvolutionResults":
        """
        Compute time evolution of two-qubit probabilities.

        Simulates the quantum system evolution over time using the measurement protocol times.
        Returns probability distributions for all two-qubit states (|00⟩, |01⟩, |10⟩, |11⟩).
        The system starts in superposition (after first rotations), evolves under
        the Hamiltonian, and probabilities are measured after the second rotations.

        Args:
            n_points: Number of time points to sample (default: 200)
            with_interaction: If True, use Hamiltonian with chi coupling.
                             If False, use Hamiltonian without chi (default: True)
            measurement_protocol: Optional custom measurement protocol to use instead of
                                 the experiment's default protocol (default: None)

        Returns:
            TimeEvolutionResults object containing:
                - times: Array of time points, shape (n_points,)
                - probabilities: Dict with 'prob_00', 'prob_01', 'prob_10', 'prob_11'
                - pulse_shape: Pulse envelope u(t), shape (n_points,)
                - measurement_times: Measurement time points
                - cavity_population: Cavity population <a†a>, shape (n_points,)
                - field_population: External field population <a_in†a_in>, shape (n_points,)

        Example:
            >>> # Get time evolution data using default measurement protocol
            >>> evolution = experiment.time_evolution(n_points=200)
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
            >>> # With cavity population displayed on secondary y-axis
            >>> fig = plot_time_evolution(evolution, show_cavity_population=True)
            >>> # Without cavity population (default)
            >>> fig = plot_time_evolution(evolution, show_cavity_population=False)
        """
        # Use provided measurement protocol or default from experimental parameters
        if measurement_protocol is None:
            measurement_protocol = self.experimental_params.measurement
        
        # Get measurement times from protocol
        measurement_times = np.array(measurement_protocol.measurement_times)
        
        # Use measurement times for start and end
        t_start = float(measurement_times[0])
        t_end = float(measurement_times[-1])
            
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

        solver = (
            self.get_solver_with_interaction()
            if with_interaction
            else self.get_solver_no_interaction()
        )

        # Apply first rotations
        rho_current = self.apply_rotation(rho0, theta1_q1, qubit=0)
        rho_current = self.apply_rotation(rho_current, theta1_q2, qubit=1)

        # Get number operators for population calculation
        a_dag = self.operators["a_dag"]
        a = self.operators["a"]
        n_cavity = a_dag * a  # Cavity number operator a†a
        
        a_in_dag = self.operators["a_in_dag"]
        a_in = self.operators["a_in"]
        n_field = a_in_dag * a_in  # Field number operator a_in†a_in
        
        args = {"sigma": self.experimental_params.inverse_pulse_width}
        
        # Check if we need to perform intermediate measurements
        intermediate_meas_times = measurement_times[(measurement_times > t_start) & (measurement_times < t_end)]
        perform_measurements = len(intermediate_meas_times) > 0
        
        # Storage for results
        all_times = []
        prob_00_list = []
        prob_01_list = []
        prob_10_list = []
        prob_11_list = []
        cavity_population_list = []
        field_population_list = []
        
        if perform_measurements:
            # Evolution with intermediate projective measurements
            segment_starts = [t_start] + list(intermediate_meas_times)
            segment_ends = list(intermediate_meas_times) + [t_end]
            
            for seg_start, seg_end in zip(segment_starts, segment_ends):
                # Number of points for this segment
                seg_fraction = (seg_end - seg_start) / (t_end - t_start)
                seg_n_points = max(2, int(n_points * seg_fraction))
                
                # Evolve segment
                seg_times = np.linspace(seg_start, seg_end, seg_n_points)
                result = solver.run(rho_current, tlist=seg_times, args=args)
                
                # Extract data for this segment
                for i, rho_t in enumerate(result.states):
                    # Apply second rotations for measurement
                    rho_meas = self.apply_rotation(rho_t, theta2_q1, qubit=0)
                    rho_meas = self.apply_rotation(rho_meas, theta2_q2, qubit=1)
                    
                    # Measure two-qubit probabilities
                    p00 = float(self.prob(rho_meas, qubits=[0, 1], state="00"))
                    p01 = float(self.prob(rho_meas, qubits=[0, 1], state="01"))
                    p10 = float(self.prob(rho_meas, qubits=[0, 1], state="10"))
                    p11 = float(self.prob(rho_meas, qubits=[0, 1], state="11"))
                    
                    all_times.append(seg_times[i])
                    prob_00_list.append(p00)
                    prob_01_list.append(p01)
                    prob_10_list.append(p10)
                    prob_11_list.append(p11)
                    
                    # Calculate populations (take real part since expectation values should be real)
                    cavity_pop = float(np.real(qt.expect(n_cavity, rho_t)))
                    field_pop = float(np.real(qt.expect(n_field, rho_t)))
                    cavity_population_list.append(cavity_pop)
                    field_population_list.append(field_pop)
                
                # Perform projective measurement at end of segment (if not the final segment)
                if seg_end != t_end:
                    rho_final = result.states[-1]
                    
                    # Measurement protocol: Ry(-theta2) -> project -> Ry(theta2)
                    rho_before_proj = self.apply_rotation(rho_final, -theta2_q1, qubit=0)
                    rho_before_proj = self.apply_rotation(rho_before_proj, -theta2_q2, qubit=1)
                    
                    # Get probabilities in basis
                    p00_basis = float(self.prob(rho_before_proj, qubits=[0, 1], state="00"))
                    p01_basis = float(self.prob(rho_before_proj, qubits=[0, 1], state="01"))
                    p10_basis = float(self.prob(rho_before_proj, qubits=[0, 1], state="10"))
                    p11_basis = float(self.prob(rho_before_proj, qubits=[0, 1], state="11"))
                    
                    # Stochastic projection (weighted by probabilities)
                    # For deterministic visualization, use weighted mixture
                    proj_00 = self.get_qubit_projector(1, "0") * self.get_qubit_projector(2, "0")
                    proj_01 = self.get_qubit_projector(1, "0") * self.get_qubit_projector(2, "1")
                    proj_10 = self.get_qubit_projector(1, "1") * self.get_qubit_projector(2, "0")
                    proj_11 = self.get_qubit_projector(1, "1") * self.get_qubit_projector(2, "1")
                    
                    rho_projected = (
                        p00_basis * (proj_00 * rho_before_proj * proj_00) +
                        p01_basis * (proj_01 * rho_before_proj * proj_01) +
                        p10_basis * (proj_10 * rho_before_proj * proj_10) +
                        p11_basis * (proj_11 * rho_before_proj * proj_11)
                    )  # type: ignore
                    
                    # Normalize
                    rho_projected = rho_projected / rho_projected.tr()
                    
                    # Apply rotations back
                    rho_current = self.apply_rotation(rho_projected, theta2_q1, qubit=0)
                    rho_current = self.apply_rotation(rho_current, theta2_q2, qubit=1)
        else:
            # Continuous evolution without intermediate measurements
            times = np.linspace(t_start, t_end, n_points)
            result = solver.run(rho_current, tlist=times, args=args)
            
            for i, rho_t in enumerate(result.states):
                # Apply second rotations
                rho_final = self.apply_rotation(rho_t, theta2_q1, qubit=0)
                rho_final = self.apply_rotation(rho_final, theta2_q2, qubit=1)

                # Measure two-qubit probabilities
                p00 = float(self.prob(rho_final, qubits=[0, 1], state="00"))
                p01 = float(self.prob(rho_final, qubits=[0, 1], state="01"))
                p10 = float(self.prob(rho_final, qubits=[0, 1], state="10"))
                p11 = float(self.prob(rho_final, qubits=[0, 1], state="11"))

                all_times.append(times[i])
                prob_00_list.append(p00)
                prob_01_list.append(p01)
                prob_10_list.append(p10)
                prob_11_list.append(p11)

                # Calculate populations (take real part since expectation values should be real)
                cavity_pop = float(np.real(qt.expect(n_cavity, rho_t)))
                field_pop = float(np.real(qt.expect(n_field, rho_t)))
                cavity_population_list.append(cavity_pop)
                field_population_list.append(field_pop)
        
        times = np.array(all_times)
        # Compute pulse shape u(t) = exp(-t^2)
        pulse_shape = np.exp(-(times**2))

        # Import at runtime to avoid circular dependency
        from qsopt.utils.results import TimeEvolutionResults

        return TimeEvolutionResults(
            times=times,
            probabilities={
                "prob_00": np.array(prob_00_list),
                "prob_01": np.array(prob_01_list),
                "prob_10": np.array(prob_10_list),
                "prob_11": np.array(prob_11_list),
            },
            pulse_shape=pulse_shape,
            measurement_times=measurement_times,
            cavity_population=np.array(cavity_population_list),
            field_population=np.array(field_population_list),
            metadata={
                "chi": self.experimental_params.chi,
                "gamma": self.experimental_params.photon_cavity_coupling,
                "with_interaction": with_interaction,
                "n_qubits": 2,
            },
        )

    def optimize_rotations(
        self,
        num_steps: int = 100,
        batch_size: int = 1,
        tolerance: float = 1e-6,
        verbose: bool = True,
        verbose_step: int = 10,
        callback: Optional[OptimizationCallback] = None,
        theta_init: Optional[List[float]] = None,
        loss_function: Optional["DetectionFromProbabilities"] = None,
    ) -> OptimizationCallback:
        """
        Optimize rotation angles to maximize sensing contrast.

        This method performs JAX-based gradient descent over four rotation angles
        (two per qubit) using the sequential measurement protocol. The detection
        criterion can be customized using the loss_function parameter.

        Args:
            num_steps: Maximum number of optimization steps (default: 100)
            batch_size: Number of random realizations for measurement uncertainty per step (default: 1)
            tolerance: Convergence threshold for gradient norm (default: 1e-6)
            verbose: Print progress information (default: True)
            verbose_step: Step interval for printing progress (default: 10)
            callback: Optional callback to track optimization progress.
                     If None, uses the experiment's default callback.
            theta_init: Optional initial rotation angles [θ1_q1, θ2_q1, θ1_q2, θ2_q2] in radians.
                       If None, uses values from trainable_params.
            loss_function: Optional custom detection criterion (DetectionFromProbabilities).
                         If None, uses default 1-P(00) detection.

        Returns:
            OptimizationCallback with full optimization history

        Example:
            >>> # Optimize with default 1-P(00) detection
            >>> callback = experiment.optimize_rotations(num_steps=200, batch_size=10)
            >>>
            >>> # With custom detection criterion
            >>> from qsopt.utils.loss_functions import DetectionFromProbabilities
            >>> loss = DetectionFromProbabilities(lambda p: p['p11'])  # Detect |11⟩
            >>> callback = experiment.optimize_rotations(
            ...     num_steps=100,
            ...     loss_function=loss
            ... )
        """
        import jax
        import optax

        # Use provided callback or default
        if callback is None:
            callback = self.callback

        # Reset callback at start of new optimization
        callback.reset()

        # Get rotation angles (must have at least 4 for two qubits)
        rotation_angles = self.trainable_params.get_rotation_angles()
        if len(rotation_angles) < 4:
            raise ValueError(
                f"Two-qubit experiment requires at least 4 rotation parameters, "
                f"got {len(rotation_angles)}"
            )

        # Get parameter names for the four rotation angles
        rotation_names = list(rotation_angles.keys())
        theta1_q1_name = rotation_names[0]
        theta2_q1_name = rotation_names[1]
        theta1_q2_name = rotation_names[2]
        theta2_q2_name = rotation_names[3]

        # Find indices for rotation parameters
        param_indices = []
        param_names = []
        for name in [theta1_q1_name, theta2_q1_name, theta1_q2_name, theta2_q2_name]:
            idx = -1
            for param in self.trainable_params.parameters:
                if param.name == name:
                    idx = param.index
                    break
            if idx == -1:
                raise ValueError(f"Could not find rotation parameter {name}")
            param_indices.append(idx)
            param_names.append(name)

        # Initialize parameter vector
        if theta_init is not None:
            if len(theta_init) != 4:
                raise ValueError(
                    "theta_init must contain exactly 4 angles [θ1_q1, θ2_q1, θ1_q2, θ2_q2]"
                )
            initial_values = theta_init
        else:
            initial_values = [
                rotation_angles[theta1_q1_name][0],
                rotation_angles[theta2_q1_name][0],
                rotation_angles[theta1_q2_name][0],
                rotation_angles[theta2_q2_name][0],
            ]

        params = jnp.array(initial_values, dtype=float)

        # Update trainable_params with initial values
        for idx, val in zip(param_indices, initial_values):
            self.trainable_params.parameters[idx].value = float(val)

        trainable_mask = jnp.array(
            [self.trainable_params.parameters[idx].trainable for idx in param_indices]
        )

        # Use optimizer from first trainable rotation parameter
        optimizer = self.trainable_params.rotation_optimizer
        opt_state = optimizer.init(params)

        # Get initial state and solvers
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")

        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()

        # Define objective function
        def objective_function(opt_params):
            """Negative sensing contrast for minimization with batch averaging."""
            # Extract parameters and apply trainability mask
            theta1_q1_raw, theta2_q1_raw, theta1_q2_raw, theta2_q2_raw = opt_params
            theta1_q1 = theta1_q1_raw if trainable_mask[0] else jax.lax.stop_gradient(theta1_q1_raw)
            theta2_q1 = theta2_q1_raw if trainable_mask[1] else jax.lax.stop_gradient(theta2_q1_raw)
            theta1_q2 = theta1_q2_raw if trainable_mask[2] else jax.lax.stop_gradient(theta1_q2_raw)
            theta2_q2 = theta2_q2_raw if trainable_mask[3] else jax.lax.stop_gradient(theta2_q2_raw)

            # Precompute rotation gates once
            rotation_gates = self._prepare_rotation_gates(
                theta1_q1, theta2_q1, theta1_q2, theta2_q2
            )

            if batch_size == 1:
                # Single realization
                measurement_times_batch = (
                    self.experimental_params.get_measurement_times_with_uncertainty()
                )

                # Simulate with photon
                prob_with = self.simulation(
                    solver_with,
                    rho0,
                    theta1_q1,
                    theta2_q1,
                    theta1_q2,
                    theta2_q2,
                    measurement_times_batch,
                    precomputed_rotations=rotation_gates,
                    loss_function=loss_function,
                )

                # Simulate without photon
                prob_without = self.simulation(
                    solver_without,
                    rho0,
                    theta1_q1,
                    theta2_q1,
                    theta1_q2,
                    theta2_q2,
                    measurement_times_batch,
                    precomputed_rotations=rotation_gates,
                    loss_function=loss_function,
                )

                contrast = prob_with - prob_without

            else:
                # Multiple realizations
                measurement_times_batch = (
                    self.experimental_params.get_measurement_times_with_uncertainty(batch_size)
                )
                prob_with_batch = jnp.zeros(batch_size)
                prob_without_batch = jnp.zeros(batch_size)

                for i in range(batch_size):
                    measurement_times = measurement_times_batch[i]

                    prob_with_batch = prob_with_batch.at[i].set(
                        self.simulation(
                            solver_with,
                            rho0,
                            theta1_q1,
                            theta2_q1,
                            theta1_q2,
                            theta2_q2,
                            measurement_times,
                            precomputed_rotations=rotation_gates,
                            loss_function=loss_function,
                        )
                    )

                    prob_without_batch = prob_without_batch.at[i].set(
                        self.simulation(
                            solver_without,
                            rho0,
                            theta1_q1,
                            theta2_q1,
                            theta1_q2,
                            theta2_q2,
                            measurement_times,
                            precomputed_rotations=rotation_gates,
                            loss_function=loss_function,
                        )
                    )

                # Average over batch
                prob_with = jnp.mean(prob_with_batch)
                prob_without = jnp.mean(prob_without_batch)
                contrast = prob_with - prob_without

            # Return negative for minimization
            return -contrast, (prob_with, prob_without, contrast)

        # Get detection description for verbose output
        detection_desc = "1 - P(00)" if loss_function is None else "custom"

        if verbose:
            theta_initial_vals = np.asarray(params, dtype=float)
            print(f"Configuration:")
            print(f"    Max iterations: {num_steps}")
            print(f"    Batch size: {batch_size}")
            print(f"    Convergence tolerance: {tolerance:.2e}")
            print(f"    Detection criterion: {detection_desc}")
            print(f"    Initial rotation parameters:")
            for i, name in enumerate(param_names):
                status = " [FIXED]" if not trainable_mask[i] else ""
                print(f"        {name}={theta_initial_vals[i]:.3f} rad{status}")

            uncertainty = self.experimental_params.initial_time_uncertainty
            if uncertainty > 0:
                spec = self.experimental_params.initial_time_uncertainty_spec
                extra = f" (specified as '{spec}')" if isinstance(spec, str) else ""
                print(f"    Measurement uncertainty: ±{uncertainty:.3f}{extra}")

            print("=" * 70)
            print(
                f"{'Step':<6}{param_names[0]:<12}{param_names[1]:<12}"
                f"{param_names[2]:<12}{param_names[3]:<12}{'Contrast':<12}{'Grad Norm'}"
            )
            print("-" * 70)

        best_contrast = -np.inf
        best_params = jnp.array(params)

        # Initialize variables
        step = 0
        grad_norm = float("inf")

        for step in range(num_steps):
            # Compute gradients using JAX autodiff
            grads, (prob_with, prob_without, sensing_contrast) = jax.grad(
                objective_function, has_aux=True
            )(params)

            # Track best parameters
            if sensing_contrast > best_contrast:
                best_contrast = sensing_contrast
                best_params = jnp.array(params)

            # Call callback to track progress
            callback(
                trainable_params=self.trainable_params,
                prob_with=float(prob_with),
                prob_without=float(prob_without),
                contrast=float(sensing_contrast),
            )

            grad_norm = float(jnp.linalg.norm(grads))
            theta_values = np.asarray(params, dtype=float)

            # Progress output
            if verbose and (step % verbose_step == 0 or grad_norm < tolerance):
                print(
                    f"{step:<6}{theta_values[0]:<12.6f}{theta_values[1]:<12.6f}"
                    f"{theta_values[2]:<12.6f}{theta_values[3]:<12.6f}"
                    f"{float(sensing_contrast):<12.6f}{grad_norm:<12.2e}"
                )

            # Convergence check
            if grad_norm < tolerance:
                break

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Update trainable parameters continuously
            for idx, val in zip(param_indices, theta_values):
                self.trainable_params.parameters[idx].value = float(val)

        # Ensure best parameters are set at the end
        best_values = np.asarray(best_params, dtype=float)
        for idx, val in zip(param_indices, best_values):
            self.trainable_params.parameters[idx].value = float(val)

        # Apply constraints at the end
        final_values = np.array([p.value for p in self.trainable_params.parameters])
        constrained_values = self.trainable_params.apply_constraints(final_values)
        for i, val in enumerate(constrained_values):
            self.trainable_params.parameters[i].value = float(val)

        if verbose:
            print("=" * 70)
            print(f"Final gradient norm: {grad_norm:.2e}")
            print(f"Best sensing contrast: {best_contrast:.6f}")
            print(f"Best parameters:")
            for i, name in enumerate(param_names):
                print(f"    {name}={best_values[i]:.3f} rad ({np.rad2deg(best_values[i]):.1f}°)")

        # Set convergence information in callback
        callback.set_convergence_info(
            converged=float(grad_norm) < tolerance, final_grad_norm=float(grad_norm)
        )

        return callback

    def sweep_chi_gamma(
        self,
        chi_interval: list = [0.1, 100.0],
        gamma_interval: list = [1.0, 100.0],
        resolution_chi: int = 20,
        resolution_gamma: int = 20,
        chi_scale: str = "linear",
        gamma_scale: str = "linear",
        batch_size: int = 1,
        verbose: bool = True,
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
        from qsopt.utils.parameters_sweep import compute_chi_gamma_sweep

        return compute_chi_gamma_sweep(
            self,
            chi_interval,
            gamma_interval,
            resolution_chi,
            resolution_gamma,
            chi_scale,
            gamma_scale,
            batch_size,
            verbose,
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
            "00": self.prob(rho, qubits=[0, 1], state="00"),
            "01": self.prob(rho, qubits=[0, 1], state="01"),
            "10": self.prob(rho, qubits=[0, 1], state="10"),
            "11": self.prob(rho, qubits=[0, 1], state="11"),
        }

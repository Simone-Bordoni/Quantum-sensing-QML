"""
N Qubit Quantum Sensing Experiment
====================================

n-qubit quantum sensing experiment implementation.
This class handles quantum sensing protocols with n qubits coupled to a shared cavity.
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import qutip as qt
import math
import time as t
import jax.numpy as jnp
import equinox
from jax import jit, lax

from qsopt.core.callback import OptimizationCallback
from qsopt.core.circuit import QuantumCircuit, create_ry_circuit
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InteractionType,
    QubitInteraction,
    MeasurementProtocol
)
from qsopt.core.loss_functions import DetectionMetric
from qsopt.utils.results import SweepResults

if TYPE_CHECKING:
    from qsopt.utils.results import TimeEvolutionResults

from .quantum_utils import (
    apply_qubit_rotation,
    build_qubit_noise_operators,
    embed_circuit_unitary,
    generate_initial_state,
    generate_system_operators,
    gu,
    measure_qubits_probability,
    u0,
)

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import

# Suppress Diffrax complex dtype warning
warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax is a work in progress*")


class Experiment:
    """
    N-qubit quantum sensing experiment.

    This class implements quantum sensing protocols with n qubits coupled dispersively
    to a shared resonator cavity. The composite Hilbert space structure is:

        input_field ⊗ resonator_cavity ⊗ qubit1 ⊗ qubit2 ⊗ ... ⊗ qubitn

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
        detection_metric: Optional[DetectionMetric] = None,
    ):
        """
        Initialize n-qubit experiment.

        Args:
            experimental_params: Physical and measurement parameters
            initial_circuit: QuantumCircuit to apply before evolution. If None, creates
                            default 2-qubit RY circuit with trainable parameters.
            final_circuit: QuantumCircuit to apply after evolution. If None, creates
                            default 2-qubit RY circuit with trainable parameters.
            detection_metric: Custom detection definition and loss metric for optimization.
                            If None, uses default detection: 1-P(0).
        """

        # Create default circuits if not provided (2-qubit RY gates)
        if initial_circuit is None:
            initial_circuit = create_ry_circuit(experimental_params.n_qubits, theta_values=np.pi/2)

        if final_circuit is None:
            final_circuit = create_ry_circuit(experimental_params.n_qubits, theta_values=-np.pi/2)

        self.experimental_params = experimental_params
        self.initial_circuit = initial_circuit
        self.final_circuit = final_circuit

        # Set detection metric, checks number of qubits given to the detection metric
        if detection_metric is None:
            self.detection_metric = DetectionMetric(n_qubits=experimental_params.n_qubits)
        else:
            if detection_metric.n_qubits != experimental_params.n_qubits:
                raise ValueError(
                    f"Detection metric n_qubits ({detection_metric.n_qubits}) must match experimental_params n_qubits ({experimental_params.n_qubits})"
                )
            self.detection_metric = detection_metric


        # Normalize qubit_levels to always be a list
        if isinstance(self.experimental_params.qubit_levels, int):
            self.experimental_params.qubit_levels = [self.experimental_params.qubit_levels] * self.experimental_params.n_qubits

        # Precompute total dimensions for QuTiP Qobj creation
        self.total_dims = [
            self.experimental_params.field_levels,
            self.experimental_params.cavity_levels
        ] + self.experimental_params.qubit_levels

        # Extract trainable parameters from both circuits
        self.trainable_params_initial = self.initial_circuit.get_trainable_parameters()
        self.trainable_params_final = self.final_circuit.get_trainable_parameters()

        # Caches
        self._cached_initial_state: Optional[qt.Qobj] = None
        self._cached_projectors: Dict[str, qt.Qobj] = {}
        self._cached_solvers: Dict[str, qt.MESolver] = {}
        self._cached_circuit_unitaries: Optional[tuple] = None

        # Callback
        self.callback = OptimizationCallback(save_every=1, save_best=True)

        # Initialize quantum objects
        self.__post_init__()

    def __post_init__(self):
        """Post-initialization to set up operators and hamiltonian."""
        # Disable auto_real_casting to avoid TracerBoolConversionError with JAX
        # When using JAX, QuTiP's trace() method tries to check `if self.isherm`
        # on traced states, which fails. Disabling this setting prevents the check.
        qt.settings.core["auto_real_casting"] = False  # type: ignore

        self._generate_operators()
        self._generate_hamiltonian()
        self._initialize_initial_state()
        self.detection_metric.initialize(self.operators["P_all"])


    @property
    def n_qubits(self) -> int:
        """Get the number of qubits in the experiment."""
        return self.experimental_params.n_qubits

    def _save_sweep_state(self) -> Dict[str, Any]:
        """
        Save current state for parameter sweeps.

        Returns:
            Dictionary with current chi, gamma, interactions, and cached objects
        """
        return {
            "chi": self.experimental_params.chi.copy() if isinstance(self.experimental_params.chi, list) else self.experimental_params.chi,
            "gamma": self.experimental_params.photon_cavity_coupling,
            "qubit_interactions": [
                QubitInteraction(
                    qubit_indices=interaction.qubit_indices,
                    chi=interaction.chi,
                    interaction_type=interaction.interaction_type
                )
                for interaction in self.experimental_params.physical_constants.qubit_interactions
            ] if self.experimental_params.physical_constants.qubit_interactions else []
        }

    def _restore_sweep_state(self, state: Dict[str, Any]) -> None:
        """
        Restore state after parameter sweep.

        Args:
            state: State dictionary from _save_sweep_state()
        """
        # Restore parameters
        self.experimental_params.physical_constants.chi = state["chi"]
        self.experimental_params.physical_constants.photon_cavity_coupling = state["gamma"]
        self.experimental_params.physical_constants.qubit_interactions = state["qubit_interactions"]

        # Regenerate Hamiltonian and clear solver caches
        self._generate_hamiltonian()
        self._cached_solvers.clear()

    def _update_chi_gamma(self, chi: Union[float, list], gamma: float, qubit_interactions: Optional[List] = None) -> None:
        """
        Temporarily update chi, gamma, and optionally qubit interactions for parameter sweeps.

        This method efficiently updates dispersive coupling, cavity decay, and qubit-qubit interactions
        without regenerating operators or initial state.

        Args:
            chi: Dispersive coupling constant(s)
            gamma: Photon-cavity coupling (cavity decay rate)
            qubit_interactions: Optional list of QubitInteraction objects
        """
        # Update parameters
        self.experimental_params.physical_constants.chi = chi
        self.experimental_params.physical_constants.photon_cavity_coupling = gamma
        if qubit_interactions is not None:
            self.experimental_params.physical_constants.qubit_interactions = qubit_interactions

        # Regenerate Hamiltonian with new parameters
        self._generate_hamiltonian()

        # Clear solver caches (they depend on Hamiltonian)
        self._cached_solvers.clear()

    def _generate_operators(self) -> None:
        """
        Generate operators for n-qubit system.

        Creates operators in composite Hilbert space:
        input_field ⊗ resonator_cavity ⊗ qubit1 ⊗ qubit2 ⊗ ... ⊗ qubitn

        Operators include:
        - Field and cavity creation/annihilation operators
        - Individual qubit Pauli operators (σx, σy, σz) for each qubit
        - Joint measurement projectors for all computational basis states
        - Individual qubit projectors based on detection criterion
        """
        # Get system dimensions
        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels
        qubit_levels = self.experimental_params.qubit_levels
        n_qubits = self.experimental_params.n_qubits

        # Generate n-qubit operators using utility function
        # Note: `generate_system_operators` expects (n_qubits, field_levels, cavity_levels, qubit_levels)
        self.operators = generate_system_operators(
            n_qubits, field_levels, cavity_levels, qubit_levels
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
                sigma1 = self.operators["sigma_z"][idx1]
                sigma2 = self.operators["sigma_z"][idx2]
            elif interaction_type == InteractionType.XX:
                # σx ⊗ σx interaction
                sigma1 = self.operators[f"sigma_x"][idx1]
                sigma2 = self.operators[f"sigma_x"][idx2]
            elif interaction_type == InteractionType.YY:
                # σy ⊗ σy interaction
                sigma1 = self.operators[f"sigma_y"][idx1]
                sigma2 = self.operators[f"sigma_y"][idx2]
            else:
                raise ValueError(f"Unknown interaction type: {interaction_type}")

            # Add interaction term: (χ/2) σᵢ ⊗ σⱼ
            H_interaction += qt.Qobj((chi / 2) * sigma1 * sigma2)  # type: ignore

        return H_interaction


    def _generate_hamiltonian(self) -> None:
        """
        Generate Hamiltonian for n-qubit system.

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
        chi_list = self.experimental_params.chi  # List of [chi1, chi2, ... , chin]
        sigma = self.experimental_params.inverse_pulse_width
        n_qubits = self.experimental_params.n_qubits

        # Extract individual chi values for each qubit
        # Type narrowing: chi is always a list for two-qubit experiments
        if isinstance(chi_list, list):
            chi = chi_list
        else:
            # Should not reach here due to __init__ validation, but type checker needs this
            chi = [chi_list] * n_qubits

        # Get operators
        a_in = self.operators["a_in"]
        a_in_dag = self.operators["a_in_dag"]
        a = self.operators["a"]
        a_dag = self.operators["a_dag"]

        # Qubit operators
        sigma_z = self.operators["sigma_z"]
        sigma_x = self.operators["sigma_x"]
        sigma_y = self.operators["sigma_y"]
        sigma_minus = self.operators["sigma_minus"]

        # Time-dependent coupling function arguments
        args = {"sigma": sigma}

        # Time-dependent cavity-field coupling Hamiltonian
        # H_c = (i/2)√γ (a_in† a - a_in a†)
        coupling_coeff = 1j / 2 * jnp.sqrt(gm)
        H_coupling = qt.Qobj(coupling_coeff * (a_in_dag * a - a_in * a_dag))  # type: ignore

        # Dispersive qubit-resonator interaction Hamiltonians
        # H_q = -Σᵢ χᵢ a†a σz_i
        H_dispersive_list = [qt.Qobj(-chi[i] * a_dag * a * sigma_z[i]) for i in range(n_qubits)]  # type: ignore
        H_dispersive = sum(H_dispersive_list)

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
        depolarizing = noise_config.depolarizing
        dephasing = noise_config.dephasing
        relaxation = noise_config.relaxation

        # Convert float parameters to lists of length n_qubits, or validate list lengths
        if isinstance(depolarizing, float):
            depolarizing = [depolarizing] * n_qubits
        elif isinstance(depolarizing, list):
            if len(depolarizing) != n_qubits:
                raise ValueError(
                    f"depolarizing list length ({len(depolarizing)}) must match n_qubits ({n_qubits})"
                )
        else:
            raise TypeError(f"depolarizing must be float or list, got {type(depolarizing)}")

        if isinstance(dephasing, float):
            dephasing = [dephasing] * n_qubits
        elif isinstance(dephasing, list):
            if len(dephasing) != n_qubits:
                raise ValueError(
                    f"dephasing list length ({len(dephasing)}) must match n_qubits ({n_qubits})"
                )
        else:
            raise TypeError(f"dephasing must be float or list, got {type(dephasing)}")

        if isinstance(relaxation, float):
            relaxation = [relaxation] * n_qubits
        elif isinstance(relaxation, list):
            if len(relaxation) != n_qubits:
                raise ValueError(
                    f"relaxation list length ({len(relaxation)}) must match n_qubits ({n_qubits})"
                )
        else:
            raise TypeError(f"relaxation must be float or list, got {type(relaxation)}")

        # Build Lindblad noise operators for each qubit using helper function
        lindblad_noise_q = [build_qubit_noise_operators(
            sigma_x=sigma_x[i],
            sigma_y=sigma_y[i],
            sigma_z=sigma_z[i],
            sigma_minus=sigma_minus[i],
            depolarizing_rate=depolarizing[i],
            dephasing_rate=dephasing[i],
            relaxation_rate=relaxation[i],
        ) for i in range(n_qubits)]

        # Combine noise operators for all qubits
        # Flatten list: collect all operators from each qubit
        lindblad_noise: List[Union[qt.Qobj, qt.QobjEvo]] = [
            op for i in range(n_qubits) for op in lindblad_noise_q[i]
        ]

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
            "dispersive_list": H_dispersive_list,
            "coupling": H_coupling,
        }

        self.lindblad_operators = {
            "interaction": interaction_ops,
            "no_interaction": no_interaction_ops,
        }

    def _initialize_initial_state(self) -> None:
        """
        Generate and cache the initial state of the system.
        """
        self._cached_initial_state = generate_initial_state(
            initial_config=self.experimental_params.initial_state,
            field_levels=self.experimental_params.field_levels,
            cavity_levels=self.experimental_params.cavity_levels,
            qubit_levels=self.experimental_params.qubit_levels,
            n_qubits=self.experimental_params.n_qubits,
        )

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

    def _prepare_circuit_unitaries(self) -> tuple:
        """
        Get embedded unitaries for initial and final circuits with their daggers.

        Computes and caches the full-space unitaries for both circuits and their
        conjugate transposes (daggers).

        Returns:
            Tuple of (initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag)
            embedded in composite space as QuTiP Qobj objects
        """
        # Get unitaries from circuits (as JAX arrays or QuTiP objects)
        initial_unitary_circuit = self.initial_circuit.get_unitary(qutip=False)
        final_unitary_circuit = self.final_circuit.get_unitary(qutip=False)

        # Embed into full composite space (JAX arrays) using utility function
        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels
        initial_unitary_jax = embed_circuit_unitary(initial_unitary_circuit, field_levels, cavity_levels)
        final_unitary_jax = embed_circuit_unitary(final_unitary_circuit, field_levels, cavity_levels)

        # Precompute daggers (conjugate transpose) in JAX
        initial_unitary_dag_jax = jnp.conj(initial_unitary_jax.T)
        final_unitary_dag_jax = jnp.conj(final_unitary_jax.T)

        # Convert to QuTiP objects once
        initial_unitary = qt.Qobj(initial_unitary_jax, dims=[self.total_dims, self.total_dims])
        initial_unitary_dag = qt.Qobj(initial_unitary_dag_jax, dims=[self.total_dims, self.total_dims])
        final_unitary = qt.Qobj(final_unitary_jax, dims=[self.total_dims, self.total_dims])
        final_unitary_dag = qt.Qobj(final_unitary_dag_jax, dims=[self.total_dims, self.total_dims])

        # Cache for reuse
        self._cached_circuit_unitaries = (initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag)

        return initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag

    def simulation(
        self,
        solver: qt.MESolver,
        rho: qt.Qobj,
        measurements: Union[List[float], np.ndarray, jnp.ndarray],
        args: Optional[Dict] = None,
        precomputed_unitaries: Optional[tuple] = None,
    ) -> jnp.ndarray:
        """
        JAX-compatible simulation for n-qubit system with customizable detection.

        Args:
            solver: Configured quantum evolution solver
            rho: Initial density matrix
            measurements: Array of measurement times (sorted)
            args: System parameters (optional)
            precomputed_unitaries: Optional tuple (U_initial, U_initial_dag, U_final, U_final_dag)
                                  to avoid recomputation

        Returns:
            Detection measure as JAX array
        """

        if args is None:
            args = {"sigma": self.experimental_params.inverse_pulse_width}

        # Get detection metric
        detection_metric = self.detection_metric

        # Get reset operators
        zipped_reset = zip(self.operators['measure_reset'], self.operators['measure_reset_dag'])

        # Get circuit unitaries
        if precomputed_unitaries is None:
            precomputed_unitaries = self._prepare_circuit_unitaries()
        initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag = precomputed_unitaries

        # Set initial state
        rho_current = rho

        # Initialise ouput list
        rho_list = []

        # Loop over measurement intervals
        for t0, t1 in zip(measurements[:-1], measurements[1:]):

            rho_after_circuit = initial_unitary * rho_current * initial_unitary_dag  # type: ignore

            evolution_result = solver.run(rho_after_circuit, [t0, t1], args=args)
           
            rho_evolved = evolution_result.states[-1]
            rho_final = final_unitary * rho_evolved * final_unitary_dag  # type: ignore

            # Reset the qubit
            rho_reset = [op * rho_final * op_dag for op,op_dag in zipped_reset]
            rho_current = sum(rho_reset)
            
            rho_list.append(rho_final)

        return rho_list


    def debug_simulation(
        self,
        solver: qt.MESolver,
        rho: qt.Qobj,
        measurements: Union[List[float], np.ndarray, jnp.ndarray],
        args: Optional[Dict] = None,
        precomputed_unitaries: Optional[tuple] = None,
    ) -> jnp.ndarray:
        """
        JAX-compatible simulation for n-qubit system with customizable detection.

        Args:
            solver: Configured quantum evolution solver
            rho: Initial density matrix
            measurements: Array of measurement times (sorted)
            args: System parameters (optional)
            precomputed_unitaries: Optional tuple (U_initial, U_initial_dag, U_final, U_final_dag)
                                  to avoid recomputation

        Returns:
            Detection measure as JAX array
        """

        if  not hasattr(self,'debug_times'):          #####################################
            self.debug_times = []
            self.step=0
        self.debug_times.append({ f'load_cached_parameters{self.step}' : t.time()})   ################################

        if args is None:
            args = {"sigma": self.experimental_params.inverse_pulse_width}

        # Get detection metric
        detection_metric = self.detection_metric

        # Get reset operators
        zipped_reset = zip(self.operators['measure_reset'], self.operators['measure_reset_dag'])

        # Get circuit unitaries
        if precomputed_unitaries is None:
            precomputed_unitaries = self._prepare_circuit_unitaries()
        initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag = precomputed_unitaries

        # Initial state
        rho_current = rho

        # Initialise ouput list
        rho_list = []

        self.debug_times.append({ f'start_simulation{self.step}' : t.time()})   ################################
        n_meas=0                      ############################

        # Loop over measurement intervals
        for t0, t1 in zip(measurements[:-1], measurements[1:]):

            rho_after_circuit = initial_unitary * rho_current * initial_unitary_dag  # type: ignore
            
            self.debug_times.append({ f'measurement{n_meas}:solver_{self.step}' : t.time()})   ################################

            evolution_result = solver.run(rho_after_circuit, [t0, t1], args=args)
            
            self.debug_times.append({ f'measurement{n_meas}:measure_{self.step}' : t.time()})   ################################
           
            rho_evolved = evolution_result.states[-1]
            rho_final = final_unitary * rho_evolved * final_unitary_dag  # type: ignore

            # Reset the qubit
            rho_reset = [op * rho_final * op_dag for op,op_dag in zipped_reset]
            rho_current = sum(rho_reset)
            
            rho_list.append(rho_final)

            n_meas+=1       ####################

        self.debug_times.append({ f'returning_simulation{self.step}' : t.time()})   ################################

        return rho_list

    def run_simulation(self, batch_size: int = 1, measurement_times = None, states_probabilities: bool = False, debug: bool=False) -> OptimizationCallback:
        """
        Run n-qubit sensing protocol with current parameters.

        This method executes the complete n-qubit quantum sensing workflow:
        - Applies rotations to all qubits independently
        - Evolves under n-qubit Hamiltonian
        - Performs measurements (joint or individual)
        - Computes detection measures with and without photon interaction

        Args:
            batch_size: Number of random realizations to average over for measurement
                       uncertainty (default: 1). Each realization uses a different
                       random shift in measurement times based on initial_time_uncertainty.
            measurement_times: Optional measurement times instead of the ones determined by the experimental parameters.
            states_probabilities: Whether to return the probabilities of the final quantum states (default: False)
            debug: Whether to enable detailed timing debug output (default: False)
        Returns:
            OptimizationCallback: Callback containing simulation results with:
                - Single epoch (epoch=1)
                - Current parameter values
                - Detection measures (detection_with, detection_without) averaged over batch
                - Metric value averaged over batch

        Raises:
            ValueError: If initial state cache is not initialized
        """
        # Get initial state and solvers

        if debug or states_probabilities:
            self.debug_times = []
            self.step=0
            self.debug_times.append({ f'initialize_solvers' : t.time()})

 
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")
        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()

        if debug:
            self.debug_times.append({ f'get_measurements' : t.time()})
 
        # Prepare measurement time realizations for batch averaging
        if measurement_times is not None:
            if isinstance(measurement_times, list):
                measurement_times = np.array(measurement_times)
            elif isinstance(measurement_times, np.ndarray):
                measurement_times = measurement_times
            elif isinstance(measurement_times, jnp.ndarray):
                measurement_times = np.array(measurement_times)
            else:
                raise TypeError(f"measurement_times must be list, np.ndarray, or jnp.ndarray, got {type(measurement_times)}")

            if measurement_times.shape[0] < 2:
                raise ValueError(f"measurement_times must be at least of lenght 2, with a starting time and a final time, got lenght {measurement_times.shape}")
            measurement_times_batch = self.experimental_params.get_measurement_times_with_uncertainty(batch_size, base_times=measurement_times)
        else:
            measurement_times_batch = self.experimental_params.get_measurement_times_with_uncertainty(
                batch_size
            )
        
        if measurement_times_batch.ndim == 1:
            measurement_sequences = [measurement_times_batch]
        else:
            measurement_sequences = [measurement_times_batch[i, :] for i in range(batch_size)]

        if states_probabilities and len(measurement_sequences[0]) != 2:
            raise ValueError("states_probabilities=True is only supported for single measurements")

        if debug:
            self.debug_times.append({ f'get_circuits' : t.time()})
 
        # Prepare circuit unitaries once for the entire batch
        circuit_unitaries = self._prepare_circuit_unitaries()

        # initialize batches
        batch_metric = []
        batch_detect_with = []
        batch_detect_without = []
        if states_probabilities:
            batch_for_prob = []

        if debug:
            self.debug_times.append({ f'start_measurement_loop' : t.time()})

        simulation_fn = self.debug_simulation if (debug or states_probabilities) else self.simulation
 
        for measurement_times in measurement_sequences:
            if debug:
                self.debug_times.append({ f'start_simulation_with{self.step}' : t.time()})
    
            # Simulation with photon interaction
            rho_with_list = simulation_fn(
                solver=solver_with,
                rho=rho0,
                measurements=measurement_times,
                precomputed_unitaries=circuit_unitaries,
            )

            if debug:
                self.debug_times.append({ f'start_simulation_no{self.step}' : t.time()})

            # Simulation without photon interaction (reference)
            rho_without_list = simulation_fn(
                solver=solver_without,
                rho=rho0,
                measurements=measurement_times,
                precomputed_unitaries=circuit_unitaries,
            )
            
            if debug:
                self.debug_times.append({ f'calculate_detection_metric{self.step}' : t.time()})

            if states_probabilities:
                batch_for_prob.append((rho_with_list, rho_without_list))

            metric_value, (detection_with, detection_without) = self.detection_metric(rho_with_list,rho_without_list)

            batch_metric.append(metric_value)            
            batch_detect_with.append(detection_with)
            batch_detect_without.append(detection_without)

            if debug:
                self.step += 1

        if debug:
            self.debug_times.append({ f'compute_means_from_batches' : t.time()})

        # Use detection metric's batching logic and then evaluate the configured metric.
        # With the default setup this metric can coincide with a simple difference (contrast),
        # but custom detection metrics may define any scalar objective.

        mean_metric = sum(batch_metric)/len(batch_metric)
        mean_detect_with = sum(batch_detect_with)/len(batch_detect_with)
        mean_detect_without = sum(batch_detect_without)/len(batch_detect_without)

        if states_probabilities:

            P_all = self.operators['P_all']
            prob_with = []
            prob_without = []

            for rho_list_with, rho_list_without in batch_for_prob:
                
                # We only compute probabilities for the first measurement in the sequence, as states_probabilities is only supported for single measurements
                rho_with = rho_list_with[0]      
                rho_without = rho_list_without[0]     
                
                prob_with.append([np.real((proj * rho_with * proj).tr()) for proj in P_all])                
                prob_without.append([np.real((proj * rho_without * proj).tr()) for proj in P_all])

            # Shape before averaging:
            #   prob_with / prob_without -> (batch_size, n_states)
            # Note: only first measurement is used (rho_list[0]), so no measurement axis.
            # Average across the batch axis only.
            prob_with = np.array(prob_with)
            prob_without = np.array(prob_without)

            # Resulting shape after mean(axis=0): (n_states,)
            avg_prob_with = np.mean(prob_with, axis=0).tolist()
            avg_prob_without = np.mean(prob_without, axis=0).tolist()
            state_prob_with = {format(i, f"0{self.n_qubits}b"): avg_prob_with[i] for i in range(len(avg_prob_with))}
            state_prob_without = {format(i, f"0{self.n_qubits}b"): avg_prob_without[i] for i in range(len(avg_prob_without))}

        if debug:
            self.debug_times.append({ f'save_callback' : t.time()})

        # Create callback with single epoch for simulation results
        callback = OptimizationCallback(save_every=1, save_best=True)

        if states_probabilities:
            
            callback(
                trainable_params_initial=self.trainable_params_initial,
                trainable_params_final=self.trainable_params_final,
                detection_with=float(mean_detect_with),
                detection_without=float(mean_detect_without),
                metric=float(mean_metric),
                state_probabilities_with=state_prob_with,
                state_probabilities_without=state_prob_without,
            )

        else:

            callback(
                trainable_params_initial=self.trainable_params_initial,
                trainable_params_final=self.trainable_params_final,
                detection_with=float(mean_detect_with),
                detection_without=float(mean_detect_without),
                metric=float(mean_metric),
            )

        if debug:
            self.debug_times.append({ f'end_time' : t.time()})

            temp = self.debug_times[0]        #############################
            print('\nDebug times for each step:')     ############################
            print('='*50)               ############################
            total_time=0
            for time in self.debug_times[1:]:                    ###############################
                                            ######################
                total_time += list(time.values())[0]-list(temp.values())[0]
                print('{:33}'.format(list(temp.keys())[0])+':'+'{:10.6f}'.format((list(time.values())[0]-list(temp.values())[0])))
                                            ######################

                temp = time                           ###########################

            print(f'\nTempo totale di simulazione = {total_time}')
            print('='*50+'\n\n')

        # Cleanup temporary debug attributes to free memory
        for _attr in ("debug_times", "step"):
            if hasattr(self, _attr):
                try:
                    delattr(self, _attr)
                except Exception:
                    pass

        return callback

    def run_simulation_with_probabilities(
        self, t_start: float = -5.0, t_end: float = 5.0
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Run simulation and return all final state probabilities and detection metrics.

        This method computes final state probabilities after evolution, then uses
        the configured metric to compute detection measures and the metric.
        Useful for parameter sweeps and reproducing notebook experiments.

        Args:
            t_start: Evolution start time (default: -5.0)
            t_end: Evolution end time (default: 5.0)

        Returns:
            Dictionary containing:
                - 'probs_with': probability of finding the qubit in excited state with photon interaction
                - 'probs_without': probability of finding the qubit in excited state without photon interaction
                - 'detection_with': Detection measure with photon
                - 'detection_without': Detection measure without photon
                - 'metric': Value of the configured optimization metric

        Example:
            >>> experiment = Experiment(exp_params)
            >>> results = experiment.run_simulation_with_probabilities()
            >>> print(f"P(11) with photon: {results['probs_with']['11']:.4f}")
            >>> print(f"Metric: {results['metric']:.4f}")
        """
        # Get initial state and solvers
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")
        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()

        # Prepare circuit unitaries as QuTiP objects
        initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag = self._prepare_circuit_unitaries()

        # Compute final state with photon after evolution
        rho_after_circuit = initial_unitary * rho0 * initial_unitary_dag  # type: ignore
        evolution_result_with = solver_with.run(rho_after_circuit, [t_start, t_end])
        rho_evolved_with = evolution_result_with.states[-1]
        rho_final_with = final_unitary * rho_evolved_with * final_unitary_dag  # type: ignore
        probs_with = self.measure_all_states(rho_final_with)

        # Compute final state without photon after evolution
        rho_after_circuit = initial_unitary * rho0 * initial_unitary_dag  # type: ignore
        evolution_result_without = solver_without.run(rho_after_circuit, [t_start, t_end])
        rho_evolved_without = evolution_result_without.states[-1]
        rho_final_without = final_unitary * rho_evolved_without * final_unitary_dag  # type: ignore
        probs_without = self.measure_all_states(rho_final_without)

        # Build detection inputs in the shape expected by the selected metric mode.
        detection_name = self.detection_metric.detection_name
        n_qubits = self.experimental_params.n_qubits

        if detection_name in ["min fidelity", "max trace distance"]:
            detection_with, detection_without = self.detection_metric.batching_logic(
                [[rho_final_with]], [[rho_final_without]]
            )
        elif detection_name == "max computational distance":
            all_states = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
            probs_with_vector = jnp.array([probs_with[state] for state in all_states], dtype=float)
            probs_without_vector = jnp.array([probs_without[state] for state in all_states], dtype=float)
            detection_with, detection_without = self.detection_metric.batching_logic(
                [[probs_with_vector]], [[probs_without_vector]]
            )
        else:
            detection_states = self.detection_metric.detection_states
            detect_with_prob = float(sum(probs_with[state] for state in detection_states))
            detect_without_prob = float(sum(probs_without[state] for state in detection_states))
            detection_with, detection_without = self.detection_metric.batching_logic(
                [detect_with_prob], [detect_without_prob]
            )

        metric = self.detection_metric.metric(detection_with, detection_without)

        return {
            "probs_with": probs_with,
            "probs_without": probs_without,
            "detection_with": float(detection_with),
            "detection_without": float(detection_without),
            "metric": float(metric),
        }

    def time_evolution(
        self,
        n_points: int = 200,
        with_interaction: bool = True,
        measurement_protocol: Optional[MeasurementProtocol] = None,
    ) -> "TimeEvolutionResults":
        """
        Compute time evolution of n-qubit probabilities.

        Simulates the quantum system evolution over time using the measurement protocol times.
        Returns probability distributions for all n-qubit states (|0...0⟩, |0...1⟩, ..., |1...1⟩).
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

                - probabilities: Dict with states as keys (e.g., '0', '1' for 1 qubit, '00', '01', '10', '11' for 2 qubits)

                - pulse_shape: Pulse envelope u(t), shape (n_points,)

                - measurement_times: Measurement time points

                - cavity_population: Cavity population <a†a>, shape (n_points,)

                - field_population: External field population <a_in†a_in>, shape (n_points,)

        Example: (for n_qubits=2)
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

        metric_name = self.detection_metric.name
        metric_name_lower = metric_name.lower()
        requires_pair = any(
            key in metric_name_lower
            for key in (
                "fidelity",
                "trace distance",
                "computational distance",
                "custom matrix distance",
            )
        )

        # Use provided measurement protocol or default from experimental parameters
        if measurement_protocol is None:
            measurement_protocol = self.experimental_params.measurement

        # Get measurement times from protocol
        measurement_times = np.array(measurement_protocol.measurement_times)
        # Use measurement times for start and end
        t_start = float(measurement_times[0])
        t_end = float(measurement_times[-1])

        # Get initial state and solvers
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")

        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()

        # Prepare circuit unitaries as QuTiP objects (including daggers)
        initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag = self._prepare_circuit_unitaries()

        # Apply initial circuit  # type: ignore

        # Get number operators for population calculation
        n_cavity = self.operators["a_dag"] * self.operators["a"]
        n_field = self.operators["a_in_dag"] * self.operators["a_in"]

        # Get measurement operators and sigma
        measure_reset = self.operators["measure_reset"]
        measure_reset_dag = self.operators["measure_reset_dag"]

        args = {"sigma": self.experimental_params.inverse_pulse_width}

        # Get number of qubits and generate all possible states
        n_qubits = self.experimental_params.n_qubits
        all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)] ################
        qubit_indices = list(range(0, n_qubits))

        # Storage for results
        all_times = []
        detection_list = []
        detection_probability_list = []
        cavity_population_list = []
        field_population_list = []

        # Set up measurements
        intermediate_meas_times = measurement_times[(measurement_times > t_start) & (measurement_times < t_end)]
        segment_starts = [t_start] + list(intermediate_meas_times)
        segment_ends = list(intermediate_meas_times) + [t_end]

        # Evolution
        rho_with = rho0
        rho_without = rho0

        for seg_start, seg_end in zip(segment_starts, segment_ends):
            # Number of points for this segment
            seg_fraction = (seg_end - seg_start) / (t_end - t_start)
            seg_n_points = max(2, int(n_points * seg_fraction))

            # Apply initial circuit for measurement
            rho_with = initial_unitary * rho_with * initial_unitary_dag
            if requires_pair or not with_interaction:
                rho_without = initial_unitary * rho_without * initial_unitary_dag

            # Evolve segment
            seg_times = np.linspace(seg_start, seg_end, seg_n_points)
            primary_solver = solver_with if (with_interaction or requires_pair) else solver_without
            result_with = primary_solver.run(rho_with, tlist=seg_times, args=args)
            result_without = None
            if requires_pair:
                result_without = solver_without.run(rho_without, tlist=seg_times, args=args)

            # Extract data for this segment
            for i, rho_t in enumerate(result_with.states):

                # Apply final circuit for measurement
                rho_meas_with = final_unitary * rho_t * final_unitary_dag  # type: ignore
                rho_meas_without = None
                if result_without is not None:
                    rho_without_t = result_without.states[i]
                    rho_meas_without = final_unitary * rho_without_t * final_unitary_dag  # type: ignore
                elif not with_interaction:
                    rho_meas_without = rho_meas_with

                # Measure detection with the configured metric
                epoch_fraction = (seg_times[i] - t_start) / (t_end - t_start)
                if requires_pair:
                    metric_value, _ = self.detection_metric(
                        [rho_meas_with],
                        [rho_meas_without],
                        epoch_fraction,
                    )
                    detection_value = metric_value
                else:
                    metric_value, (detect_value, _) = self.detection_metric(
                        [rho_meas_with],
                        [rho_meas_with],
                        epoch_fraction,
                    )
                    detection_value = detect_value
                    detection_probability_list.append(detect_value)

                detection_list.append(detection_value)

                all_times.append(seg_times[i])

                # Calculate populations (take real part since expectation values should be real)
                population_rho = rho_t if with_interaction else (rho_without_t if rho_meas_without is not None else rho_t)
                cavity_pop = float(np.real(qt.expect(n_cavity, population_rho)))
                field_pop = float(np.real(qt.expect(n_field, population_rho)))
                cavity_population_list.append(cavity_pop)
                field_population_list.append(field_pop)

            # Update system after actual measurement
            reset_with = [op * rho_meas_with * op_dag for op, op_dag in zip(measure_reset, measure_reset_dag)]
            rho_with = sum(reset_with)
            if rho_meas_without is not None:
                reset_without = [op * rho_meas_without * op_dag for op, op_dag in zip(measure_reset, measure_reset_dag)]
                rho_without = sum(reset_without)

        times = np.array(all_times)
        # Compute pulse shape using the same u0 function as visualization
        pulse_shape = np.array([float(u0(t, sigma=args["sigma"])) for t in times])

        # Build probabilities dictionary
        probabilities = {"detection_measure": np.array(detection_list)}
        if detection_probability_list:
            probabilities["detection_probability"] = np.array(detection_probability_list)

        # Import at runtime to avoid circular dependency
        from qsopt.utils.results import TimeEvolutionResults

        return TimeEvolutionResults(
            times=times,
            probabilities=probabilities,
            pulse_shape=pulse_shape,
            measurement_times=measurement_times,
            cavity_population=np.array(cavity_population_list),
            field_population=np.array(field_population_list),
            metadata={
                "chi": self.experimental_params.chi,
                "gamma": self.experimental_params.photon_cavity_coupling,
                "with_interaction": with_interaction,
                "n_qubits": n_qubits,
                "detection_metric" : self.detection_metric.name,
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
        initial_values: Optional[List[float]] = None,
        optimizer = None,
        optimize_measurement_times: bool = False,
        renormalize_grad: Optional[Union[bool,float]] = False,
        noisy_training: Optional[float] = None,
        final_results: bool = True,
        hot_start: bool = False,
        tot_steps: Optional[int] = None
    ) -> OptimizationCallback:
        """
        Optimize rotation angles to maximize the detection metric.

        This method performs JAX-based gradient descent over rotation angles
        using the sequential measurement protocol.

        Args:
            num_steps: Maximum number of optimization steps (default: 100)
            batch_size: Number of random realizations for measurement uncertainty per step (default: 1)
            tolerance: Convergence threshold for gradient norm (default: 1e-6)
            verbose: Print progress information (default: True)
            verbose_step: Step interval for printing progress (default: 10)
            callback: Optional callback to track optimization progress.
                    If None, uses the experiment's default callback.
            initial_values: Optional initial circuit parameters as list of floats.
                    If None, uses current values from circuits.
            optimizer: Optional optax optimizer (e.g., optax.adam(0.01), optax.sgd(0.5)).
                    If None, uses SGD.
            optimize_measurement_times: If True, also optimizes the measurement times along with the circuit parameters. (default: False)
            renormalize_grad: Renormalizes the gradients to be within a certain radius. (default: 1)
                    If False (0), does not renormalize the gradients.
            noisy_training: float, adds noise to the gradients during optimization.
                    If a float is given, it is used as the standard deviation relative to the average gradient. (default: None)
            hot_start: If True, continues optimization from the last parameters and optimizer state in the callback.
                    If either the optimizer or the params are given they override the hot start values. (default: False)
            tot_steps: Total number of optimization steps to run, it's used to give the epoch percentage to the detection metric. 
                    It's useful if the optimization is divided in multiple runs.
                    If None, uses num_steps. (default: None)
            final_results: If True, stores the final optimization results in the callback. (default: True)

        Returns:
            OptimizationCallback with full optimization history, including
            per-step metric values and detection measures.

        Example:
        >>> # Optimize with default 1-P(00) detection
        >>> callback = experiment.optimize_rotations(num_steps=200, batch_size=10)
        >>>
        >>> # With custom detection criterion
        >>> from qsopt.utils.loss_functions import DetectionMetric
        >>> detection = DetectionMetric(metric=(lambda x: x), name='state list', detection_param=['11'])  # Detect |11⟩
        >>> callback = experiment.optimize_rotations(
        ...     num_steps=100,
        ...     detection_metric=detection
        ... )
        """
        import jax
        import optax

        start_time = t.time()

        # Use provided callback or default
        if callback is None:
            callback = self.callback
                
        loaded_grads = None

        # Reset callback only at start of new optimizations
        if hot_start:

            if verbose: print("Starting hot start optimization, trying to load last parameters, optimizer state, and gradients from callback:")

            if initial_values is not None:
                warnings.warn("Starting parameters were given but where overwritten by the hot start.")
            
            loaded_initial, loaded_final, epoch = callback.get_params()
            initial_values = [float(p) for p in np.asarray(loaded_initial, dtype=float).reshape(-1)] + [
                float(p) for p in np.asarray(loaded_final, dtype=float).reshape(-1)
            ]
            if verbose: print("- Parameters LOADED")

            opt_state, loaded_grads = callback.get_opt_state()
            if verbose: print(
                f"- Gradients LOADED\n- Optimizer state LOADED")

            start_step = epoch
            num_steps = start_step + num_steps
            if verbose: print(f"Resuming from epoch {start_step}, running until epoch {num_steps}")
            
        else:
            start_step = 0
            callback.reset()

        if tot_steps is None:
            tot_steps = num_steps
        elif tot_steps < num_steps:
            raise ValueError(f"tot_steps should be greater than or equal to num_steps, got tot_steps={tot_steps} and num_steps={num_steps}")

        if isinstance(noisy_training, (int, float)) and noisy_training > 0.01:
            raise ValueError(f"noisy_training should be a boolean or a float representing the standard deviation of the noise relative to the gradient norm. It shouldn't exceed 1% Got value: {noisy_training}")
        elif noisy_training is None:
            noisy_training=0

        # Count total trainable parameters from both circuits
        n_initial = self.initial_circuit.count_trainable_parameters()
        n_final = self.final_circuit.count_trainable_parameters()
        n_total = n_initial + n_final

        if n_total == 0:
            raise ValueError("No trainable parameters found in circuits")

        if self.experimental_params.measurement_times.ndim != 1 or self.experimental_params.measurement_times.size < 2:
            raise ValueError("measurement_times must be a 1D array with at least 2 time points")

        # Initialize parameter vector
        if initial_values is not None:
            if len(initial_values) != n_total:
                raise ValueError(
                    f"initial_values must contain exactly {n_total} angles, got {len(initial_values)}"
                )
            self.initial_circuit.set_trainable_parameters(initial_values[:n_initial])
            self.final_circuit.set_trainable_parameters(initial_values[n_initial:])
        else:
            # Get current values from circuits
            initial_params = self.initial_circuit.get_trainable_parameters()
            final_params = self.final_circuit.get_trainable_parameters()
            initial_values = [float(p) for p in initial_params] + [float(p) for p in final_params]

        params = jnp.array(initial_values, dtype=float)
            
        # Initialize optimizer for new optimizations (default to SGD with lr=0.5 if not provided)
        if optimizer is None:
            optimizer = optax.sgd(learning_rate=0.5)
        if not hot_start:
            opt_state = optimizer.init(params)
        elif opt_state is None:
            warnings.warn(
                "No optimizer state available for hot start; reinitializing optimizer state."
            )
            opt_state = optimizer.init(params)
        elif loaded_grads is None:
            warnings.warn(
                "No gradients available for hot start; continuing from loaded optimizer state without pre-update. One epoch will be repeated."
            )
        else:
            try:
                updates, opt_state = optimizer.update(loaded_grads, opt_state, params)
                params = optax.apply_updates(params, updates)
            except Exception as e:
                warnings.warn(f"An error occurred while using the hot start optimizer state to update the given optimizer, the optimizer state will be ignored and the optimizer will be reinitialized:\n {e}")
                opt_state = optimizer.init(params)

        # Get initial state, solvers and detection metric
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")

        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()
        detection_metric = self.detection_metric

        # Define objective function with explicit uncertainty input.
        # Signature order is kept future-proof for optional optimization over times.
        def coupled_simulation(circuit_unitaries, measurement_times, measurement_noise, epoch_fraction: float):
            """Single-realization of the two simulations with the same parameters and noise."""

            noisy_measurement_times = measurement_times + measurement_noise

            rho_with_list = self.simulation(
                solver_with,
                rho0,
                noisy_measurement_times,
                precomputed_unitaries=circuit_unitaries,
            )

            rho_without_list = self.simulation(
                solver_without,
                rho0,
                noisy_measurement_times,
                precomputed_unitaries=circuit_unitaries,
            )

            metric_value, (detection_with, detection_without) = self.detection_metric(rho_with_list, rho_without_list, epoch_fraction)

            return metric_value, detection_with , detection_without


        static_args = [3]  # objective_function static arg: epoch_fraction
        time_uncertainty = float(self.experimental_params.initial_time_uncertainty)

        if not optimize_measurement_times:
            static_args.append(1)  # add objective_function static arg: measurement_times
            base_measurement_times = tuple(
                float(x) for x in np.asarray(self.experimental_params.measurement_times, dtype=float)
            )
        else:
            base_measurement_times = jnp.asarray(self.experimental_params.measurement_times, dtype=float)

        if time_uncertainty == 0:

            if batch_size != 1:
                if verbose:
                    warnings.warn(f"Batch size > 1 has no effect when there is no measurement uncertainty. Setting batch size to 1.")
                batch_size = 1

            static_args.append(2)  # objective_function arg: measurement_noise_batch
            zero_uncertainty_batch = 0.0

            def get_noise_batch():
                    return zero_uncertainty_batch

            def objective_function(circuit_params, measurement_times, measurement_noise_batch, epoch_fraction: float):
                """Objective with no uncertainty."""

                measurement_times = np.asarray(measurement_times, dtype=float)

                # Compute circuit unitaries
                self.initial_circuit.set_trainable_parameters(circuit_params[:n_initial])
                self.final_circuit.set_trainable_parameters(circuit_params[n_initial:])
                circuit_unitaries = self._prepare_circuit_unitaries()

                metric, detect_with, detect_without = coupled_simulation(circuit_unitaries, measurement_times, measurement_noise_batch, epoch_fraction)

                return -metric, (detect_with, detect_without, metric)

        else:
            
            if batch_size < 16 and verbose:
                warnings.warn(f"Using a small batch size of {batch_size} for optimization with measurement uncertainty may lead to noisy gradients and slow convergence. Consider increasing the batch size for better performance.")

            def get_noise_batch():
                measurement_uncertainty_batch = jnp.asarray(
                    np.random.uniform(-time_uncertainty, time_uncertainty, size=batch_size),
                    dtype=float,
                )
                return measurement_uncertainty_batch
            
            def objective_function(circuit_params, measurement_times, measurement_noise_batch, epoch_fraction: float):
                """Batch vmapped objective where vectorization happens only over uncertainty."""

                measurement_times = jnp.asarray(measurement_times, dtype=float)

                # Compute circuit unitaries
                self.initial_circuit.set_trainable_parameters(circuit_params[:n_initial])
                self.final_circuit.set_trainable_parameters(circuit_params[n_initial:])
                circuit_unitaries = self._prepare_circuit_unitaries()

                batch_metric, batch_detect_with, batch_detect_without = jax.vmap(
                    coupled_simulation,
                    in_axes=(None, None, 0, None),
                )(circuit_unitaries, measurement_times, measurement_noise_batch, epoch_fraction)

                mean_metric = jnp.mean(batch_metric)
                mean_detect_with = jnp.mean(batch_detect_with)
                mean_detect_without = jnp.mean(batch_detect_without)

                return -mean_metric, (mean_detect_with, mean_detect_without, mean_metric)

        jitted_objective = jit(objective_function, static_argnums=tuple(static_args))


        # Get detection description for verbose output
        detection_metric_name = detection_metric.name

        if verbose:
            print(f"Configuration:")
            print(f"    Max iterations: {num_steps}")
            print(f"    Batch size: {batch_size}")
            print(f"    Convergence tolerance: {tolerance:.2e}")
            print(f"    Detection metric:\n{detection_metric_name}")
            print(f"    Trainable parameters: {n_total} ({n_initial} initial circuit + {n_final} final circuit)")
            print(f"    Initial parameter values:")

            initial_vals = np.asarray(params, dtype=float)
            setup_gates = [gate for gate in self.initial_circuit._gates if gate.has_parameter() and gate._parameter.trainable]
            reset_gates = [gate for gate in self.final_circuit._gates if gate.has_parameter() and gate._parameter.trainable]
            
            for i, val in enumerate(initial_vals):
                if i < n_initial :
                    circuit_type = "setup" 
                    print(f"        param{(f"{i}"+"."):<3} {(f"{circuit_type}_{setup_gates[i].__repr__(params=False)}"):<13}= {val:<6.3f} rad ({np.rad2deg(val):.1f}°)")
                else:
                    circuit_type = "reset"
                    print(f"        param{(f"{i}"+"."):<3} {(f"{circuit_type}_{reset_gates[i-n_initial].__repr__(params=False)}"):<13}= {val:<6.3f} rad ({np.rad2deg(val):.1f}°)")

            uncertainty = time_uncertainty
            if uncertainty > 0:
                spec = self.experimental_params.initial_time_uncertainty_spec
                extra = f" (specified as '{spec}')" if isinstance(spec, str) else ""
                print(f"    Measurement uncertainty: ±{uncertainty:.3f}{extra}")

            # Build header based on number of parameters (up to 4 each)
            header_parts = [f"{'Step':<6}"]
            n_init_show = min(n_initial, 4)
            n_final_show = min(n_final, 4)
            for i in range(n_init_show):
                header_parts.append(f"setup{i}_{setup_gates[i].__repr__(params=False):<8}")
            for i in range(n_final_show):
                header_parts.append(f"reset{i}_{reset_gates[i].__repr__(params=False):<8}")
            header_parts.extend([f"{'Metric':<12}", f"{'Grad Norm':<12}", "Time"])

            header = "".join(header_parts)
            print("=" * (5+len(header)))
            print(header)
            print("-" * (5+len(header)))

        best_metric = -np.inf
        best_params = jnp.array(params)

        # Initialize variables
        step = start_step
        grad_norm = float("inf")

        for step in range(start_step, num_steps):

            measurement_uncertainty_batch = get_noise_batch()

            # Compute gradients using JAX autodiff
            grads, (detection_with, detection_without, step_metric) = jax.grad(
                jitted_objective, has_aux=True
            )(params, base_measurement_times, measurement_uncertainty_batch,epoch_fraction=step/tot_steps)

            # Track best parameters
            if step_metric > best_metric:
                best_metric = step_metric
                best_params = jnp.array(params)

            #Renormalize gradient inside a set interval, to avoid too large steps in the limited (2pi)^n_params parameter space.
            grad_norm = float(jnp.linalg.norm(grads))
            if renormalize_grad and grad_norm > 0:
                new_norm = jnp.tanh(grad_norm/renormalize_grad) * renormalize_grad
                grads = grads * new_norm/grad_norm
                grad_norm = new_norm

            # Call callback to track progress
            callback(
                trainable_params_initial=params[:n_initial], 
                trainable_params_final=params[n_initial:],
                detection_with=float(detection_with),
                detection_without=float(detection_without),
                metric=float(step_metric),
                optimizer_state=opt_state,
                grads=grads,
            )

            # Progress output
            if verbose and (step % verbose_step == 0 or grad_norm < tolerance or step-start_step <3):
                new_time = t.time() - start_time
                # Build parameter display (up to 4 each)
                n_init_show = min(n_initial, 4)
                n_final_show = min(n_final, 4)
                param_vals = np.asarray(params, dtype=float)

                output_parts = [f"{step:<6}"]
                for i in range(n_init_show):
                    output_parts.append(f"{param_vals[i]:<15.6f}")
                for i in range(n_final_show):
                    output_parts.append(f"{param_vals[n_initial + i]:<15.6f}")
                output_parts.extend([f"{float(step_metric):<12.6f}", f"{grad_norm:<12.2e}",f"{t.strftime("%Hh%Mm%Ss", t.gmtime(new_time))}"])
                print("".join(output_parts))

            # Convergence check
            if grad_norm < tolerance:
                break

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            if noisy_training!= 0:
                params += jnp.asarray(np.random.normal(0, noisy_training*grad_norm, size=params.shape), dtype=float)

        # Ensure best parameters are set at the end
        best_values = np.asarray(best_params, dtype=float)
        best_initial = [best_values[i] for i in range(n_initial)]
        best_final = [best_values[i] for i in range(n_initial, n_total)]
        self.initial_circuit.set_trainable_parameters(best_initial)
        self.final_circuit.set_trainable_parameters(best_final)

        # Run simulation to get probabilities for each state with the best parameters
        if time_uncertainty != 0 and batch_size < 16:
            batch_size = 16 # Use a larger batch size for final evaluation to reduce noise in results when uncertainty is present

        final_results_callback = self.run_simulation(batch_size=batch_size, 
                                            measurement_times=[base_measurement_times[0], base_measurement_times[-1]], # run_simulation only accepts 1 measurement
                                            states_probabilities=True,
                                            debug=False
                                            )

        state_probs_with = final_results_callback.state_probabilities_with
        state_probs_without = final_results_callback.state_probabilities_without
            
        callback.set_measurement_protocol(state_probs_with, state_probs_without)

        if verbose:
            print("=" * (5+len(header)))
            print(f"Final gradient norm: {grad_norm:.2e}")
            print(f"Best metric: {best_metric:.6f}")
            print(f"Best parameters:")
            for i, val in enumerate(best_values):
                if i < n_initial:
                    circuit_type = "setup"  
                    print(f"    param{i}. {circuit_type}_{setup_gates[i]}={val:.3f} rad ({np.rad2deg(val):.1f}°)")
                else:
                    circuit_type = "reset"
                    print(f"    param{i}. {circuit_type}_{reset_gates[i-n_initial]}={val:.3f} rad ({np.rad2deg(val):.1f}°)")

        # Set convergence information in callback
        callback.set_convergence_info(
            converged=float(grad_norm) < tolerance, final_grad_norm=float(grad_norm)
        )


        return callback


    def optimize_measurement_times(
        self,
        resolution: Optional[int] = None,
        mode: str = "continuous",
        batch_size: int = 1,
        verbose: bool = True,
        min_interval: Optional[float] = None,
        max_interval: Optional[float] = None,
    ) -> Dict[str, Union[np.ndarray, float, str, int]]:
        """Optimize measurement interval via landscape search.

        This helper uses the :meth:`compute_time_interval_landscape` method
        and applies the best-performing interval to the experiment configuration.
        Current rotation angles from circuits are used automatically.

        Args:
            resolution: Number of interval samples to evaluate (minimum 2).
                Default: 50 if None.
            mode: Interval sampling mode, ``'continuous'`` or ``'discrete'``.
            batch_size: Number of uncertainty realizations per interval.
            verbose: Print progress feedback when True.
            min_interval: Optional lower bound on the interval sweep. If None,
                uses total_time/100 for continuous mode or total_time/resolution
                for discrete mode.
            max_interval: Optional upper bound on the interval sweep. If None,
                uses total evolution time.

        Returns:
            Dictionary returned by :meth:`compute_time_interval_landscape` with additional keys:
                - ``'best_interval'``: Interval delivering the highest metric.
                - ``'best_metric'``: Maximum metric observed.
                - ``'best_index'``: Index of the optimal interval in the sampled array.

        Example:
            >>> # Optimize measurement interval with current rotation angles
            >>> time_callback = experiment.optimize_measurement_times(
            ...     resolution=30,
            ...     mode='discrete',
            ...     batch_size=10
            ... )
            >>> print(f"Best interval: {time_callback['best_interval']:.3f}")
            >>> print(f"Best metric: {time_callback['best_metric']:.6f}")
        """

        # Set default resolution
        resolved_resolution = resolution if resolution is not None else 50
        resolved_resolution = int(resolved_resolution)

        # Resolve min/max intervals
        resolved_min_interval = float(min_interval) if min_interval is not None else None
        resolved_max_interval = float(max_interval) if max_interval is not None else None

        if (
            resolved_min_interval is not None
            and resolved_max_interval is not None
            and resolved_min_interval > resolved_max_interval
        ):
            resolved_min_interval, resolved_max_interval = (
                resolved_max_interval,
                resolved_min_interval,
            )

        # Compute landscape using the class method
        results = self.compute_time_interval_landscape(
            resolution=resolved_resolution,
            mode=mode,
            batch_size=batch_size,
            verbose=verbose,
            min_interval=resolved_min_interval,
            max_interval=resolved_max_interval,
        )

        # Find best interval
        metric_vals_np = np.asarray(results["metric_vals"], dtype=float)
        interval_vals_np = np.asarray(results["interval_vals"], dtype=float)
        best_index = int(np.argmax(metric_vals_np))
        best_interval = float(interval_vals_np[best_index])
        best_metric = float(metric_vals_np[best_index])

        # Apply best interval to experimental parameters
        self.experimental_params.measurement.time_interval = best_interval
        self.experimental_params.measurement.measurement_times = None
        self.experimental_params._update_measurement_times()

        # Add best results to output
        results_with_best = dict(results)
        results_with_best["best_interval"] = best_interval
        results_with_best["best_metric"] = best_metric
        results_with_best["best_index"] = best_index

        if verbose:
            n_measurements = np.asarray(results["n_measurements"], dtype=int)
            print(f"\nOptimization complete:")
            print(f"  Best interval: {best_interval:.4f}")
            print(f"  Best metric: {best_metric:.6f}")
            print(f"  Number of measurements: {n_measurements[best_index]}")

        return results_with_best

    def compute_time_interval_landscape(
        self,
        resolution: int = 50,
        mode: str = "continuous",
        batch_size: int = 1,
        verbose: bool = True,
        min_interval: Optional[float] = None,
        max_interval: Optional[float] = None,
    ) -> Dict[str, Union[np.ndarray, float, str, int]]:
        """
        Compute detection metric landscape vs measurement time interval.

        This method evaluates how the detection metric varies with the time interval
        between measurements, keeping circuit parameters fixed. Two modes
        are supported:

        1. **Continuous mode**: Time interval varies continuously from a minimum
           value to the full evolution time (final_time - initial_time).

        2. **Discrete mode**: Time interval is restricted to integer fractions
           of the full evolution time (e.g., T/2, T/3, T/4, ..., T/N).

        The method supports batch averaging to account for initial_time_uncertainty,
        providing more realistic simulations that include timing jitter effects.

        Workflow for each time interval:
            1. Set time_interval in experimental_params
            2. Recompute measurement times based on initial_time, final_time, interval
            3. Run quantum simulation with batch averaging (if batch_size > 1)
            4. Calculate average metric value across realizations
            5. Store results in 1D array

        Args:
            resolution: Number of time interval values to evaluate. Default: 50.
            mode: Computation mode - either 'continuous' or 'discrete'.
                - 'continuous': Linearly spaced intervals from min to max
                - 'discrete': Integer fractions of total time (1/2, 1/3, ..., 1/N)
                Default: 'continuous'.
            batch_size: Number of random realizations to average over for
                measurement uncertainty. Recommended: ≥10 for realistic results
                when initial_time_uncertainty > 0. Default: 1.
            verbose: Print progress information. Default: True.
            min_interval: Minimum interval to consider.
                - Continuous mode: defaults to total_time / 100 when None.
                - Discrete mode: defaults to total_time / resolution when None.
            max_interval: Maximum interval to consider.
                Defaults to total_time when None. In discrete mode, constraints
                are enforced by rounding up to the nearest valid measurement count.

        Returns:
            Dictionary containing:
                - 'interval_vals': 1D array of time interval values (shape: [resolution])
                - 'metric_vals': 1D array of metric values (shape: [resolution])
                - 'detection_with': 1D array of detection measures with photon (shape: [resolution])
                - 'detection_without': 1D array of detection measures without photon (shape: [resolution])
                - 'n_measurements': 1D array of number of measurements per interval (shape: [resolution])
                - 'mode': Computation mode used (str)
                - 'batch_size': Batch size used (int)
                - 'initial_time_uncertainty': Resolved uncertainty value from exp_params (float)
                - 'initial_time_uncertainty_spec': Raw specification (float or str)

        Raises:
            ValueError: If mode is not 'continuous' or 'discrete'
            ValueError: If resolution < 2

        Example:
            >>> # Create experiment with circuits
            >>> from qsopt.core.circuit import create_ry_circuit_layer
            >>> exp_params = ExperimentalParameters()
            >>> exp_params.measurement.initial_time = -5.0
            >>> exp_params.measurement.final_time = 5.0
            >>> exp_params.measurement.initial_time_uncertainty = 0.1
            >>>
            >>> initial_circuit = create_ry_circuit_layer(n_qubits=1, theta_values=[np.pi/2])
            >>> final_circuit = create_ry_circuit_layer(n_qubits=1, theta_values=[-np.pi/2])
            >>> experiment = Experiment(exp_params, initial_circuit, final_circuit)
            >>>
            >>> # Continuous mode with uncertainty
            >>> data = experiment.compute_time_interval_landscape(
            ...     resolution=50,
            ...     mode='continuous',
            ...     batch_size=20  # Average over 20 realizations
            ... )
            >>>
            >>> # Find optimal interval
            >>> optimal_idx = np.argmax(data['metric_vals'])
            >>> optimal_interval = data['interval_vals'][optimal_idx]
            >>> print(f"Optimal interval: {optimal_interval:.4f}")

        Notes:
            - When batch_size > 1 and initial_time_uncertainty > 0, each simulation
              point averages over multiple realizations with random timing shifts.
            - The original measurement.time_interval is restored after computation.
            - In discrete mode, intervals are chosen as T/N where N = 2, 3, ..., resolution+1
            - In continuous mode, the minimum interval ensures at least 2 measurements
        """
        import time

        # Validate inputs
        if mode not in ["continuous", "discrete"]:
            raise ValueError(f"mode must be 'continuous' or 'discrete', got '{mode}'")
        if resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {resolution}")

        # Get experimental parameters
        exp_params = self.experimental_params

        # Store original interval to restore later
        original_interval = exp_params.measurement.time_interval

        initial_time = exp_params.measurement.initial_time
        final_time = exp_params.measurement.final_time
        total_time = final_time - initial_time

        if verbose:
            # Get current circuit parameters for display
            initial_params = self.initial_circuit.get_trainable_parameters()
            final_params = self.final_circuit.get_trainable_parameters()

            print(f"Computing time interval landscape (mode: {mode})...")
            if len(initial_params) > 0 and len(final_params) > 0:
                print(f"  Circuit parameters: θ_init={np.degrees(initial_params[0]):.1f}°, θ_final={np.degrees(final_params[0]):.1f}°")
            print(f"  Resolution: {resolution} points")
            print(f"  Batch size: {batch_size} realizations")
            print(f"  Total evolution time: {total_time:.4f}")
            if min_interval is not None or max_interval is not None:
                print(
                    "  Requested interval bounds: "
                    f"[{(min_interval if min_interval is not None else 'default')}, "
                    f"{(max_interval if max_interval is not None else 'default')}]"
                )
            uncertainty_val = exp_params.initial_time_uncertainty
            if uncertainty_val > 0:
                spec = exp_params.initial_time_uncertainty_spec
                extra = f" (specified as '{spec}')" if isinstance(spec, str) else ""
                print(f"  Initial time uncertainty: ±{uncertainty_val:.4f}{extra}")

        # Generate time interval values based on mode
        # Helper to select approximately uniform samples from a sorted array.
        def _sample_uniform(values: np.ndarray, count: int) -> np.ndarray:
            """Select ``count`` approximately uniform samples from ``values``."""
            if values.size == 0:
                raise ValueError("No candidate intervals available within the requested bounds")
            if count == 1:
                return np.array([values[values.size // 2]])
            if values.size == 1:
                return np.repeat(values, count)

            positions = np.linspace(0, values.size - 1, count)
            indices = np.round(positions).astype(int)
            indices = np.clip(indices, 0, values.size - 1)
            # Ensure non-decreasing indices to keep the sequence sorted
            for idx in range(1, len(indices)):
                if indices[idx] < indices[idx - 1]:
                    indices[idx] = indices[idx - 1]
            return values[indices]

        if mode == "continuous":
            min_val = total_time / 100.0 if min_interval is None else float(min_interval)
            max_val = total_time if max_interval is None else float(max_interval)
            if min_val <= 0:
                raise ValueError(f"min_interval must be > 0, got {min_val}")
            if max_val <= 0 or max_val > total_time:
                raise ValueError(f"max_interval must be in (0, {total_time}], got {max_val}")
            if min_val >= max_val:
                raise ValueError(f"min_interval ({min_val}) must be less than max_interval ({max_val})")

            # Generate ideal continuous targets and approximate using available spacing.
            target_vals = np.linspace(min_val, max_val, resolution)

            # Derive feasible intervals based on integer partitions of the total time
            n_min = max(1, int(math.ceil(total_time / max_val)))
            n_max = int(math.floor(total_time / min_val))
            candidate_ns = np.arange(n_min, n_max + 1, dtype=int)
            candidate_intervals = total_time / candidate_ns.astype(float)
            candidate_intervals = np.sort(candidate_intervals)

            if candidate_intervals.size == 0:
                interval_vals = target_vals
            else:
                selected = np.empty_like(target_vals)
                prev_idx = 0
                for i, target in enumerate(target_vals):
                    idx = int(np.abs(candidate_intervals - target).argmin())
                    if i > 0 and idx < prev_idx:
                        idx = prev_idx
                    prev_idx = idx
                    selected[i] = candidate_intervals[idx]
                interval_vals = selected
        else:  # mode == 'discrete'
            max_val = total_time if max_interval is None else float(max_interval)
            if max_val <= 0 or max_val > total_time:
                raise ValueError(f"max_interval must be in (0, {total_time}], got {max_val}")

            if min_interval is None:
                min_val = total_time / float(resolution)
            else:
                min_val = float(min_interval)

            if min_val <= 0:
                raise ValueError(f"min_interval must be > 0, got {min_val}")
            if min_val > max_val:
                raise ValueError(
                    f"min_interval ({min_val}) must be less than or equal to max_interval ({max_val})"
                )

            n_start = max(1, int(math.ceil(total_time / max_val)))
            n_end = int(math.floor(total_time / min_val))

            if n_end < n_start:
                raise ValueError(
                    "No discrete intervals satisfy the requested min/max bounds. "
                    f"Computed n_start={n_start}, n_end={n_end}."
                )

            candidate_ns = np.arange(n_start, n_end + 1, dtype=int)
            candidate_intervals = total_time / candidate_ns.astype(float)
            candidate_intervals = np.sort(candidate_intervals)

            interval_vals = _sample_uniform(candidate_intervals, resolution)

        # Initialize result arrays
        metric_vals = np.zeros(resolution)
        detection_with = np.zeros(resolution)
        detection_without = np.zeros(resolution)
        n_measurements = np.zeros(resolution, dtype=int)

        start_time = time.time()

        # Evaluate each time interval
        for i, interval in enumerate(interval_vals):
            # Update time interval in exp_params
            exp_params.measurement.time_interval = interval
            exp_params.measurement.measurement_times = None  # Force recomputation
            exp_params._update_measurement_times()

            # Store number of measurements for this interval
            meas_times_list = exp_params._measurement_times_list
            n_measurements[i] = len(meas_times_list) if meas_times_list is not None else 0

            # Run simulation with batch averaging
            callback = self.run_simulation(batch_size=batch_size)

            # Store results (averaged over batch)
            # Clip values to ensure they're in valid ranges (handle numerical precision issues)
            metric_vals[i] = np.clip(callback.history["metric"][-1], 0.0, 1.0)
            detection_with[i] = np.clip(callback.history["detection_with"][-1], 0.0, 1.0)
            detection_without[i] = np.clip(callback.history["detection_without"][-1], 0.0, 1.0)

            # Progress update
            if verbose:
                progress = (i + 1) / resolution * 100
                print(
                    f"  Progress: {progress:.1f}% "
                    f"(interval={interval:.4f}, n_meas={n_measurements[i]}, "
                    f"metric={metric_vals[i]:.6f})",
                    end="\r",
                )

        # Restore original time interval
        exp_params.measurement.time_interval = original_interval
        exp_params.measurement.measurement_times = None
        exp_params._update_measurement_times()

        if verbose:
            elapsed = time.time() - start_time
            print(f"\nCompleted in {elapsed:.1f}s " f"({elapsed/resolution:.3f}s per point)")

            # Report optimal interval
            optimal_idx = np.argmax(metric_vals)
            optimal_interval = interval_vals[optimal_idx]
            optimal_metric = metric_vals[optimal_idx]
            optimal_n_meas = n_measurements[optimal_idx]
            print(
                f"  Optimal interval: {optimal_interval:.4f} "
                f"(n_meas={optimal_n_meas}, metric={optimal_metric:.6f})"
            )

        return {
            "interval_vals": interval_vals,
            "metric_vals": metric_vals,
            "detection_with": detection_with,
            "detection_without": detection_without,
            "n_measurements": n_measurements,
            "mode": mode,
            "batch_size": batch_size,
            "initial_time_uncertainty": exp_params.initial_time_uncertainty,
            "initial_time_uncertainty_spec": exp_params.initial_time_uncertainty_spec,
        }

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
    ) -> SweepResults:
        """
        Sweep over chi and gamma parameters for n-qubit system.

        This method evaluates the sensing metric and detection measure across
        a 2D grid of chi (dispersive coupling) and gamma (cavity decay rate)
        values. The experiment instance is reused with temporary parameter
        updates for efficiency, and the original state is restored after completion.

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
            SweepResults object containing chi_vals, gamma_vals, metric_map,
            detection_map, detection_without_map, and metadata.

        Example:
            >>> results = experiment.sweep_chi_gamma(
            ...     chi_interval=[0.1, 50.0],
            ...     resolution_chi=15,
            ...     resolution_gamma=15,
            ...     chi_scale='log'
            ... )
            >>> max_idx = np.unravel_index(
            ...     np.argmax(results['metric_map']),
            ...     results['metric_map'].shape
            ... )
            >>> print(f"Optimal chi: {results['chi_vals'][max_idx[1]]:.3f}")

        Note:
            For multi-qubit experiments, chi is set equal for all qubits.
            For n_qubits >= 2, probability maps are stored for all computational
            basis states using keys like ``p00``, ``p11`` (2 qubits) or
            ``p000``, ``p001``, ... (n qubits).
            Experiment state is automatically restored after the sweep.
        """
        import time

        # Validate scale parameters
        if chi_scale not in ["linear", "log"]:
            raise ValueError(f"chi_scale must be 'linear' or 'log', got '{chi_scale}'")
        if gamma_scale not in ["linear", "log"]:
            raise ValueError(f"gamma_scale must be 'linear' or 'log', got '{gamma_scale}'")

        if verbose:
            print("Computing χ-γ parameter sweep...")
            print(
                f"  Resolution: {resolution_chi}×{resolution_gamma} = {resolution_chi * resolution_gamma} points"
            )
            print(f"  χ range: [{chi_interval[0]:.2f}, {chi_interval[1]:.2f}] ({chi_scale} scale)")
            print(
                f"  γ range: [{gamma_interval[0]:.2f}, {gamma_interval[1]:.2f}] ({gamma_scale} scale)"
            )

        # Create parameter grids with specified scales
        if chi_scale == "log":
            chi_vals = np.logspace(np.log10(chi_interval[0]), np.log10(chi_interval[1]), resolution_chi)
        else:
            chi_vals = np.linspace(chi_interval[0], chi_interval[1], resolution_chi)

        if gamma_scale == "log":
            gamma_vals = np.logspace(
                np.log10(gamma_interval[0]), np.log10(gamma_interval[1]), resolution_gamma
            )
        else:
            gamma_vals = np.linspace(gamma_interval[0], gamma_interval[1], resolution_gamma)

        # Initialize result arrays
        metric_map = np.zeros((resolution_gamma, resolution_chi))
        detection_map = np.zeros((resolution_gamma, resolution_chi))
        detection_without_map = np.zeros((resolution_gamma, resolution_chi))

        # Determine number of qubits
        n_qubits = self.experimental_params.n_qubits
        store_state_prob_maps = n_qubits >= 2
        all_states = [format(k, f"0{n_qubits}b") for k in range(2**n_qubits)]

        # For n-qubit experiments (n >= 2), track probabilities for all basis states.
        if store_state_prob_maps:
            prob_maps = {
                f"p{state}": np.zeros((resolution_gamma, resolution_chi))
                for state in all_states
            }

        start_time = time.time()
        total_points = resolution_chi * resolution_gamma

        # Save current state
        saved_state = self._save_sweep_state()

        try:
            # Compute sweep
            for i, chi in enumerate(chi_vals):
                for j, gamma_val in enumerate(gamma_vals):
                    # Update physical constants with new chi and gamma
                    chi_list = [chi] * n_qubits

                    # Temporarily update experiment parameters
                    self._update_chi_gamma(chi_list, gamma_val)

                    # Run simulation
                    if store_state_prob_maps:
                        # For n >= 2 qubits, get full basis-state probability information.
                        results = self.run_simulation_with_probabilities()

                        # Store detection and metric results
                        metric_map[j, i] = results["metric"]
                        detection_map[j, i] = results["detection_with"]
                        detection_without_map[j, i] = results["detection_without"]

                        # Store individual probability maps for all basis states
                        for state in all_states:
                            prob_maps[f"p{state}"][j, i] = results["probs_with"][state]
                    else:
                        # Run simulation with batch averaging
                        callback = self.run_simulation(batch_size=batch_size)

                        # Store results (j,i indexing for correct orientation in plots)
                        metric_map[j, i] = callback.history["metric"][-1]
                        detection_map[j, i] = callback.history["detection_with"][-1]
                        detection_without_map[j, i] = callback.history["detection_without"][-1]

                    # Progress indicator
                    if verbose and (i * resolution_gamma + j + 1) % max(1, total_points // 10) == 0:
                        elapsed = time.time() - start_time
                        progress = (i * resolution_gamma + j + 1) / total_points
                        print(f"  Progress: {progress*100:.1f}% | " f"Elapsed: {elapsed:.1f}s")

        finally:
            # Always restore original state
            self._restore_sweep_state(saved_state)

        if verbose:
            total_time = time.time() - start_time
            print(f"Sweep completed in {total_time:.1f}s")
            print(f"  Max metric: {np.max(metric_map):.6f}")
            max_idx = np.unravel_index(np.argmax(metric_map), metric_map.shape)
            print(f"  Optimal χ: {chi_vals[max_idx[1]]:.3f}")
            print(f"  Optimal γ: {gamma_vals[max_idx[0]]:.3f}")

        # Prepare results dictionary
        results_dict = {
            "metric_map": metric_map,
            "detection_map": detection_map,
            "detection_without_map": detection_without_map,
        }

        # Add probability maps for n-qubit experiments with n >= 2
        if store_state_prob_maps:
            results_dict.update(prob_maps)

        # Prepare metadata
        max_idx = np.unravel_index(np.argmax(metric_map), metric_map.shape)

        # Get measurement times, handling None case
        meas_times = self.experimental_params.measurement.measurement_times
        n_measurements = len(meas_times) if meas_times is not None else 0

        # Get noise rates (could be list or float)
        depol = self.experimental_params.noise_config.depolarizing
        depol_val = depol[0] if isinstance(depol, list) else depol
        dephasing = self.experimental_params.noise_config.dephasing
        dephasing_val = dephasing[0] if isinstance(dephasing, list) else dephasing
        relax = self.experimental_params.noise_config.relaxation
        relax_val = relax[0] if isinstance(relax, list) else relax

        metadata = {
            "optimal_chi": chi_vals[max_idx[1]],
            "optimal_gamma": gamma_vals[max_idx[0]],
            "max_metric": metric_map[max_idx],
            "optimal_idx": max_idx,
            # System characteristics
            "n_qubits": n_qubits,
            "cavity_levels": self.experimental_params.system_dims.cavity_levels,
            "qubit_levels": self.experimental_params.system_dims.qubit_levels,
            "field_levels": self.experimental_params.system_dims.field_levels,
            "n_measurements": n_measurements,
            "measurement_times": meas_times,
            "initial_time_uncertainty": self.experimental_params.measurement.initial_time_uncertainty,
            "depolarizing_rate": depol_val,
            "dephasing_rate": dephasing_val,
            "relaxation_rate": relax_val,
            "initial_state": self.experimental_params.initial_state.state_type.name,
            "inverse_pulse_width": self.experimental_params.physical_constants.inverse_pulse_width,
        }

        return SweepResults(
            param1_name="gamma",
            param1_vals=gamma_vals,
            param1_scale=gamma_scale,
            param2_name="chi",
            param2_vals=chi_vals,
            param2_scale=chi_scale,
            results=results_dict,
            metadata=metadata,
        )

    def measure_all_states(self, rho: qt.Qobj) -> Dict[str, float]:
        """
        Measure probabilities for all computational-basis qubit states.

        Convenience method to get all joint measurement outcomes at once for
        an arbitrary number of qubits.

        Args:
            rho: State to measure

        Returns:
            Dictionary with probabilities for all $2^n$ basis states,
            keyed by bitstrings like ``'0'``, ``'1'`` (1 qubit) or
            ``'00'``, ``'01'``, ``'10'``, ``'11'`` (2 qubits), etc.

        Example:
            >>> probs = experiment.measure_all_states(rho)
            >>> print(f"P(00) = {probs['00']:.4f}")
        """
        from .quantum_utils import measure_qubits_probability

        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels
        qubit_levels = self.experimental_params.qubit_levels
        n_qubits = self.experimental_params.n_qubits
        all_states = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]

        return {
            state: measure_qubits_probability(
                rho,
                "all",
                self.operators,
                state=state,
                field_levels=field_levels,
                cavity_levels=cavity_levels,
                q_levels=qubit_levels,
            )
            for state in all_states
        }

"""
N Qubit Quantum Sensing Experiment
====================================

n-qubit quantum sensing experiment implementation.
This class handles quantum sensing protocols with n qubits coupled to a shared cavity.
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
import qutip_jax
import math
import time as t
from qutip import settings

from qsopt.core.callback import OptimizationCallback
from qsopt.core.circuit import QuantumCircuit, create_ry_circuit_layer
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

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import

from .quantum_utils import (
    apply_qubit_rotation,
    build_qubit_noise_operators,
    embed_circuit_unitary,
    generate_initial_state,
    generate_n_qubit_operators,
    gu,
    measure_qubits_probability,
    u0,
)

# Suppress Diffrax complex dtype warning
warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax is a work in progress*")

@jax.tree_util.register_pytree_node_class
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
        time_trainable: bool = False
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
            initial_circuit = create_ry_circuit_layer(experimental_params.n_qubits, theta_values=np.pi/2)

        if final_circuit is None:
            final_circuit = create_ry_circuit_layer(experimental_params.n_qubits, theta_values=-np.pi/2)

        self.experimental_params = experimental_params
        self.initial_circuit = initial_circuit
        self.final_circuit = final_circuit
        self.detection_metric = detection_metric if detection_metric is not None else DetectionMetric(n_qubits=experimental_params.n_qubits)

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

        # Define trainable parameters for pytree
        self._dynamic_fields = ("initial_circuit","final_circuit")
        if time_trainable == True:
            self._dynamic_fields = self._dynamic_fields + ("experimental_params",) #experimental params is not yet defined as a pytree

        # Initialize quantum objects
        self.__post_init__()

    def __post_init__(self):
        """Post-initialization to set up operators and hamiltonian."""
        settings.core["auto_real_casting"] = False
        self._generate_operators()
        self._generate_hamiltonian()
        self._initialize_initial_state()

    # Pytree construction from class for jax
    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in self._dynamic_fields)
        aux_data = {k: v for k, v in self.__dict__.items()
                    if k not in self._dynamic_fields}
        return children, aux_data

    # Class reconstruction from pytree for jax
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        # restore static data first
        obj.__dict__.update(aux_data)

        # restore dynamic fields
        for name, value in zip(obj._dynamic_fields, children):
            setattr(obj, name, value)

        return obj

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
        qubit_levels = self.experimental_params.qubit_levels
        n_qubits = self.experimental_params.n_qubits
        detection_states = self.detection_metric.detection_states

        # Generate n-qubit operators using utility function
        self.operators = generate_n_qubit_operators(
            field_levels, cavity_levels, qubit_levels, n_qubits, detection_states
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

    #@partial(jax.jit, static_argnames=["solver","args"])
    def simulation(
        self,
        solver: qt.MESolver,
        rho: qt.Qobj,
        measurements: Union[List[float], np.ndarray],
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
            Detection probability as JAX array
        """
        if  not hasattr(self,'debug_times'):          #####################################
            self.debug_times = []
            self.step=0
        self.debug_times.append({ f'load_cached_parameters{self.step}' : t.time()})   ################################

        if args is None:
            args = {"sigma": self.experimental_params.inverse_pulse_width}

        # Get detection metric
        detection_metric = self.detection_metric

        # Get measurement operators
        measure_reset = self.operators["measure_reset"]
        measure_reset_dag = self.operators["measure_reset_dag"]

        measurement_array = jnp.asarray(measurements, dtype=float)
        if measurement_array.ndim != 1 or measurement_array.size < 2:
            raise ValueError("measurements must be a 1D array with at least 2 time points")

        # Get circuit unitaries (precomputed or compute from circuits)
        # precomputed_unitaries are already QuTiP objects for efficiency
        if precomputed_unitaries is None:
            initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag = self._prepare_circuit_unitaries()
        else:
            initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag = precomputed_unitaries

        # Initial state
        rho_current = rho

        # Track cumulative probability of non-detection
        prob = jnp.array(1.0)

        self.debug_times.append({ f'start_simulation{self.step}' : t.time()})   ################################
        n_meas=0

        # Loop over measurement intervals
        for t0, t1 in zip(measurement_array[:-1], measurement_array[1:]):

            rho_after_circuit = initial_unitary * rho_current * initial_unitary_dag  # type: ignore

            self.debug_times.append({ f'measurement{n_meas}:solver_{self.step}' : t.time()})   ################################

            evolution_result = solver.run(rho_after_circuit, [t0, t1], args=args)

            self.debug_times.append({ f'measurement{n_meas}:measure_{self.step}' : t.time()})   ################################
            print(evolution_result)
            rho_evolved = evolution_result.states[-1]
            rho_final = final_unitary * rho_evolved * final_unitary_dag  # type: ignore

            # Measure probability of non detection and reset the qubit
            rho_reset = measure_reset * rho_final * measure_reset_dag
            prob_no_detect = jnp.real(rho_reset.tr())
            rho_current = rho_reset if prob_no_detect == 0 else rho_reset / prob_no_detect
            prob = prob * prob_no_detect


            n_meas+=1

        self.debug_times.append({ f'returning_simulation{self.step}' : t.time()})   ################################


        return 1-prob

    def run_simulation(self, batch_size: int = 1, measure_qubit: Optional[Union[int,List[int]]] = None) -> OptimizationCallback:
        """
        Run n-qubit sensing protocol with current parameters.

        This method executes the complete n-qubit quantum sensing workflow:
        - Applies rotations to all qubits independently
        - Evolves under n-qubit Hamiltonian
        - Performs measurements (joint or individual)
        - Computes detection probabilities with and without photon interaction

        Args:
            batch_size: Number of random realizations to average over for measurement
                       uncertainty (default: 1). Each realization uses a different
                       random shift in measurement times based on initial_time_uncertainty.
            measure_qubit: Which qubit to measure (None => both jointly, index int: one qubit only, index list: qubits in the list only)

        Returns:
            OptimizationCallback: Callback containing simulation results with:
                - Single epoch (epoch=1)
                - Current parameter values
                - Detection probabilities (prob_with, prob_without) averaged over batch
                - Sensing contrast averaged over batch

        Raises:
            ValueError: If initial state cache is not initialized
        """
        # Get initial state and solvers
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")
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

        # Prepare circuit unitaries once for the entire batch
        circuit_unitaries = self._prepare_circuit_unitaries()

        # Run simulations with batch averaging over uncertainty realizations
        prob_with_list = []
        prob_without_list = []

        for measurement_times in measurement_sequences:

            # Simulation with photon interaction
            prob_with = self.simulation(
                solver=solver_with,
                rho=rho0,
                measurements=jnp.array(measurement_times),
                precomputed_unitaries=circuit_unitaries,
            )
            prob_with_list.append(prob_with)

            # Simulation without photon interaction (reference)
            prob_without = self.simulation(
                solver=solver_without,
                rho=rho0,
                measurements=jnp.array(measurement_times),
                precomputed_unitaries=circuit_unitaries,
            )
            prob_without_list.append(prob_without)

        # Average over batch
        prob_with = jnp.mean(jnp.array(prob_with_list))
        prob_without = jnp.mean(jnp.array(prob_without_list))
        contrast = jnp.abs(prob_with - prob_without)

        # Create callback with single epoch for simulation results
        callback = OptimizationCallback(save_every=1, save_best=True)
        callback(
            trainable_params_initial=self.trainable_params_initial,
            trainable_params_final=self.trainable_params_final,
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
        the configured metric to compute detection probabilities and contrast.
        Useful for parameter sweeps and reproducing notebook experiments.

        Args:
            t_start: Evolution start time (default: -5.0)
            t_end: Evolution end time (default: 5.0)

        Returns:
            Dictionary containing:
                - 'probs_with': probability of finding the qubit in excited state with photon interaction
                - 'probs_without': probability of finding the qubit in excited state without photon interaction
                - 'detection_with': Detection probability with photon
                - 'detection_without': Detection probability without photon
                - 'contrast': Sensing contrast

        Example:
            >>> experiment = Experiment(exp_params)
            >>> results = experiment.run_simulation_with_probabilities()
            >>> print(f"P(11) with photon: {results['probs_with']['p11']:.4f}")
            >>> print(f"Contrast: {results['contrast']:.4f}")
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

        # Use metric to compute detection probabilities
        n_qubits = self.experimental_params.n_qubits
        detection_with = float(self.detection_metric(probs_with))
        detection_without = float(self.detection_metric(probs_without))

        # Compute contrast using metric's method
        contrast = float(detection_with - detection_without)

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

        # Use provided measurement protocol or default from experimental parameters
        if measurement_protocol is None:
            measurement_protocol = self.experimental_params.measurement

        # Get measurement times from protocol
        measurement_times = np.array(measurement_protocol.measurement_times)
        # Use measurement times for start and end
        t_start = float(measurement_times[0])
        t_end = float(measurement_times[-1])

        # Get initial state and solver
        rho = self._cached_initial_state
        if rho is None:
            raise RuntimeError("Initial state cache is not initialized.")


        solver = (
            self.get_solver_with_interaction()
            if with_interaction
            else self.get_solver_no_interaction()
        )

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
        cavity_population_list = []
        field_population_list = []

        # Set up measurements
        intermediate_meas_times = measurement_times[(measurement_times > t_start) & (measurement_times < t_end)]
        segment_starts = [t_start] + list(intermediate_meas_times)
        segment_ends = list(intermediate_meas_times) + [t_end]

        # Evolution
        for seg_start, seg_end in zip(segment_starts, segment_ends):
            # Number of points for this segment
            seg_fraction = (seg_end - seg_start) / (t_end - t_start)
            seg_n_points = max(2, int(n_points * seg_fraction))

            # Apply initial circuit for measurement
            rho = initial_unitary * rho * initial_unitary_dag

            # Evolve segment
            seg_times = np.linspace(seg_start, seg_end, seg_n_points)
            result = solver.run(rho, tlist=seg_times, args=args)

            # Extract data for this segment
            for i, rho_t in enumerate(result.states):

                # Apply final circuit for measurement
                rho_meas = final_unitary * rho_t * final_unitary_dag  # type: ignore

                # Measure detection
                rho_reset = measure_reset * rho_meas * measure_reset_dag
                prob_no_detect = jnp.real(rho_reset.tr())

                detection_list.append(1-prob_no_detect)

                all_times.append(seg_times[i])

                # Calculate populations (take real part since expectation values should be real)
                cavity_pop = float(np.real(qt.expect(n_cavity, rho_t)))
                field_pop = float(np.real(qt.expect(n_field, rho_t)))
                cavity_population_list.append(cavity_pop)
                field_population_list.append(field_pop)

            # Update system after actual measurement
            rho = rho_reset #if prob_no_detect == 0 else rho_reset/prob_no_detect

        times = np.array(all_times)
        # Compute pulse shape using the same u0 function as visualization
        pulse_shape = np.array([float(u0(t, sigma=args["sigma"])) for t in times])

        # Build probabilities dictionary
        probabilities = { "detection_probability" : np.array(detection_list)}

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
                "detection_criterion" : self.detection_metric.detection_name,
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
        renormalize_grad: Optional[Union[bool,float]] = False,
    ) -> OptimizationCallback:
        """
        Optimize rotation angles to maximize sensing contrast.

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
            learning_rate: Optional learning rate for the optimizer.
                    If None defaults to 0.5.
            renormalize_grad: Radious of the sphere inside which the gradients are renormalized. (default: 1)
                    If False (0), does not renormalize the gradients.

        Returns:
            OptimizationCallback with full optimization history

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

        self.debug_times = []
        self.debug_times.append({ 'opt_start' : t.time()})   ################################

        # Use provided callback or default
        if callback is None:
            callback = self.callback

        # Reset callback at start of new optimization
        callback.reset()

        # Count total trainable parameters from both circuits
        n_initial = self.initial_circuit.count_trainable_parameters()
        n_final = self.final_circuit.count_trainable_parameters()
        n_total = n_initial + n_final

        if n_total == 0:
            raise ValueError("No trainable parameters found in circuits")

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

        # Initialize optimizer (default to SGD with lr=0.5 if not provided)
        if optimizer is None:
            optimizer = optax.sgd(learning_rate=0.5)
        opt_state = optimizer.init(params)

        # Get initial state, solvers and detection metric
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")

        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()
        detection_metric = self.detection_metric

        # Define objective function
        #@jax.jit
        def objective_function(opt_params):
            """Negative sensing contrast for minimization with batch averaging."""

            self.debug_times.append({ f'setup_circuits{self.step}' : t.time()})   ################################

            self.initial_circuit.set_trainable_parameters(opt_params[:n_initial])
            self.final_circuit.set_trainable_parameters(opt_params[n_initial:])

            # Compute circuit unitaries
            circuit_unitaries = self._prepare_circuit_unitaries()

            self.debug_times.append({ f'setup_measurements{self.step}' : t.time()})   ################################

            # Get measurement times batch
            measurement_times_batch = (
                self.experimental_params.get_measurement_times_with_uncertainty(batch_size)
            )

            # Handle both single and multiple realizations uniformly
            if measurement_times_batch.ndim == 1:
                measurement_times_batch = measurement_times_batch[jnp.newaxis, :]

            detect_with_batch = jnp.zeros(batch_size)
            detect_without_batch = jnp.zeros(batch_size)

            self.debug_times.append({ f'setup_simulations{self.step}' : t.time()})   ################################

            for i in range(batch_size):
                measurement_times = measurement_times_batch[i]

                self.debug_times.append({ f'start 1st simulation{self.step}' : t.time()})   ################################

                detect_without_batch = detect_without_batch.at[i].set(
                    self.simulation(
                        solver_without,
                        rho0,
                        measurement_times,
                        precomputed_unitaries=circuit_unitaries
                    )
                )

                self.debug_times.append({ f'start 2nd simulation{self.step}' : t.time()})   ################################

                detect_with_batch = detect_with_batch.at[i].set(
                    self.simulation(
                        solver_with,
                        rho0,
                        measurement_times,
                        precomputed_unitaries=circuit_unitaries
                    )
                )

            self.debug_times.append({ f'batch_averaging{self.step}' : t.time()})   ################################

            # Average over batch
            detect_with = jnp.mean(detect_with_batch)
            detect_without = jnp.mean(detect_without_batch)
            contrast = detect_with - detect_without
            loss = detection_metric.metric(detect_with, detect_without)

            self.debug_times.append({ f'exit_objective{self.step}' : t.time()})   ################################

            # Return negative for minimization
            return loss, (detect_with, detect_without, contrast)

        #jitted_value_grad = jax.jit(jax.value_and_grad(
        #        objective_function, has_aux=True
        #    ))

        # Get detection description for verbose output
        detection_desc = detection_metric.detection_name

        if verbose:
            print(f"Configuration:")
            print(f"    Max iterations: {num_steps}")
            print(f"    Batch size: {batch_size}")
            print(f"    Convergence tolerance: {tolerance:.2e}")
            print(f"    Detection criterion: {detection_desc}")
            print(f"    Trainable parameters: {n_total} ({n_initial} initial circuit + {n_final} final circuit)")
            print(f"    Initial parameter values:")

            initial_vals = np.asarray(params, dtype=float)
            for i, val in enumerate(initial_vals):
                circuit_type = "setup" if i < n_initial else "reset"
                print(f"        param{i}. {circuit_type}_={val:.3f} rad ({np.rad2deg(val):.1f}°)")

            uncertainty = self.experimental_params.initial_time_uncertainty
            if uncertainty > 0:
                spec = self.experimental_params.initial_time_uncertainty_spec
                extra = f" (specified as '{spec}')" if isinstance(spec, str) else ""
                print(f"    Measurement uncertainty: ±{uncertainty:.3f}{extra}")

            print("=" * 75)
            # Build header based on number of parameters (up to 4 each)
            header_parts = [f"{'Step':<6}"]
            n_init_show = min(n_initial, 4)
            n_final_show = min(n_final, 4)
            for i in range(n_init_show):
                header_parts.append(f"setup_{i:<6}")
            for i in range(n_final_show):
                header_parts.append(f"reset_{i:<6}")
            header_parts.extend([f"{'Contrast':<12}", "Grad Norm"])
            print("".join(header_parts))
            print("-" * 75)

        best_contrast = -np.inf
        best_params = jnp.array(params)

        # Initialize variables
        step = 0
        grad_norm = float("inf")

        self.debug_times.append({ 'start_training' : t.time()})   ################################

        for step in range(num_steps):

            self.step=step      ############################

            self.debug_times.append({ f'compute_gradients{step}' : t.time()})   ################################

            # Compute gradients using JAX autodiff
            (loss, (prob_with, prob_without, sensing_contrast)), grads = jax.value_and_grad(objective_function, has_aux=True)(params)
            #(loss, (prob_with, prob_without, sensing_contrast)), grads = jitted_value_grad(params)
            self.debug_times.append({ f'callback_tracking{step}' : t.time()})   ################################

            # Track best parameters
            if sensing_contrast > best_contrast:
                best_contrast = sensing_contrast
                best_params = jnp.array(params)

            # Call callback to track progress
            callback(
                trainable_params_initial=params[:n_initial], #self.initial_circuit.get_trainable_parameters(),
                trainable_params_final=params[n_initial:], #self.final_circuit.get_trainable_parameters(),
                prob_with=float(prob_with),
                prob_without=float(prob_without),
                contrast=float(sensing_contrast),
                loss=float(loss),
            )

            self.debug_times.append({ f'apply grads{step}' : t.time()})   ################################

            #Check for gradient renormalization 
            grad_norm = float(jnp.linalg.norm(grads))
            if renormalize_grad != False:
                new_norm = jnp.tanh(grad_norm/renormalize_grad) * renormalize_grad
                grads = grads * new_norm/grad_norm
                grad_norm = new_norm
                
            
            # Progress output
            if verbose and (step % verbose_step == 0 or grad_norm < tolerance):
                # Build parameter display (up to 4 each)
                n_init_show = min(n_initial, 4)
                n_final_show = min(n_final, 4)
                param_vals = np.asarray(params, dtype=float)

                output_parts = [f"{step:<6}"]
                for i in range(n_init_show):
                    output_parts.append(f"{param_vals[i]:<12.6f}")
                for i in range(n_final_show):
                    output_parts.append(f"{param_vals[n_initial + i]:<12.6f}")
                output_parts.extend([f"{float(sensing_contrast):<12.6f}", f"{grad_norm:<12.2e}"])
                print("".join(output_parts))

            
            # Convergence check
            if grad_norm < tolerance:
                break

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)


        self.debug_times.append({ 'end' : t.time()})   ################################

        # Ensure best parameters are set at the end
        best_values = np.asarray(best_params, dtype=float)
        best_initial = [best_values[i] for i in range(n_initial)]
        best_final = [best_values[i] for i in range(n_initial, n_total)]
        self.initial_circuit.set_trainable_parameters(best_initial)
        self.final_circuit.set_trainable_parameters(best_final)

        if verbose:
            print("=" * 75)
            print(f"Final gradient norm: {grad_norm:.2e}")
            print(f"Best sensing contrast: {best_contrast:.6f}")
            print(f"Best parameters:")
            for i, val in enumerate(best_values):
                circuit_type = "setup" if i < n_initial else "reset"
                print(f"    param{i}. {circuit_type}_[name]={val:.3f} rad ({np.rad2deg(val):.1f}°)")

        # Set convergence information in callback
        callback.set_convergence_info(
            converged=float(grad_norm) < tolerance, final_grad_norm=float(grad_norm)
        )

        temp = self.debug_times[0]        #############################

        print('\n\n'+'='*75 + '\n\n')               ############################

        for time in self.debug_times[1:]:                    ###############################
                                        ######################
            print('{:33}'.format(list(temp.keys())[0])+':'+'{:10.6f}'.format((list(time.values())[0]-list(temp.values())[0])))
                                        ######################

            temp = time                           ###########################

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
                - ``'best_interval'``: Interval delivering the highest contrast.
                - ``'best_contrast'``: Maximum contrast observed.
                - ``'best_index'``: Index of the optimal interval in the sampled array.

        Example:
            >>> # Optimize measurement interval with current rotation angles
            >>> time_callback = experiment.optimize_measurement_times(
            ...     resolution=30,
            ...     mode='discrete',
            ...     batch_size=10
            ... )
            >>> print(f"Best interval: {time_callback['best_interval']:.3f}")
            >>> print(f"Best contrast: {time_callback['best_contrast']:.6f}")
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
        contrast_vals_np = np.asarray(results["contrast_vals"], dtype=float)
        interval_vals_np = np.asarray(results["interval_vals"], dtype=float)
        best_index = int(np.argmax(contrast_vals_np))
        best_interval = float(interval_vals_np[best_index])
        best_contrast = float(contrast_vals_np[best_index])

        # Apply best interval to experimental parameters
        self.experimental_params.measurement.time_interval = best_interval
        self.experimental_params.measurement.measurement_times = None
        self.experimental_params._update_measurement_times()

        # Add best results to output
        results_with_best = dict(results)
        results_with_best["best_interval"] = best_interval
        results_with_best["best_contrast"] = best_contrast
        results_with_best["best_index"] = best_index

        if verbose:
            n_measurements = np.asarray(results["n_measurements"], dtype=int)
            print(f"\nOptimization complete:")
            print(f"  Best interval: {best_interval:.4f}")
            print(f"  Best contrast: {best_contrast:.6f}")
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
        Compute contrast landscape vs measurement time interval.

        This method evaluates how sensing contrast varies with the time interval
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
            4. Calculate average sensing contrast across realizations
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
                - 'contrast_vals': 1D array of sensing contrast (shape: [resolution])
                - 'detection_with': 1D array of detection prob with photon (shape: [resolution])
                - 'detection_without': 1D array of detection prob without photon (shape: [resolution])
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
            >>> optimal_idx = np.argmax(data['contrast_vals'])
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
        contrast_vals = np.zeros(resolution)
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
            contrast_vals[i] = np.clip(callback.history["contrast"][-1], 0.0, 1.0)
            detection_with[i] = np.clip(callback.history["prob_with"][-1], 0.0, 1.0)
            detection_without[i] = np.clip(callback.history["prob_without"][-1], 0.0, 1.0)

            # Progress update
            if verbose:
                progress = (i + 1) / resolution * 100
                print(
                    f"  Progress: {progress:.1f}% "
                    f"(interval={interval:.4f}, n_meas={n_measurements[i]}, "
                    f"contrast={contrast_vals[i]:.6f})",
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
            optimal_idx = np.argmax(contrast_vals)
            optimal_interval = interval_vals[optimal_idx]
            optimal_contrast = contrast_vals[optimal_idx]
            optimal_n_meas = n_measurements[optimal_idx]
            print(
                f"  Optimal interval: {optimal_interval:.4f} "
                f"(n_meas={optimal_n_meas}, contrast={optimal_contrast:.6f})"
            )

        return {
            "interval_vals": interval_vals,
            "contrast_vals": contrast_vals,
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

        This method evaluates sensing contrast and detection probability across
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
            SweepResults object containing chi_vals, gamma_vals, contrast_map,
            detection_map, detection_without_map, and metadata.

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
            For multi-qubit experiments, chi is set equal for all qubits.
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
        contrast_map = np.zeros((resolution_gamma, resolution_chi))
        detection_map = np.zeros((resolution_gamma, resolution_chi))
        detection_without_map = np.zeros((resolution_gamma, resolution_chi))

        # Determine number of qubits
        n_qubits = self.experimental_params.n_qubits
        is_two_qubit = (n_qubits == 2)

        # For two-qubit experiments, also track individual probabilities
        if is_two_qubit:
            prob_maps = {
                "p00": np.zeros((resolution_gamma, resolution_chi)),
                "p01": np.zeros((resolution_gamma, resolution_chi)),
                "p10": np.zeros((resolution_gamma, resolution_chi)),
                "p11": np.zeros((resolution_gamma, resolution_chi)),
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
                    if is_two_qubit:
                        # For two-qubit, get full probability information
                        results = self.run_simulation_with_probabilities()

                        # Store detection and contrast results
                        contrast_map[j, i] = results["contrast"]
                        detection_map[j, i] = results["detection_with"]
                        detection_without_map[j, i] = results["detection_without"]

                        # Store individual probability maps
                        for key in ["p00", "p01", "p10", "p11"]:
                            prob_maps[key][j, i] = results["probs_with"][key]
                    else:
                        # Run simulation with batch averaging
                        callback = self.run_simulation(batch_size=batch_size)

                        # Store results (j,i indexing for correct orientation in plots)
                        contrast_map[j, i] = callback.history["contrast"][-1]
                        detection_map[j, i] = callback.history["prob_with"][-1]
                        detection_without_map[j, i] = callback.history["prob_without"][-1]

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
            print(f"  Max contrast: {np.max(contrast_map):.6f}")
            max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
            print(f"  Optimal χ: {chi_vals[max_idx[1]]:.3f}")
            print(f"  Optimal γ: {gamma_vals[max_idx[0]]:.3f}")

        # Prepare results dictionary
        results_dict = {
            "contrast_map": contrast_map,
            "detection_map": detection_map,
            "detection_without_map": detection_without_map,
        }

        # Add probability maps for two-qubit experiments
        if is_two_qubit:
            results_dict.update(prob_maps)

        # Prepare metadata
        max_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)

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
            "max_contrast": contrast_map[max_idx],
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
        from .quantum_utils import measure_qubits_probability

        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels
        qubit_levels = self.experimental_params.qubit_levels

        return {
            "00": measure_qubits_probability(rho, [0, 1], self.operators, state="00",
                                            field_levels=field_levels, cavity_levels=cavity_levels, q_levels=qubit_levels),
            "01": measure_qubits_probability(rho, [0, 1], self.operators, state="01",
                                            field_levels=field_levels, cavity_levels=cavity_levels, q_levels=qubit_levels),
            "10": measure_qubits_probability(rho, [0, 1], self.operators, state="10",
                                            field_levels=field_levels, cavity_levels=cavity_levels, q_levels=qubit_levels),
            "11": measure_qubits_probability(rho, [0, 1], self.operators, state="11",
                                            field_levels=field_levels, cavity_levels=cavity_levels, q_levels=qubit_levels),
        }

def print_type(value):
    print(type(value))
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
import time as t
import jax
import jax.numpy as jnp
import diffrax

from qsopt.core.callback import OptimizationCallback
from qsopt.core.circuit import QuantumCircuit, create_ry_circuit
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    MeasurementProtocol,
    SystemConfiguration,
)
from qsopt.core.loss_functions import DetectionMetric
from qsopt.core.core_utils import (
    adaptive_map,
    annealing_weight,
    classify_sweep_axis,
    sweep_key_types,
    make_confusion_matrix,
    confusion_quality,
    state_probs_to_dict
)
from qsopt.utils.results import SweepResults

if TYPE_CHECKING:
    from qsopt.utils.results import TimeEvolutionResults

from .quantum_utils import (
    build_hamiltonians,
    embed_circuit_unitary,
    generate_initial_state,
    generate_system_operators,
)

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import

# Suppress Diffrax complex dtype warning
warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax is a work in progress*")

# Tight (final) ODE-solver tolerances == qutip_jax diffrax defaults. Tolerance annealing
# loosens these by _TOL_ANNEAL_FACTOR early in training and tightens back to them by the end.
_SOLVER_ATOL = 1e-8
_SOLVER_RTOL = 1e-6
_TOL_ANNEAL_FACTOR = 100.0  # start 100x (2 decades) looser than the tight tolerances


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

        # Save locally config names for easy access
        self.config_names = [config.name for config in self.experimental_params.configuration_set]

        # Set detection metric, checks number of qubits given to the detection metric
        if detection_metric is None:
            self.detection_metric = DetectionMetric(n_cavities=experimental_params.n_cavities,\
                                                    n_qubits=experimental_params.n_qubits,\
                                                    n_fields=experimental_params.n_fields,\
                                                    config_names=self.config_names,\
                                                    perturbation_type=experimental_params.perturbation_type)
        else:
            # Validate the detection metric by making sure core parameters have been set to the same values
            if detection_metric.n_qubits != experimental_params.n_qubits or\
                detection_metric.n_subsystems != (experimental_params.n_cavities + experimental_params.n_fields + experimental_params.n_qubits):
                raise ValueError(
                    f"Detection metric n_qubits ({detection_metric.n_qubits}) must match experimental_params n_qubits ({experimental_params.n_qubits})"
                )
            
            if detection_metric.perturbation_type != experimental_params.perturbation_type:
                raise ValueError(
                    f"Detection metric perturbation_type ({detection_metric.perturbation_type}) must match"
                    f"the experiment's perturbation type ({experimental_params.perturbation_type})"
                )


            if set(detection_metric.config_names) != set(self.config_names):
                raise ValueError(
                    f"Detection metric config_names ({detection_metric.config_names}) must match "
                    f"the experiment's configuration names ({self.config_names})"
                )
            self.detection_metric = detection_metric

        # Precompute total dimensions for QuTiP Qobj creation
        self.total_dims = self.experimental_params.cavity_levels \
                            + self.experimental_params.field_levels \
                            + self.experimental_params.qubit_levels

        # Extract trainable parameters from both circuits
        self.trainable_params_initial = self.initial_circuit.get_trainable_parameters()
        self.trainable_params_final = self.final_circuit.get_trainable_parameters()

        # Caches
        self._cached_initial_states: Optional[Dict[str, qt.Qobj]] = None
        self._cached_projectors: Dict[str, qt.Qobj] = {} # IS THIS USED??????????? TO BE CHECKED
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
        self._cached_initial_states = {
            config.name: self._initialize_initial_state(system_configuration=config) \
                for config in self.experimental_params.configuration_set
        }

        if len(self._cached_initial_states) != len(self.experimental_params.configuration_set):
            raise RuntimeError("Not all initial states were cached. Check experimental parameters.")

        self.detection_metric.initialize(self.operators["P_all"])


    @property
    def n_cavities(self) -> int:
        """Get the number of cavity modes in the experiment."""
        return self.experimental_params.n_cavities
    @property
    def n_fields(self) -> int:
        """Get the number of field modes in the experiment."""
        return self.experimental_params.n_fields
    @property
    def n_qubits(self) -> int:
        """Get the number of qubits in the experiment."""
        return self.experimental_params.n_qubits
    @property
    def cavity_levels(self) -> int:
        """Get the number of cavity levels in the experiment."""
        return self.experimental_params.cavity_levels
    @property
    def field_levels(self) -> int:
        """Get the number of field levels in the experiment."""
        return self.experimental_params.field_levels
    @property
    def qubit_levels(self) -> int:
        """Get the number of qubit levels in the experiment."""
        return self.experimental_params.qubit_levels
    @property
    def ground_config(self) -> int:
        """Get the name of the ground config in the experiment."""
        return self.experimental_params.ground
    


    def _generate_operators(self) -> None:
        """
        Generate operators for n-qubit system.

        Operators live in the composite Hilbert space ordered as:
        (cavity_1..M) ⊗ (field_1..L) ⊗ (qubit_1..N)
    
        Operators include:
        - Field and cavity creation/annihilation operators
        - Individual qubit Pauli operators (σx, σy, σz) for each qubit
        - Joint measurement projectors for all computational basis states
        - Individual qubit projectors based on detection criterion
        """
        # Get system dimensions
        cavity_levels = self.cavity_levels
        field_levels = self.field_levels
        qubit_levels = self.qubit_levels
        n_cavities = self.n_cavities
        n_fields = self.n_fields
        n_qubits = self.n_qubits

        # Generate system operators using utility function
        # Note: `generate_system_operators` expects (n_cavities, n_fields, n_qubits, cavity_levels, field_levels, qubit_levels)
        self.operators = generate_system_operators(
            n_cavities, n_fields, n_qubits, cavity_levels, field_levels, qubit_levels
        )

    def _generate_hamiltonian(self) -> None:
        """
        Generate Hamiltonians and Lindblad operators for the system and store them on ``self``.

        Delegates to :func:`build_hamiltonians` with the configured parameter values, then assigns:
        - self.hamiltonians: per-configuration ``QobjEvo`` Hamiltonians
        - self.lindblad_operators: per-configuration collapse-operator lists
        - self.global_args: resolved parameter values keyed by global key
        """
        self.hamiltonians, self.lindblad_operators, self.global_args = build_hamiltonians(
            self.operators, self.experimental_params, self.n_cavities, self.n_fields, self.n_qubits)

    def _initialize_initial_state(self, system_configuration: SystemConfiguration) -> qt.Qobj:
        """
        Generate and cache the initial state of the system.
        """
        return generate_initial_state(
            system_configuration=system_configuration,
            field_levels=self.field_levels,
            cavity_levels=self.cavity_levels,
            qubit_levels=self.qubit_levels,
            n_cavities=self.n_cavities,
            n_fields=self.n_fields,
            n_qubits=self.n_qubits,
        )

    def get_solvers(self) -> qt.MESolver:
        """Get Lindblad master equation solvers for all configurations (cached)."""

        for config in self.experimental_params.configuration_set:
            if config.name not in self._cached_solvers:
                self._cached_solvers[config.name] = qt.MESolver(
                    self.hamiltonians[config.name],
                    self.lindblad_operators[config.name],
                    options={
                        "method": "diffrax",
                        "progress_bar": False,
                        "normalize_output": False,
                        "stepsize_controller": diffrax.PIDController(atol=_SOLVER_ATOL, rtol=_SOLVER_RTOL),
                    },
                )

        return self._cached_solvers

    @staticmethod
    def _set_solver_tolerances(solvers: Dict[str, qt.MESolver], atol = _SOLVER_ATOL, rtol = _SOLVER_RTOL) -> None:
        """Inject a diffrax PIDController with the given tolerances into each cached solver.

        atol/rtol may be traced (epoch-dependent) values: they enter the compiled graph as
        dynamic leaves, so annealing the tolerance over training costs no recompilation.
        """
        for solver in solvers.values():
            solver._integrator._options["stepsize_controller"] = diffrax.PIDController(atol=atol, rtol=rtol)
    
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
        field_levels = self.field_levels
        cavity_levels = self.cavity_levels
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
            args = self.global_args

        # Get detection metric
        detection_metric = self.detection_metric

        # Get reset operators (pre-zipped at construction; see generate_system_operators)
        reset_list = self.operators['measure_reset_pairs']

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
            rho_reset = [op * rho_final * op_dag for op,op_dag in reset_list]
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
            args = self.global_args

        # Get detection metric
        detection_metric = self.detection_metric

        # Get reset operators (pre-zipped at construction; see generate_system_operators)
        reset_list = self.operators['measure_reset_pairs']

        # Get circuit unitaries
        if precomputed_unitaries is None:
            precomputed_unitaries = self._prepare_circuit_unitaries()
        initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag = precomputed_unitaries

        # Set initial state
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
            rho_reset = [op * rho_final * op_dag for op,op_dag in reset_list]
            rho_current = sum(rho_reset)
            
            rho_list.append(rho_final)

            n_meas+=1       ####################

        self.debug_times.append({ f'returning_simulation{self.step}' : t.time()})   ################################

        return rho_list

    def run_simulation(self, batch_size: int = 1, measurement_times = None, detection_states: bool = False, debug: bool=False) -> OptimizationCallback:
        """
        Run n-qubit sensing protocol with current parameters.

        This method executes the complete n-qubit quantum sensing workflow:
        - Applies rotations to all qubits independently
        - Evolves under n-qubit Hamiltonian
        - Performs measurements (joint or individual)
        - Computes detection measures with and without photon interaction

        Args:
            batch_size: Number of random realizations to average over for measurement
                       uncertainty (default: 1). Each realization draws a different timing
                       offset (collective shift + per-measurement jitter) from the protocol.
            measurement_times: Optional measurement times instead of the ones determined by the experimental parameters.
            detection_states: Whether to return the probabilities of the final quantum states (default: False)
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

        if debug or detection_states:
            self.debug_times = []
            self.step=0
            self.debug_times.append({ f'initialize_solvers' : t.time()})

 
        init_states = self._cached_initial_states
        if init_states is None:
            raise RuntimeError("Initial states cache is not initialized.")
        
        solvers = self.get_solvers() 
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
                raise ValueError(f"measurement_times must have at least 2 entries (a start and a measurement), got length {measurement_times.shape}")
            # explicit override: used as the deterministic timestamp sequence [t_start, *measurements]
            measurement_sequences = [measurement_times]
        else:
            # timestamps from the protocol, with the configured uncertainty applied over the batch
            timestamps = self.experimental_params.get_timestamps(batch_size)
            measurement_sequences = [timestamps[i] for i in range(timestamps.shape[0])]

        if debug:
            self.debug_times.append({ f'get_circuits' : t.time()})
 
        # Prepare circuit unitaries once for the entire batch
        circuit_unitaries = self._prepare_circuit_unitaries()

        # initialize batches
        batch_metric = []
        batch_detect = { config.name : [] for config in self.experimental_params.configuration_set }
        batch_validation = []

        if detection_states:
            batch_for_prob = []
            batch_weights = []

        if debug:
            self.debug_times.append({ f'start_measurement_loop' : t.time()})

        simulation_fn = self.simulation if not debug else self.debug_simulation
 
        for measurement_times in measurement_sequences:
            if debug:
                self.debug_times.append({ f'start_simulations_{self.step}' : t.time()})
            
            rho_lists = {}

            for config in self.experimental_params.configuration_set:

                # Simulation for each configuration with its specific solver and initial state
                rho_lists[config.name] = simulation_fn(
                    solver=solvers[config.name],
                    rho=init_states[config.name],
                    measurements=measurement_times,
                    precomputed_unitaries=circuit_unitaries,
                )
                if debug:
                    self.debug_times.append({ f'end_simulation_{config.name}_{self.step}' : t.time()})
            
            if debug:
                self.debug_times.append({ f'calculate_detection_metric{self.step}' : t.time()})


            weights = self.experimental_params.measurement.measurement_weights(measurement_times[1:])
            metric_value, (detection_dict, validation_value) = self.detection_metric(rho_lists, weights=weights)

            if detection_states:
                batch_for_prob.append(rho_lists)
                batch_weights.append(weights)


            batch_metric.append(metric_value)
            batch_validation.append(validation_value)
            for name, value in detection_dict.items():
                batch_detect[name].append(float(value))

            if debug:
                self.step += 1

        if debug:
            self.debug_times.append({ f'compute_means_from_batches' : t.time()})

        # Use detection metric's batching logic and then evaluate the configured metric.
        # With the default setup this metric can coincide with a simple difference (contrast),
        # but custom detection metrics may define any scalar objective.

        mean_metric = sum(batch_metric)/len(batch_metric)
        mean_validation = sum(batch_validation)/len(batch_validation)
        mean_detect_dict = {name: sum(values)/len(values) for name, values in batch_detect.items()}

        if detection_states:

            # P_all[i] is a diagonal projector, so P(state i | rho) = Tr(P_i @ rho) = <diag(P_i), diag(rho)>.
            # Stack the diagonals into (n_states, D): one mat-vec against diag(rho) gives the full prob vector.
            # Real cast: qutip returns diagonals as complex128 and would otherwise promote the result.
            proj_diags = np.real(np.stack([np.asarray(proj.diag()) for proj in self.operators['P_all']]))

            config_names = self.experimental_params.get_configuration_names()
            n_measurements = len(batch_for_prob[0][config_names[0]])
            n_states = proj_diags.shape[0]
            avg_weights = np.asarray(sum(batch_weights))

            # Sum probability vectors over the batch, then divide once. Axes: (config, measurement, state).
            prob_sum = np.zeros((len(config_names), n_measurements, n_states))
            for rho_lists in batch_for_prob:
                for c, name in enumerate(config_names):
                    for m, rho in enumerate(rho_lists[name]):
                        prob_sum[c, m] += proj_diags @ np.real(rho.diag())
            avg_probs = prob_sum / len(batch_for_prob)

            # Flatten to per-measurement, per-config {binary_state: prob} only at the boundary.
            state_prob_data = [state_probs_to_dict(avg_probs[:,m], config_names) for m in range(n_measurements)]

            if self.experimental_params.perturbation_type == 'persistent':

                weighted_probs = (avg_probs * avg_weights[None, :, None]).sum(axis=1) / avg_weights.sum()

                main_confusion_matrix, states_map = make_confusion_matrix(weighted_probs, config_names)

            else: #transient

                # Create a confusion matrix for each measurement
                evaluation_matrices, states_maps = list(zip(*[make_confusion_matrix(measurement_probs)
                                            for measurement_probs in state_prob_data]))

                # Each matrix is evaluated and the highest is used to determine which map to use
                confusion_scores = np.array([confusion_quality(matrix) for matrix in evaluation_matrices])*avg_weights
                winning_map = np.argmax(confusion_scores)

                main_confusion_matrix = evaluation_matrices[winning_map]
                states_map = states_maps[winning_map]

            confusion_matrices = [make_confusion_matrix(avg_probs[:, m], config_names, states_map)[0] for m in range(n_measurements)]
            
            false_signal_dict = {config:
                                    [sum([matrix[(config,prediction)] for prediction in config_names if not prediction in [config, self.experimental_params.ground]])
                                     for matrix in confusion_matrices]
                                 for config in config_names}


        if debug:
            self.debug_times.append({ f'save_callback' : t.time()})

        # Create callback with single epoch for simulation results
        callback = OptimizationCallback(save_every=1, save_best=True)

        if detection_states:
            
            callback(
                trainable_params_initial=self.trainable_params_initial,
                trainable_params_final=self.trainable_params_final,
                detection_dict=mean_detect_dict,
                metric=float(mean_metric),
                validation=float(mean_validation),
                state_probabilities=avg_probs,
                states_map=states_map,
                confusion_matrix=main_confusion_matrix,
                false_signal=false_signal_dict
            )

        else:

            callback(
                trainable_params_initial=self.trainable_params_initial,
                trainable_params_final=self.trainable_params_final,
                detection_dict=mean_detect_dict,
                metric=float(mean_metric),
                validation=float(mean_validation),
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

    def time_evolution(
        self,
        n_points: int = 200,
        measurement_protocol: Optional[MeasurementProtocol] = None,
        callback: Optional[OptimizationCallback] = None,
        partial_configs: Optional[List[str]] = None
    ) -> "TimeEvolutionResults":
        """
        Compute time evolution of n-qubit probabilities.

        Simulates the quantum system evolution over time using the measurement protocol times.
        Returns probability distributions for all n-qubit states (|0...0⟩, |0...1⟩, ..., |1...1⟩).
        The system starts in superposition (after first rotations), evolves under
        the Hamiltonian, and probabilities are measured after the second rotations.

        Args:
            n_points: Number of time points to sample (default: 200)
            measurement_protocol: Optional custom measurement protocol to use instead of
                    the experiment's default protocol (default: None)
            callback: Optional OptimizationCallback to determine detection states
                    If None, a new one will be created with detection_states=True (default: None)
            partial_configs: Optional list of configuration names to include in the evolution.
                    If None, all configurations will be included (default: None)

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

        # Deterministic full timestamp sequence [t_start, *measurements] (no offset, no jitter:
        # time evolution never uses noise). t_start is the unmeasured simulation start.
        measurement_times = np.asarray(self.experimental_params.get_timestamps(1, offset=False, jitter=False)[0], dtype=float)
        t_start = float(measurement_times[0])
        t_end = float(measurement_times[-1])

        # Get initial state and solvers
        init_states = self._cached_initial_states
        if init_states is None:
            raise RuntimeError("Initial state cache is not initialized.")

        solvers_dict = self.get_solvers()

        # Prepare circuit unitaries as QuTiP objects (including daggers)
        initial_unitary, initial_unitary_dag, final_unitary, final_unitary_dag = self._prepare_circuit_unitaries()

        n_cavities = self.n_cavities
        n_fields = self.n_fields
        n_qubits = self.n_qubits

        if partial_configs is None:
            config_names = self.config_names
        elif isinstance(partial_configs, list):
            config_names = [name for name in self.config_names if name in partial_configs]
            if len(config_names) == 0:
                raise ValueError("No valid configuration names found in partial_configs. \
                 Available configuration names: " + ", ".join(self.config_names) + \
                f"\nGot partial_configs: " + ", ".join(partial_configs))
            elif len(config_names) < len(partial_configs):
                missing = set(partial_configs) - set(config_names)
                print(f"Warning: The following names in partial_configs were not found and will be ignored: {missing}")
        else:
            raise TypeError("partial_configs must be a list of strings or None.")

        if callback is None or not callback.states_map:
            if callback is not None and not callback.states_map:
                print("Warning: Callback provided without a states map. Creating new one.")
            callback = self.run_simulation(measurement_times=(t_start, t_end), detection_states=True)

        
        # Get number operators for population calculation
        op_n_cavity = [self.operators["a_c_dag"][i] * self.operators["a_c"][i] for i in range(len(n_cavities))]
        op_n_field = [self.operators["a_f_dag"][i] * self.operators["a_f"][i] for i in range(len(n_fields))]

        # Get measurement operators and sigma
        measure_reset = self.operators["measure_reset"]
        measure_reset_dag = self.operators["measure_reset_dag"]

        args = self.global_args
        args["n_qubits"] = n_qubits
        args["detection_metric"] = self.detection_metric.name

        # Generate all possible qubit's computational states
        n_qubits = self.n_qubits 
        all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)] ################
        qubit_indices = list(range(0, n_qubits))

        # Storage for results
        all_times = []
        detection_lists = {name: [] for name in config_names}
        cavities_populations = {name: [[]]*n_cavities for name in config_names}
        fields_populations = {name: [[]]*n_fields for name in config_names}

        # Set up measurements
        intermediate_meas_times = measurement_times[(measurement_times > t_start) & (measurement_times < t_end)]
        segment_starts = [t_start] + list(intermediate_meas_times)
        segment_ends = list(intermediate_meas_times) + [t_end]

        # Evolution
        rho0 = init_states

        for seg_start, seg_end in zip(segment_starts, segment_ends):
            # Number of points for this segment
            seg_fraction = (seg_end - seg_start) / (t_end - t_start)
            seg_n_points = max(2, int(n_points * seg_fraction))

            # Apply initial circuit for measurement
            rho_circuit = {name: initial_unitary * rho0[name] * initial_unitary_dag for name in self.config_names}

            # Evolve segment
            seg_times = np.linspace(seg_start, seg_end, seg_n_points)
            results_dict = {name: solvers_dict[name].run(rho_circuit[name], tlist=seg_times, args=args) for name in self.config_names }

            # Extract data for this segment
            for i, rho_t in enumerate(results_dict[config_names[0]].states):

                # Apply final circuit for measurement
                rho_meas = {name: [final_unitary * results_dict[name].states[i] * final_unitary_dag] for name in self.config_names}

                # Measure detection with the configured metric
                epoch_fraction = (seg_times[i] - t_start) / (t_end - t_start)

                weights = self.experimental_params.measurement.measurement_weights(seg_times[i:i + 1])
                metric_value, (detect_dict, _) = self.detection_metric(rho_meas, epoch_fraction, weights=weights)

                detection_lists.update((name, list + [float(detect_dict[name])]) for name, list in detection_lists.items())

                all_times.append(seg_times[i])

                # Calculate populations (take real part since expectation values should be real)
                cavities_populations.update((name, [list_of_lists[j] + [float(np.real(qt.expect(op_n_cavity[j], results_dict[name].states[i])))] for j in range(n_cavities)]) for name, list_of_lists in cavities_populations.items())
                fields_populations.update((name, [list_of_lists[j] + [float(np.real(qt.expect(op_n_field[j], results_dict[name].states[i])))] for j in range(n_fields)]) for name, list_of_lists in fields_populations.items())

            # Update system after actual measurement
            #reset_with = [op * rho_meas_with * op_dag for op, op_dag in zip(measure_reset, measure_reset_dag)]
            rho0 = rho_meas

        times = np.array(all_times)
        # pulse shape plotting is outdated; to be reworked to plot the experiment's time modulations
        pulse_shape = None

        # Turn lists into arrays
        detection_lists.update((name, np.array(list)) for name, list in detection_lists.items())
        cavities_populations.update((name, [np.array(list) for list in list_of_lists]) for name, list_of_lists in cavities_populations.items())
        fields_populations.update((name, [np.array(list) for list in list_of_lists]) for name, list_of_lists in fields_populations.items())

        # Import at runtime to avoid circular dependency
        from qsopt.utils.results import TimeEvolutionResults

        return TimeEvolutionResults(
            times=times,
            probabilities=detection_lists,
            pulse_shape=pulse_shape,
            measurement_times=measurement_times,
            cavity_population=cavities_populations,
            field_population=fields_populations,
            metadata=args,
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
        tot_steps: Optional[int] = None,
        anneal_tolerances: Union[bool, float] = True,
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
            final_results: If True, stores the final optimization results in the callback. (default: True)
            hot_start: If True, continues optimization from the last parameters and optimizer state in the callback.
                    If either the optimizer or the params are given they override the hot start values. (default: False)
            tot_steps: Total number of optimization steps to run, it's used to give the epoch percentage to the detection metric.
                    It's useful if the optimization is divided in multiple runs.
                    If None, uses num_steps. (default: None)
            anneal_tolerances: If True, anneals the ODE-solver tolerances over training from loose
                    (fast, approximate gradients early) to the tight final tolerances (accurate near
                    convergence), following annealing_weight(epoch_fraction). Costs no recompilation.
                    True uses the default 100x loosening at the start; pass a float > 1 to set a
                    custom starting loosening factor; False disables annealing. (default: True)

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

        if isinstance(anneal_tolerances, bool):
            tolerance_scale = _TOL_ANNEAL_FACTOR if anneal_tolerances else 1.0

        elif isinstance(anneal_tolerances, (int, float)) and anneal_tolerances > 1:
            tolerance_scale = float(anneal_tolerances)
            anneal_tolerances = True

        else:
            raise ValueError("anneal_tolerances must either be a bool to toggle the tolerance annealing "
                             f"or a float > 1 to toggle it on with a custom starting loosening factor. "
                             f"Value given: {anneal_tolerances}")

        if tot_steps is None:
            tot_steps = num_steps
        elif tot_steps < num_steps:
            raise ValueError(f"tot_steps should be greater than or equal to num_steps+start_step, got tot_steps={tot_steps} and num_steps={num_steps-start_step}, start_step={start_step}")

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
        rho0 = self._cached_initial_states
        if rho0 is None:
            raise RuntimeError("Initial states cache is not initialized.")

        solvers = self.get_solvers()
        detection_metric = self.detection_metric

        # Define objective function with explicit uncertainty input.
        # Signature order is kept future-proof for optional optimization over times.
        def parallel_simulations(circuit_unitaries, timestamps, noise, epoch_fraction: float):
            """Single-realization of all configuration simulations at the given (noisy) timestamps."""

            noisy_timestamps = timestamps + noise

            rho_dict = {config: self.simulation(
                solvers[config],
                rho0[config],
                noisy_timestamps,
                precomputed_unitaries=circuit_unitaries,
            ) for config in self.config_names}

            weights = self.experimental_params.measurement.measurement_weights(noisy_timestamps[1:])
            metric_value, (detection_dict, validation_value) = self.detection_metric(rho_dict, epoch_fraction, weights=weights)

            return metric_value, detection_dict, validation_value

        static_args = []  # initialize objective_function static args list
        mp = self.experimental_params.measurement
        has_uncertainty = bool(mp.max_measurements_offset) or (mp.per_measurement_jitter is not None)
        # deterministic base timestamps [t_start, *measurements]; uncertainty enters only as the noise.
        base_timestamps = np.asarray(self.experimental_params.get_timestamps(1, offset=False, jitter=False)[0], dtype=float)

        if not optimize_measurement_times:
            static_args.append(1)  # add objective_function static arg: timestamps
            timestamps_arg = tuple(float(x) for x in base_timestamps)
        else:
            timestamps_arg = jnp.asarray(base_timestamps, dtype=float)

        if not has_uncertainty:

            if batch_size != 1:
                if verbose:
                    warnings.warn(f"Batch size > 1 has no effect when there is no measurement uncertainty. Setting batch size to 1.")
                batch_size = 1

            static_args.append(2)  # objective_function arg: noise_batch
            zero_uncertainty_batch = 0.0

            def get_noise_batch():
                    return zero_uncertainty_batch

            def objective_function(circuit_params, timestamps, noise_batch, epoch_fraction: float):
                """Objective with no uncertainty."""

                timestamps = np.asarray(timestamps, dtype=float)

                # Compute circuit unitaries
                self.initial_circuit.set_trainable_parameters(circuit_params[:n_initial])
                self.final_circuit.set_trainable_parameters(circuit_params[n_initial:])
                circuit_unitaries = self._prepare_circuit_unitaries()

                metric, detect_dict, validation = parallel_simulations(circuit_unitaries, timestamps, noise_batch, epoch_fraction)

                return -metric, (detect_dict, metric, validation)

        else:

            if batch_size < 16 and verbose:
                warnings.warn(f"Using a small batch size of {batch_size} for optimization with measurement uncertainty may lead to noisy gradients and slow convergence. Consider increasing the batch size for better performance.")

            def get_noise_batch():
                # guarded timing uncertainties (batch_size, M+1), sampled independently of the base
                # timestamps so the base times stay free to be optimized.
                return jnp.asarray(self.experimental_params.get_measurement_uncertainties(batch_size), dtype=float)

            vmapped_simulations = jax.vmap(parallel_simulations, in_axes=(None, None, 0, None))

            def objective_function(circuit_params, timestamps, noise_batch, epoch_fraction: float):
                """Batch vmapped objective where vectorization happens only over uncertainty."""

                timestamps = jnp.asarray(timestamps, dtype=float)

                # Compute circuit unitaries
                self.initial_circuit.set_trainable_parameters(circuit_params[:n_initial])
                self.final_circuit.set_trainable_parameters(circuit_params[n_initial:])
                circuit_unitaries = self._prepare_circuit_unitaries()

                batch_metric, batch_detect_dict, batch_validation = vmapped_simulations(circuit_unitaries, timestamps, noise_batch, epoch_fraction)

                mean_metric = jnp.mean(batch_metric)
                mean_detect_dict = {name: jnp.mean(batch) for name, batch in batch_detect_dict.items()}
                mean_validation = jnp.mean(batch_validation)

                return -mean_metric, (mean_detect_dict, mean_metric, mean_validation)

        jitted_grad = jax.jit(
            jax.grad(objective_function, has_aux=True, argnums=0),
            static_argnums=tuple(static_args),
        )


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

            if has_uncertainty:
                offset_desc = "off" if not mp.max_measurements_offset else (
                    "custom" if callable(mp.max_measurements_offset) else f"uniform(-{mp.collective_offset_width():.3g}, 0)")
                jitter = mp.per_measurement_jitter
                jit_desc = "off" if jitter is None else ("custom" if callable(jitter) else f"Gaussian std {float(jitter):.3g}")
                print(f"    Measurement uncertainty: collective offset {offset_desc}, per-measurement jitter {jit_desc}")

            # Build header based on number of parameters (up to 4 each)
            header_parts = [f"{'Step':<6}"]
            n_init_show = min(n_initial, 4)
            n_final_show = min(n_final, 4)
            for i in range(n_init_show):
                header_parts.append(f"setup{i}_{setup_gates[i].__repr__(params=False):<8}")
            for i in range(n_final_show):
                header_parts.append(f"reset{i}_{reset_gates[i].__repr__(params=False):<8}")
            header_parts.extend([f"{'Metric':<12}", f"{'Validation':<12}", f"{'Grad Norm':<12}", "Time"])

            header = "".join(header_parts)
            print("=" * (5+len(header)))
            print(header)
            print("-" * (5+len(header)))

        best_validation = -np.inf
        best_metric = -np.inf
        best_params = jnp.array(params)

        # Initialize variables
        step = start_step
        grad_norm = float("inf")
        scaling_speedup = 1.0

        for step in range(start_step, num_steps):

            epoch_fraction = step / tot_steps

            # Anneal ODE-solver tolerances loose -> tight over training (epoch_fraction is
            # unbatched here, so this sets one scalar tolerance per step; no recompilation).
            if anneal_tolerances:                
                scale = tolerance_scale ** annealing_weight(epoch_fraction*scaling_speedup)
                self._set_solver_tolerances(solvers, _SOLVER_ATOL * scale, _SOLVER_RTOL * scale)

            noise_batch = get_noise_batch()

            # Compute gradients using JAX autodiff
            grads, (detection_dict, step_metric, step_validation) = jitted_grad(
                params, timestamps_arg, noise_batch, epoch_fraction
            )

            step_metric_value = float(step_metric)
            step_validation_value = float(step_validation)

            # Track best parameters
            if step_validation_value > best_validation:
                best_validation = step_validation_value
                best_metric = step_metric_value
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
                detection_dict={name: float(detection) for name, detection in detection_dict.items()},
                metric=step_metric_value,
                validation=step_validation_value,
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
                output_parts.extend([
                    f"{step_metric_value:<12.6f}",
                    f"{step_validation_value:<12.6f}",
                    f"{grad_norm:<12.2e}",
                    f"{t.strftime("%Hh%Mm%Ss", t.gmtime(new_time))}",
                ])
                print("".join(output_parts))

            # Convergence check
            if grad_norm < tolerance:
                if epoch_fraction*scaling_speedup <= 8: # se è la terza volta che triggero questa condizione al massimo dell'apprendimento
                    scaling_speedup *= 2
                else:
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

        # Restore concrete tight tolerances on the cached solvers: annealing left their
        # controllers holding (now dead) traced tolerances, and the final run_simulation
        # below reuses these same solvers and wants full accuracy.
        if anneal_tolerances:
            self._set_solver_tolerances(solvers)

        # Run simulation to get probabilities for each state with the best parameters
        if has_uncertainty and batch_size < 16:
            if verbose:
                warnings.warn(f"Temporarily raising batch size from {batch_size} to 16 for the final evaluation to reduce uncertainty noise in the reported probabilities.")
            batch_size = 16 # Use a larger batch size for final evaluation to reduce noise in results when uncertainty is present

        final_results_callback = self.run_simulation(batch_size=batch_size,
                                            detection_states=True,
                                            debug=False
                                            )
            
        callback.set_measurement_protocol(*final_results_callback.get_measurement_protocol())

        if verbose:
            print("=" * (5+len(header)))
            print(f"Total optimization time: {t.strftime('%Hh%Mm%Ss', t.gmtime(t.time() - start_time))}")
            print(f"Final gradient norm: {grad_norm:.2e}")
            print(f"Best validation: {best_validation:.6f}")
            print(f"Best metric (at best validation): {best_metric:.6f}")
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


    # ---------------------------- generic N-dimensional sweep ----------------------------

    def sweep(self, param_grid: Dict[str, Any], *, measurement_protocol=None, batch_size: int = 1,
              verbose: bool = True) -> SweepResults:
        """N-dimensional sweep over global-arg parameters and/or the measurement ``time_interval``.

        Args:


        ``param_grid`` maps each name (a parameter name, a full global-arg key, or 'time_interval')
        to a 1D array of values; the metric and per-configuration detections are evaluated at every
        grid combination and returned as N-D arrays in a :class:`SweepResults`. When the measurement
        protocol defines uncertainty (collective offset and/or per-measurement jitter) each grid point
        is averaged over ``batch_size`` jittered realizations.
        """
        import itertools

        if not param_grid:
            raise ValueError("param_grid must be a non-empty {name: 1D values} mapping")

        names = list(param_grid)
        vals = [np.asarray(param_grid[n], dtype=float).reshape(-1) for n in names]
        protocol = measurement_protocol if measurement_protocol is not None else self.experimental_params.measurement
        has_uncertainty = bool(protocol.max_measurements_offset) or (protocol.per_measurement_jitter is not None)
        if has_uncertainty and protocol.jitter_is_time_dependent:
            raise ValueError("per_measurement_jitter as a time-dependent f(t) is not supported in sweeps "
                             "(swept times make a pre-drawn noise batch inconsistent);"
                             "Use a float for a gaussian jitter or a custom non time-dependent distribution f().")
        if not has_uncertainty and batch_size != 1:
            batch_size = 1  # no uncertainty -> a batch would just repeat the same point
        if has_uncertainty and batch_size < 16:
            warnings.warn(f"batch_size={batch_size} is too small to average measurement uncertainty "
                          f"consistently; using 16 instead.")
            batch_size = 16
        key_types = sweep_key_types(self)

        lanes, keys_per = [], []
        for n in names:
            lane, ks = classify_sweep_axis(self, n, key_types)
            lanes.append(lane)
            keys_per.append(ks)

        if verbose:
            total = int(np.prod([len(v) for v in vals]))
            unc = f"uncertainty on, {batch_size} realizations/point" if has_uncertainty else "deterministic"
            print(f"Sweeping {len(names)} axes over {total} grid points ({unc}):")
            for n, v, lane, ks in zip(names, vals, lanes, keys_per):
                tgt = "measurement times" if lane == "measurement" else ", ".join(ks)
                print(f"  - {n}: {len(v)} pts in [{v.min():.3g}, {v.max():.3g}]  lane={lane}  ({tgt})")

        # measurement lane (time_interval): M measurements fixed, with the unmeasured sim start first.
        M = len(self.experimental_params.measurement_times)
        t0 = float(self.experimental_params.t_simulation_start)
        meas_axes = [i for i, l in enumerate(lanes) if l == "measurement"]
        if meas_axes and protocol.window_start is not None:
            need = float(protocol.window_end - t0)  # lead + window_length
            for i in meas_axes:
                span = M * float(np.min(vals[i]))
                if span < need:
                    raise ValueError(
                        f"time_interval sweep: n_measurements*min(time_interval)={span:.3g} < "
                        f"window_end-initial_time={need:.3g}; increase M, the smallest interval, or shrink the window.")

        fast_idx = [i for i, l in enumerate(lanes) if l != "rebuild"]
        loop_idx = [i for i, l in enumerate(lanes) if l == "rebuild"]
        promote_keys = sorted({k for i in fast_idx if lanes[i] == "promote" for k in keys_per[i]}) or None

        if fast_idx:
            mesh = np.meshgrid(*[vals[i] for i in fast_idx], indexing="ij")
            fast_grid = np.stack([m.ravel() for m in mesh], axis=1)
            fast_shape = tuple(len(vals[i]) for i in fast_idx)
        else:
            fast_grid = np.zeros((1, 0))
            fast_shape = ()

        base_meas = jnp.asarray(self.experimental_params.timestamps, dtype=float)  # [t0, *measurements]
        noise_batch = (jnp.asarray(self.experimental_params.get_measurement_uncertainties(batch_size), dtype=float)
                       if has_uncertainty else None)
        rho0 = self._cached_initial_states
        unis = self._prepare_circuit_unitaries()
        opts = {"method": "diffrax", "progress_bar": False, "normalize_output": False,
                "stepsize_controller": diffrax.PIDController(atol=_SOLVER_ATOL, rtol=_SOLVER_RTOL)}

        shape = tuple(len(v) for v in vals)
        res_metric = np.empty(shape)
        res_val = np.empty(shape)
        res_det = {c: np.empty(shape) for c in self.config_names}

        def make_eval(solvers, base_args):
            def evaluate_at(meas_t, args):
                """Metric/detections/validation for one timestamp realization."""
                rho = {c: self.simulation(solvers[c], rho0[c], meas_t, args=args, precomputed_unitaries=unis)
                       for c in self.config_names}
                weights = protocol.measurement_weights(meas_t[1:])
                m, (det, val) = self.detection_metric(rho, weights=weights)
                return m, jnp.array([det[c] for c in self.config_names]), val
            def evaluate(point):
                args = dict(base_args)
                meas_t = base_meas
                for j, i in enumerate(fast_idx):
                    v = point[j]
                    if lanes[i] == "measurement":
                        meas_t = t0 + jnp.arange(M + 1, dtype=float) * v  # [t0, t0+v, ..., t0+M*v]
                    else:
                        for kk in keys_per[i]:
                            args[kk] = v
                if noise_batch is None:
                    return evaluate_at(meas_t, args)
                # average over the pre-drawn jittered realizations
                m, det, val = jax.vmap(lambda nz: evaluate_at(meas_t + nz, args))(noise_batch)
                return jnp.mean(m), jnp.mean(det, axis=0), jnp.mean(val)
            return evaluate

        loop_ranges = [range(len(vals[i])) for i in loop_idx]
        for loop_combo in (itertools.product(*loop_ranges) if loop_idx else [()]):
            overrides = {k: float(vals[i][ix]) for i, ix in zip(loop_idx, loop_combo) for k in keys_per[i]} or None
            H, L, ga = build_hamiltonians(
                self.operators, self.experimental_params, self.n_cavities, self.n_fields, self.n_qubits,
                overrides=overrides, dynamic_keys=set(promote_keys) if promote_keys else None)
            solvers = {c: qt.MESolver(H[c], L[c], options=opts) for c in self.config_names}
            out_m, out_det, out_val = adaptive_map(make_eval(solvers, ga), fast_grid, verbose)

            slc = [slice(None)] * len(names)
            for i, ix in zip(loop_idx, loop_combo):
                slc[i] = ix
            slc = tuple(slc)
            res_metric[slc] = np.asarray(out_m).reshape(fast_shape)
            res_val[slc] = np.asarray(out_val).reshape(fast_shape)
            out_det = np.asarray(out_det).reshape(fast_shape + (len(self.config_names),))
            for ci, c in enumerate(self.config_names):
                res_det[c][slc] = out_det[..., ci]

        results = {"metric": res_metric, "validation": res_val}
        results.update({f"detection_{c}": res_det[c] for c in self.config_names})
        best = np.unravel_index(int(np.nanargmax(res_metric)), shape)
        metadata = {"best_index": best, "best_metric": float(res_metric[best]),
                    "best_point": {n: float(vals[i][best[i]]) for i, n in enumerate(names)}}
        if verbose:
            print(f"Best metric {metadata['best_metric']:.6g} at {metadata['best_point']}")
        return SweepResults(axis_names=names, axis_vals=[np.asarray(v) for v in vals],
                            axis_scales=["linear"] * len(names), results=results, metadata=metadata)
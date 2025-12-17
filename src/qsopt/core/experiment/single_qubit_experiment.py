"""
Quantum Sensing Experiment Class
================================

Main experiment class that orchestrates quantum sensing protocols with configurable
parameters, noise models, and optimization strategies.

Note: This module uses JAX arrays extensively. Type checker warnings about JAX array
operations (unsubscriptable-object, unsupported-operand-type, etc.) are false positives
and are disabled in .pylintrc. The code executes correctly at runtime.
"""

"""Quantum sensing experiment module."""
# type: ignore  # Suppress Pylance type warnings for JAX arrays
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qutip as qt

from qsopt.core.callback import OptimizationCallback
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import Parameter, ParameterType, TrainableParameters

if TYPE_CHECKING:
    from qsopt.utils.results import TimeEvolutionResults

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import

from .base import Experiment
from .quantum_utils import (
    build_qubit_noise_operators,
    create_measurement_projector,
    generate_initial_state,
    generate_single_qubit_operators,
    gu,
)

# Suppress Diffrax complex dtype warning
warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax is a work in progress*")


class SingleQubitExperiment(Experiment):
    """
    A class representing a single qubit photon detection experiment.

    This class implements the quantum sensing protocol with a three-system composite
    Hilbert space: input_cavity ⊗ resonator_cavity ⊗ qubit.

    The system workflow:
    |ψ₀⟩ → Ry(θ₁) → H(t) Evolution → Ry(θ₂) → Measurement → Detection Probability
    """

    def __init__(
        self, experimental_params: ExperimentalParameters, trainable_params: TrainableParameters
    ):
        # Call parent constructor
        super().__init__(experimental_params, trainable_params)

        # Cache for frequently used objects (significant speedup during optimization)
        self._cached_initial_state: Optional[qt.Qobj] = None
        self._cached_projector_0: Optional[qt.Qobj] = None
        self._cached_projector_1: Optional[qt.Qobj] = None
        self._cached_solvers: Dict[str, qt.MESolver] = {}

        # Initialize quantum objects
        self.__post_init__()

    def __post_init__(self):
        """Post-initialization to set up operators and hamiltonian."""
        self._generate_operators()
        self._generate_hamiltonian()
        self._initialize_caches()
        self._ensure_measurement_interval_sync()

    def _generate_operators(self):
        """
        Generate the necessary operators for the experiment in the composite space.

        Uses utility function to create operators for the three-subsystem composite
        Hilbert space: input_field ⊗ resonator_cavity ⊗ qubit.

        This allows for easy extension to multi-qubit systems in the future.
        """
        # Get system dimensions
        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels
        qubit_levels_list = self.experimental_params.qubit_levels

        # For single qubit experiment, use the first (and only) qubit's levels
        qubit_levels = (
            qubit_levels_list[0] if isinstance(qubit_levels_list, list) else qubit_levels_list
        )

        # Use utility function to generate all operators
        self.operators = generate_single_qubit_operators(field_levels, cavity_levels, qubit_levels)

    def _generate_hamiltonian(self):
        """
        Generate the time-dependent Hamiltonian and Lindblad operators for the experiment.

        Creates:
        1. Time-dependent Hamiltonian
        2. Dispersive qubit-resonator interaction Hamiltonian
        3. Lindblad operators for noise processes (relaxation, dephasing, depolarization)
        """
        if self.operators is None:
            raise RuntimeError("Operators must be generated before Hamiltonian")

        # Extract coupling constants
        gm = self.experimental_params.photon_cavity_coupling
        chi_list = self.experimental_params.chi
        # For single qubit experiment, use the first (and only) qubit's chi
        chi = chi_list[0] if isinstance(chi_list, list) else chi_list
        sigma = self.experimental_params.inverse_pulse_width

        # Get operators
        a_in = self.operators["a_in"]
        a_in_dag = self.operators["a_in_dag"]
        a = self.operators["a"]
        a_dag = self.operators["a_dag"]
        sigma_z = self.operators["sigma_z"]
        sigma_x = self.operators["sigma_x"]
        sigma_y = self.operators["sigma_y"]
        sigma_minus = self.operators["sigma_minus"]

        # Time-dependent coupling function arguments
        args = {"sigma": sigma}

        # Time-dependent cavity-cavity coupling Hamiltonian
        coupling_coeff = 1j / 2 * jnp.sqrt(gm)
        H_coupling = qt.Qobj(coupling_coeff * (a_in_dag * a - a_in * a_dag))  # type: ignore

        # Dispersive qubit-resonator interaction Hamiltonian
        H_dispersive = qt.Qobj(-chi * a_dag * a * sigma_z)  # type: ignore

        # Complete time-dependent Hamiltonian
        H_total = qt.QobjEvo([H_dispersive, [H_coupling, gu]], args=args)

        # Noise configuration
        noise_config = self.experimental_params.noise_config

        # Extract noise rates for the first (and only) qubit
        depolarizing = (
            noise_config.depolarizing[0]
            if isinstance(noise_config.depolarizing, list)
            else noise_config.depolarizing
        )
        dephasing = (
            noise_config.dephasing[0]
            if isinstance(noise_config.dephasing, list)
            else noise_config.dephasing
        )
        relaxation = (
            noise_config.relaxation[0]
            if isinstance(noise_config.relaxation, list)
            else noise_config.relaxation
        )

        # Build Lindblad noise operators using helper function
        lindblad_noise = build_qubit_noise_operators(
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            sigma_minus=sigma_minus,
            depolarizing_rate=depolarizing,
            dephasing_rate=dephasing,
            relaxation_rate=relaxation,
        )

        # Add custom Lindblad operators if provided
        if noise_config.custom_operators is not None:
            lindblad_noise.extend(noise_config.custom_operators)

        # Lindblad interaction operators
        L_int = qt.QobjEvo([a_in, gu], args=args) + np.sqrt(gm) * a

        interaction_ops: List[Union[qt.Qobj, qt.QobjEvo]] = [L_int] + lindblad_noise
        no_interaction_ops: List[Union[qt.Qobj, qt.QobjEvo]] = lindblad_noise

        # Store Hamiltonians and Lindblad operators
        self.hamiltonians = {"total": H_total, "dispersive": H_dispersive, "coupling": H_coupling}

        self.lindblad_operators = {
            "interaction": interaction_ops,
            "no_interaction": no_interaction_ops,
        }

    def _initialize_caches(self):
        """
        Initialize cached objects for performance optimization.

        Caches frequently-used objects that don't change during optimization:
        - Measurement projectors P0 and P1
        - Initial state (computed once per experiment)

        This provides significant speedup during optimization loops.
        """
        # Cache projectors (used in every measurement step)
        self._cached_projector_0 = create_measurement_projector(
            0,
            self.experimental_params.field_levels,
            self.experimental_params.cavity_levels,
            self.experimental_params.qubit_levels,
        )
        self._cached_projector_1 = create_measurement_projector(
            1,
            self.experimental_params.field_levels,
            self.experimental_params.cavity_levels,
            self.experimental_params.qubit_levels,
        )

        # Cache initial state (doesn't change during optimization)
        self._cached_initial_state = generate_initial_state(
            self.experimental_params.initial_state,
            self.experimental_params.field_levels,
            self.experimental_params.cavity_levels,
            self.experimental_params.qubit_levels,
            num_qubits=1,
        )

    def get_initial_state(self) -> qt.Qobj:
        """
        Get the cached initial state density matrix.

        Returns:
            qt.Qobj: Initial density matrix for the experiment
        """
        if self._cached_initial_state is None:
            raise RuntimeError(
                "Initial state has not been initialized. This should not happen after __post_init__."
            )
        return self._cached_initial_state

    def get_solver_with_interaction(self) -> qt.MESolver:
        """
        Get Lindblad master equation solver WITH input photon interaction (cached).

        Solver is created once and cached for performance during optimization loops.

        Returns:
            qt.MESolver: Configured solver for signal case evolution
        """
        if "with_interaction" not in self._cached_solvers:
            if self.hamiltonians is None or self.lindblad_operators is None:
                raise RuntimeError("Hamiltonian and operators must be generated first")

            self._cached_solvers["with_interaction"] = qt.MESolver(
                self.hamiltonians["total"],
                self.lindblad_operators["interaction"],
                options={"method": "diffrax", "normalize_output": False},
            )

        return self._cached_solvers["with_interaction"]

    def get_solver_no_interaction(self) -> qt.MESolver:
        """
        Get Lindblad master equation solver WITHOUT input photon interaction (cached).

        Solver is created once and cached for performance during optimization loops.

        Returns:
            qt.MESolver: Configured solver for reference case evolution
        """
        if "no_interaction" not in self._cached_solvers:
            if self.hamiltonians is None or self.lindblad_operators is None:
                raise RuntimeError("Hamiltonian and operators must be generated first")

            self._cached_solvers["no_interaction"] = qt.MESolver(
                self.hamiltonians["dispersive"],
                self.lindblad_operators["no_interaction"],
                options={"method": "diffrax", "normalize_output": False},
            )

        return self._cached_solvers["no_interaction"]

    def _build_rotation_gate(self, axis: str, theta: float) -> qt.Qobj:
        """Construct embedded single-qubit rotation gate for the specified axis."""
        if self.operators is None:
            raise RuntimeError("Operators not initialized")

        axis_key = f"sigma_{axis.lower()}"
        if axis_key not in self.operators:
            raise ValueError(f"Unsupported rotation axis '{axis}'. Expected one of x, y, z.")

        generator = self.operators[axis_key]
        return (-1j * generator * theta / 2).expm()

    def _prepare_rotation_gates(self, theta1: float, theta2: float) -> Tuple[qt.Qobj, qt.Qobj]:
        """Build rotation gates for optimization angles θ₁ and θ₂."""
        rotation_theta1 = self._build_rotation_gate("y", theta1)
        rotation_theta2 = self._build_rotation_gate("y", theta2)
        return rotation_theta1, rotation_theta2

    def ry_rotation(self, rho: qt.Qobj, theta: float) -> qt.Qobj:
        """
        Apply Ry rotation to qubit in the three-system composite space.

        Uses utility function to apply rotation around the Y-axis for quantum state
        manipulation. The rotation is applied only to the qubit subsystem while
        preserving the cavity states in the composite Hilbert space.

        Args:
            rho: QuTiP Qobj density matrix in composite space (input ⊗ resonator ⊗ qubit)
            theta: float or JAX array, Ry rotation angle in radians

        Returns:
            QuTiP Qobj: Rotated density matrix
        """
        rotation_gate = self._build_rotation_gate("y", theta)
        return rotation_gate * rho * rotation_gate.dag()  # type: ignore

    def proj0(self, rho: qt.Qobj) -> qt.Qobj:
        """
        Project density matrix onto qubit |0⟩ state.

        Uses cached projector for performance (avoids repeated computation).

        Args:
            rho: QuTiP Qobj density matrix in composite space

        Returns:
            QuTiP Qobj: Projected density matrix P₀ρP₀† (unnormalized)
        """
        P0 = self._cached_projector_0
        return P0 * rho * P0.dag()  # type: ignore

    def prob0(self, rho: qt.Qobj):
        """
        Calculate probability of measuring qubit in |0⟩ state.

        Uses cached projector for performance (avoids repeated computation).

        Args:
            rho: QuTiP Qobj density matrix in composite space

        Returns:
            float: Real probability value Tr(P₀ρ) ∈ [0,1]
        """
        P0 = self._cached_projector_0
        return jnp.real((P0 * rho * P0.dag()).tr())  # type: ignore

    def prob1(self, rho: qt.Qobj):
        """
        Calculate probability of measuring qubit in |1⟩ state.

        Uses cached projector for performance (avoids repeated computation).

        Args:
            rho: QuTiP Qobj density matrix in composite space

        Returns:
            float: Real probability value Tr(P₁ρ) ∈ [0,1]
        """
        P1 = self._cached_projector_1
        return jnp.real((P1 * rho * P1.dag()).tr())  # type: ignore

    def simulation(
        self,
        solver: qt.MESolver,
        rho: qt.Qobj,
        theta1: float,
        theta2: float,
        measurements: Union[List[float], np.ndarray],
        args: Optional[Dict] = None,
        precomputed_rotations: Optional[Tuple[qt.Qobj, qt.Qobj]] = None,
    ) -> jnp.ndarray:
        """
        Complete quantum photon detection simulation workflow.

        Workflow Steps:
        1. Apply first rotation Ry(θ₁) for initial qubit preparation
        2. Time evolution under cavity-qubit Hamiltonian H(t)
        3. Apply second rotation Ry(θ₂) for measurement optimization
        4. Sequential projective measurements with conditional state updates
        5. Calculate cumulative detection probability

        Args:
            solver: QuTiP MESolver, Configured quantum evolution solver
            rho: QuTiP Qobj, Initial density matrix in composite space
            theta1: float/JAX array, First Ry rotation angle
            theta2: float/JAX array, Second Ry rotation angle
            measurements: List or array of measurement times (sorted)
            args: dict, System parameters (optional, uses experimental_params if None)
            precomputed_rotations: Optional tuple of rotation gates ``(R_y(θ₁), R_y(θ₂))``
                to avoid recomputing exponentials when shared across simulations.

        Returns:
            float: Probability of detecting at least one excited state
                P(detection) = 1 - ∏ᵢ P(|0⟩ᵢ) ∈ [0,1]
        """
        if args is None:
            args = {"sigma": self.experimental_params.inverse_pulse_width}
        if self._cached_projector_0 is None:
            raise RuntimeError("Measurement projectors are not initialized.")

        measurement_array = np.asarray(measurements, dtype=float)
        if measurement_array.ndim != 1 or measurement_array.size < 2:
            raise ValueError("Measurement times must be a 1D array with at least two entries.")

        if precomputed_rotations is None:
            rotation_theta1, rotation_theta2 = self._prepare_rotation_gates(theta1, theta2)
        else:
            rotation_theta1, rotation_theta2 = precomputed_rotations

        rotation_theta1_dag = rotation_theta1.dag()
        rotation_theta2_dag = rotation_theta2.dag()
        projector_0 = self._cached_projector_0

        rho_current = rho
        prob_all_ground = jnp.array(1.0)

        for t0, t1 in zip(measurement_array[:-1], measurement_array[1:]):
            rho_after_ry = rotation_theta1 * rho_current * rotation_theta1_dag  # type: ignore
            evolution_result = solver.run(rho_after_ry, [t0, t1], args=args)
            rho_evolved = evolution_result.states[-1]

            rho_final = rotation_theta2 * rho_evolved * rotation_theta2_dag  # type: ignore
            prob_ground = jnp.real((projector_0 * rho_final * projector_0).tr())  # type: ignore
            prob_all_ground = prob_all_ground * prob_ground

            rho_projected = projector_0 * rho_final * projector_0  # type: ignore  # Always project to |0⟩
            trace_val = rho_projected.tr()
            rho_current = rho_projected if trace_val == 0 else rho_projected / trace_val

        prob_detection = 1 - prob_all_ground
        return prob_detection

    def run_simulation(self, batch_size: int = 1) -> OptimizationCallback:
        """
        Run simulation with current parameter values without updating them.

        This method provides a convenient way to test the system with the current
        trainable parameter values, computing detection probabilities both with
        and without photon interaction.

        Args:
            batch_size: Number of random realizations to average over for measurement
                       uncertainty (default: 1). Each realization uses a different
                       random shift in measurement times based on initial_time_uncertainty.

        Returns:
            OptimizationCallback: Callback containing simulation results with:
                - Single epoch (epoch=1)
                - Current parameter values
                - Detection probabilities (prob_with, prob_without) averaged over batch
                - Sensing contrast averaged over batch
                - Optimization-related attributes set to None (converged, final_grad_norm)

        Raises:
            ValueError: If fewer than 2 rotation parameters are defined
        """
        self._ensure_measurement_interval_sync()

        # Get rotation parameters using the dedicated method
        rotation_angles = self.trainable_params.get_rotation_angles()

        if len(rotation_angles) < 2:
            raise ValueError("Need at least 2 rotation angle parameters")

        # Extract first two rotation parameters (order preserved from TrainableParameters)
        param_names = list(rotation_angles.keys())
        param1_name = param_names[0]
        param2_name = param_names[1]
        theta1 = float(rotation_angles[param1_name][0])
        theta2 = float(rotation_angles[param2_name][0])

        # Get initial state and solvers
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")
        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()

        # Prepare measurement time realizations once for the entire batch
        measurement_times_batch = self.experimental_params.get_measurement_times_with_uncertainty(
            batch_size
        )
        if measurement_times_batch.ndim == 1:
            measurement_sequences = [measurement_times_batch]
        else:
            measurement_sequences = [
                measurement_times_batch[i] for i in range(measurement_times_batch.shape[0])
            ]

        rotation_pair = self._prepare_rotation_gates(theta1, theta2)

        # Run simulations with batch averaging over uncertainty realizations
        prob_with_list = []
        prob_without_list = []

        for measurement_times in measurement_sequences:
            prob_with_batch = self.simulation(
                solver_with,
                rho0,
                theta1,
                theta2,
                measurement_times,
                precomputed_rotations=rotation_pair,
            )
            prob_without_batch = self.simulation(
                solver_without,
                rho0,
                theta1,
                theta2,
                measurement_times,
                precomputed_rotations=rotation_pair,
            )

            prob_with_list.append(prob_with_batch)
            prob_without_list.append(prob_without_batch)

        # Average over batch
        prob_with = jnp.mean(jnp.array(prob_with_list))
        prob_without = jnp.mean(jnp.array(prob_without_list))
        contrast = prob_with - prob_without

        # Create a callback with single epoch for simulation results
        callback = OptimizationCallback(save_every=1, save_best=True)
        callback(
            trainable_params=self.trainable_params,
            prob_with=float(prob_with),
            prob_without=float(prob_without),
            contrast=float(contrast),
        )

        # Keep optimization-related attributes as None/False (not from optimization)
        # converged and final_grad_norm remain as initialized (False, None)

        return callback

    def time_evolution(
        self,
        n_points: int = 200,
        with_interaction: bool = True,
        measurement_protocol: Optional["MeasurementProtocol"] = None,
    ) -> "TimeEvolutionResults":
        """
        Compute time evolution of qubit probabilities with optional projective measurements.

        Simulates the quantum system evolution over time using the measurement protocol times.
        If intermediate measurement times exist in the protocol, performs projective measurements
        at those times using the same protocol as run_simulation(): invert rotation Ry(-theta2),
        project onto basis states, then reapply Ry(theta2).

        This method is useful for understanding the temporal dynamics and creating
        time evolution plots for quantum sensing protocols.

        Args:
            n_points: Number of time points to sample (default: 200)
            with_interaction: If True, use Hamiltonian with chi coupling.
                             If False, use Hamiltonian without chi (default: True)
            measurement_protocol: Optional custom measurement protocol to use instead of
                                 the experiment's default protocol (default: None)

        Returns:
            TimeEvolutionResults object containing:
                - times: Array of time points, shape (n_points,)
                - probabilities: Dict with 'prob_0' and 'prob_1' arrays
                - pulse_shape: Pulse envelope u(t), shape (n_points,)
                - measurement_times: Measurement time points
                - cavity_population: Cavity population <a†a>, shape (n_points,)
                - field_population: External field population <a_in†a_in>, shape (n_points,)
                - cumulative_detection: Cumulative detection probability (only when intermediate
                  measurements are present). Monotonically increasing probability of having
                  detected |1⟩ at least once up to each time point.

        Example:
            >>> # Get time evolution data using default measurement protocol
            >>> evolution = experiment.time_evolution(n_points=200)
            >>>
            >>> # Plot with matplotlib
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(evolution['times'], evolution['prob_0'], label='P(|0⟩)')
            >>> plt.plot(evolution['times'], evolution['prob_1'], label='P(|1⟩)')
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
            
        # Get current rotation angles
        rotation_angles = self.trainable_params.get_rotation_angles()
        if len(rotation_angles) < 2:
            raise ValueError("Need at least 2 rotation angle parameters")

        param_names = list(rotation_angles.keys())
        theta1 = float(rotation_angles[param_names[0]][0])
        theta2 = float(rotation_angles[param_names[1]][0])

        # Get initial state and solver
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")

        solver = (
            self.get_solver_with_interaction()
            if with_interaction
            else self.get_solver_no_interaction()
        )

        # Prepare rotation gates
        rotation_theta1, rotation_theta2 = self._prepare_rotation_gates(theta1, theta2)
        rotation_theta1_dag = rotation_theta1.dag()
        rotation_theta2_dag = rotation_theta2.dag()

        # Start with initial state (apply rotation_theta1 at start of each segment)
        rho_current = rho0

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
        prob_0_list = []
        prob_1_list = []
        cavity_population_list = []
        field_population_list = []
        cumulative_prob = 0.0  # Running sum of detection probabilities
        
        if perform_measurements:
            # Evolution with intermediate projective measurements
            segment_starts = [t_start] + list(intermediate_meas_times)
            segment_ends = list(intermediate_meas_times) + [t_end]
            
            for seg_start, seg_end in zip(segment_starts, segment_ends):
                # Number of points for this segment
                seg_fraction = (seg_end - seg_start) / (t_end - t_start)
                seg_n_points = max(2, int(n_points * seg_fraction))
                
                # Apply first rotation before evolution (as in simulation method)
                rho_after_ry = rotation_theta1 * rho_current * rotation_theta1_dag  # type: ignore
                
                # Evolve segment
                seg_times = np.linspace(seg_start, seg_end, seg_n_points)
                result = solver.run(rho_after_ry, tlist=seg_times, args=args)
                
                # Extract data for this segment
                for i, rho_t in enumerate(result.states):
                    # Apply second rotation for measurement
                    rho_meas = rotation_theta2 * rho_t * rotation_theta2_dag  # type: ignore
                    
                    # Measure qubit probabilities
                    p0 = float(self.prob0(rho_meas))
                    p1 = float(self.prob1(rho_meas))
                    
                    all_times.append(seg_times[i])
                    prob_0_list.append(p0*(1-cumulative_prob))  # Cumulative prob
                    prob_1_list.append(cumulative_prob + p1*(1-cumulative_prob))  # Cumulative prob
                    
                    # Calculate populations (take real part since expectation values should be real)
                    cavity_pop = float(np.real(qt.expect(n_cavity, rho_t))*(1-cumulative_prob))
                    field_pop = float(np.real(qt.expect(n_field, rho_t))*(1-cumulative_prob))
                    cavity_population_list.append(cavity_pop)
                    field_population_list.append(field_pop)
                
                # Perform projective measurement at end of segment (if not the final segment)
                if seg_end != t_end:
                    rho_evolved = result.states[-1]
                    
                    # Apply second rotation for measurement (as in simulation method)
                    rho_final = rotation_theta2 * rho_evolved * rotation_theta2_dag  # type: ignore
                    
                    # Get probability of detecting |1⟩ at this measurement
                    p1_measurement = float(self.prob1(rho_final))
                    
                    # Update cumulative detection probability
                    # P(detect at least once) = P(already detected) + P(not yet detected) * P(detect now)
                    cumulative_prob = cumulative_prob + (1.0 - cumulative_prob) * p1_measurement
                    
                    # Project onto |0⟩ state (matches simulation protocol)
                    proj_0 = self._cached_projector_0
                    rho_projected = proj_0 * rho_final * proj_0  # type: ignore
                    
                    # Normalize
                    trace_val = rho_projected.tr()
                    rho_current = rho_projected if trace_val == 0 else rho_projected / trace_val
        else:
            # Continuous evolution without intermediate measurements
            # Apply first rotation before evolution
            rho_after_ry = rotation_theta1 * rho_current * rotation_theta1_dag  # type: ignore
            
            times = np.linspace(t_start, t_end, n_points)
            result = solver.run(rho_after_ry, tlist=times, args=args)
            
            for i, rho_t in enumerate(result.states):
                # Apply second rotation
                rho_final = rotation_theta2 * rho_t * rotation_theta2_dag  # type: ignore

                # Measure qubit probabilities
                p0 = float(self.prob0(rho_final))
                p1 = float(self.prob1(rho_final))

                all_times.append(times[i])
                prob_0_list.append(p0)
                prob_1_list.append(p1)

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
            probabilities={"prob_0": np.array(prob_0_list), "prob_1": np.array(prob_1_list)},
            pulse_shape=pulse_shape,
            measurement_times=measurement_times,
            cavity_population=np.array(cavity_population_list),
            field_population=np.array(field_population_list),            metadata={
                "chi": self.experimental_params.chi[0],
                "gamma": self.experimental_params.photon_cavity_coupling,
                "with_interaction": with_interaction,
            },
        )

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
        Sweep over chi and gamma parameters to find optimal values.

        This method evaluates sensing contrast and detection probability across
        a 2D grid of chi (dispersive coupling) and gamma (cavity decay rate)
        values.

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

    def _get_cached_measurement_times(self) -> List[float]:
        """Return cached measurement times, ensuring they are up to date."""
        self._ensure_measurement_interval_sync()
        times_list = self.experimental_params._measurement_times_list
        if times_list is None:
            self.experimental_params._update_measurement_times()
            times_list = self.experimental_params._measurement_times_list
        if times_list is None:
            return []
        return [float(t) for t in times_list]

    def _get_measurement_interval_parameter(self) -> Optional[Parameter]:
        """Return the first measurement-interval parameter if configured."""
        for param in self.trainable_params.parameters:
            if param.param_type == ParameterType.MEASUREMENT_TIME:
                return param
        return None

    def _ensure_measurement_interval_sync(self) -> float:
        """Ensure experiment and trainable parameter time intervals stay aligned."""
        param = self._get_measurement_interval_parameter()
        if param is not None:
            interval = float(param.value)
            if interval <= 0:
                raise ValueError("Measurement interval in trainable parameters must be positive")
            current = float(self.experimental_params.measurement.time_interval)
            if not np.isclose(current, interval):
                self.experimental_params.measurement.time_interval = interval
                self.experimental_params.measurement.measurement_times = None
                self.experimental_params._update_measurement_times()
        else:
            interval = float(self.experimental_params.measurement.time_interval)
            if interval <= 0:
                raise ValueError("Measurement interval must be positive")
        return interval

    def get_measurement_interval(self) -> float:
        """Return the unified measurement interval value used by the experiment."""
        return self._ensure_measurement_interval_sync()

    def optimize_rotations(
        self,
        num_steps: int = 100,
        batch_size: int = 1,
        tolerance: float = 1e-6,
        verbose: bool = True,
        verbose_step: int = 10,
        callback: Optional[OptimizationCallback] = None,
        theta_init: Optional[List[float]] = None,
    ) -> OptimizationCallback:
        """
        Optimize rotation angles to maximize sensing contrast.

        This routine keeps measurement times fixed and performs gradient-based
        optimization only over the first two rotation angles defined in
        ``TrainableParameters``. Measurement-time refinements are handled by
        :meth:`optimize_measurement_times`.

        Args:
            batch_size: Number of random realizations for measurement uncertainty per step
            num_steps: Maximum number of optimization steps
            tolerance: Convergence threshold for gradient norm
            verbose: Print progress information
            verbose_step: Step interval for printing progress
            callback: Optional callback to track optimization progress.
                     If None, uses the experiment's default callback (saves every epoch).
            theta_init: Optional initial rotation angles [θ₁, θ₂] in radians.
                       If None, uses values from ``trainable_params``.

        Returns:
            OptimizationCallback with full optimization history.
        """
        # Use provided callback or default to self.callback
        if callback is None:
            callback = self.callback

        # Reset callback at start of new optimization
        callback.reset()

        self._ensure_measurement_interval_sync()

        # Get initial state
        rho0 = self._cached_initial_state
        if rho0 is None:
            raise RuntimeError("Initial state cache is not initialized.")

        # Get solvers
        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()

        # Get rotation angles (must have at least 2)
        rotation_angles = self.trainable_params.get_rotation_angles()
        if len(rotation_angles) < 2:
            raise ValueError("Need at least 2 rotation angle parameters")

        # Get parameter names for the first two rotation angles
        rotation_names = list(rotation_angles.keys())
        theta1_name = rotation_names[0]
        theta2_name = rotation_names[1]

        # Find indices for rotation parameters
        theta1_idx = -1
        theta2_idx = -1

        for param in self.trainable_params.parameters:
            if param.name == theta1_name:
                theta1_idx = param.index
            elif param.name == theta2_name:
                theta2_idx = param.index

        if theta1_idx == -1 or theta2_idx == -1:
            raise ValueError(
                f"Could not find rotation parameters {theta1_name} and/or {theta2_name}"
            )

        # Initialize parameter vector
        # Use provided theta_init or current values
        if theta_init is not None:
            if len(theta_init) != 2:
                raise ValueError("theta_init must contain exactly 2 angles [θ₁, θ₂]")
            initial_theta1 = theta_init[0]
            initial_theta2 = theta_init[1]
        else:
            initial_theta1 = rotation_angles[theta1_name][0]
            initial_theta2 = rotation_angles[theta2_name][0]

        params = jnp.array([initial_theta1, initial_theta2], dtype=float)
        param_indices = [theta1_idx, theta2_idx]

        # Update trainable_params with initial values
        self.trainable_params.parameters[theta1_idx].value = float(initial_theta1)
        self.trainable_params.parameters[theta2_idx].value = float(initial_theta2)
        trainable_mask = jnp.array(
            [self.trainable_params.parameters[idx].trainable for idx in param_indices]
        )

        # Use optimizer from first trainable rotation parameter
        optimizer = self.trainable_params.rotation_optimizer
        opt_state = optimizer.init(params)

        # Define objective function that returns probabilities along with loss
        def objective_function(opt_params):
            """Negative sensing contrast for minimization with batch averaging.

            Args:
                opt_params: Array of parameters [theta1, theta2]

            Returns:
                tuple: (loss, (prob_with, prob_without, contrast))
                      All values are averaged over batch_size realizations
            """
            # pylint: disable=unsubscriptable-object
            # Extract parameters based on what we're optimizing
            theta0_raw, theta1_raw = opt_params
            theta0 = theta0_raw if trainable_mask[0] else jax.lax.stop_gradient(theta0_raw)
            theta1 = theta1_raw if trainable_mask[1] else jax.lax.stop_gradient(theta1_raw)

            rotation_pair = self._prepare_rotation_gates(theta0, theta1)

            if batch_size == 1:
                # Single realization: use current measurement times (supports uncertainty)
                measurement_times_batch = (
                    self.experimental_params.get_measurement_times_with_uncertainty()
                )
                # Calculate sensing contrast for this realization
                prob_with = self.simulation(
                    solver_with,
                    rho0,
                    theta0,
                    theta1,
                    measurement_times_batch,
                    precomputed_rotations=rotation_pair,
                )
                prob_without = self.simulation(
                    solver_without,
                    rho0,
                    theta0,
                    theta1,
                    measurement_times_batch,
                    precomputed_rotations=rotation_pair,
                )
                sensing_contrast = prob_with - prob_without

            else:
                # Multiple realizations: generate uncertainty realizations at once
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
                            theta0,
                            theta1,
                            measurement_times,
                            precomputed_rotations=rotation_pair,
                        )
                    )
                    prob_without_batch = prob_without_batch.at[i].set(
                        self.simulation(
                            solver_without,
                            rho0,
                            theta0,
                            theta1,
                            measurement_times,
                            precomputed_rotations=rotation_pair,
                        )
                    )

                # Average over batch using JAX operations (efficient)
                prob_with = jnp.mean(prob_with_batch)
                prob_without = jnp.mean(prob_without_batch)
                sensing_contrast = prob_with - prob_without

            # Return negative for minimization (we want to maximize contrast)
            # Also return aux data (probabilities and contrast)
            return -sensing_contrast, (prob_with, prob_without, sensing_contrast)

        if verbose:
            theta_initial_vals = np.asarray(params, dtype=float)
            print(f"Configuration:")
            print(f"    Max iterations: {num_steps}")
            print(f"    Batch size: {batch_size}")
            print(f"    Convergence tolerance: {tolerance:.2e}")
            theta1_status = " [FIXED]" if not trainable_mask[0] else ""
            theta2_status = " [FIXED]" if not trainable_mask[1] else ""
            print(
                f"    Initial rotation parameters: {theta1_name}={theta_initial_vals[0]:.3f} rad{theta1_status}, "
                f"{theta2_name}={theta_initial_vals[1]:.3f} rad{theta2_status}"
            )
            uncertainty = self.experimental_params.initial_time_uncertainty
            if uncertainty > 0:
                spec = self.experimental_params.initial_time_uncertainty_spec
                extra = f" (specified as '{spec}')" if isinstance(spec, str) else ""
                print(f"    Measurement uncertainty: ±{uncertainty:.3f}{extra}")
            measurement_params = [
                param
                for param in self.trainable_params.parameters
                if param.param_type == ParameterType.MEASUREMENT_TIME
            ]
            if measurement_params:
                print("    Measurement interval:")
                for param in measurement_params:
                    status = " [FIXED]" if not param.trainable else ""
                    print(f"        {param.name}={param.value:.6f}{status}")
            print("=" * 70)
            print(f"{'Step':<6}{theta1_name:<12}{theta2_name:<12}{'Contrast':<12}{'Grad Norm'}")
            print("-" * 70)

        best_contrast = -np.inf
        best_params = jnp.array(params)  # Make a copy using jnp

        # Initialize variables
        step = 0
        grad_norm = float("inf")

        for step in range(num_steps):
            # Compute gradients using JAX autodiff with auxiliary data
            # This computes the simulation only once and returns probabilities
            grads, (prob_with, prob_without, sensing_contrast) = jax.grad(
                objective_function, has_aux=True
            )(params)

            # Track best parameters
            if sensing_contrast > best_contrast:
                best_contrast = sensing_contrast
                best_params = jnp.array(params)  # Copy using jnp

            # Call callback to track progress
            callback(
                trainable_params=self.trainable_params,
                prob_with=float(prob_with),
                prob_without=float(prob_without),
                contrast=float(sensing_contrast),
            )

            grad_norm = float(jnp.linalg.norm(grads))
            theta_values = np.asarray(params, dtype=float)
            theta0_val = float(theta_values[0])
            theta1_val = float(theta_values[1])

            # Progress output
            if verbose and (step % verbose_step == 0 or grad_norm < tolerance):
                print(
                    f"{step:<6}{theta0_val:<12.6f}{theta1_val:<12.6f}"
                    f"{float(sensing_contrast):<12.6f}{grad_norm:<12.2e}"
                )

            # Convergence check
            if grad_norm < tolerance:
                break

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Update trainable parameters continuously
            # pylint: disable=unsubscriptable-object
            self.trainable_params.parameters[theta1_idx].value = theta0_val
            self.trainable_params.parameters[theta2_idx].value = theta1_val

        # Ensure best parameters are set at the end
        # pylint: disable=unsubscriptable-object
        best_values = np.asarray(best_params, dtype=float)
        for idx, param_idx in enumerate(param_indices):
            self.trainable_params.parameters[param_idx].value = float(best_values[idx])

        # Apply constraints at the end
        final_values = np.array([p.value for p in self.trainable_params.parameters])
        constrained_values = self.trainable_params.apply_constraints(final_values)
        for i, val in enumerate(constrained_values):
            self.trainable_params.parameters[i].value = float(val)

        if verbose:
            print("=" * 70)
            print(f"Final gradient norm: {grad_norm:.2e}")
            print(f"Best sensing contrast: {best_contrast:.6f}")
            print(
                f"Best parameters: {theta1_name}={best_values[0]:.3f} rad, "
                f"{theta2_name}={best_values[1]:.3f} rad"
            )

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

        This helper mirrors :func:`qsopt.utils.landscape_analysis.compute_time_interval_landscape`
        and applies the best-performing interval to the experiment configuration.
        Current rotation angles are used automatically.

        Args:
            resolution: Number of interval samples to evaluate (minimum 2). When None,
                falls back to defaults stored in ``TrainableParameters`` or 50.
            mode: Interval sampling mode, ``'continuous'`` or ``'discrete'``.
            batch_size: Number of uncertainty realizations per interval.
            verbose: Print progress feedback when True.
            min_interval: Optional lower bound on the interval sweep. Defaults to stored
                measurement-interval settings when available.
            max_interval: Optional upper bound on the interval sweep. Defaults to stored
                measurement-interval settings when available.

        Returns:
            Dictionary returned by ``compute_time_interval_landscape`` with additional keys:
                - ``'best_interval'``: Interval delivering the highest contrast.
                - ``'best_contrast'``: Maximum contrast observed.
                - ``'best_index'``: Index of the optimal interval in the sampled array.
        """

        from qsopt.utils.landscape_analysis import compute_time_interval_landscape

        self._ensure_measurement_interval_sync()

        rotation_angles = self.rotation_angles
        if len(rotation_angles) < 2:
            raise ValueError("Need at least 2 rotation angle parameters")

        theta_values = list(rotation_angles.values())
        theta1 = theta_values[0]
        theta2 = theta_values[1]

        interval_defaults = self.trainable_params.get_measurement_interval_defaults()
        default_resolution = interval_defaults.get("grid_resolution") if interval_defaults else None
        resolved_resolution = resolution or default_resolution or 50
        resolved_resolution = int(resolved_resolution)

        resolved_min_interval = min_interval
        resolved_max_interval = max_interval
        if resolved_min_interval is None and interval_defaults:
            resolved_min_interval = interval_defaults.get("grid_min")
        if resolved_max_interval is None and interval_defaults:
            resolved_max_interval = interval_defaults.get("grid_max")

        if resolved_min_interval is not None:
            resolved_min_interval = float(resolved_min_interval)
        if resolved_max_interval is not None:
            resolved_max_interval = float(resolved_max_interval)
        if (
            resolved_min_interval is not None
            and resolved_max_interval is not None
            and resolved_min_interval > resolved_max_interval
        ):
            resolved_min_interval, resolved_max_interval = (
                resolved_max_interval,
                resolved_min_interval,
            )

        results = compute_time_interval_landscape(
            self.experimental_params,
            theta1=theta1,
            theta2=theta2,
            resolution=resolved_resolution,
            mode=mode,
            batch_size=batch_size,
            verbose=verbose,
            min_interval=resolved_min_interval,
            max_interval=resolved_max_interval,
        )

        contrast_vals_np = np.asarray(results["contrast_vals"], dtype=float)
        interval_vals_np = np.asarray(results["interval_vals"], dtype=float)
        best_index = int(np.argmax(contrast_vals_np))
        best_interval = float(interval_vals_np[best_index])
        best_contrast = float(contrast_vals_np[best_index])

        # Apply best interval to experimental parameters
        self.experimental_params.measurement.time_interval = best_interval
        self.experimental_params.measurement.measurement_times = None
        self.experimental_params._update_measurement_times()

        # Keep trainable parameters in sync if measurement interval exists
        for param in self.trainable_params.parameters:
            if param.param_type == ParameterType.MEASUREMENT_TIME:
                param.value = best_interval

        results_with_best = dict(results)
        results_with_best["best_interval"] = best_interval
        results_with_best["best_contrast"] = best_contrast
        results_with_best["best_index"] = best_index

        self._ensure_measurement_interval_sync()

        return results_with_best

    def optimize(self, *args, **kwargs) -> OptimizationCallback:
        """Deprecated wrapper for :meth:`optimize_rotations`."""
        warnings.warn(
            "SingleQubitExperiment.optimize is deprecated; use optimize_rotations instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.optimize_rotations(*args, **kwargs)

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
        Save a comprehensive experiment report to a JSON file.

        This method creates a detailed report containing:
        - All experimental parameters (physical constants, dimensions, noise config, etc.)
        - Trainable parameters and their current values
        - Latest callback information (if available)
        - For optimization callbacks: saves detailed results to NPZ file

        Args:
            save_path: Path where the JSON report will be saved (default: 'results/report.json')
                      The directory will be created if it doesn't exist.
        """
        import json
        from pathlib import Path

        # Create results directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Build report dictionary
        self._ensure_measurement_interval_sync()
        cached_times = self._get_cached_measurement_times()

        # Handle chi and qubit_levels which may be lists
        chi = self.experimental_params.chi
        if isinstance(chi, list):
            chi = chi[0]  # Single qubit uses first element

        qubit_levels = self.experimental_params.qubit_levels
        if isinstance(qubit_levels, list):
            qubit_levels = qubit_levels[0]

        report = {
            "experiment_type": "SingleQubitExperiment",
            "version": "0.1.0",
            "experimental_parameters": {
                "physical_constants": {
                    "chi": float(chi),
                    "photon_cavity_coupling": float(
                        self.experimental_params.photon_cavity_coupling
                    ),
                    "inverse_pulse_width": float(self.experimental_params.inverse_pulse_width),
                },
                "system_dimensions": {
                    "cavity_levels": int(self.experimental_params.cavity_levels),
                    "qubit_levels": int(qubit_levels),
                    "field_levels": int(self.experimental_params.field_levels),
                },
                "measurement_protocol": {
                    # Store the mode (explicit list vs interval-based)
                    "mode": (
                        "explicit"
                        if self.experimental_params.measurement.measurement_times is not None
                        else "interval"
                    ),
                    # If explicit mode, store the list
                    "measurement_times": (
                        [float(t) for t in self.experimental_params.measurement.measurement_times]
                        if self.experimental_params.measurement.measurement_times is not None
                        else None
                    ),
                    # If interval mode, store the interval parameters
                    "initial_time": (
                        float(self.experimental_params.measurement.initial_time)
                        if self.experimental_params.measurement.measurement_times is None
                        else None
                    ),
                    "final_time": (
                        float(self.experimental_params.measurement.final_time)
                        if self.experimental_params.measurement.measurement_times is None
                        else None
                    ),
                    "time_interval": (
                        float(self.experimental_params.measurement.time_interval)
                        if self.experimental_params.measurement.measurement_times is None
                        else None
                    ),
                    # Always store uncertainty settings
                    "initial_time_uncertainty": float(
                        self.experimental_params.initial_time_uncertainty
                    ),
                    # Computed times for reference
                    "computed_times": cached_times,
                    "num_measurements": len(cached_times),
                },
                "initial_state": {
                    "state_type": self.experimental_params.initial_state.state_type.value,
                    "coherent_alpha": (
                        None
                        if self.experimental_params.initial_state.coherent_alpha is None
                        else float(abs(self.experimental_params.initial_state.coherent_alpha))
                    ),
                    "coherent_alpha_phase": (
                        None
                        if self.experimental_params.initial_state.coherent_alpha is None
                        else float(np.angle(self.experimental_params.initial_state.coherent_alpha))
                    ),
                    "thermal_n_bar": (
                        None
                        if self.experimental_params.initial_state.thermal_n_bar is None
                        else float(self.experimental_params.initial_state.thermal_n_bar)
                    ),
                    "has_custom_amplitudes": self.experimental_params.initial_state.custom_amplitudes
                    is not None,
                },
                "noise_configuration": {
                    "depolarizing": (
                        float(self.experimental_params.noise_config.depolarizing[0])
                        if isinstance(self.experimental_params.noise_config.depolarizing, list)
                        else float(self.experimental_params.noise_config.depolarizing)
                    ),
                    "dephasing": (
                        float(self.experimental_params.noise_config.dephasing[0])
                        if isinstance(self.experimental_params.noise_config.dephasing, list)
                        else float(self.experimental_params.noise_config.dephasing)
                    ),
                    "relaxation": (
                        float(self.experimental_params.noise_config.relaxation[0])
                        if isinstance(self.experimental_params.noise_config.relaxation, list)
                        else float(self.experimental_params.noise_config.relaxation)
                    ),
                },
            },
            "trainable_parameters": {
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.param_type.value,
                        "value": (
                            float(param.value)
                            if not isinstance(param.value, list)
                            else [float(v) for v in param.value]
                        ),
                        "trainable": param.trainable,
                    }
                    for param in self.trainable_params.parameters
                ],
                "num_parameters": len(self.trainable_params.parameters),
                "num_trainable": len(self.trainable_params.get_trainable_indices()),
            },
            "callback_info": None,
        }

        # Add callback information if available
        if self.callback is not None and self.callback.epoch > 0:
            # Determine if this was an optimization or simulation
            is_optimization = self.callback.epoch > 1 or (
                self.callback.epoch == 1 and self.callback.converged is not False
            )

            callback_info = {
                "mode": "optimization" if is_optimization else "simulation",
                "total_epochs": int(self.callback.epoch),
                "converged": bool(self.callback.converged),
                "final_gradient_norm": (
                    None
                    if self.callback.final_grad_norm is None
                    else float(self.callback.final_grad_norm)
                ),
            }

            # Add best metrics if available
            if self.callback.best_metrics is not None:
                callback_info["best_metrics"] = {
                    "epoch": int(self.callback.best_metrics["epoch"]),
                    "contrast": float(self.callback.best_metrics["contrast"]),
                    "prob_with": float(self.callback.best_metrics["prob_with"]),
                    "prob_without": float(self.callback.best_metrics["prob_without"]),
                }

                # Add best parameters
                if self.callback.best_trainable_params is not None:
                    best_angles = self.callback.best_trainable_params.get_rotation_angles()
                    callback_info["best_parameters"] = {
                        name: {"value_rad": float(val[0]), "value_deg": float(np.rad2deg(val[0]))}
                        for name, val in best_angles.items()
                    }

            # For optimization runs, save detailed callback data to NPZ file
            if is_optimization:
                # Generate callback save path based on report path
                callback_save_path = save_path_obj.with_stem(
                    save_path_obj.stem + "_callback"
                ).with_suffix(".npz")

                # Save callback data
                self.callback.save(str(callback_save_path))
                callback_info["callback_data_path"] = str(callback_save_path)

                # Add summary statistics
                history = self.callback.get_history()
                if len(history["contrast"]) > 0:
                    callback_info["optimization_summary"] = {
                        "initial_contrast": float(history["contrast"][0]),
                        "final_contrast": float(history["contrast"][-1]),
                        "best_contrast": float(max(history["contrast"])),
                        "improvement": float(max(history["contrast"]) - history["contrast"][0]),
                    }

            report["callback_info"] = callback_info

        # Save report to JSON
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def plot_pulse_shape(
        self, save_path: Optional[str] = None, dpi: int = 300, batch_size: int = 1
    ):
        """
        Plot Gaussian pulse envelope with measurement time markers.

        Convenience method that visualizes the temporal shape of the Gaussian input pulse
        along with vertical markers indicating when measurements are performed. This helps
        understand the relationship between the pulse envelope and the measurement protocol.

        Args:
            save_path: Optional path to save the figure
            dpi: Resolution for saved figure (default: 300)
            batch_size: Number of measurement realizations to visualize. If > 1,
                measurement times are drawn using distinct colors (default: 1)

        Returns:
            matplotlib.figure.Figure: Figure object containing the plot

        Example:
            >>> experiment = SingleQubitExperiment(exp_params, train_params)
            >>> fig = experiment.plot_pulse_shape(save_path="results/pulse_shape.png")
            >>> # Plot shows pulse shape with measurement markers

        Note:
            - Uses current trainable parameters (theta1, theta2)
            - Pulse shape computed using u0() function from quantum_utils
            - Measurement times extracted from experimental_params.measurement
        """
        from qsopt.utils.visualization import plot_pulse_shape_with_measurements

        return plot_pulse_shape_with_measurements(
            self.experimental_params, save_path=save_path, dpi=dpi, batch_size=batch_size
        )

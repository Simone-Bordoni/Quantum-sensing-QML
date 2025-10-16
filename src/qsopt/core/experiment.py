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
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qutip as qt
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import TrainableParameters, ParameterType
from qsopt.core.quantum_utils import gu
from qsopt.core.callback import OptimizationCallback
from qsopt.core.quantum_utils import (
    generate_single_qubit_operators,
    generate_initial_state,
    apply_single_qubit_rotation,
    create_measurement_projector,
)

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import

# Suppress Diffrax complex dtype warning
warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax is a work in progress*")

class SingleQubitExperiment:
    """
    A class representing a single qubit photon detection experiment.
    
    This class implements the quantum sensing protocol with a three-system composite 
    Hilbert space: input_cavity ⊗ resonator_cavity ⊗ qubit.
    
    The system workflow:
    |ψ₀⟩ → Ry(θ₁) → H(t) Evolution → Ry(θ₂) → Measurement → Detection Probability
    """
    
    def __init__(self, experimental_params: ExperimentalParameters, trainable_params: TrainableParameters):
        self.experimental_params = experimental_params
        self.trainable_params = trainable_params
        
        # Storage for operators and Hamiltonians
        self.operators: Optional[Dict[str, qt.Qobj]] = None
        self.hamiltonians: Optional[Dict[str, Union[qt.QobjEvo, qt.Qobj]]] = None
        self.lindblad_operators: Optional[Dict[str, List[Union[qt.Qobj, qt.QobjEvo]]]] = None
        
        # Optimization callback (default: save every epoch)
        self.callback: OptimizationCallback = OptimizationCallback(save_every=1, save_best=True)
        
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
        qubit_levels = self.experimental_params.qubit_levels
        
        # Use utility function to generate all operators
        self.operators = generate_single_qubit_operators(
            field_levels,
            cavity_levels,
            qubit_levels
        )

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
        chi = self.experimental_params.chi
        sigma = self.experimental_params.inverse_pulse_width
        
        # Get operators
        a_in = self.operators['a_in']
        a_in_dag = self.operators['a_in_dag']
        a = self.operators['a']
        a_dag = self.operators['a_dag']
        sigma_z = self.operators['sigma_z']
        sigma_x = self.operators['sigma_x']
        sigma_y = self.operators['sigma_y']
        sigma_minus = self.operators['sigma_minus']
        
        # Time-dependent coupling function arguments
        args = {'sigma': sigma}
        
        # Time-dependent cavity-cavity coupling Hamiltonian
        coupling_coeff = 1j/2 * jnp.sqrt(gm)
        H_coupling = qt.Qobj( coupling_coeff * (a_in_dag * a - a_in * a_dag) )  # type: ignore

        # Dispersive qubit-resonator interaction Hamiltonian
        H_dispersive = qt.Qobj( -chi * a_dag * a * sigma_z )  # type: ignore

        # Complete time-dependent Hamiltonian
        H_total = qt.QobjEvo([H_dispersive, [H_coupling, gu]], args=args)
        
        # Noise configuration
        noise_config = self.experimental_params.noise_config
        
        # Build Lindblad noise operators
        lindblad_noise: List[Union[qt.Qobj, qt.QobjEvo]] = []
        
        # Depolarizing noise (σx, σy, σz components)
        if noise_config.depolarizing != 0.0:
            gamma_depol = noise_config.depolarizing
            lindblad_noise.extend([
                np.sqrt(gamma_depol/3) * sigma_x,  # σx component
                np.sqrt(gamma_depol/3) * sigma_y,  # σy component  
                np.sqrt(gamma_depol/3) * sigma_z   # σz component
            ])
        
        # Pure dephasing noise (σz)
        if noise_config.dephasing != 0.0:
            lindblad_noise.append(np.sqrt(noise_config.dephasing) * sigma_z)
        
        # Relaxation noise (σ-)
        if noise_config.relaxation != 0.0:
            lindblad_noise.append(np.sqrt(noise_config.relaxation) * sigma_minus)
        
        # Add custom Lindblad operators if provided
        if noise_config.custom_operators is not None:
            lindblad_noise.extend(noise_config.custom_operators)
        
        # Lindblad interaction operators
        L_int = qt.QobjEvo([a_in, gu], args=args) + np.sqrt(gm) * a
    
        interaction_ops: List[Union[qt.Qobj, qt.QobjEvo]] = [L_int] + lindblad_noise
        no_interaction_ops: List[Union[qt.Qobj, qt.QobjEvo]] = lindblad_noise
        
        # Store Hamiltonians and Lindblad operators
        self.hamiltonians = {
            'total': H_total,
            'dispersive': H_dispersive,
            'coupling': H_coupling
        }
        
        self.lindblad_operators = {
            'interaction': interaction_ops,
            'no_interaction': no_interaction_ops,
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
            self.experimental_params.qubit_levels
        )
        self._cached_projector_1 = create_measurement_projector(
            1,
            self.experimental_params.field_levels,
            self.experimental_params.cavity_levels,
            self.experimental_params.qubit_levels
        )
        
        # Cache initial state (doesn't change during optimization)
        self._cached_initial_state = generate_initial_state(
            self.experimental_params.initial_state,
            self.experimental_params.field_levels,
            self.experimental_params.cavity_levels,
            self.experimental_params.qubit_levels,
            num_qubits=1
        )
    
    def get_initial_state(self) -> qt.Qobj:
        """
        Get the cached initial state density matrix.
        
        Returns:
            qt.Qobj: Initial density matrix for the experiment
        """
        if self._cached_initial_state is None:
            raise RuntimeError("Initial state has not been initialized. This should not happen after __post_init__.")
        return self._cached_initial_state
    
    def get_solver_with_interaction(self) -> qt.MESolver:
        """
        Get Lindblad master equation solver WITH input photon interaction (cached).
        
        Solver is created once and cached for performance during optimization loops.
        
        Returns:
            qt.MESolver: Configured solver for signal case evolution
        """
        if 'with_interaction' not in self._cached_solvers:
            if self.hamiltonians is None or self.lindblad_operators is None:
                raise RuntimeError("Hamiltonian and operators must be generated first")
            
            self._cached_solvers['with_interaction'] = qt.MESolver(
                self.hamiltonians['total'], 
                self.lindblad_operators['interaction'],
                options={"method": "diffrax", "normalize_output": False}
            )
        
        return self._cached_solvers['with_interaction']
    
    def get_solver_no_interaction(self) -> qt.MESolver:
        """
        Get Lindblad master equation solver WITHOUT input photon interaction (cached).
        
        Solver is created once and cached for performance during optimization loops.
        
        Returns:
            qt.MESolver: Configured solver for reference case evolution
        """
        if 'no_interaction' not in self._cached_solvers:
            if self.hamiltonians is None or self.lindblad_operators is None:
                raise RuntimeError("Hamiltonian and operators must be generated first")
            
            self._cached_solvers['no_interaction'] = qt.MESolver(
                self.hamiltonians['dispersive'], 
                self.lindblad_operators['no_interaction'],
                options={"method": "diffrax", "normalize_output": False}
            )
        
        return self._cached_solvers['no_interaction']
    
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
        if self.operators is None:
            raise RuntimeError("Operators not initialized")
        
        # Use utility function for rotation
        return apply_single_qubit_rotation(
            rho,
            theta,
            'y',
            self.operators['I_field'],
            self.operators['I_cavity']
        )
    
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
    
    def simulation(self, solver: qt.MESolver, rho: qt.Qobj, 
                   theta1: float, theta2: float, 
                   measurements: Union[List[float], np.ndarray],
                   args: Optional[Dict] = None):
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
                
        Returns:
            float: Probability of detecting at least one excited state
                P(detection) = 1 - ∏ᵢ P(|0⟩ᵢ) ∈ [0,1]
        """
        if args is None:
            args = {'sigma': self.experimental_params.inverse_pulse_width}
            
        tmeas = measurements
        probability_list = []
        
        # Process each measurement interval sequentially
        for kt in range(len(tmeas)-1):
            t0, t1 = tmeas[kt], tmeas[kt+1]
            
            # Step 1: Apply first rotation Ry(θ₁) for state preparation
            rho_after_ry = self.ry_rotation(rho, theta1)
            
            # Step 2: Time evolution under system Hamiltonian H(t)
            evolution_result = solver.run(rho_after_ry, [t0, t1], args=args)
            rho_evolved = evolution_result.states[-1]
            
            # Step 3: Apply second rotation Ry(θ₂) for measurement optimization
            rho_final = self.ry_rotation(rho_evolved, theta2)
            
            # Step 4: Measure qubit in |0⟩ state (ground state probability)
            prob_ground = self.prob0(rho_final)
            probability_list.append(prob_ground)
            
            # Step 5: Project onto measurement result for conditional evolution
            rho = self.proj0(rho_final)  # Always project to |0⟩
            rho = rho / rho.tr()          # Normalize

        # Calculate detection probability: P(at least one |1⟩) = 1 - P(all |0⟩)
        prob_all_ground = jnp.prod(jnp.array(probability_list))
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
        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()
        
        # Run simulations with batch averaging over uncertainty realizations
        prob_with_list = []
        prob_without_list = []
        
        for _ in range(batch_size):
            # Get measurement times with uncertainty (uses random_seed if set)
            # Each iteration gets a different realization if uncertainty > 0
            measurement_times = self.experimental_params.get_measurement_times_with_uncertainty()
            
            # Run simulations for this realization
            prob_with_batch = self.simulation(solver_with, rho0, theta1, theta2, measurement_times)
            prob_without_batch = self.simulation(solver_without, rho0, theta1, theta2, measurement_times)
            
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
            contrast=float(contrast)
        )
        
        # Keep optimization-related attributes as None/False (not from optimization)
        # converged and final_grad_norm remain as initialized (False, None)
        
        return callback
    
    def optimize(self, 
                 num_steps: int = 100,
                 batch_size: int = 1,
                 tolerance: float = 1e-6,
                 verbose: bool = True,
                 verbose_step: int = 10,
                 callback: Optional[OptimizationCallback] = None,
                 theta_init: Optional[List[float]] = None) -> OptimizationCallback:
        """
        Optimize trainable parameters to maximize sensing contrast.
        
        Uses JAX automatic differentiation to find optimal parameter values that
        maximize the difference between detection probabilities with and without
        photon present. Supports optimization of rotation angles, time_interval, 
        and custom parameters.
        
        Args:
            batch_size: Number of random realizations for measurement uncertainty per step
            num_steps: Maximum number of optimization steps
            tolerance: Convergence threshold for gradient norm
            verbose: Print progress information
            verbose_step: Step interval for printing progress
            callback: Optional callback to track optimization progress.
                     If None, uses the experiment's default callback (saves every epoch).
                     Pass a custom OptimizationCallback for different behavior.
            theta_init: Optional initial rotation angles [θ₁, θ₂] in radians.
                       If None, uses values from trainable_params.
            
        Returns:
            OptimizationCallback: The callback instance containing all optimization data:
                - callback.epoch: Total number of iterations performed
                - callback.converged: Whether optimization converged (gradient norm < tolerance)
                - callback.final_grad_norm: Final gradient norm value
                - callback.best_contrast: Best contrast value achieved
                - callback.best_trainable_params: Best parameters found
                - callback.best_metrics: Metrics at best parameters (epoch, contrast, probs)
                - callback.history: Complete optimization history (epochs, contrast, probs, params)
        """
        # Use provided callback or default to self.callback
        if callback is None:
            callback = self.callback
        
        # Reset callback at start of new optimization
        callback.reset()
        
        # Get initial state
        rho0 = self._cached_initial_state
        
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
        
        # Find indices for all parameters
        theta1_idx = -1
        theta2_idx = -1
        time_interval_idx = -1
        
        for param in self.trainable_params.parameters:
            if param.name == theta1_name:
                theta1_idx = param.index
            elif param.name == theta2_name:
                theta2_idx = param.index
            elif param.param_type == ParameterType.MEASUREMENT_TIME:
                time_interval_idx = param.index
        
        if theta1_idx == -1 or theta2_idx == -1:
            raise ValueError(f"Could not find rotation parameters {theta1_name} and/or {theta2_name}")
        
        # Check if we're optimizing time_interval
        has_trainable_interval = (time_interval_idx != -1 and 
                                  self.trainable_params.parameters[time_interval_idx].trainable)
        
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
        
        # Build parameter vector: [theta1, theta2, time_interval (if trainable)]
        if has_trainable_interval:
            initial_time_interval = self.trainable_params.parameters[time_interval_idx].value
            params = jnp.array([initial_theta1, initial_theta2, initial_time_interval], dtype=float)
            param_indices = [theta1_idx, theta2_idx, time_interval_idx]
        else:
            params = jnp.array([initial_theta1, initial_theta2], dtype=float)
            param_indices = [theta1_idx, theta2_idx]
        
        # Update trainable_params with initial values
        self.trainable_params.parameters[theta1_idx].value = float(initial_theta1)
        self.trainable_params.parameters[theta2_idx].value = float(initial_theta2)
        
        # Check which parameters are trainable
        trainable_mask = jnp.array([self.trainable_params.parameters[idx].trainable 
                                   for idx in param_indices])
        
        # Get optimizers from trainable_params
        if len(self.trainable_params.optimizers) == 0:
            raise ValueError("No optimizer defined in trainable_params.")
        
        # Use optimizer from first trainable rotation parameter
        optimizer = self.trainable_params.optimizers[theta1_idx]
        opt_state = optimizer.init(params)
        
        # Base random key for measurement uncertainty (fixed seed for reproducibility)
        base_rng_key = jax.random.PRNGKey(42)
        
        # Step counter for generating random keys (accessible in objective function)
        step_counter = [0]  # Use list to allow modification in nested function
        
        # Helper function to apply measurement uncertainty (JAX-compatible)
        def apply_measurement_uncertainty(times: jnp.ndarray, key) -> jnp.ndarray:
            """
            Apply random time shift to measurement times (JAX-compatible).
            
            Args:
                times: Base measurement times
                key: JAX random key
                
            Returns:
                Measurement times with uncertainty shift applied
            """
            uncertainty = self.experimental_params.measurement.initial_time_uncertainty
            if uncertainty > 0:
                # Generate random shift using JAX random (traceable by JAX)
                shift = jax.random.uniform(key, minval=-uncertainty, maxval=uncertainty)
                return times + shift
            else:
                return times
        
        # Helper function to update time_interval in both locations (avoids repetition)
        def update_time_interval(new_interval: float) -> None:
            """
            Update time_interval in both trainable_params and experimental_params.
            
            This ensures synchronization between:
            - trainable_params.parameters[time_interval_idx].value
            - experimental_params.measurement.time_interval
            - experimental_params._measurement_times_list (recomputed)
            
            Must be called outside JAX tracing (after gradient computation).
            """
            self.trainable_params.parameters[time_interval_idx].value = float(new_interval)
            self.experimental_params.measurement.time_interval = float(new_interval)
            self.experimental_params._update_measurement_times()
        
        # Define objective function that returns probabilities along with loss
        def objective_function(opt_params):
            """Negative sensing contrast for minimization with batch averaging.
            
            Args:
                opt_params: Array of parameters [theta1, theta2, time_interval (optional)]
                
            Returns:
                tuple: (loss, (prob_with, prob_without, contrast))
                      All values are averaged over batch_size realizations
            """
            # pylint: disable=unsubscriptable-object
            # Extract parameters based on what we're optimizing
            if has_trainable_interval:
                theta0_raw, theta1_raw, time_interval_raw = opt_params
                
                # Apply stop_gradient to non-trainable parameters
                theta0 = theta0_raw if trainable_mask[0] else jax.lax.stop_gradient(theta0_raw)
                theta1 = theta1_raw if trainable_mask[1] else jax.lax.stop_gradient(theta1_raw)
                time_interval = time_interval_raw if trainable_mask[2] else jax.lax.stop_gradient(time_interval_raw)
                
            else:
                theta0_raw, theta1_raw = opt_params
                
                # Apply stop_gradient to non-trainable parameters  
                theta0 = theta0_raw if trainable_mask[0] else jax.lax.stop_gradient(theta0_raw)
                theta1 = theta1_raw if trainable_mask[1] else jax.lax.stop_gradient(theta1_raw)
                time_interval = None
            
            if batch_size == 1:
                # Single realization: no need for batching overhead
                if has_trainable_interval:
                    # Compute measurement times directly from current time_interval
                    # This allows JAX autodiff to compute gradients w.r.t. time_interval
                    t_start = self.experimental_params.measurement.initial_time
                    t_end = self.experimental_params.measurement.final_time
                    measurement_times_batch = jnp.arange(t_start, t_end + time_interval/2, time_interval)
                else:
                    # Use pre-computed measurement times (supports uncertainty)
                    measurement_times_batch = self.experimental_params.get_measurement_times_with_uncertainty()
                
                # Calculate sensing contrast for this realization
                prob_with = self.simulation(solver_with, rho0, theta0, theta1, measurement_times_batch)
                prob_without = self.simulation(solver_without, rho0, theta0, theta1, measurement_times_batch)
                sensing_contrast = prob_with - prob_without
                
            else:
                # Multiple realizations: use vectorization for better performance
                # Pre-allocate JAX arrays
                if has_trainable_interval:
                    # For trainable interval, generate base times then apply uncertainty per batch element
                    t_start = self.experimental_params.measurement.initial_time
                    t_end = self.experimental_params.measurement.final_time
                    base_measurement_times = jnp.arange(t_start, t_end + time_interval/2, time_interval)
                    
                    # Each batch element gets different uncertainty realization
                    prob_with_batch = jnp.zeros(batch_size)
                    prob_without_batch = jnp.zeros(batch_size)
                    
                    # Generate random key for this step (reproducible based on step counter)
                    step_key = jax.random.fold_in(base_rng_key, step_counter[0])
                    
                    for i in range(batch_size):
                        # Generate different random shift for each batch element
                        subkey = jax.random.fold_in(step_key, i)
                        measurement_times = apply_measurement_uncertainty(base_measurement_times, subkey)
                        
                        prob_with_batch = prob_with_batch.at[i].set(
                            self.simulation(solver_with, rho0, theta0, theta1, measurement_times)
                        )
                        prob_without_batch = prob_without_batch.at[i].set(
                            self.simulation(solver_without, rho0, theta0, theta1, measurement_times)
                        )
                else:
                    # For fixed interval, generate all uncertainty realizations at once
                    # Use JAX arrays directly for better performance
                    measurement_times_batch = self.experimental_params.get_measurement_times_with_uncertainty(batch_size)
                    
                    prob_with_batch = jnp.zeros(batch_size)
                    prob_without_batch = jnp.zeros(batch_size)
                    
                    for i in range(batch_size):
                        # Extract measurement times for this batch element
                        measurement_times = measurement_times_batch[i]
                        prob_with_batch = prob_with_batch.at[i].set(
                            self.simulation(solver_with, rho0, theta0, theta1, measurement_times)
                        )
                        prob_without_batch = prob_without_batch.at[i].set(
                            self.simulation(solver_without, rho0, theta0, theta1, measurement_times)
                        )
                
                # Average over batch using JAX operations (efficient)
                prob_with = jnp.mean(prob_with_batch)
                prob_without = jnp.mean(prob_without_batch)
                sensing_contrast = prob_with - prob_without
            
            # Return negative for minimization (we want to maximize contrast)
            # Also return aux data (probabilities and contrast)
            return -sensing_contrast, (prob_with, prob_without, sensing_contrast)
        
        if verbose:
            print(f"Configuration:")
            print(f"    Max iterations: {num_steps}")
            print(f"    Batch size: {batch_size}")
            print(f"    Convergence tolerance: {tolerance:.2e}")
            theta1_status = " [FIXED]" if not trainable_mask[0] else ""
            theta2_status = " [FIXED]" if not trainable_mask[1] else ""
            print(f"    Initial rotation parameters: {theta1_name}={params[0]:.3f} rad{theta1_status}, {theta2_name}={params[1]:.3f} rad{theta2_status}")
            if has_trainable_interval:
                interval_status = " [FIXED]" if not trainable_mask[2] else ""
                print(f"    Initial time interval: {params[2]:.6f}{interval_status}")
            print(f"    Optimizer: {type(optimizer).__name__}")
            if self.experimental_params.measurement.initial_time_uncertainty > 0:
                print(f"    Measurement uncertainty: ±{self.experimental_params.measurement.initial_time_uncertainty:.3f}")
            print("="*70)
            if has_trainable_interval:
                print(f"{'Step':<6}{theta1_name:<12}{theta2_name:<12}{'Δt':<12}{'Contrast':<12}{'Grad Norm'}")
            else:
                print(f"{'Step':<6}{theta1_name:<12}{theta2_name:<12}{'Contrast':<12}{'Grad Norm'}")
            print("-"*70)
        
        best_contrast = -np.inf
        best_params = jnp.array(params)  # Make a copy using jnp
        
        # Initialize variables
        step = 0
        
        for step in range(num_steps):
            # Update step counter for random key generation
            step_counter[0] = step
            
            # Compute gradients using JAX autodiff with auxiliary data
            # This computes the simulation only once and returns probabilities
            grads, (prob_with, prob_without, sensing_contrast) = \
                jax.grad(objective_function, has_aux=True)(params)
            
            print(grads)
            # Track best parameters
            if sensing_contrast > best_contrast:
                best_contrast = sensing_contrast
                best_params = jnp.array(params)  # Copy using jnp
            
            # Call callback to track progress
            callback(
                trainable_params=self.trainable_params,
                prob_with=float(prob_with),
                prob_without=float(prob_without),
                contrast=float(sensing_contrast)
            )
            
            grad_norm = jnp.linalg.norm(grads)
            
            # Progress output
            if verbose and (step % verbose_step == 0 or grad_norm < tolerance):
                # pylint: disable=unsubscriptable-object
                if has_trainable_interval:
                    print(f"{step:<6}{params[0]:<12.6f}{params[1]:<12.6f}{params[2]:<12.6f}"
                          f"{sensing_contrast:<12.6f}{grad_norm:<12.2e}")
                else:
                    print(f"{step:<6}{params[0]:<12.6f}{params[1]:<12.6f}"
                          f"{sensing_contrast:<12.6f}{grad_norm:<12.2e}")
            
            # Convergence check
            if grad_norm < tolerance:
                break
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Apply constraints immediately after update
            if has_trainable_interval:
                # Enforce positive constraint on time_interval
                min_interval = self.trainable_params.constraints[time_interval_idx].min_value or 1e-6
                params = params.at[2].set(jnp.maximum(params[2], min_interval))
            
            # Update trainable parameters continuously
            # pylint: disable=unsubscriptable-object
            self.trainable_params.parameters[theta1_idx].value = float(params[0])
            self.trainable_params.parameters[theta2_idx].value = float(params[1])
            if has_trainable_interval:
                # Update time_interval in both experimental_params and trainable_params
                # This must be done outside JAX tracing (after gradient computation)
                update_time_interval(float(params[2]))
        
        # Ensure best parameters are set at the end
        # pylint: disable=unsubscriptable-object
        for idx, param_idx in enumerate(param_indices):
            self.trainable_params.parameters[param_idx].value = float(best_params[idx])
        
        # Update experimental_params with best time_interval if it was trainable
        if has_trainable_interval:
            update_time_interval(float(best_params[2]))
        
        # Apply constraints at the end
        final_values = np.array([p.value for p in self.trainable_params.parameters])
        constrained_values = self.trainable_params.apply_constraints(final_values)
        for i, val in enumerate(constrained_values):
            self.trainable_params.parameters[i].value = float(val)

        if verbose:
            print("="*70)
            print(f"Final gradient norm: {grad_norm:.2e}")
            print(f"Best sensing contrast: {best_contrast:.6f}")
            if has_trainable_interval:
                print(f"Best parameters: {theta1_name}={best_params[0]:.3f} rad, {theta2_name}={best_params[1]:.3f} rad, Δt={best_params[2]:.6f}")
            else:
                print(f"Best parameters: {theta1_name}={best_params[0]:.3f} rad, {theta2_name}={best_params[1]:.3f} rad")
        
        # Set convergence information in callback
        callback.set_convergence_info(
            converged=float(grad_norm) < tolerance,
            final_grad_norm=float(grad_norm)
        )
        
        return callback
    
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
        report = {
            "experiment_type": "SingleQubitExperiment",
            "version": "0.1.0",
            "experimental_parameters": {
                "physical_constants": {
                    "chi": float(self.experimental_params.chi),
                    "photon_cavity_coupling": float(self.experimental_params.photon_cavity_coupling),
                    "inverse_pulse_width": float(self.experimental_params.inverse_pulse_width)
                },
                "system_dimensions": {
                    "cavity_levels": int(self.experimental_params.cavity_levels),
                    "qubit_levels": int(self.experimental_params.qubit_levels),
                    "field_levels": int(self.experimental_params.field_levels)
                },
                "measurement_protocol": {
                    # Store the mode (explicit list vs interval-based)
                    "mode": "explicit" if self.experimental_params.measurement.measurement_times is not None else "interval",
                    # If explicit mode, store the list
                    "measurement_times": [float(t) for t in self.experimental_params.measurement.measurement_times] 
                                        if self.experimental_params.measurement.measurement_times is not None else None,
                    # If interval mode, store the interval parameters
                    "initial_time": float(self.experimental_params.measurement.initial_time) 
                                   if self.experimental_params.measurement.measurement_times is None else None,
                    "final_time": float(self.experimental_params.measurement.final_time)
                                 if self.experimental_params.measurement.measurement_times is None else None,
                    "time_interval": float(self.experimental_params.measurement.time_interval)
                                    if self.experimental_params.measurement.measurement_times is None else None,
                    # Always store uncertainty settings
                    "initial_time_uncertainty": float(self.experimental_params.measurement.initial_time_uncertainty),
                    # Computed times for reference
                    "computed_times": [float(t) for t in self.experimental_params._measurement_times_list],
                    "num_measurements": len(self.experimental_params._measurement_times_list)
                },
                "initial_state": {
                    "state_type": self.experimental_params.initial_state.state_type.value,
                    "coherent_alpha": None if self.experimental_params.initial_state.coherent_alpha is None 
                                     else float(abs(self.experimental_params.initial_state.coherent_alpha)),
                    "coherent_alpha_phase": None if self.experimental_params.initial_state.coherent_alpha is None
                                           else float(np.angle(self.experimental_params.initial_state.coherent_alpha)),
                    "thermal_n_bar": None if self.experimental_params.initial_state.thermal_n_bar is None
                                    else float(self.experimental_params.initial_state.thermal_n_bar),
                    "has_custom_amplitudes": self.experimental_params.initial_state.custom_amplitudes is not None
                },
                "noise_configuration": {
                    "depolarizing": float(self.experimental_params.noise_config.depolarizing),
                    "dephasing": float(self.experimental_params.noise_config.dephasing),
                    "relaxation": float(self.experimental_params.noise_config.relaxation)
                }
            },
            "trainable_parameters": {
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.param_type.value,
                        "value": float(param.value),
                        "trainable": param.trainable
                    }
                    for param in self.trainable_params.parameters
                ],
                "num_parameters": len(self.trainable_params.parameters),
                "num_trainable": len(self.trainable_params.get_trainable_indices())
            },
            "callback_info": None
        }
        
        # Add callback information if available
        if self.callback is not None and self.callback.epoch > 0:
            # Determine if this was an optimization or simulation
            is_optimization = self.callback.epoch > 1 or (
                self.callback.epoch == 1 and 
                self.callback.converged is not False
            )
            
            callback_info = {
                "mode": "optimization" if is_optimization else "simulation",
                "total_epochs": int(self.callback.epoch),
                "converged": bool(self.callback.converged),
                "final_gradient_norm": None if self.callback.final_grad_norm is None 
                                       else float(self.callback.final_grad_norm)
            }
            
            # Add best metrics if available
            if self.callback.best_metrics is not None:
                callback_info["best_metrics"] = {
                    "epoch": int(self.callback.best_metrics['epoch']),
                    "contrast": float(self.callback.best_metrics['contrast']),
                    "prob_with": float(self.callback.best_metrics['prob_with']),
                    "prob_without": float(self.callback.best_metrics['prob_without'])
                }
                
                # Add best parameters
                if self.callback.best_trainable_params is not None:
                    best_angles = self.callback.best_trainable_params.get_rotation_angles()
                    callback_info["best_parameters"] = {
                        name: {
                            "value_rad": float(val[0]),
                            "value_deg": float(np.rad2deg(val[0]))
                        }
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
                if len(history['contrast']) > 0:
                    callback_info["optimization_summary"] = {
                        "initial_contrast": float(history['contrast'][0]),
                        "final_contrast": float(history['contrast'][-1]),
                        "best_contrast": float(max(history['contrast'])),
                        "improvement": float(max(history['contrast']) - history['contrast'][0])
                    }
            
            report["callback_info"] = callback_info
        
        # Save report to JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    
    def plot_pulse_shape(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Plot Gaussian pulse envelope with measurement time markers.
        
        Convenience method that visualizes the temporal shape of the Gaussian input pulse
        along with vertical markers indicating when measurements are performed. This helps
        understand the relationship between the pulse envelope and the measurement protocol.
        
        Args:
            save_path: Optional path to save the figure
            dpi: Resolution for saved figure (default: 300)
            
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
        from ..utils.visualization import plot_pulse_shape_with_measurements
        
        return plot_pulse_shape_with_measurements(
            self.experimental_params,
            save_path=save_path,
            dpi=dpi
        )
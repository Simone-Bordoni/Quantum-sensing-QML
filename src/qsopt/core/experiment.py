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
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qutip as qt
from jax.scipy.special import erfc
from qsopt.core.experimental_parameters import ExperimentalParameters, InitialStateType
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.core.quantum_utils import gu
from qsopt.core.callback import OptimizationCallback
from qsopt.core.quantum_utils import (
    generate_single_qubit_operators,
    generate_initial_state,
    apply_single_qubit_rotation,
    create_measurement_projector,
    project_and_measure,
    measure_qubit_probability
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
    
    def get_initial_state(self) -> qt.Qobj:
        """
        Get the initial state (cached for performance).
        
        Returns the cached initial state computed during initialization.
        The state is computed once based on configuration and reused.
        
        Supported state types:
        - VACUUM: |0,0,0⟩ (vacuum field, vacuum cavity, qubit ground)
        - SINGLE_PHOTON: |1,0,0⟩ (one photon in field, vacuum cavity, qubit ground)
        - COHERENT: |α,0,0⟩ (coherent state in field, vacuum cavity, qubit ground)
        - THERMAL: Thermal state in cavity with specified average photon number
        - CUSTOM: User-defined state from amplitude dictionary
        
        Returns:
            qt.Qobj: Initial density matrix in composite space (field ⊗ cavity ⊗ qubit)
        """
        return self._cached_initial_state
    
    def simulation(self, solver: qt.MESolver, rho: qt.Qobj, 
                   theta1: float, theta2: float, 
                   measurements: Dict[float, float],
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
            measurements: dict, Measurement protocol specification
            args: dict, System parameters (optional, uses experimental_params if None)
                
        Returns:
            float: Probability of detecting at least one excited state
                P(detection) = 1 - ∏ᵢ P(|0⟩ᵢ) ∈ [0,1]
        """
        if args is None:
            args = {'sigma': self.experimental_params.inverse_pulse_width}
            
        tmeas = list(measurements.keys())
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
    
    def run_simulation(self) -> OptimizationCallback:
        """
        Run simulation with current parameter values without updating them.
        
        This method provides a convenient way to test the system with the current
        trainable parameter values, computing detection probabilities both with
        and without photon interaction.
        
        Returns:
            OptimizationCallback: Callback containing simulation results with:
                - Single epoch (epoch=1)
                - Current parameter values
                - Detection probabilities (prob_with, prob_without)
                - Sensing contrast
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
        
        # Get initial state and measurement protocol
        rho0 = self.get_initial_state()
        measurement_times = self.experimental_params.measurement.measurement_times
        measurements = {t: 0.0 for t in measurement_times}
        
        # Get solvers
        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()
        
        # Run simulations
        prob_with = self.simulation(solver_with, rho0, theta1, theta2, measurements)
        prob_without = self.simulation(solver_without, rho0, theta1, theta2, measurements)
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
                 tolerance: float = 1e-6,
                 verbose: bool = True,
                 verbose_step: int = 10,
                 callback: Optional[OptimizationCallback] = None,
                 theta_init: Optional[List[float]] = None) -> OptimizationCallback:
        """
        Optimize rotation parameters to maximize sensing contrast.
        
        Uses JAX automatic differentiation to find optimal rotation angles that
        maximize the difference between detection probabilities with and without
        photon present. Uses optimizers defined in trainable_params.
        
        Args:
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
        rho0 = self.get_initial_state()
        
        # Get measurement protocol
        measurement_times = self.experimental_params.measurement.measurement_times
        measurements = {t: 0.0 for t in measurement_times}  # All measurements expect |0⟩
        
        # Get solvers
        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()
        
        # Get rotation angles using the correct method
        rotation_angles = self.trainable_params.get_rotation_angles()
        
        if len(rotation_angles) < 2:
            raise ValueError("Need at least 2 rotation angle parameters")
        
        # Get parameter names and values
        param_names = list(rotation_angles.keys())
        theta1_name = param_names[0]
        theta2_name = param_names[1]
        
        # Get parameter indices for rotation angles
        theta1_idx = -1
        theta2_idx = -1
        for param in self.trainable_params.parameters:
            if param.name == theta1_name:
                theta1_idx = param.index
            elif param.name == theta2_name:
                theta2_idx = param.index
        
        if theta1_idx == -1 or theta2_idx == -1:
            raise ValueError(f"Could not find rotation parameters {theta1_name} and/or {theta2_name}")
        
        # Use provided theta_init, property theta_init, or trainable_params values (in that order)
        if theta_init is not None:
            if len(theta_init) != 2:
                raise ValueError("theta_init must contain exactly 2 angles [θ₁, θ₂]")
            initial_angles = theta_init
        else:
            theta1_value = rotation_angles[theta1_name][0]
            theta2_value = rotation_angles[theta2_name][0]
            initial_angles = [theta1_value, theta2_value]
        
        params = jnp.array(initial_angles, dtype=float)
        # Update trainable_params with initial values
        self.trainable_params.parameters[theta1_idx].value = float(initial_angles[0])
        self.trainable_params.parameters[theta2_idx].value = float(initial_angles[1])
        
        # Get optimizers from trainable_params
        if len(self.trainable_params.optimizers) == 0:
            raise ValueError("No optimizer defined in trainable_params. Use add_rotation_angles() with optimizer parameter.")
        
        # Use optimizer from first rotation parameter
        optimizer = self.trainable_params.optimizers[theta1_idx]
        opt_state = optimizer.init(params)
        
        # Define objective function that returns probabilities along with loss
        def objective_function(theta_params):
            """Negative sensing contrast for minimization.
            
            Args:
                theta_params: Array of [theta1, theta2] rotation angles
                
            Returns:
                tuple: (loss, (prob_with, prob_without, contrast))
            """
            # pylint: disable=unsubscriptable-object
            theta0, theta1 = theta_params
            
            # Calculate sensing contrast: P(with photon) - P(without photon)
            prob_with = self.simulation(solver_with, rho0, theta0, theta1, measurements)
            prob_without = self.simulation(solver_without, rho0, theta0, theta1, measurements)
            sensing_contrast = prob_with - prob_without
            
            # Return negative for minimization (we want to maximize contrast)
            # Also return aux data (probabilities and contrast)
            return -sensing_contrast, (prob_with, prob_without, sensing_contrast)
        
        if verbose:
            print(f"Configuration:")
            print(f"    Max iterations: {num_steps}")
            print(f"    Convergence tolerance: {tolerance:.2e}")
            print(f"    Initial rotation parameters: {theta1_name}={params[0]:.3f} rad, {theta2_name}={params[1]:.3f} rad")
            print(f"    Optimizer: {type(optimizer).__name__}")
            print("="*70)
            print(f"{'Step':<6}{theta1_name:<12}{theta2_name:<12}{'Contrast':<12}{'Grad Norm'}")
            print("-"*70)
        
        best_contrast = -np.inf
        best_params = params.copy()
        
        # Initialize variables
        step = 0
        
        for step in range(num_steps):
            # Compute gradients using JAX autodiff with auxiliary data
            # This computes the simulation only once and returns probabilities
            grads, (prob_with, prob_without, sensing_contrast) = \
                jax.grad(objective_function, has_aux=True)(params)
            
            # Track best parameters
            if sensing_contrast > best_contrast:
                best_contrast = sensing_contrast
                best_params = params.copy()
            
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
                print(f"{step:<6}{params[0]:<12.6f}{params[1]:<12.6f}"
                      f"{sensing_contrast:<12.6f}{grad_norm:<12.2e}")
            
            # Convergence check
            if grad_norm < tolerance:
                break
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Update trainable parameters continuously
            # pylint: disable=unsubscriptable-object
            self.trainable_params.parameters[theta1_idx].value = float(params[0])
            self.trainable_params.parameters[theta2_idx].value = float(params[1])
        
        # Ensure best parameters are set at the end
        # pylint: disable=unsubscriptable-object
        self.trainable_params.parameters[theta1_idx].value = float(best_params[0])
        self.trainable_params.parameters[theta2_idx].value = float(best_params[1])
        
        # Apply constraints at the end (angles between 0 and 2π)
        final_values = np.array([
            self.trainable_params.parameters[theta1_idx].value,
            self.trainable_params.parameters[theta2_idx].value
        ])
        constrained_values = self.trainable_params.apply_constraints(final_values)
        self.trainable_params.parameters[theta1_idx].value = float(constrained_values[0])
        self.trainable_params.parameters[theta2_idx].value = float(constrained_values[1])

        if verbose:
            print("="*70)
            print(f"Final gradient norm: {grad_norm:.2e}")
            print(f"Best sensing contrast: {best_contrast:.6f}")
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
        
        Example:
            >>> experiment = SingleQubitExperiment(exp_params, trainable_params)
            >>> history = experiment.optimize(num_steps=50)
            >>> experiment.save_experiment_report('results/my_experiment.json')
            >>> # This creates:
            >>> # - results/my_experiment.json (experiment metadata)
            >>> # - results/my_experiment_callback.npz (detailed optimization data)
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
                    "measurement_times": [float(t) for t in self.experimental_params.measurement_times],
                    "num_measurements": len(self.experimental_params.measurement_times)
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
                "rotation_angles": {
                    name: float(val[0]) 
                    for name, val in self.trainable_params.get_rotation_angles().items()
                },
                "num_parameters": len(self.trainable_params.parameters)
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
        
        print(f"✓ Experiment report saved to: {save_path}")
        if report["callback_info"] and "callback_data_path" in report["callback_info"]:
            print(f"✓ Optimization data saved to: {report['callback_info']['callback_data_path']}")
    
    @classmethod
    def load_experiment_report(cls, json_path: str) -> Dict[str, Any]:
        """
        Load experiment configuration from a JSON report file.
        
        This method loads the experiment report and reconstructs the configuration
        parameters. It does NOT reconstruct the full experiment object (as quantum
        operators cannot be serialized), but provides all the information needed
        to recreate the experiment.
        
        Args:
            json_path: Path to the JSON report file
        
        Returns:
            Dictionary containing:
                - 'experimental_params_dict': Dictionary with all experimental parameters
                - 'trainable_params_dict': Dictionary with trainable parameter values
                - 'callback_info': Callback information (if available)
                - 'callback_data': Loaded NPZ data (if optimization was performed)
        
        Example:
            >>> # Load experiment configuration
            >>> loaded = SingleQubitExperiment.load_experiment_report('results/report.json')
            >>> 
            >>> # Recreate experimental parameters
            >>> from qsopt import *
            >>> exp_params = ExperimentalParameters(
            ...     physical_constants=PhysicalConstants(**loaded['experimental_params_dict']['physical_constants']),
            ...     system_dims=SystemDimensions(**loaded['experimental_params_dict']['system_dimensions']),
            ...     # ... etc
            ... )
            >>> 
            >>> # Access optimization data if available
            >>> if 'callback_data' in loaded:
            ...     epochs = loaded['callback_data']['epochs']
            ...     contrast = loaded['callback_data']['contrast']
        """
        import json
        from pathlib import Path
        
        # Load JSON report
        with open(json_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        result = {
            'experiment_type': report.get('experiment_type'),
            'version': report.get('version'),
            'experimental_params_dict': report.get('experimental_parameters'),
            'trainable_params_dict': report.get('trainable_parameters'),
            'callback_info': report.get('callback_info')
        }
        
        # Load callback data if available
        if (result['callback_info'] is not None and 
            'callback_data_path' in result['callback_info']):
            
            callback_path = result['callback_info']['callback_data_path']
            if Path(callback_path).exists():
                callback_data = OptimizationCallback.load(callback_path)
                result['callback_data'] = callback_data
                print(f"✓ Loaded callback data from: {callback_path}")
            else:
                print(f"⚠ Warning: Callback data file not found: {callback_path}")
        
        print(f"✓ Experiment report loaded from: {json_path}")
        return result


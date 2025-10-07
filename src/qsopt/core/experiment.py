"""
Quantum Sensing Experiment Class
================================

Main experiment class that orchestrates quantum sensing protocols with configurable
parameters, noise models, and optimization strategies.
"""

"""Quantum sensing experiment module."""
import warnings
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qutip as qt
from jax.scipy.special import erfc
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import TrainableParameters

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import

# Suppress Diffrax complex dtype warning
warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax is a work in progress*")

@jax.jit
def gu(t, **kwargs):  
    """
    Time-dependent coupling function for input cavity transparency.
    
    Args:
        t: float or JAX array, time variable
        **kwargs: Dictionary containing 'sigma' parameter (pulse bandwidth)
        
    Returns:
        JAX array: Normalized coupling strength g(t)
    """
    sigma = kwargs.get("sigma", 0.1)
    dx = sigma * t
    coupling = jnp.sqrt(2*sigma/jnp.sqrt(jnp.pi)*jnp.exp(-dx**2)/erfc(dx))
    return jnp.array(coupling, float)

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
        
        # Initialize quantum objects
        self.__post_init__()

    def __post_init__(self):
        """Post-initialization to set up operators and hamiltonian."""
        self._generate_operators()
        self._generate_hamiltonian()

    def _generate_operators(self):
        """
        Generate the necessary operators for the experiment in the composite space.
        
        Creates operators for the three-subsystem composite Hilbert space:
        - Input cavity: Annihilation/creation operators
        - Resonator cavity: Annihilation/creation operators  
        - Qubit: Pauli matrices and projection operators
        
        All operators are embedded in the full three-system tensor product space.
        """
        # Get system dimensions
        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels
        qubit_levels = self.experimental_params.qubit_levels
        
        # Identity operators for each subsystem
        I_field = qt.identity(field_levels)     # Input field identity
        I_cavity = qt.identity(cavity_levels)   # Resonator cavity identity  
        I_qubit = qt.identity(qubit_levels)     # Qubit identity
        
        # Individual subsystem operators
        # Input field operators
        a_field = qt.destroy(field_levels)      # Field annihilation
        
        # Resonator cavity operators
        a_cavity = qt.destroy(cavity_levels)    # Cavity annihilation
        
        # Qubit operators
        sigma_z = qt.sigmaz()                   # Pauli-Z
        sigma_x = qt.sigmax()                   # Pauli-X
        sigma_y = qt.sigmay()                   # Pauli-Y
        sigma_minus = qt.destroy(qubit_levels)  # Qubit lowering
        
        # Qubit measurement projectors
        P0 = qt.Qobj([[1, 0], [0, 0]])         # Ground state |0⟩⟨0|
        P1 = qt.Qobj([[0, 0], [0, 1]])         # Excited state |1⟩⟨1|
        
        # Composite system operators (input_field ⊗ resonator_cavity ⊗ qubit)
        self.operators = {
            # Input field operators in composite space
            'a_in': qt.tensor(a_field, I_cavity, I_qubit),
            'a_in_dag': qt.tensor(a_field.dag(), I_cavity, I_qubit),
            
            # Resonator cavity operators in composite space
            'a': qt.tensor(I_field, a_cavity, I_qubit),
            'a_dag': qt.tensor(I_field, a_cavity.dag(), I_qubit),
            
            # Qubit operators in composite space
            'sigma_z': qt.tensor(I_field, I_cavity, sigma_z),
            'sigma_x': qt.tensor(I_field, I_cavity, sigma_x),
            'sigma_y': qt.tensor(I_field, I_cavity, sigma_y),
            'sigma_minus': qt.tensor(I_field, I_cavity, sigma_minus),
            'sigma_plus': qt.tensor(I_field, I_cavity, sigma_minus.dag()),
            
            # Qubit measurement projectors in composite space
            'P0': qt.tensor(I_field, I_cavity, P0),
            'P1': qt.tensor(I_field, I_cavity, P1),
            
            # Identity operators for reference
            'I_field': I_field,
            'I_cavity': I_cavity,
            'I_qubit': I_qubit,
        }

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
        
        interaction_ops: List[Union[qt.Qobj, qt.QobjEvo]] = [L_int] + list(lindblad_noise)
        no_interaction_ops: List[Union[qt.Qobj, qt.QobjEvo]] = list(lindblad_noise)
        
        # Store Hamiltonians and Lindblad operators
        self.hamiltonians = {
            'total': H_total,
            'dispersive': H_dispersive,
            'coupling': H_coupling
        }
        
        # Fix type annotations for mypy
        noise_only_list: List[Union[qt.Qobj, qt.QobjEvo]] = list(lindblad_noise)
        
        self.lindblad_operators = {
            'interaction': interaction_ops,
            'no_interaction': no_interaction_ops,
            'noise_only': noise_only_list
        }
    
    def get_solver_with_interaction(self) -> qt.MESolver:
        """
        Create a Lindblad master equation solver WITH input photon interaction.
        
        Returns:
            qt.MESolver: Configured solver for signal case evolution
        """
        if self.hamiltonians is None or self.lindblad_operators is None:
            raise RuntimeError("Hamiltonian and operators must be generated first")
            
        return qt.MESolver(
            self.hamiltonians['total'], 
            self.lindblad_operators['interaction'],
            options={"method": "adams", "normalize_output": False}
        )
    
    def get_solver_no_interaction(self) -> qt.MESolver:
        """
        Create a Lindblad master equation solver WITHOUT input photon interaction.
        
        Returns:
            qt.MESolver: Configured solver for reference case evolution
        """
        if self.hamiltonians is None or self.lindblad_operators is None:
            raise RuntimeError("Hamiltonian and operators must be generated first")
            
        return qt.MESolver(
            self.hamiltonians['dispersive'], 
            self.lindblad_operators['no_interaction'],
            options={"method": "adams", "normalize_output": False}
        )
    
    def ry_rotation(self, rho: qt.Qobj, theta: float) -> qt.Qobj:
        """
        Apply Ry rotation to qubit in the three-system composite space.
        
        Implements a Ry rotation gate around the Y-axis for quantum state manipulation.
        The rotation is applied only to the qubit subsystem while preserving the 
        cavity states in the composite Hilbert space.
        
        Args:
            rho: QuTiP Qobj density matrix in composite space (input ⊗ resonator ⊗ qubit)
            theta: float or JAX array, Ry rotation angle in radians
            
        Returns:
            QuTiP Qobj: Rotated density matrix
        """
        if self.operators is None:
            raise RuntimeError("Operators not initialized")
        I_field = self.operators['I_field']
        I_cavity = self.operators['I_cavity']
        
        Sy_jax = qt.sigmay()
        ry_gate = (-1j * Sy_jax * theta / 2).expm()
        r = qt.tensor(I_field, I_cavity, ry_gate)
        return r * rho * r.dag()  # type: ignore
    
    def proj0(self, rho: qt.Qobj) -> qt.Qobj:
        """
        Project density matrix onto qubit |0⟩ state.
        
        Args:
            rho: QuTiP Qobj density matrix in composite space
            
        Returns:
            QuTiP Qobj: Projected density matrix P₀ρP₀† (unnormalized)
        """
        if self.operators is None:
            raise RuntimeError("Operators not initialized")
        P0 = self.operators['P0']
        return P0 * rho * P0.dag()  # type: ignore
    
    def prob0(self, rho: qt.Qobj) -> float:
        """
        Calculate probability of measuring qubit in |0⟩ state.
        
        Args:
            rho: QuTiP Qobj density matrix in composite space
            
        Returns:
            float: Real probability value Tr(P₀ρ) ∈ [0,1]
        """
        return float(jnp.real(self.proj0(rho).tr()))
    
    def prob1(self, rho: qt.Qobj) -> float:
        """
        Calculate probability of measuring qubit in |1⟩ state.
        
        Args:
            rho: QuTiP Qobj density matrix in composite space
            
        Returns:
            float: Real probability value Tr(P₁ρ) ∈ [0,1]
        """
        if self.operators is None:
            raise RuntimeError("Operators not initialized")
        P1 = self.operators['P1']
        return float(jnp.real((P1 * rho * P1.dag()).tr()))  # type: ignore
    
    def get_initial_state(self) -> qt.Qobj:
        """
        Generate the initial state based on configuration.
        
        Returns:
            qt.Qobj: Initial density matrix in composite space
        """
        field_levels = self.experimental_params.field_levels
        cavity_levels = self.experimental_params.cavity_levels
        qubit_levels = self.experimental_params.qubit_levels
        
        initial_config = self.experimental_params.initial_state
        
        # For SINGLE_PHOTON: |0,1,0⟩ (vacuum input, 1 photon resonator, qubit ground)
        psi = qt.tensor(
            qt.basis(field_levels, 0),
            qt.basis(cavity_levels, 1),
            qt.basis(qubit_levels, 0)
        )
        return psi * psi.dag()  # type: ignore
    
    def simulation(self, solver: qt.MESolver, rho: qt.Qobj, 
                   theta1: float, theta2: float, 
                   measurements: Dict[float, float],
                   args: Optional[Dict] = None) -> float:
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

        return float(prob_detection)
    
    def run_simulation(self, with_interaction: bool = True) -> Dict[str, float]:
        """
        Run simulation with current parameter values without updating them.
        
        This method provides a convenient way to test the system with the current
        trainable parameter values, computing detection probabilities both with
        and without photon interaction.
        
        Args:
            with_interaction: If True, use solver with interaction. If False, use both
                            solvers to compute contrast.
        
        Returns:
            dict: Dictionary containing:
                - 'prob_with': Detection probability with interaction
                - 'prob_without': Detection probability without interaction  
                - 'contrast': Sensing contrast (prob_with - prob_without)
                - 'theta1': First rotation angle used
                - 'theta2': Second rotation angle used
        
        Raises:
            ValueError: If fewer than 2 rotation parameters are defined
        """
        # Get rotation parameters
        rotation_params = [p for p in self.trainable_params.parameters 
                          if p.param_type.value == 'rotation_angle']
        
        if len(rotation_params) < 2:
            raise ValueError("Need at least 2 rotation angle parameters")
        
        theta1 = rotation_params[0].value
        theta2 = rotation_params[1].value
        
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
        
        return {
            'prob_with': float(prob_with),
            'prob_without': float(prob_without),
            'contrast': float(prob_with - prob_without),
            'theta1': float(theta1),
            'theta2': float(theta2)
        }
    
    def optimize(self, num_steps: int = 100, 
                 learning_rate: float = 0.05,
                 tolerance: float = 1e-6,
                 verbose: bool = True,
                 callback: Optional[Any] = None) -> Dict[str, Any]:
        """
        Optimize rotation parameters to maximize sensing contrast.
        
        Uses JAX automatic differentiation to find optimal rotation angles that
        maximize the difference between detection probabilities with and without
        photon present.
        
        Args:
            num_steps: Maximum number of optimization steps
            learning_rate: Learning rate for gradient descent
            tolerance: Convergence threshold for gradient norm
            verbose: Print progress information
            callback: Optional callback function to track optimization progress.
                     Should accept (parameters, loss, prob_with, prob_without, contrast)
            
        Returns:
            dict: Optimization results including optimal parameters and history
        """
        # Get initial state
        rho0 = self.get_initial_state()
        
        # Get measurement protocol
        measurement_times = self.experimental_params.measurement.measurement_times
        measurements = {t: 0.0 for t in measurement_times}  # All measurements expect |0⟩
        
        # Get solvers
        solver_with = self.get_solver_with_interaction()
        solver_without = self.get_solver_no_interaction()
        
        # Get initial parameter values from trainable_params
        rotation_params = [p for p in self.trainable_params.parameters 
                          if p.param_type.value == 'rotation_angle']
        
        if len(rotation_params) < 2:
            raise ValueError("Need at least 2 rotation angle parameters")
        
        # Get initial values and optimizer
        theta1_param = rotation_params[0]
        theta2_param = rotation_params[1]
        
        params = jnp.array([theta1_param.value, theta2_param.value], dtype=float)
        
        # Get optimizer - use the trainable_params optimizer or create default
        if hasattr(self.trainable_params, 'optimizers') and len(self.trainable_params.optimizers) > 0:
            optimizer = self.trainable_params.optimizers[0]
        else:
            optimizer = optax.adam(learning_rate)
        
        opt_state = optimizer.init(params)
        
        # Define objective function
        def objective_function(theta_params):
            """Negative sensing contrast for minimization.
            
            Args:
                theta_params: Array of [theta1, theta2] rotation angles
                
            Returns:
                float: Negative contrast for minimization
            """
            theta0, theta1 = theta_params
            
            # Calculate sensing contrast: P(with photon) - P(without photon)
            prob_with = self.simulation(solver_with, rho0, theta0, theta1, measurements)
            prob_without = self.simulation(solver_without, rho0, theta0, theta1, measurements)
            sensing_contrast = prob_with - prob_without
            
            # Return negative for minimization (we want to maximize contrast)
            return -sensing_contrast
        
        # Optimization history
        history = {
            'loss': [],
            'contrast': [],
            'theta1': [],
            'theta2': [],
            'gradients': [],
            'prob_with': [],
            'prob_without': []
        }
        
        if verbose:
            print("Starting optimization...")
            print(f"Initial: θ₁={params[0]:.3f} rad, θ₂={params[1]:.3f} rad")
            print("="*70)
            print(f"{'Step':<6}{'θ₁':<12}{'θ₂':<12}{'Contrast':<12}{'Loss':<12}{'Grad Norm'}")
            print("-"*70)
        
        best_contrast = -np.inf
        best_params = params.copy()
        
        # Initialize variables
        grad_norm = tolerance + 1
        step = 0
        
        for step in range(num_steps):
            # Compute loss and gradients using JAX autodiff
            loss_value, grads = jax.value_and_grad(objective_function)(params)
            
            # Calculate metrics
            theta0, theta1 = params
            prob_with = self.simulation(solver_with, rho0, theta0, theta1, measurements)
            prob_without = self.simulation(solver_without, rho0, theta0, theta1, measurements)
            sensing_contrast = prob_with - prob_without
            
            # Track best parameters
            if sensing_contrast > best_contrast:
                best_contrast = sensing_contrast
                best_params = params.copy()
            
            # Store history
            history['loss'].append(float(loss_value))
            history['contrast'].append(float(sensing_contrast))
            history['theta1'].append(float(params[0]))
            history['theta2'].append(float(params[1]))
            history['gradients'].append([float(grads[0]), float(grads[1])])
            history['prob_with'].append(float(prob_with))
            history['prob_without'].append(float(prob_without))
            
            # Call callback if provided
            if callback is not None:
                callback(
                    parameters=np.array(params),
                    loss=float(loss_value),
                    prob_with=float(prob_with),
                    prob_without=float(prob_without),
                    contrast=float(sensing_contrast)
                )
            
            grad_norm = jnp.linalg.norm(grads)
            
            # Progress output
            if verbose and (step % 20 == 0 or grad_norm < tolerance):
                print(f"{step:<6}{params[0]:<12.6f}{params[1]:<12.6f}"
                      f"{sensing_contrast:<12.6f}{loss_value:<12.6f}{grad_norm:<12.2e}")
            
            # Convergence check
            if grad_norm < tolerance:
                if verbose:
                    print(f"\nConverged after {step+1} iterations!")
                    print(f"Final gradient norm: {grad_norm:.2e}")
                    print(f"Best sensing contrast: {best_contrast:.6f}")
                break
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Update trainable parameters continuously
            theta1_param.value = float(params[0])
            theta2_param.value = float(params[1])
        
        # Ensure best parameters are set at the end
        theta1_param.value = float(best_params[0])
        theta2_param.value = float(best_params[1])
        
        return {
            'optimal_params': best_params,
            'optimal_contrast': best_contrast,
            'history': history,
            'converged': float(grad_norm) < tolerance,
            'iterations': step + 1
        }


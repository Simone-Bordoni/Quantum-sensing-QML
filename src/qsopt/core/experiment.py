"""
Quantum Sensing Experiment Class
================================

Main experiment class that orchestrates quantum sensing protocols with configurable
parameters, noise models, and optimization strategies.
"""

import jax
import jax.numpy as jnp
import qutip as qt
import qutip_jax
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import warnings

# Suppress Diffrax complex dtype warning
warnings.filterwarnings('ignore', message='Complex dtype support in Diffrax is a work in progress*')


class Experiment:
    """
    Main experiment class for quantum sensing protocols.
    
    This class serves as the central orchestrator for quantum sensing experiments,
    providing a high-level interface that coordinates between system parameters,
    noise models, quantum evolution, and optimization strategies.
    
    The class is designed to be flexible and extensible, allowing for different
    types of quantum sensing protocols while maintaining a consistent interface.
    
    Attributes:
        experimental_parameters: ExperimentalParameters object containing system configuration
            (cavity levels, coupling constants, measurement setup, physical parameters)
        trainable_parameters: TrainableParameters object containing optimization parameters
            (rotation angles, pulse parameters, protocol parameters)
        noise_model: NoiseModel object for decoherence simulation
        _initialized: Whether the experiment has been properly initialized
        _operators: Dictionary of quantum operators in composite space
        _solvers: Dictionary of evolution solvers (with/without interaction)
        _initial_state: Initial quantum state for the experiment
    """
    
    def __init__(self, experimental_parameters=None, trainable_parameters=None, noise_model=None):
        """
        Initialize quantum sensing experiment.
        
        Args:
            experimental_parameters: ExperimentalParameters object containing system configuration
                (cavity levels, coupling constants, measurement times, physical constants)
            trainable_parameters: TrainableParameters object containing optimization parameters
                (rotation angles, pulse parameters, protocol settings)
            noise_model: NoiseModel object for decoherence simulation
        """
        self.experimental_parameters = experimental_parameters
        self.trainable_parameters = trainable_parameters
        self.noise_model = noise_model
        self._initialized = False
        
        # Internal storage for quantum objects
        self._operators = {}
        self._solvers = {}
        self._initial_state = None
        self._measurement_times = None
        
        # Results storage
        self._last_results = {}
        
    def initialize(self) -> None:
        """
        Initialize the quantum system with current parameters and noise model.
        
        This method constructs all necessary quantum operators, Hamiltonians,
        and evolution solvers based on the current configuration.
        
        Raises:
            ValueError: If experimental_parameters are not properly configured
        """
        if self.experimental_parameters is None:
            raise ValueError("ExperimentalParameters must be set before initialization")
        
        # Build quantum operators in composite Hilbert space
        self._build_operators()
        
        # Construct Hamiltonians
        self._build_hamiltonians()
        
        # Setup evolution solvers
        self._build_solvers()
        
        # Prepare initial state
        self._prepare_initial_state()
        
        # Setup measurement times
        self._setup_measurement_times()
        
        self._initialized = True
        
    def _build_operators(self) -> None:
        """Build quantum operators for the composite Hilbert space."""
        if not hasattr(self.experimental_parameters, 'nlev') or not hasattr(self.experimental_parameters, 'qlev'):
            raise ValueError("ExperimentalParameters must define nlev and qlev")
            
        nlev = self.experimental_parameters.nlev
        qlev = self.experimental_parameters.qlev
        
        with qt.CoreOptions(default_dtype="jax"):
            # Identity operators
            Iq = qt.identity(qlev)
            In = qt.identity(nlev)
            
            # Cavity operators
            An = qt.destroy(nlev)
            
            # Pauli matrices
            Sz = qt.sigmaz()
            Sx = qt.sigmax()
            Sy = qt.sigmay()
            Sp = qt.sigmap()
            Sm = qt.sigmam()
            
            # Measurement projectors
            P0 = qt.Qobj([[1,0],[0,0]])
            P1 = qt.Qobj([[0,0],[0,1]])
            
            # Composite system operators: input ⊗ resonator ⊗ qubit
            self._operators.update({
                # Input cavity
                'ain': qt.tensor(An, In, Iq),
                'ainc': qt.tensor(An, In, Iq).dag(),
                
                # Resonator cavity  
                'a': qt.tensor(In, An, Iq),
                'ac': qt.tensor(In, An, Iq).dag(),
                
                # Qubit operators
                'sz': qt.tensor(In, In, Sz),
                'sx': qt.tensor(In, In, Sx),
                'sy': qt.tensor(In, In, Sy),
                'sp': qt.tensor(In, In, Sp),
                'sm': qt.tensor(In, In, Sm),
                
                # Measurement projectors
                'p0': qt.tensor(In, In, P0),
                'p1': qt.tensor(In, In, P1),
                
                # Identity operators
                'In': In,
                'Iq': Iq
            })
    
    def _build_hamiltonians(self) -> None:
        """Construct time-dependent and time-independent Hamiltonians."""
        if not hasattr(self.experimental_parameters, 'chi') or not hasattr(self.experimental_parameters, 'gm'):
            raise ValueError("ExperimentalParameters must define chi and gm")
            
        with qt.CoreOptions(default_dtype="jax"):
            # Dispersive qubit-resonator interaction
            H_dispersive = -self.experimental_parameters.chi * self._operators['ac'] * self._operators['a'] * self._operators['sz']
            
            # Time-dependent cavity-cavity coupling
            coupling_strength = 1j/2 * jnp.sqrt(self.experimental_parameters.gm)
            H_coupling = qt.Qobj(coupling_strength * (
                self._operators['ainc'] * self._operators['a'] - 
                self._operators['ain'] * self._operators['ac']
            ))
            
            # Time-dependent coefficient function
            def coupling_function(t, **kwargs):
                sigma = kwargs.get('sigma', self.experimental_parameters.sigma)
                dx = sigma * t
                from jax.scipy.special import erfc
                coupling = jnp.sqrt(2*sigma/jnp.sqrt(jnp.pi)*jnp.exp(-dx**2))/erfc(dx)
                return float(coupling)
            
            # Complete Hamiltonian
            args = {'sigma': self.experimental_parameters.sigma}
            self._operators.update({
                'H_dispersive': H_dispersive,
                'H_coupling': H_coupling,
                'H_total': qt.QobjEvo([H_dispersive, [H_coupling, coupling_function]], args=args),
                'coupling_args': args
            })
    
    def _build_solvers(self) -> None:
        """Setup quantum evolution solvers with and without photon interaction."""
        # Lindblad operators for noise
        lindblad_ops = []
        
        if self.noise_model is not None:
            lindblad_ops = self.noise_model.get_lindblad_operators(self._operators)
        
        # Photon interaction operator
        L_photon = qt.QobjEvo([self._operators['ain'], self._get_coupling_function()], 
                             args=self._operators['coupling_args']) + jnp.sqrt(self.experimental_parameters.gm) * self._operators['a']
        
        # Solver WITH photon interaction
        self._solvers['interaction'] = qt.MESolver(
            self._operators['H_total'], 
            lindblad_ops + [L_photon],
            options={"method": "diffrax", "normalize_output": False}
        )
        
        # Solver WITHOUT photon interaction
        self._solvers['no_interaction'] = qt.MESolver(
            self._operators['H_dispersive'],
            lindblad_ops,
            options={"method": "diffrax", "normalize_output": False}
        )
    
    def _get_coupling_function(self):
        """Get the coupling function for time-dependent evolution."""
        def gu(t, **kwargs):
            sigma = kwargs.get('sigma', self.experimental_parameters.sigma)
            dx = sigma * t
            from jax.scipy.special import erfc
            coupling = jnp.sqrt(2*sigma/jnp.sqrt(jnp.pi)*jnp.exp(-dx**2))/erfc(dx)
            return float(coupling)
        return gu
    
    def _prepare_initial_state(self) -> None:
        """Prepare the initial quantum state based on parameters."""
        if not hasattr(self.experimental_parameters, 'initial_state_config'):
            # Default: |0,1,0⟩ (vacuum input, 1 photon resonator, qubit ground)
            with qt.CoreOptions(default_dtype="jax"):
                psi = qt.tensor(
                    qt.basis(self.experimental_parameters.nlev, 0),  # Input cavity: vacuum  
                    qt.basis(self.experimental_parameters.nlev, 1),  # Resonator: 1 photon
                    qt.basis(self.experimental_parameters.qlev, 0)   # Qubit: ground state
                )
                self._initial_state = psi * psi.dag()
        else:
            # Custom initial state from parameters
            self._initial_state = self._build_custom_initial_state()
    
    def _build_custom_initial_state(self):
        """Build custom initial state from parameters configuration."""
        # Placeholder for custom state construction
        # This would be implemented based on specific parameter configurations
        pass
    
    def _setup_measurement_times(self) -> None:
        """Setup measurement times based on parameters."""
        if hasattr(self.experimental_parameters, 'N_meas') and hasattr(self.experimental_parameters, 'tstep'):
            N_meas = self.experimental_parameters.N_meas
            tstep = self.experimental_parameters.tstep
            tstart = -(N_meas-1) * tstep / 2
            tend = -tstart
            self._measurement_times = np.linspace(tstart, tend, N_meas)
        else:
            # Default measurement times
            self._measurement_times = np.array([-5.0, 5.0])
    
    def run_sensing_protocol(self, rotation_angles: List[float], 
                           with_photon: bool = True, 
                           random_timing: bool = False,
                           protocol_type: str = "theta_delta") -> float:
        """
        Execute the quantum sensing protocol with specified rotation angles.
        
        Args:
            rotation_angles: List of rotation angles in radians
                - For protocol_type="theta_delta": [theta, delta] where rotations are Ry(θ±δ) 
                - For protocol_type="theta1_theta2": [theta1, theta2] for separate rotations
            with_photon: Whether to include input photon interaction
            random_timing: Whether to use random measurement timing
            protocol_type: Either "theta_delta" or "theta1_theta2"
            
        Returns:
            float: Detection probability
            
        Raises:
            RuntimeError: If experiment is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Experiment must be initialized before running protocols")
        
        solver = self._solvers['interaction'] if with_photon else self._solvers['no_interaction']
        
        return self._simulate_protocol(
            solver, self._initial_state, rotation_angles, 
            self._measurement_times, random_timing, protocol_type
        )
    
    def _simulate_protocol(self, solver, rho, rotation_angles, tmeas, random_timing, protocol_type="theta_delta"):
        """
        Internal method to simulate the sensing protocol.
        
        This implements the core sensing workflow:
        1. Apply first rotation 
        2. Time evolution under Hamiltonian
        3. Apply second rotation
        4. Measure and update state
        
        Args:
            solver: QuTiP solver for time evolution
            rho: Initial density matrix
            rotation_angles: Rotation parameters
            tmeas: Measurement times
            random_timing: Whether to use random timing
            protocol_type: "theta_delta" for Ry(θ±δ) or "theta1_theta2" for separate rotations
        """
        if protocol_type == "theta_delta":
            theta, delta = rotation_angles
            theta1, theta2 = theta + delta, theta - delta
        else:  # theta1_theta2
            theta1, theta2 = rotation_angles
            
        probability_list = []
        
        for kt in range(len(tmeas[:-1])):
            t0, t1 = tmeas[kt], tmeas[kt+1]
            
            # Apply first rotation Ry(θ₁)
            rho_rotated = self._apply_rotation(rho, theta1)
            
            # Time evolution
            evolution_result = solver.run(
                rho_rotated, [t0, t1], 
                args=self._operators.get('coupling_args', {})
            )
            rho_evolved = evolution_result.states[-1]
            
            # Apply second rotation Ry(θ₂)
            rho_final = self._apply_rotation(rho_evolved, theta2)
            
            # Measurement
            prob_0 = self._measure_probability(rho_final, '0')
            probability_list.append(prob_0)
            
            # Update state (project to measured outcome)
            rho = self._project_state(rho_final, '0')
        
        # Calculate detection probability: 1 - P(all ground states)
        prob_all_ground = jnp.prod(jnp.array(probability_list))
        prob_detection = 1 - prob_all_ground
        
        return prob_detection
        prob_all_ground = jnp.prod(jnp.array(probability_list))
        return 1 - prob_all_ground
    
    def _apply_rotation(self, rho, theta):
        """Apply Ry rotation to qubit subsystem."""
        with qt.CoreOptions(default_dtype="jax"):
            Sy_jax = qt.sigmay()
            ry_gate = (-1j * Sy_jax * theta / 2).expm()
            r = qt.tensor(self._operators['In'], self._operators['In'], ry_gate)
            return r * rho * r.dag()
    
    def _measure_probability(self, rho, state):
        """Calculate measurement probability for specified state."""
        if state == '0':
            return jnp.real((self._operators['p0'] * rho).tr())
        elif state == '1':
            return jnp.real((self._operators['p1'] * rho).tr())
        else:
            raise ValueError(f"Invalid state specification: {state}")
    
    def _project_state(self, rho, state):
        """Project state onto measurement outcome."""
        if state == '0':
            return self._operators['p0'] * rho * self._operators['p0'].dag()
        elif state == '1':
            return self._operators['p1'] * rho * self._operators['p1'].dag()
        else:
            raise ValueError(f"Invalid state specification: {state}")
    
    def calculate_sensing_contrast(self, rotation_angles: List[float]) -> float:
        """
        Calculate the sensing contrast for given rotation angles.
        
        Args:
            rotation_angles: List of [theta, delta] rotation angles
            
        Returns:
            float: Sensing contrast (P_with_photon - P_without_photon)
        """
        if not self._initialized:
            raise RuntimeError("Experiment must be initialized")
        
        prob_with = self.run_sensing_protocol(rotation_angles, with_photon=True)
        prob_without = self.run_sensing_protocol(rotation_angles, with_photon=False)
        
        contrast = prob_with - prob_without
        
        # Store results for analysis
        self._last_results = {
            'rotation_angles': rotation_angles,
            'prob_with_photon': float(prob_with),
            'prob_without_photon': float(prob_without),
            'sensing_contrast': float(contrast)
        }
        
        return contrast
    
    def get_last_results(self) -> Dict[str, Any]:
        """Get results from the last sensing protocol run."""
        return self._last_results.copy()
    
    def get_operators(self) -> Dict[str, qt.Qobj]:
        """Get dictionary of quantum operators."""
        if not self._initialized:
            raise RuntimeError("Experiment must be initialized to access operators")
        return self._operators.copy()
    
    def get_solvers(self) -> Dict[str, qt.MESolver]:
        """Get dictionary of evolution solvers."""
        if not self._initialized:
            raise RuntimeError("Experiment must be initialized to access solvers")
        return self._solvers.copy()
    
    def update_experimental_parameters(self, new_experimental_parameters) -> None:
        """
        Update experimental parameters and re-initialize if needed.
        
        Args:
            new_experimental_parameters: New ExperimentalParameters object
        """
        self.experimental_parameters = new_experimental_parameters
        if self._initialized:
            # Re-initialize with new parameters
            self._initialized = False
            self.initialize()
    
    def update_trainable_parameters(self, new_trainable_parameters) -> None:
        """
        Update trainable parameters.
        
        Args:
            new_trainable_parameters: New TrainableParameters object
        """
        self.trainable_parameters = new_trainable_parameters
        # Note: Trainable parameters don't require re-initialization of the quantum system
    
    def update_noise_model(self, new_noise_model) -> None:
        """
        Update noise model and re-initialize solvers.
        
        Args:
            new_noise_model: New NoiseModel object
        """
        self.noise_model = new_noise_model
        if self._initialized:
            # Rebuild solvers with new noise model
            self._build_solvers()
    
    def __repr__(self) -> str:
        """String representation of the experiment."""
        status = "initialized" if self._initialized else "not initialized"
        return (f"Experiment("
                f"experimental_parameters={type(self.experimental_parameters).__name__ if self.experimental_parameters else None}, "
                f"trainable_parameters={type(self.trainable_parameters).__name__ if self.trainable_parameters else None}, "
                f"noise_model={type(self.noise_model).__name__ if self.noise_model else None}, "
                f"status={status})")

"""
Quantum Sensing Experiment Class
================================

Main experiment class that orchestrates quantum sensing protocols with configurable
parameters, noise models, and optimization strategies.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
from jax.scipy.special import erfc
from qsopt.core.experimental_parameters import ExperimentalParameters
from qsopt.core.trainable_parameters import TrainableParameters

# Suppress Diffrax complex dtype warning
warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax is a work in progress*")


def gu(t, **kwargs):  
    """
    Time-dependent coupling function for input cavity transparency.
    
    Args:
        t: float or JAX array, time variable
        **kwargs: Dictionary containing 'sigma' parameter (pulse bandwidth)
        
    Returns:
        float: Normalized coupling strength g(t)
    """
    sigma = kwargs.get("sigma", 0.1)
    dx = sigma * t
    coupling = jnp.sqrt(2*sigma/jnp.sqrt(jnp.pi)*jnp.exp(-dx**2)/erfc(dx))
    return float(coupling)

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
        1. Time-dependent cavity-cavity coupling Hamiltonian with Gaussian pulse
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
        # From notebook: Hc = qt.Qobj(1j/2*jnp.sqrt(gm)*(ainc*a - ain*ac))
        # Build using standard QuTiP operations
        coupling_coeff = 1j/2 * np.sqrt(gm)
        H_coupling = coupling_coeff * (a_in_dag * a - a_in * a_dag)  # type: ignore
        
        # Dispersive qubit-resonator interaction Hamiltonian  
        # From notebook: Hq = qt.Qobj(-chi*ac*a*sz1)
        H_dispersive = -chi * a_dag * a * sigma_z  # type: ignore
        
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
        # From notebook: L_int = qt.QobjEvo([ain, gu], args=args) + jnp.sqrt(gm) * a
        L_time_dep = qt.QobjEvo([a_in, gu], args=args)  
        L_cavity = np.sqrt(gm) * a
        
        interaction_ops: List[Union[qt.Qobj, qt.QobjEvo]] = [L_time_dep, L_cavity] + list(lindblad_noise)
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

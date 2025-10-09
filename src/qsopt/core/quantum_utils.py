"""
Quantum System Utilities
=========================

Utility functions for quantum system setup, operator generation, and initial state preparation.
Designed to support both single and multi-qubit quantum sensing experiments.

This module provides reusable components that can be composed for different experiment types.
"""

from typing import Dict, List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
from jax.scipy.special import erfc
from qsopt.core.experimental_parameters import ExperimentalParameters, InitialStateType


@jax.jit
def gu(t, **kwargs):  
    """
    Time-dependent coupling function for input cavity transparency.
    
    This function calculates the time-dependent coupling strength for a cavity
    with Gaussian temporal mode matching. The coupling is normalized to ensure
    proper energy exchange during the pulse interaction.
    
    Args:
        t: float or JAX array, time variable
        **kwargs: Dictionary containing 'sigma' parameter (pulse bandwidth)
        
    Returns:
        JAX array: Normalized coupling strength g(t)
        
    Physical Interpretation:
        The coupling function represents the effective interaction strength between
        the input field and the cavity mode, accounting for the temporal profile
        of the input pulse. The erfc function ensures causality and smooth turn-on.
    """
    sigma = kwargs.get("sigma", 0.1)
    dx = sigma * t
    coupling = jnp.sqrt(2*sigma/jnp.sqrt(jnp.pi)*jnp.exp(-dx**2)/erfc(dx))
    return jnp.array(coupling, float)


def generate_single_qubit_operators(
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int
) -> Dict[str, qt.Qobj]:
    """
    Generate operators for a single-qubit composite system (field ⊗ cavity ⊗ qubit).
    
    Creates all necessary operators embedded in the three-subsystem tensor product space:
    - Field (input cavity) operators: creation/annihilation
    - Resonator cavity operators: creation/annihilation
    - Qubit operators: Pauli matrices, ladder operators, projectors
    
    Args:
        field_levels: Number of Fock levels for input field mode
        cavity_levels: Number of Fock levels for resonator cavity mode
        qubit_levels: Number of levels for qubit (typically 2)
        
    Returns:
        Dictionary containing all operators in composite space
        
    Example:
        >>> operators = generate_single_qubit_operators(2, 2, 2)
        >>> sigma_z = operators['sigma_z']
        >>> P0 = operators['P0']
    """
    # Generate operators with JAX backend for autodiff compatibility
    with qt.CoreOptions(default_dtype="jax"):
        # Identity operators for each subsystem
        I_field = qt.identity(field_levels)
        I_cavity = qt.identity(cavity_levels)
        I_qubit = qt.identity(qubit_levels)
        
        # Individual subsystem operators
        a_field = qt.destroy(field_levels)        # Field annihilation
        a_cavity = qt.destroy(cavity_levels)      # Cavity annihilation
        
        # Qubit operators (for 2-level system)
        sigma_z = qt.sigmaz()
        sigma_x = qt.sigmax()
        sigma_y = qt.sigmay()
        sigma_minus = qt.destroy(qubit_levels)
        
        # Qubit measurement projectors
        P0 = qt.Qobj([[1, 0], [0, 0]])  # Ground state |0⟩⟨0|
        P1 = qt.Qobj([[0, 0], [0, 1]])  # Excited state |1⟩⟨1|
        
        # Embed operators in composite space (input_field ⊗ resonator_cavity ⊗ qubit)
        operators = {
            # Input field operators
            'a_in': qt.tensor(a_field, I_cavity, I_qubit),
            'a_in_dag': qt.tensor(a_field.dag(), I_cavity, I_qubit),
            
            # Resonator cavity operators
            'a': qt.tensor(I_field, a_cavity, I_qubit),
            'a_dag': qt.tensor(I_field, a_cavity.dag(), I_qubit),
            
            # Qubit operators
            'sigma_z': qt.tensor(I_field, I_cavity, sigma_z),
            'sigma_x': qt.tensor(I_field, I_cavity, sigma_x),
            'sigma_y': qt.tensor(I_field, I_cavity, sigma_y),
            'sigma_minus': qt.tensor(I_field, I_cavity, sigma_minus),
            'sigma_plus': qt.tensor(I_field, I_cavity, sigma_minus.dag()),
            
            # Qubit measurement projectors
            'P0': qt.tensor(I_field, I_cavity, P0),
            'P1': qt.tensor(I_field, I_cavity, P1),
            
            # Identity operators for reference
            'I_field': I_field,
            'I_cavity': I_cavity,
            'I_qubit': I_qubit,
        }
        
        return operators


def generate_two_qubit_operators(
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int
) -> Dict[str, qt.Qobj]:
    """
    Generate operators for a two-qubit composite system.
    
    NOT IMPLEMENTED YET - Placeholder for future two-qubit experiments.
    
    Will create operators for (field ⊗ cavity ⊗ qubit1 ⊗ qubit2) composite space.
    
    Args:
        field_levels: Number of Fock levels for input field mode
        cavity_levels: Number of Fock levels for resonator cavity mode
        qubit_levels: Number of levels for each qubit (typically 2)
        
    Returns:
        Dictionary containing all operators in composite space
        
    Raises:
        NotImplementedError: This function is not yet implemented
    """
    raise NotImplementedError(
        "Two-qubit operator generation is not yet implemented. "
        "This will be added in a future version to support multi-qubit experiments."
    )


def generate_initial_state(
    initial_config,
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int,
    num_qubits: int = 1
) -> qt.Qobj:
    """
    Generate initial density matrix based on configuration and system type.
    
    Supports multiple initial state types:
    - VACUUM: All subsystems in ground state
    - SINGLE_PHOTON: One photon in field, vacuum cavity, qubits in ground
    - COHERENT: Coherent state in field
    - THERMAL: Thermal state in cavity
    - CUSTOM: User-defined superposition
    
    Args:
        initial_config: InitialStateConfig object with state type and parameters
        field_levels: Number of Fock levels for input field
        cavity_levels: Number of Fock levels for resonator cavity
        qubit_levels: Number of levels for each qubit
        num_qubits: Number of qubits in the system (1 or 2)
        
    Returns:
        Initial density matrix in composite Hilbert space
        
    Raises:
        ValueError: If required parameters are missing or invalid
        NotImplementedError: If num_qubits > 1 (not yet supported)
        
    Example:
        >>> from qsopt.core.experimental_parameters import InitialStateConfig, InitialStateType
        >>> config = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
        >>> rho0 = generate_initial_state(config, 2, 2, 2, num_qubits=1)
    """
    if num_qubits > 1:
        raise NotImplementedError(
            f"Initial state generation for {num_qubits} qubits is not yet implemented. "
            "Currently only single-qubit systems are supported."
        )
    
    state_type = initial_config.state_type
    
    # Use JAX backend for compatibility
    with qt.CoreOptions(default_dtype="jax"):
        if state_type == InitialStateType.VACUUM:
            # Vacuum state: |0,0,0⟩
            return _create_vacuum_state(field_levels, cavity_levels, qubit_levels)
            
        elif state_type == InitialStateType.SINGLE_PHOTON:
            # Single photon in field: |1,0,0⟩
            return _create_single_photon_state(field_levels, cavity_levels, qubit_levels)
            
        elif state_type == InitialStateType.COHERENT:
            # Coherent state in field: |α,0,0⟩
            alpha = initial_config.coherent_alpha
            if alpha is None:
                raise ValueError("coherent_alpha must be specified for COHERENT state type")
            return _create_coherent_state(alpha, field_levels, cavity_levels, qubit_levels)
            
        elif state_type == InitialStateType.THERMAL:
            # Thermal state in cavity with average photon number n_bar
            n_bar = initial_config.thermal_n_bar
            if n_bar is None:
                raise ValueError("thermal_n_bar must be specified for THERMAL state type")
            return _create_thermal_state(n_bar, field_levels, cavity_levels, qubit_levels)
            
        elif state_type == InitialStateType.CUSTOM:
            # Custom state from user-provided amplitudes
            custom_amps = initial_config.custom_amplitudes
            if custom_amps is None:
                raise ValueError("custom_amplitudes must be specified for CUSTOM state type")
            return _create_custom_state(
                custom_amps, field_levels, cavity_levels, qubit_levels
            )
            
        else:
            raise ValueError(f"Unknown initial state type: {state_type}")


# ==================== Private Helper Functions ====================

def _create_vacuum_state(
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int
) -> qt.Qobj:
    """Create vacuum state |0,0,0⟩."""
    psi = qt.tensor(
        qt.basis(field_levels, 0),
        qt.basis(cavity_levels, 0),
        qt.basis(qubit_levels, 0)
    )
    return psi * psi.dag()  # type: ignore


def _create_single_photon_state(
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int
) -> qt.Qobj:
    """Create single photon state |1,0,0⟩."""
    psi = qt.tensor(
        qt.basis(field_levels, 1),
        qt.basis(cavity_levels, 0),
        qt.basis(qubit_levels, 0)
    )
    return psi * psi.dag()  # type: ignore


def _create_coherent_state(
    alpha: complex,
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int
) -> qt.Qobj:
    """Create coherent state |α,0,0⟩ in field mode."""
    coherent_field = qt.coherent(field_levels, alpha)
    vacuum_cavity = qt.basis(cavity_levels, 0)
    ground_qubit = qt.basis(qubit_levels, 0)
    
    psi = qt.tensor(coherent_field, vacuum_cavity, ground_qubit)
    return psi * psi.dag()  # type: ignore


def _create_thermal_state(
    n_bar: float,
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int
) -> qt.Qobj:
    """Create thermal state in cavity with average photon number n_bar."""
    # Create thermal state in cavity
    thermal_cavity = qt.thermal_dm(cavity_levels, n_bar)
    
    # Vacuum field and ground qubit
    vacuum_field = qt.basis(field_levels, 0)
    ground_qubit = qt.basis(qubit_levels, 0)
    vacuum_field_dm = vacuum_field * vacuum_field.dag()  # type: ignore
    ground_qubit_dm = ground_qubit * ground_qubit.dag()  # type: ignore
    
    # Tensor product of density matrices
    return qt.tensor(vacuum_field_dm, thermal_cavity, ground_qubit_dm)


def _create_custom_state(
    custom_amplitudes: Dict[Tuple[int, int, int], complex],
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int
) -> qt.Qobj:
    """Create custom state from user-provided amplitudes."""
    # Initialize zero state vector
    total_dim = field_levels * cavity_levels * qubit_levels
    psi_array = np.zeros((total_dim,), dtype=complex)
    
    # Fill in amplitudes from dictionary
    # Dictionary keys are tuples (field_n, cavity_n, qubit_n)
    for (n_field, n_cavity, n_qubit), amplitude in custom_amplitudes.items():
        # Validate indices
        if not (0 <= n_field < field_levels):
            raise ValueError(f"Field index {n_field} out of range [0, {field_levels})")
        if not (0 <= n_cavity < cavity_levels):
            raise ValueError(f"Cavity index {n_cavity} out of range [0, {cavity_levels})")
        if not (0 <= n_qubit < qubit_levels):
            raise ValueError(f"Qubit index {n_qubit} out of range [0, {qubit_levels})")
        
        # Compute flat index: field ⊗ cavity ⊗ qubit ordering
        idx = n_field * (cavity_levels * qubit_levels) + n_cavity * qubit_levels + n_qubit
        psi_array[idx] = amplitude
    
    # Normalize the state
    norm = np.linalg.norm(psi_array)
    if norm < 1e-10:
        raise ValueError("Custom state has zero norm")
    psi_array = psi_array / norm
    
    # Create QuTiP state
    psi = qt.Qobj(psi_array, dims=[[field_levels, cavity_levels, qubit_levels], [1, 1, 1]])
    return psi * psi.dag()  # type: ignore


def apply_single_qubit_rotation(
    rho: qt.Qobj,
    theta: float,
    axis: str,
    I_field: qt.Qobj,
    I_cavity: qt.Qobj
) -> qt.Qobj:
    """
    Apply rotation gate to qubit in composite space.
    
    Applies rotation only to qubit subsystem while preserving field and cavity states.
    
    Args:
        rho: Density matrix in composite space (field ⊗ cavity ⊗ qubit)
        theta: Rotation angle in radians
        axis: Rotation axis ('x', 'y', or 'z')
        I_field: Identity operator for field subsystem
        I_cavity: Identity operator for cavity subsystem
        
    Returns:
        Rotated density matrix
        
    Raises:
        ValueError: If axis is not 'x', 'y', or 'z'
        
    Example:
        >>> rho_rotated = apply_single_qubit_rotation(rho, np.pi/2, 'y', I_field, I_cavity)
    """
    with qt.CoreOptions(default_dtype="jax"):
        # Select Pauli matrix based on axis
        if axis.lower() == 'x':
            pauli = qt.sigmax()
        elif axis.lower() == 'y':
            pauli = qt.sigmay()
        elif axis.lower() == 'z':
            pauli = qt.sigmaz()
        else:
            raise ValueError(f"Invalid rotation axis: {axis}. Must be 'x', 'y', or 'z'.")
        
        # Create rotation gate: exp(-i * σ * θ / 2)
        rotation_gate = (-1j * pauli * theta / 2).expm()
        
        # Embed in composite space
        r = qt.tensor(I_field, I_cavity, rotation_gate)
        
        # Apply rotation: R ρ R†
        return r * rho * r.dag()  # type: ignore


def create_measurement_projector(
    outcome: int,
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int
) -> qt.Qobj:
    """
    Create measurement projector for qubit in composite space.
    
    Args:
        outcome: Measurement outcome (0 for ground state |0⟩, 1 for excited state |1⟩)
        field_levels: Number of field levels
        cavity_levels: Number of cavity levels  
        qubit_levels: Number of qubit levels
        
    Returns:
        Projector operator in composite space
        
    Example:
        >>> P0 = create_measurement_projector(0, 2, 2, 2)  # Project onto |0⟩
        >>> P1 = create_measurement_projector(1, 2, 2, 2)  # Project onto |1⟩
    """
    with qt.CoreOptions(default_dtype="jax"):
        I_field = qt.identity(field_levels)
        I_cavity = qt.identity(cavity_levels)
        
        if outcome == 0:
            # Ground state projector |0⟩⟨0|
            P = qt.Qobj([[1, 0], [0, 0]])
        elif outcome == 1:
            # Excited state projector |1⟩⟨1|
            P = qt.Qobj([[0, 0], [0, 1]])
        else:
            raise ValueError(f"Invalid measurement outcome: {outcome}. Must be 0 or 1.")
        
        return qt.tensor(I_field, I_cavity, P)


def project_and_measure(
    rho: qt.Qobj,
    outcome: int,
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int
):
    """
    Project density matrix onto measurement outcome and calculate probability.
    
    Performs a projective measurement on the qubit subsystem, returning both
    the projected (unnormalized) state and the measurement probability.
    
    Args:
        rho: Density matrix in composite space (field ⊗ cavity ⊗ qubit)
        outcome: Measurement outcome (0 or 1)
        field_levels: Number of field levels
        cavity_levels: Number of cavity levels
        qubit_levels: Number of qubit levels
        
    Returns:
        Tuple of (projected_state, probability):
            - projected_state: P|ψ⟩⟨ψ|P† (unnormalized)
            - probability: Tr(Pρ) as JAX array (for autodiff compatibility)
            
    Note:
        Returns JAX array for probability to maintain autodiff compatibility.
        Convert to float outside JAX-traced functions if needed.
            
    Example:
        >>> rho_proj, prob = project_and_measure(rho, 0, 2, 2, 2)
        >>> rho_normalized = rho_proj / rho_proj.tr()  # Normalize for next step
    """
    import jax.numpy as jnp
    
    P = create_measurement_projector(outcome, field_levels, cavity_levels, qubit_levels)
    rho_projected = P * rho * P.dag()  # type: ignore
    probability = jnp.real(rho_projected.tr())
    
    return rho_projected, probability


def measure_qubit_probability(
    rho: qt.Qobj,
    outcome: int,
    field_levels: int,
    cavity_levels: int,
    qubit_levels: int
):
    """
    Calculate probability of measuring qubit in specified state.
    
    Computes Tr(P_outcome ρ) where P_outcome is the projector onto |outcome⟩.
    
    Args:
        rho: Density matrix in composite space (field ⊗ cavity ⊗ qubit)
        outcome: Measurement outcome (0 for |0⟩, 1 for |1⟩)
        field_levels: Number of field levels
        cavity_levels: Number of cavity levels
        qubit_levels: Number of qubit levels
        
    Returns:
        Measurement probability ∈ [0,1] as JAX array (for autodiff compatibility)
        
    Note:
        Returns JAX array to maintain autodiff compatibility.
        Convert to float outside JAX-traced functions if needed.
        
    Example:
        >>> prob0 = measure_qubit_probability(rho, 0, 2, 2, 2)
        >>> prob1 = measure_qubit_probability(rho, 1, 2, 2, 2)
        >>> assert abs(prob0 + prob1 - 1.0) < 1e-10  # Probabilities sum to 1
    """
    import jax.numpy as jnp
    
    P = create_measurement_projector(outcome, field_levels, cavity_levels, qubit_levels)
    probability = jnp.real((P * rho * P.dag()).tr())  # type: ignore
    
    return probability

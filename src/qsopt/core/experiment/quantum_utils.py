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


@jax.jit
def u0(t, **kwargs):
    """
    Normalized Gaussian pulse envelope function.
    
    This function represents the amplitude envelope of a Gaussian input pulse,
    useful for visualizing the temporal shape of the input field. Unlike gu(),
    this is a simple Gaussian without normalization factors.
    
    Args:
        t: float or JAX array, time variable(s)
        **kwargs: Dictionary containing 'sigma' parameter (pulse bandwidth)
        
    Returns:
        JAX array: Gaussian pulse amplitude at time t
        
    Example:
        >>> import numpy as np
        >>> t_vals = np.linspace(-5, 5, 100)
        >>> sigma = 0.1
        >>> pulse = u0(t_vals, sigma=sigma)
        >>> # pulse has maximum value of 1.0 at t=0
        
    Physical Interpretation:
        Represents the unnormalized envelope of a Gaussian pulse centered at t=0.
        The pulse width is determined by 1/sigma, where sigma is the inverse pulse width.
    """
    sigma = kwargs.get("sigma", 0.1)
    dx = sigma * t
    return jnp.exp(-dx**2)


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
    qubit_levels: Union[int, List[int]]
) -> Dict[str, qt.Qobj]:
    """
    Generate operators for a two-qubit composite system.
    
    Creates operators for (field ⊗ cavity ⊗ qubit1 ⊗ qubit2) composite space.
    Each qubit can have different level truncation for flexibility.
    
    Args:
        field_levels: Number of Fock levels for input field mode
        cavity_levels: Number of Fock levels for resonator cavity mode
        qubit_levels: Number of levels for each qubit. Can be:
                     - int: Same levels for both qubits (typically 2)
                     - List[int]: Individual levels [qubit1_levels, qubit2_levels]
        
    Returns:
        Dictionary containing all operators in composite space:
        - Field operators: a_in, a_in_dag
        - Cavity operators: a, a_dag
        - Qubit1 operators: sigma_z1, sigma_x1, sigma_y1, sigma_minus1, sigma_plus1
        - Qubit2 operators: sigma_z2, sigma_x2, sigma_y2, sigma_minus2, sigma_plus2
        - Measurement projectors: P00, P01, P10, P11 (joint qubit states)
        - Individual projectors: P0_q1, P1_q1, P0_q2, P1_q2
        - Rotation operators: 
            * roty_q1: Y-rotation on qubit1 only
            * roty_q2: Y-rotation on qubit2 only
            * roty: Simultaneous Y-rotation on both qubits
        
    Example:
        >>> ops = generate_two_qubit_operators(2, 2, 2)
        >>> # Access operators for each qubit
        >>> sz1 = ops['sigma_z1']
        >>> sz2 = ops['sigma_z2']
        >>> # Joint measurement projector |00⟩⟨00|
        >>> P00 = ops['P00']
    """
    # Handle qubit_levels as list or int
    if isinstance(qubit_levels, int):
        q1_levels = qubit_levels
        q2_levels = qubit_levels
    elif isinstance(qubit_levels, list):
        if len(qubit_levels) < 2:
            raise ValueError(f"qubit_levels list must have at least 2 elements, got {len(qubit_levels)}")
        q1_levels = qubit_levels[0]
        q2_levels = qubit_levels[1]
    else:
        raise TypeError(f"qubit_levels must be int or list, got {type(qubit_levels)}")
    
    # Generate operators with JAX backend for autodiff compatibility
    with qt.CoreOptions(default_dtype="jax"):
        # Identity operators for each subsystem
        I_field = qt.identity(field_levels)
        I_cavity = qt.identity(cavity_levels)
        I_q1 = qt.identity(q1_levels)
        I_q2 = qt.identity(q2_levels)
        
        # Individual subsystem operators
        a_field = qt.destroy(field_levels)        # Field annihilation
        a_cavity = qt.destroy(cavity_levels)      # Cavity annihilation
        
        # Qubit 1 operators
        sigma_z1 = qt.sigmaz() if q1_levels == 2 else qt.jmat(q1_levels-1, 'z')
        sigma_x1 = qt.sigmax() if q1_levels == 2 else qt.jmat(q1_levels-1, 'x')
        sigma_y1 = qt.sigmay() if q1_levels == 2 else qt.jmat(q1_levels-1, 'y')
        sigma_minus1 = qt.destroy(q1_levels)
        
        # Qubit 2 operators
        sigma_z2 = qt.sigmaz() if q2_levels == 2 else qt.jmat(q2_levels-1, 'z')
        sigma_x2 = qt.sigmax() if q2_levels == 2 else qt.jmat(q2_levels-1, 'x')
        sigma_y2 = qt.sigmay() if q2_levels == 2 else qt.jmat(q2_levels-1, 'y')
        sigma_minus2 = qt.destroy(q2_levels)
        
        # Measurement projectors for 2-level qubits
        P0 = qt.Qobj([[1, 0], [0, 0]])  # Ground state |0⟩⟨0|
        P1 = qt.Qobj([[0, 0], [0, 1]])  # Excited state |1⟩⟨1|
        
        # Rotation operators (Y-rotation by π/2)
        # Ry(π/2) = [[cos(π/4), -sin(π/4)], [sin(π/4), cos(π/4)]]
        # = (1/√2)[[1, -1], [1, 1]]
        rot_single = qt.Qobj([[1, -1], [1, 1]]) / jnp.sqrt(2.0)
        
        # Embed operators in composite space (input_field ⊗ cavity ⊗ qubit1 ⊗ qubit2)
        operators = {
            # Input field operators
            'a_in': qt.tensor(a_field, I_cavity, I_q1, I_q2),
            'a_in_dag': qt.tensor(a_field.dag(), I_cavity, I_q1, I_q2),
            
            # Resonator cavity operators
            'a': qt.tensor(I_field, a_cavity, I_q1, I_q2),
            'a_dag': qt.tensor(I_field, a_cavity.dag(), I_q1, I_q2),
            
            # Qubit 1 operators
            'sigma_z1': qt.tensor(I_field, I_cavity, sigma_z1, I_q2),
            'sigma_x1': qt.tensor(I_field, I_cavity, sigma_x1, I_q2),
            'sigma_y1': qt.tensor(I_field, I_cavity, sigma_y1, I_q2),
            'sigma_minus1': qt.tensor(I_field, I_cavity, sigma_minus1, I_q2),
            'sigma_plus1': qt.tensor(I_field, I_cavity, sigma_minus1.dag(), I_q2),
            
            # Qubit 2 operators
            'sigma_z2': qt.tensor(I_field, I_cavity, I_q1, sigma_z2),
            'sigma_x2': qt.tensor(I_field, I_cavity, I_q1, sigma_x2),
            'sigma_y2': qt.tensor(I_field, I_cavity, I_q1, sigma_y2),
            'sigma_minus2': qt.tensor(I_field, I_cavity, I_q1, sigma_minus2),
            'sigma_plus2': qt.tensor(I_field, I_cavity, I_q1, sigma_minus2.dag()),
            
            # Joint measurement projectors (for 2-level qubits)
            'P00': qt.tensor(I_field, I_cavity, P0, P0),  # |00⟩⟨00|
            'P01': qt.tensor(I_field, I_cavity, P0, P1),  # |01⟩⟨01|
            'P10': qt.tensor(I_field, I_cavity, P1, P0),  # |10⟩⟨10|
            'P11': qt.tensor(I_field, I_cavity, P1, P1),  # |11⟩⟨11|
            
            # Individual qubit projectors
            'P0_q1': qt.tensor(I_field, I_cavity, P0, I_q2),  # |0⟩⟨0| ⊗ I for qubit1
            'P1_q1': qt.tensor(I_field, I_cavity, P1, I_q2),  # |1⟩⟨1| ⊗ I for qubit1
            'P0_q2': qt.tensor(I_field, I_cavity, I_q1, P0),  # I ⊗ |0⟩⟨0| for qubit2
            'P1_q2': qt.tensor(I_field, I_cavity, I_q1, P1),  # I ⊗ |1⟩⟨1| for qubit2
            
            # Rotation operators (Y-rotation by π/2, can be applied independently)
            'roty_q1': qt.tensor(I_field, I_cavity, rot_single, I_q2),  # Ry on qubit1 only
            'roty_q2': qt.tensor(I_field, I_cavity, I_q1, rot_single),  # Ry on qubit2 only
            'roty': qt.tensor(I_field, I_cavity, rot_single, rot_single),  # Simultaneous Ry on both
            
            # Identity operators for reference
            'I_field': I_field,
            'I_cavity': I_cavity,
            'I_q1': I_q1,
            'I_q2': I_q2,
        }
        
        return operators


def build_qubit_noise_operators(
    sigma_x: qt.Qobj,
    sigma_y: qt.Qobj,
    sigma_z: qt.Qobj,
    sigma_minus: qt.Qobj,
    depolarizing_rate: float,
    dephasing_rate: float,
    relaxation_rate: float
) -> List[qt.Qobj]:
    """
    Build Lindblad noise operators for a single qubit.
    
    Creates noise operators for common decoherence channels:
    - Depolarizing: Equal probability of X, Y, Z errors (√(γ/3) σᵢ)
    - Dephasing: Pure Z errors causing phase decoherence (√γ σz)
    - Relaxation: Energy decay from |1⟩ to |0⟩ (√γ σ₋)
    
    Args:
        sigma_x: Pauli X operator in composite space
        sigma_y: Pauli Y operator in composite space
        sigma_z: Pauli Z operator in composite space
        sigma_minus: Lowering operator σ₋ = |0⟩⟨1| in composite space
        depolarizing_rate: Depolarizing noise rate γ_depol
        dephasing_rate: Pure dephasing noise rate γ_deph
        relaxation_rate: Relaxation (T1) noise rate γ_relax
        
    Returns:
        List of Lindblad operators for this qubit
        
    Example:
        >>> # For single qubit system
        >>> ops = generate_single_qubit_operators(2, 2, 2)
        >>> noise_ops = build_qubit_noise_operators(
        ...     ops['sigma_x'], ops['sigma_y'], ops['sigma_z'], ops['sigma_minus'],
        ...     depolarizing_rate=0.01, dephasing_rate=0.005, relaxation_rate=0.002
        ... )
        
        >>> # For two-qubit system, qubit 1
        >>> ops = generate_two_qubit_operators(2, 2, 2)
        >>> noise_ops_q1 = build_qubit_noise_operators(
        ...     ops['sigma_x1'], ops['sigma_y1'], ops['sigma_z1'], ops['sigma_minus1'],
        ...     depolarizing_rate=0.01, dephasing_rate=0.005, relaxation_rate=0.002
        ... )
    """
    noise_operators = []
    
    # Depolarizing noise: equal probability of X, Y, Z errors
    if depolarizing_rate != 0.0:
        noise_operators.extend([
            np.sqrt(depolarizing_rate / 3) * sigma_x,
            np.sqrt(depolarizing_rate / 3) * sigma_y,
            np.sqrt(depolarizing_rate / 3) * sigma_z
        ])
    
    # Pure dephasing noise: Z errors only (phase decoherence)
    if dephasing_rate != 0.0:
        noise_operators.append(np.sqrt(dephasing_rate) * sigma_z)
    
    # Relaxation noise: energy decay |1⟩ → |0⟩
    if relaxation_rate != 0.0:
        noise_operators.append(np.sqrt(relaxation_rate) * sigma_minus)
    
    return noise_operators


def generate_initial_state(
    initial_config,
    field_levels: int,
    cavity_levels: int,
    qubit_levels: Union[int, List[int]],
    num_qubits: int = 1
) -> qt.Qobj:
    """
    Generate initial density matrix based on configuration and system type.
    
    Supports multiple initial state types:
    - VACUUM: All subsystems in ground state
    - SINGLE_PHOTON: One photon in field, vacuum cavity, qubits in ground or superposition
    - COHERENT: Coherent state in field
    - THERMAL: Thermal state in cavity
    - CUSTOM: User-defined superposition
    
    For multi-qubit systems, qubits are initialized in equal superposition (|0⟩+|1⟩)/√2
    unless otherwise specified.
    
    Args:
        initial_config: InitialStateConfig object with state type and parameters
        field_levels: Number of Fock levels for input field
        cavity_levels: Number of Fock levels for resonator cavity
        qubit_levels: Number of levels for each qubit (int or list)
        num_qubits: Number of qubits in the system (1 or 2)
        
    Returns:
        Initial density matrix in composite Hilbert space
        
    Raises:
        ValueError: If required parameters are missing or invalid
        NotImplementedError: If num_qubits > 2
        
    Example:
        >>> from qsopt.core.experimental_parameters import InitialStateConfig, InitialStateType
        >>> config = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
        >>> rho0 = generate_initial_state(config, 2, 2, 2, num_qubits=1)
    """
    if num_qubits > 2:
        raise NotImplementedError(
            f"Initial state generation for {num_qubits} qubits is not yet implemented. "
            "Currently supports 1-2 qubit systems."
        )
    
    state_type = initial_config.state_type
    
    # Use JAX backend for compatibility
    with qt.CoreOptions(default_dtype="jax"):
        if num_qubits == 1:
            # Single qubit system - use original logic
            if state_type == InitialStateType.VACUUM:
                return _create_vacuum_state(field_levels, cavity_levels, qubit_levels)
            elif state_type == InitialStateType.SINGLE_PHOTON:
                return _create_single_photon_state(field_levels, cavity_levels, qubit_levels)
            elif state_type == InitialStateType.COHERENT:
                alpha = initial_config.coherent_alpha
                if alpha is None:
                    raise ValueError("coherent_alpha must be specified for COHERENT state type")
                return _create_coherent_state(alpha, field_levels, cavity_levels, qubit_levels)
            elif state_type == InitialStateType.THERMAL:
                n_bar = initial_config.thermal_n_bar
                if n_bar is None:
                    raise ValueError("thermal_n_bar must be specified for THERMAL state type")
                return _create_thermal_state(n_bar, field_levels, cavity_levels, qubit_levels)
            elif state_type == InitialStateType.CUSTOM:
                custom_amps = initial_config.custom_amplitudes
                if custom_amps is None:
                    raise ValueError("custom_amplitudes must be specified for CUSTOM state type")
                return _create_custom_state(custom_amps, field_levels, cavity_levels, qubit_levels)
            else:
                raise ValueError(f"Unknown initial state type: {state_type}")
                
        elif num_qubits == 2:
            # Two qubit system
            if state_type == InitialStateType.SINGLE_PHOTON:
                # |1⟩_field ⊗ |0⟩_cavity ⊗ (|0⟩+|1⟩)/√2_q1 ⊗ (|0⟩+|1⟩)/√2_q2
                return _create_two_qubit_single_photon_state(field_levels, cavity_levels, qubit_levels)
            elif state_type == InitialStateType.VACUUM:
                return _create_two_qubit_vacuum_state(field_levels, cavity_levels, qubit_levels)
            else:
                # For other state types, use single qubit logic and extend to 2 qubits
                raise NotImplementedError(
                    f"Initial state type {state_type} not yet implemented for two-qubit systems. "
                    "Currently only SINGLE_PHOTON and VACUUM are supported."
                )
        
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


def _create_two_qubit_vacuum_state(
    field_levels: int,
    cavity_levels: int,
    qubit_levels: Union[int, List[int]]
) -> qt.Qobj:
    """
    Create vacuum state for two-qubit system: |0⟩_field ⊗ |0⟩_cavity ⊗ |0⟩_q1 ⊗ |0⟩_q2.
    
    Args:
        field_levels: Number of field levels
        cavity_levels: Number of cavity levels
        qubit_levels: Qubit levels (int or list of 2 ints)
    
    Returns:
        Density matrix for vacuum state in 4-subsystem composite space
    """
    if isinstance(qubit_levels, int):
        q1_levels = qubit_levels
        q2_levels = qubit_levels
    else:
        q1_levels = qubit_levels[0]
        q2_levels = qubit_levels[1]
    
    psi = qt.tensor(
        qt.basis(field_levels, 0),
        qt.basis(cavity_levels, 0),
        qt.basis(q1_levels, 0),
        qt.basis(q2_levels, 0)
    )
    return psi * psi.dag()  # type: ignore


def _create_two_qubit_single_photon_state(
    field_levels: int,
    cavity_levels: int,
    qubit_levels: Union[int, List[int]]
) -> qt.Qobj:
    """
    Create single photon state for two-qubit system.
    
    State: |1⟩_field ⊗ |0⟩_cavity ⊗ (|0⟩+|1⟩)/√2_q1 ⊗ (|0⟩+|1⟩)/√2_q2
    
    Both qubits are initialized in equal superposition as per Fabio's notebook.
    
    Args:
        field_levels: Number of field levels
        cavity_levels: Number of cavity levels
        qubit_levels: Qubit levels (int or list of 2 ints)
    
    Returns:
        Density matrix for single photon state with qubits in superposition
    """
    if isinstance(qubit_levels, int):
        q1_levels = qubit_levels
        q2_levels = qubit_levels
    else:
        q1_levels = qubit_levels[0]
        q2_levels = qubit_levels[1]
    
    # Field: single photon |1⟩
    field_state = qt.basis(field_levels, 1)
    
    # Cavity: vacuum |0⟩
    cavity_state = qt.basis(cavity_levels, 0)
    
    # Qubits: equal superposition (|0⟩ + |1⟩)/√2
    qubit1_state = (qt.basis(q1_levels, 0) + qt.basis(q1_levels, 1)) / jnp.sqrt(2.0)
    qubit2_state = (qt.basis(q2_levels, 0) + qt.basis(q2_levels, 1)) / jnp.sqrt(2.0)
    
    psi = qt.tensor(field_state, cavity_state, qubit1_state, qubit2_state)
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

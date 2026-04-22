"""
Quantum System Utilities
=========================

Utility functions for quantum system setup, operator generation, and initial state preparation.
Designed to support both single and multi-qubit quantum sensing experiments.

This module provides reusable components that can be composed for different experiment types.
"""

from typing import Dict, List, Optional, Tuple, Union
import math

import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
from jax.scipy.special import erfc

from qsopt.core.experimental_parameters import InitialStateType


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
    coupling = jnp.sqrt(2 * sigma / jnp.sqrt(jnp.pi) * jnp.exp(-(dx**2)) / erfc(dx))
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
    return jnp.exp(-(dx**2))

def generate_n_qubit_operators(
    n_cavities: int,
    n_qubits: int, 
    cavity_levels: Union[int, List[int]],
    qubit_levels: Union[int, List[int]], 
    detection_states: Optional[[List[str]]] = None
) -> Dict[str, qt.Qobj]:
    """
    Generate operators for an n-qubit composite system.

    Creates operators for (field ⊗ cavity ⊗ qubit1 ⊗ ... ⊗ qubitn) composite space.
    Each qubit can have different level truncation for flexibility.

    Args:
        n_cavities: Number of cavities in the system
        n_qubits: Number of qubits in the system
        cavity_levels: Number of Fock levels for cavities. Can be:
                     - int: Same levels for all cavities (typically 2)
                     - List[int]: Individual levels [cavity1_levels, cavity2_levels, ..., cavityn_levels]
        qubit_levels: Number of levels for each qubit. Can be:
                     - int: Same levels for all qubits (typically 2)
                     - List[int]: Individual levels [qubit1_levels, qubit2_levels, ..., qubitn_levels]
        detection_states: Optional list of binary strings representing qubit states that correspond to "detection" outcomes. 
                          For example, for 2 qubits, detection_states could be ['01', '10', '11'] to indicate that any state with at least one qubit in |1⟩ is a detection.
                          If None, defaults to all non-zero states as detection states.

    Returns:
        Dictionary containing all operators in composite space:
        - Lists of cavities operators: a, a_dag
        - Lists of qubits operators: sigma_z, sigma_x, sigma_y, sigma_minus, sigma_plus
        - Dictionary of all 2^n measurement projectors: P (matrix of joint qubit states)
        - Lists of individual qubit projectors: P0_q, P1_q
        - Global projectors: P_all0 (joint ground state), P_detect, P_no_detect
        - Identity operators: I_c (cavities), I_q (qubits)
        - Rotation operators:
            * roty_q: list of Y-rotations on individual qubits
            * roty: Simultaneous Y-rotation on all qubits
        - Reset operators:
            * reset_q: list of reset operators for each qubit
            * reset_all: Global reset operator for all qubits
            * measure_reset: Measurement operator that also resets qubits (depends on detection states)
            * measure_reset_dag: Adjoint of measure_reset for use in Lindblad terms
    """
    # Handle cavity_levels as list or int
    if isinstance(cavity_levels, int):
        c_levels = [cavity_levels] * n_cavities
    elif isinstance(cavity_levels, list):
        if len(cavity_levels) != n_cavities:
            raise ValueError(
                f"cavity_levels list must have n_cavities={n_cavities} elements, got {len(cavity_levels)}"
            )
        c_levels = cavity_levels
    else:
        raise TypeError(f"cavity_levels must be int or list, got {type(cavity_levels)}")

    # Handle qubit_levels as list or int
    if isinstance(qubit_levels, int):
        q_levels = [qubit_levels] * n_qubits
    elif isinstance(qubit_levels, list):
        if len(qubit_levels) != n_qubits:
            raise ValueError(
                f"qubit_levels list must have n_qubits={n_qubits} elements, got {len(qubit_levels)}"
            )
        q_levels = qubit_levels
    else:
        raise TypeError(f"qubit_levels must be int or list, got {type(qubit_levels)}")

    # Generate operators with JAX backend for autodiff compatibility
    with qt.CoreOptions(default_dtype="jax"):

        # Lists of identity operators for each cavity
        I_c = [qt.identity(c_levels[i]) for i in range(n_cavities)]

        # Lists of operators for each cavity
        a_c = [qt.destroy(c_levels[i]) for i in range(n_cavities)]  # Cavity annihilation

        # Lists of qubits operators
        I_q = [qt.identity(q_levels[i]) for i in range(n_qubits)]
        sigma_z = [qt.sigmaz() if q_levels[i] == 2 else qt.jmat(q_levels[i] - 1, "z") for i in range(n_qubits)]
        sigma_x = [qt.sigmax() if q_levels[i] == 2 else qt.jmat(q_levels[i] - 1, "x") for i in range(n_qubits)]
        sigma_y = [qt.sigmay() if q_levels[i] == 2 else qt.jmat(q_levels[i] - 1, "y") for i in range(n_qubits)]
        sigma_minus = [qt.destroy(q_levels[i]) for i in range(n_qubits)]

        # Rotation operator (Y-rotation by π/2)
        # Ry(π/2) = [[cos(π/4), -sin(π/4)], [sin(π/4), cos(π/4)]]
        # = (1/√2)[[1, -1], [1, 1]]
        rot_single = qt.Qobj([[1, -1], [1, 1]]) / jnp.sqrt(2.0)

        # Lists of measurement projectors for n qubits with q-levels
        P0 = [qt.Qobj([[1, 0] + [0]*(l-2)] + [[0]*l]*(l-1)) for l in q_levels]    # Ground state |0⟩⟨0|
        P1 = [qt.Qobj([[0]*l] + [[0, 1] + [0]*(l-2)] + [[0]*l]*(l-2)) for l in q_levels] # Excited state |1⟩⟨1|
        
        # Reset operators, individual qubits and global reset
        reset_q = [qt.Qobj([[1]*l] + [[0]*l]*(l-1)) for l in q_levels]
        reset_all = qt.tensor(I_c + reset_q)

        # Helper function to embed single-qubit operator in composite space
        def embed_qubit_op(op, qubit_idx):
            """Embed operator acting on qubit qubit_idx into full composite space."""
            ops_list = I_c + I_q[:qubit_idx] + [op] + I_q[qubit_idx+1:]
            return qt.tensor(ops_list)

        # Helper function to embed single-cavity operator in composite space
        def embed_cavity_op(op, cavity_idx):
            """Embed operator acting on cavity cavity_idx into full composite space."""
            ops_list = I_c[:cavity_idx] + [op] + I_c[cavity_idx+1:] + I_q
            return qt.tensor(ops_list)

        # Embed operators in composite space (input_field ⊗ cavity ⊗ qubits)
        operators = {
            # Lists of cavity operators
            "a": [embed_cavity_op(a_c[i],i) for i in range(n_cavities)],
            "a_dag": [embed_cavity_op(a_c[i].dag(),i) for i in range(n_cavities)],
            # Lists of qubits operators
            "sigma_z": [embed_qubit_op(sigma_z[i], i) for i in range(n_qubits)],
            "sigma_x": [embed_qubit_op(sigma_x[i], i) for i in range(n_qubits)],
            "sigma_y": [embed_qubit_op(sigma_y[i], i) for i in range(n_qubits)],
            "sigma_minus": [embed_qubit_op(sigma_minus[i], i) for i in range(n_qubits)],
            "sigma_plus": [embed_qubit_op(sigma_minus[i].dag(), i) for i in range(n_qubits)],
            # Individual qubit projectors on |0⟩, |1⟩ and the reset operator
            "P0_q": [embed_qubit_op(P0[i], i) for i in range(n_qubits)],
            "P1_q": [embed_qubit_op(P1[i], i) for i in range(n_qubits)],
            # Global qubit projectors
            "P_all0": qt.tensor(I_c + P0),  # Joint projector onto |00...0⟩
            # Reset operators
            "reset_q": reset_q, 
            "reset_all": reset_all, 
            # Rotation operators (Y-rotation by π/2, can be applied independently)
            "roty_q": [embed_qubit_op(rot_single, i) for i in range(n_qubits)],
            "roty": qt.tensor(I_c + [rot_single]*n_qubits),  # Simultaneous Ry on all qubits
            # Identity operators for reference
            "I_c": I_c,
            "I_q": I_q,
        }

        
        Ptemp = [P0,P1]
        
        if detection_states == 'all states':

            all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
            P_all = [qt.tensor(I_c + [Ptemp[q_state][qb] for qb,q_state in enumerate(list(map(int,state)))]) for state in all_states]
            
            # Insert in dictionary all states projectors
            operators['Pall'] = P_all
            
            # Calculate measure reset operators
            measure_reset = [reset_all*p for p in P_all]
            measure_reset_dag = [x.dag() for x in measure_reset]

        else:

            # Detection and non-detection projectors using the detection states:
                # Default detection states are all the non zero states
            if detection_states is None:
                detection_states = [format(i, f'0{n_qubits}b') for i in range(1,2**n_qubits)]

            # Calculate detection projectors
            P_detection = sum([ \
                qt.tensor(I_c + [Ptemp[q_state][qb] for qb,q_state in enumerate(list(map(int,state)))]) \
                for state in detection_states])
            P_no_detection = sum([ \
                qt.tensor(I_c + [Ptemp[q_state][qb] \
                    for qb,q_state in enumerate(list(map(int,format(i, f'0{n_qubits}b'))))]) \
                for i in range(2**n_qubits) if not any(state == format(i, f'0{n_qubits}b') for state in detection_states)])

            # Insert in dictionary detection projectors
            operators["P_detect"] = P_detection
            operators["P_no_detect"] = P_no_detection
            
            # Calculate measure reset operators            
            measure_reset = reset_all*P_no_detection
            measure_reset_dag = measure_reset.dag()


        # Insert in dictionary measure reset operators
        operators["measure_reset"] = measure_reset
        operators["measure_reset_dag"] = measure_reset_dag

        return operators


def build_qubit_noise_operators(
    sigma_x: qt.Qobj,
    sigma_y: qt.Qobj,
    sigma_z: qt.Qobj,
    sigma_minus: qt.Qobj,
    depolarizing_rate: float,
    dephasing_rate: float,
    relaxation_rate: float,
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
        noise_operators.extend(
            [
                np.sqrt(depolarizing_rate / 3) * sigma_x,
                np.sqrt(depolarizing_rate / 3) * sigma_y,
                np.sqrt(depolarizing_rate / 3) * sigma_z,
            ]
        )

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
    n_qubits: int = 1,
) -> qt.Qobj:
    """
    Generate initial density matrix based on configuration and system type.

    Supports multiple initial state types:
    - VACUUM: All subsystems in ground state
    - SINGLE_PHOTON: One photon in field, vacuum cavity, qubits in ground or superposition
    - COHERENT: Coherent state in field
    - THERMAL: Thermal state in field
    - CUSTOM: User-defined superposition

    Qubits are initialized in equal superposition (|1⟩+|2⟩+...+|n⟩)/√n
    unless otherwise specified.

    Args:
        initial_config: InitialStateConfig object with state type and parameters
        field_levels: Number of Fock levels for input field
        cavity_levels: Number of Fock levels for resonator cavity
        qubit_levels: Number of levels for each qubit (int or list)
        n_qubits: Number of qubits in the system
    Returns:
        Initial density matrix in composite Hilbert space

    Raises:
        ValueError: If required parameters are missing or invalid

    Example:
        >>> from qsopt.core.experimental_parameters import InitialStateConfig, InitialStateType
        >>> config = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
        >>> rho0 = generate_initial_state(config, 2, 2, 2, n_qubits=1)
    """

    state_type = initial_config.state_type

    # Use JAX backend for compatibility
    with qt.CoreOptions(default_dtype="jax"):
        # Create ground state base (cavity + qubits always in ground state)
        ground_base = _create_ground_state_base(cavity_levels, qubit_levels, n_qubits)

        # Create field state (varies by experiment)
        if state_type == InitialStateType.VACUUM:
            field_dm = _create_field_vacuum(field_levels)

        elif state_type == InitialStateType.SINGLE_PHOTON:
            field_dm = _create_field_single_photon(field_levels)

        elif state_type == InitialStateType.COHERENT:
            alpha = initial_config.coherent_alpha
            if alpha is None:
                raise ValueError("coherent_alpha must be specified for COHERENT state type")
            field_dm = _create_field_coherent(field_levels, alpha)

        elif state_type == InitialStateType.CUSTOM:
            custom_amps = initial_config.custom_amplitudes
            if custom_amps is None:
                raise ValueError("custom_amplitudes must be specified for CUSTOM state type")
            return _create_custom_state(
                custom_amps, field_levels, cavity_levels, qubit_levels, n_qubits
            )

        else:
            raise ValueError(f"Unknown initial state type: {state_type}")

        # Combine field state with ground state base (field ⊗ cavity ⊗ qubits)
        return qt.tensor(field_dm, ground_base)


# ==================== Private Helper Functions ====================


def _create_ground_state_base(
    cavity_levels: int, qubit_levels: Union[int, List[int]], n_qubits: int
) -> qt.Qobj:
    """
    Create ground state for cavity and qubits: |0⟩_cavity ⊗ |0⟩_q1 ⊗ |0⟩_q2 ⊗ ...

    The cavity and qubits are always initialized in ground state. Only the input
    field state varies depending on the experiment.

    Args:
        cavity_levels: Number of cavity levels
        qubit_levels: Number of levels for each qubit (int or list)
        n_qubits: Number of qubits

    Returns:
        Ground state density matrix for cavity + qubits subsystem
    """
    # Extract qubit levels for each qubit
    if isinstance(qubit_levels, int):
        q_levels = [qubit_levels] * n_qubits
    else:
        q_levels = qubit_levels[:n_qubits]

    # Start with cavity ground state
    cavity_ground = qt.basis(cavity_levels, 0)

    # Build ground state for all qubits
    qubit_grounds = [qt.basis(q_levels[i], 0) for i in range(n_qubits)]

    # Create state vector: cavity ⊗ qubit1 ⊗ qubit2 ⊗ ... ⊗ qubitn
    psi = qt.tensor(cavity_ground, *qubit_grounds)
    return psi * psi.dag()  # type: ignore


def _create_field_vacuum(field_levels: int) -> qt.Qobj:
    """Create vacuum state for input field: |0⟩_field."""
    field_state = qt.basis(field_levels, 0)
    return field_state * field_state.dag()  # type: ignore


def _create_field_single_photon(field_levels: int) -> qt.Qobj:
    """Create single photon state for input field: |1⟩_field."""
    field_state = qt.basis(field_levels, 1)
    return field_state * field_state.dag()  # type: ignore


def _create_field_coherent(field_levels: int, alpha: complex) -> qt.Qobj:
    """Create coherent state for input field: |α⟩_field."""
    coherent_field = qt.coherent(field_levels, alpha)
    return coherent_field * coherent_field.dag()  # type: ignore


def _create_field_thermal(cavity_levels: int, n_bar: float) -> qt.Qobj:
    """
    Create thermal state in cavity with average photon number n_bar.

    Note: This is a special case where the cavity is NOT in ground state.
    """
    return qt.thermal_dm(cavity_levels, n_bar)


def _create_custom_state(
    custom_amplitudes: Dict[Tuple[int, int, Union[int, Tuple[int]]], complex],
    field_levels: int,
    cavity_levels: int,
    qubit_levels: Union[int, List[int]],
    n_qubits: int,
) -> qt.Qobj:
    """
    Create custom state from user-provided amplitudes.

    For custom states, we need to support arbitrary configurations,
    so we maintain the full flexibility of the original implementation.
    """
    # Handle qubit_levels
    if isinstance(qubit_levels, int):
        q_levels = [qubit_levels] * n_qubits
    else:
        q_levels = qubit_levels[:n_qubits]

    # For now, only support single qubit custom states
    #if n_qubits != 1:
    #    raise NotImplementedError("Custom states only supported for single qubit systems")

    # Initialize zero state vector
    total_dim = field_levels * cavity_levels * math.prod(q_levels)
    psi_array = np.zeros((total_dim,), dtype=complex)

    # Fill in amplitudes from dictionary
    # Dictionary keys are tuples (field, cavity, qubit)

    if all(isinstance(z, int) for (x,y,z),e in custom_amplitudes.items()):
        for (field, cavity, qubit), amplitude in custom_amplitudes.items():

            # Validate indices
            if not (0 <= field < field_levels):
                raise ValueError(f"Field index {field} out of range [0, {field_levels})")
            if not (0 <= cavity < cavity_levels):
                raise ValueError(f"Cavity index {cavity} out of range [0, {cavity_levels})")
            qubit = [qubit]*len(q_levels)
            if not all(0 <= x < y for x,y in zip(qubit,q_levels)):
                raise ValueError(f"Qubit index {qubit[0]} is out of range [0, {min(q_levels)})")

            # Compute flat index: field ⊗ cavity ⊗ qubit1 ⊗ ... ⊗ qubitn ordering
            idx = [field * (cavity_levels * math.prod(q_levels)) + cavity * math.prod(q_levels) + qubit[i]*math.prod(q_levels[i+1:]) for i in range(len(qubit))]
            psi_array[idx] = amplitude

    elif all(isinstance(z, tuple) for (x,y,z),e in custom_amplitudes.items()):
        for (field, cavity, qubit), amplitude in custom_amplitudes.items():

            # Validate indices
            if not (0 <= field < field_levels):
                raise ValueError(f"Field index {field} out of range [0, {field_levels})")
            if not (0 <= cavity < cavity_levels):
                raise ValueError(f"Cavity index {cavity} out of range [0, {cavity_levels})")
            if not all(0 <= x < y for x,y in zip(list(qubit),q_levels)):
                raise ValueError("At least one of the qubit indexes is out of range [0, qubit levels)")

            # Compute flat index: field ⊗ cavity ⊗ qubit ordering
            idx = [field * (cavity_levels * math.prod(q_levels)) + cavity * math.prod(q_levels) + qubit[i]*math.prod(q_levels[i+1:]) for i in range(len(qubit))]
            psi_array[idx] = amplitude


    # Normalize the state
    norm = np.linalg.norm(psi_array)
    if norm < 1e-10:
        raise ValueError("Custom state has zero norm (<1e-10)")
    psi_array = psi_array / norm

    # Create QuTiP state
    psi = qt.Qobj(psi_array, dims=[[field_levels, cavity_levels] + q_levels, [1, 1] + [1]*len(q_levels)])
    return psi * psi.dag()  # type: ignore


def measure_qubits_probability(
    rho: qt.Qobj, qubit_indices: Union[int, str, List[int]], operators: Dict[str, qt.Qobj], state: str = "0",\
    field_levels: Optional[int] = None, cavity_levels: Optional[int] = None, q_levels: Optional[List[int]] = None
) -> float:
    """
    Measure probability for specific qubits in multi-qubit system.

    Generic measurement function supporting:
    - Single qubit measurement: qubit_indices=i measures qubit i
    - Full multi-qubit measurement: qubit_indices='all' measures all qubits jointly
    - Partial multi-qubit measurement: qubit_indices=list[i,j,...,k] measures qubits i,j,...,k jointly

    Args:
        rho: Density matrix in composite space
        qubit_indices: List of qubit indices to measure (0-based)
        operators: Dictionary of operators from generate_*_operators()
        state: State to measure:
                - For single qubit: '0' or '1'
                - For multiple qubits: '00...0', '10...0', '01...0', ... , '11...1'

    Returns:
        Measurement probability ∈ [0,1]

    Example:
        >>> # Measure qubit 0 only
        >>> p0_q0 = measure_qubits_probability(rho, 0, ops, state='0')
        >>>
        >>> # Measure multiple qubits jointly
        >>> p000 = measure_qubits_probability(rho, 'all', ops, state='000')
    """
    import jax.numpy as jnp

    n_qubits= len((operators['P1_q']))


    if qubit_indices == 'all':
        # Joint measurement - only all-ground state supported
        if len(state) != n_qubits:
            raise ValueError(
                f"State string length ({len(state)}) must match the total number of qubits in the system {n_qubits}"
            )
        if state == '0' * n_qubits:
            # All qubits in ground state
            P = operators['P_all0']
        else:
            # For other states, need to construct projector manually
            if (field_levels is None) or (cavity_levels is None):
                raise ValueError("Measurement of arbitrary joint states requires field_levels and cavity_levels")
            I_field = qt.identity(field_levels)
            I_cavity = qt.identity(cavity_levels)
            if q_levels is None:
                q_levels = [2]*n_qubits
            elif isinstance(q_levels, int):
                q_levels = [q_levels] * n_qubits

            # Build projector for each qubit based on state string
            qubit_projectors = []
            for i, s in enumerate(state):
                if s == '0':
                    qubit_projectors.append(qt.Qobj([[1, 0] + [0]*(q_levels[i]-2)] + [[0]*q_levels[i]]*(q_levels[i]-1)))
                elif s == '1':
                    qubit_projectors.append(qt.Qobj([[0]*q_levels[i]] + [[0, 1] + [0]*(q_levels[i]-2)] + [[0]*q_levels[i]]*(q_levels[i]-2)))
                else:
                    raise ValueError(f"Invalid state character '{s}', must be '0' or '1'")

            P = qt.tensor([I_field, I_cavity] + qubit_projectors)

    elif (isinstance(qubit_indices, list) and len(qubit_indices) == 1) or (isinstance(qubit_indices, int) and qubit_indices < n_qubits):
        # Single qubit measurement
        if isinstance(qubit_indices, list):
            qubit_indices = qubit_indices[0]
        projector_key = f"P{state}_q"
        if projector_key not in operators:
            raise ValueError(f"Projector {projector_key} not found in operators")
        if len(state) != 1:
            raise ValueError(f"In single qubit projections the state {state} must be either 0 or 1")
        P = operators[projector_key][qubit_indices]

    elif (len(qubit_indices) <= n_qubits):
        # Joint measurement
        if (field_levels is None) or (cavity_levels is None):
            raise ValueError("Non cached measurement of non fixed number of qubits require both the field_levels and cavity_levels")
        if len(state) != len(qubit_indices):
            raise ValueError(f"Lenght of state string ({len(state)}) must match lenght of qubit_indices ({len(qubit_indices)})")
        I_field = qt.identity(field_levels)
        I_cavity = qt.identity(cavity_levels)
        if q_levels is None:
            q_levels = [2]*n_qubits
        elif isinstance(q_levels, int):
            q_levels = [q_levels] * n_qubits
        elif len(q_levels) != n_qubits:
            raise ValueError(f"q_levels were passed, but the lenght is different than the number of qubits ({n_qubits})")

            #Generate the projector: (l is the number of qubit levels of a qubit)
            # IF qubit is in the indices -> generates the matrix lxl that projects on the state at the same index
            # ELSE -> generates the identity lxl
        qubit_projector = [ \
            qt.Qobj([[1 - int(state[qubit_indices.index(i)]), 0] + [0]*(l-2)] +\
                [[0, 0 + int(state[qubit_indices.index(i)])] + [0]*(l-2)] +\
                [[0]*l]*(l-2)) \
            if i in qubit_indices \
            else qt.identity(l) \
            for i,l in enumerate(q_levels) \
                ]
        P = qt.tensor([I_field, I_cavity] + qubit_projector)

    else:
        raise ValueError(f'qubit indices must either be:\n \
        - int or list of int with lenght 1 ->   single qubit measurement in the interval [0,{n_qubits})\n\
        - a string = "all" ->                   joint measurement on all qubits\n \
        - a list of lenght between [2,{n_qubits}] ->     non cached measurement of non fixed number of qubits')

    probability = jnp.real((P * rho * P.dag()).tr())  # type: ignore
    return float(probability)


def embed_circuit_unitary(
    circuit_unitary: jnp.ndarray,
    cavity_levels: int
) -> jnp.ndarray:
    """
    Embed an n-qubit circuit unitary into the full composite Hilbert space using JAX.

    The composite space is: input_field ⊗ resonator_cavity ⊗ qubit1 ⊗ qubit2 ⊗ ... ⊗ qubitn
    The circuit acts only on the qubit subspace (qubit1 ⊗ qubit2 ⊗ ... ⊗ qubitn).

    Args:
        circuit_unitary: (Σ qubit_levels)x(Σ qubit_levels) unitary matrix for the n-qubit circuit as JAX array
        cavity_levels: List of integers representing the number of levels in each resonator cavity

    Returns:
        Full-space unitary as JAX array: I_field ⊗ I_cavity ⊗ circuit_unitary

    Example:
        >>> # 2-qubit circuit unitary (4x4 for 2-level qubits)
        >>> U_circuit = jnp.eye(4, dtype=jnp.complex128)
        >>> U_full = embed_circuit_unitary(U_circuit, field_levels=2, cavity_levels=3)
        >>> # U_full is now (2*3*4)x(2*3*4) = 24x24
    """
    # Build full operator using JAX Kronecker products
    # I_cavities ⊗ U_circuit
    I_cavities = jnp.eye(np.prod(cavity_levels), dtype=jnp.complex128)

    # Kronecker product: I_cavities ⊗ U_circuit
    U_full_jax = jnp.kron(I_cavities, circuit_unitary)

    return U_full_jax

"""
Quantum System Utilities
=========================

Utility functions for quantum system setup, operator generation, and initial state preparation.
Designed to support both single and multi-qubit quantum sensing experiments.

This module provides reusable components that can be composed for different experiment types.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
from jax.scipy.special import erfc

from qsopt.core.experimental_parameters import InitialStateType, InitialState


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

def generate_system_operators(
    n_cavities: int,\
    n_fields: int, \
    n_qubits: int, \
    cavity_levels: Union[int, List[int]], \
    field_levels: Union[int, List[int]], \
    qubit_levels: Union[int, List[int]]
) -> Dict[str, qt.Qobj]:
    """
    Generate operators for an n-qubit composite system.

    Creates operators for (field ⊗ cavity ⊗ qubit1 ⊗ ... ⊗ qubitn) composite space.
    Each qubit can have different level truncation for flexibility.

    Args:
        n_qubits: Number of qubits in the system
        field_levels: Number of Fock levels for input field mode
        cavity_levels: Number of Fock levels for resonator cavity mode
        qubit_levels: Number of levels for each qubit. Can be:
                     - int: Same levels for all qubits (typically 2)
                     - List[int]: Individual levels [qubit1_levels, qubit2_levels, ..., qubitn_levels]

    Returns:
        Dictionary containing all operators in composite space:
        - Field operators: a_in, a_in_dag (annihilation/creation for input field)
        - Cavity operators: a, a_dag (annihilation/creation for cavity)
        - Qubit operators: sigma_z, sigma_x, sigma_y, sigma_minus, sigma_plus (lists of embeddings)
        - Joint measurement projectors:
            * P_all0: Projector onto |00...0⟩ state
            * P_all: List of projectors for all 2^n computational basis states
        - Individual qubit projectors: P0_q (ground states), P1_q (excited states)
        - Measurement/reset operators: measure_reset, measure_reset_dag
        - Rotation operators:
            * roty_q: List of Y-rotation gates on individual qubits
            * roty: Simultaneous Y-rotation on all qubits
        - Identity operators: I_c, I_f, I_q (per-mode identity lists for composite space construction)
    """
    # Handle levels as list or int
    # Cavities
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

    # Fields
    if isinstance(field_levels, int):
        f_levels = [field_levels] * n_fields
    elif isinstance(field_levels, list):
        if len(field_levels) != n_fields:
            raise ValueError(
                f"field_levels list must have n_fields={n_fields} elements, got {len(field_levels)}"
            )
        f_levels = field_levels
    else:
        raise TypeError(f"field_levels must be int or list, got {type(field_levels)}")
        
    # Qubits
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

        # Identity operators for each subsystem
        I_c = [qt.identity(level) for level in c_levels]
        I_f = [qt.identity(level) for level in f_levels]
        I_q = [qt.identity(q_levels[i]) for i in range(n_qubits)]

        # Individual subsystem operators
        a_c = [qt.destroy(level) for level in c_levels]  # Cavity annihilation
        a_f = [qt.destroy(level) for level in f_levels]  # Field annihilation
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

        # Projectors for all 2^n qubit states (joint projectors)
        Ptemp = [P0,P1]
        all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]            
        P_all = [qt.tensor(I_c + I_f + [Ptemp[q_state][qb] for qb,q_state in enumerate(list(map(int,state)))]) for state in all_states]
            
        # Reset operators, individual qubits and global reset
        reset_q = [qt.Qobj([[1]*l] + [[0]*l]*(l-1)) for l in q_levels]
        reset_all = qt.tensor(I_c + I_f + reset_q)
        measure_reset = [reset_all*p for p in P_all]
        measure_reset_dag = [x.dag() for x in measure_reset]

        # Helper functions to embed subsystem operators in composite space
        def embed_cavity_op(op, cavity_idx):
            """Embed operator acting on cavity cavity_idx into full composite space."""
            ops_list = I_c[:cavity_idx] + [op] + I_c[cavity_idx+1:] + I_f + I_q
            return qt.tensor(ops_list)
        
        def embed_field_op(op, field_idx):
            """Embed operator acting on field field_idx into full composite space."""
            ops_list = I_c + I_f[:field_idx] + [op] + I_f[field_idx+1:] + I_q
            return qt.tensor(ops_list)
        
        def embed_qubit_op(op, qubit_idx):
            """Embed operator acting on qubit qubit_idx into full composite space."""
            ops_list = I_c + I_f + I_q[:qubit_idx] + [op] + I_q[qubit_idx+1:]
            return qt.tensor(ops_list)

        # Embed operators in composite space (input_field ⊗ cavity ⊗ qubits)
        operators = {
            # Resonator cavity operators
            "a_c":[embed_cavity_op(a_c[i], i) for i in range(n_cavities)],
            "a_c_dag": [embed_cavity_op(a_c[i].dag(), i) for i in range(n_cavities)],
            # Input field operators
            "a_f": [embed_field_op(a_f[i], i) for i in range(n_fields)],
            "a_f_dag": [embed_field_op(a_f[i].dag(), i) for i in range(n_fields)],
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
            "P_all0": qt.tensor(I_c + I_f + P0),  # Joint projector onto |00...0⟩
            "P_all": P_all,  # List of all joint projectors for 2^n states
            # Reset operators
            "reset_q": reset_q, 
            "reset_all": reset_all, 
            "measure_reset": measure_reset,
            "measure_reset_dag": measure_reset_dag,
            # Rotation operators (Y-rotation by π/2, can be applied independently)
            "roty_q": [embed_qubit_op(rot_single, i) for i in range(n_qubits)],
            "roty": qt.tensor(I_c + I_f + [rot_single]*n_qubits),  # Simultaneous Ry on all qubits
            # Identity operators for reference
            "I_c": I_c,
            "I_f": I_f,
            "I_q": I_q,
            "identity": embed_qubit_op(I_q[0], 0)  # Identity of the whole system
        }

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
    initial_state: InitialState,
    cavity_levels: Union[int, List[int]],
    field_levels: Union[int, List[int]],
    qubit_levels: Union[int, List[int]],
    n_cavities: int = 1,
    n_fields: int = 1,
    n_qubits: int = 1,
) -> qt.Qobj:
    """
    Generate initial density matrix based on configuration and system type.

    Supports various possible initial state types for each field and cavity subsystems,
    (qubit subsystems are always initialized in ground state the circuit is used to prepare the qubits).
    Possible subsystem states:
    - VACUUM: ground state
    - SINGLE_PHOTON: One photon Fock state
    - COHERENT: Coherent state
    - THERMAL: Thermal state
    - CUSTOM: User-defined superposition

    Args:
        initial_state: InitialState object with dictionaries of subsystems states or custom initial density_matrix
        cavity_levels: Number of Fock levels for each resonator cavity
        field_levels: Number of Fock levels for each input field
        qubit_levels: Number of levels for each qubit (int or list)
        n_cavities: Number of cavities in the system
        n_fields: Number of input fields in the system
        n_qubits: Number of qubits in the system
    Returns:
        Initial density matrix in composite Hilbert space

    Raises:
        ValueError: If required parameters are missing or invalid

    Example:
        >>> from qsopt.core.experimental_parameters import InitialState, SubsystemState, InitialStateType
        >>> config = InitialState(
        ...     field_states={0: SubsystemState(state_type=InitialStateType.FOCK, parameters={"n": 1})}
        ... )
        >>> rho0 = generate_initial_state(
        ...     config, cavity_levels=2, field_levels=2, qubit_levels=2,
        ...     n_cavities=1, n_fields=1, n_qubits=1,
        ... )
    """

    cavity_states = initial_state.cavity_states
    field_states = initial_state.field_states
    density_matrix = initial_state.density_matrix
    if (cavity_states is None or field_states is None) and density_matrix is None:
        raise ValueError("Either cavity_states and field_states dictionaries or a custom density_matrix must be provided in initial_state.")

    if isinstance(cavity_levels, int):
        c_levels = [cavity_levels] * n_cavities
    else:
        c_levels = cavity_levels[:n_cavities]

    if isinstance(field_levels, int):
        f_levels = [field_levels] * n_fields
    else:
        f_levels = field_levels[:n_fields]

    if isinstance(qubit_levels, int):
        q_levels = [qubit_levels] * n_qubits
    else:
        q_levels = qubit_levels[:n_qubits]

    # Use JAX backend for compatibility
    with qt.CoreOptions(default_dtype="jax"):
        # Create ground state base for qubits (qubits always in ground state)
        qubits_ground = _create_qubit_ground_state(q_levels, n_qubits)

        if density_matrix is not None:
            # Validate dimensions of provided density matrix
            expected_dim = math.prod(c_levels + f_levels)
            if density_matrix.shape != (expected_dim, expected_dim):
                raise ValueError(f"Custom density matrix was provided, expected dimensions ({expected_dim}, {expected_dim}), but got {density_matrix.shape}.\n\
                                Please ensure the custom density matrix is defined for the correct subsystem {'{cavities} ⊗ {fields}'}, qubits are always initialized in ground state.")
            else:
                return qt.tensor(density_matrix, qubits_ground)

        cavity_keys = cavity_states.keys()
        field_keys = field_states.keys()

        state_matrix_list = []

        for i in range(n_cavities):
            if i in cavity_keys:
                state = cavity_states[i]
                if state.state_type == InitialStateType.THERMAL:
                    n_avg = state.parameters["n_avg"]
                    state_matrix = _create_thermal(c_levels[i], n_avg)
                elif state.state_type == InitialStateType.FOCK:
                    n = state.parameters["n"]
                    state_matrix = _create_fock(c_levels[i], n)
                elif state.state_type == InitialStateType.COHERENT:
                    alpha = state.parameters["alpha"]
                    state_matrix = _create_coherent(c_levels[i], alpha)
                elif state.state_type == InitialStateType.CUSTOM:
                    state_matrix = _create_custom_state(c_levels[i], state.parameters["amplitudes"])
                else:
                    state_matrix = _create_vacuum(c_levels[i])
                state_matrix_list.append(state_matrix)
            else:
                vacuum = _create_vacuum(c_levels[i])
                state_matrix_list.append(vacuum)
            
        for i in range(n_fields):
            if i in field_keys:
                state = field_states[i]
                if state.state_type == InitialStateType.THERMAL:
                    n_avg = state.parameters["n_avg"]
                    state_matrix = _create_thermal(f_levels[i], n_avg)
                elif state.state_type == InitialStateType.FOCK:
                    n = state.parameters["n"]
                    state_matrix = _create_fock(f_levels[i], n)
                elif state.state_type == InitialStateType.COHERENT:
                    alpha = state.parameters["alpha"]
                    state_matrix = _create_coherent(f_levels[i], alpha)
                elif state.state_type == InitialStateType.CUSTOM:
                    state_matrix = _create_custom_state(f_levels[i], state.parameters["amplitudes"])
                else:
                    state_matrix = _create_vacuum(f_levels[i])
                state_matrix_list.append(state_matrix)
            else:
                vacuum = _create_vacuum(f_levels[i])
                state_matrix_list.append(vacuum)

        # Combine subsystem states with the qubit ground state base.
        # Ordering matches generate_system_operators: cavities ⊗ fields ⊗ qubits.
        return qt.tensor(*state_matrix_list, qubits_ground)


# ==================== Private Helper Functions ====================


def _create_qubit_ground_state(qubit_levels: Union[int, List[int]], n_qubits: int
) -> qt.Qobj:
    """
    Create ground state for qubits: |0⟩_q1 ⊗ |0⟩_q2 ⊗ ... ⊗ |0⟩_qn

    The qubits are always initialized in ground state and are 
    prepared by the circuit during the simulation.

    Args:
        qubit_levels: Number of levels for each qubit (int or list)
        n_qubits: Number of qubits

    Returns:
        Ground state density matrix for qubits subsystem
    """
    # Extract qubit levels for each qubit
    if isinstance(qubit_levels, int):
        q_levels = [qubit_levels] * n_qubits
    else:
        q_levels = qubit_levels[:n_qubits]

    # Build ground state for all qubits
    qubit_grounds = [qt.basis(q_levels[i], 0) for i in range(n_qubits)]

    # Create state vector: qubit1 ⊗ qubit2 ⊗ ... ⊗ qubitn
    psi = qt.tensor(*qubit_grounds)
    return psi * psi.dag()  # type: ignore


def _create_vacuum(levels: int) -> qt.Qobj:
    """Create vacuum state: |0⟩."""
    state = qt.basis(levels, 0)
    return state * state.dag()  # type: ignore


def _create_fock(levels: int, n: int) -> qt.Qobj:
    """Create fock state with n photons: |n⟩."""
    state = qt.basis(levels, n)
    return state * state.dag()  # type: ignore


def _create_coherent(levels: int, alpha: complex) -> qt.Qobj:
    """Create coherent state: |α⟩."""
    coherent_state = qt.coherent(levels, alpha)
    return coherent_state * coherent_state.dag()  # type: ignore


def _create_thermal(levels: int, n_bar: float) -> qt.Qobj:
    """Create thermal state with average photon number n_bar."""
    return qt.thermal_dm(levels, n_bar)


def _create_custom_state(levels: int, amplitudes: Union[List[complex], np.ndarray]) -> qt.Qobj:
    """
    Create a pure single-mode custom state from Fock-basis amplitudes: |ψ⟩ = Σ aₙ|n⟩.

    Args:
        levels: Fock truncation of the mode.
        amplitudes: 1D sequence of complex amplitudes (length <= levels). High Fock
            components beyond ``len(amplitudes)`` are padded with zeros and the state
            is renormalized.

    Returns:
        Density matrix |ψ⟩⟨ψ| for the single mode.
    """
    amp = np.asarray(amplitudes, dtype=complex).reshape(-1)
    if amp.size > levels:
        raise ValueError(
            f"Custom state has {amp.size} amplitudes but the mode only has {levels} levels"
        )
    vec = np.zeros(levels, dtype=complex)
    vec[: amp.size] = amp
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Custom state has zero norm (<1e-12)")
    psi = qt.Qobj(vec / norm)
    return psi * psi.dag()  # type: ignore


def apply_single_qubit_rotation(
    rho: qt.Qobj, theta: float, axis: str, I_field: qt.Qobj, I_cavity: qt.Qobj
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
        if axis.lower() == "x":
            pauli = qt.sigmax()
        elif axis.lower() == "y":
            pauli = qt.sigmay()
        elif axis.lower() == "z":
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
    outcome: int, field_levels: int, cavity_levels: int, qubit_levels: int
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
    rho: qt.Qobj, outcome: int, field_levels: int, cavity_levels: int, qubit_levels: int
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
    rho: qt.Qobj, outcome: int, field_levels: int, cavity_levels: int, qubit_levels: int
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


def apply_qubit_rotation(
    rho: qt.Qobj, theta: float, qubit_index: int, operators: Dict[str, qt.Qobj], axis: str = "y"
) -> qt.Qobj:
    """
    Apply a rotation to a specific qubit in a multi-qubit composite system.

    Generic function for rotating individual qubits in the composite Hilbert space
    (cavity ⊗ field ⊗ qubits). Works for any number of cavities, fields and qubits.

    Args:
        rho: Density matrix in composite space
        theta: Rotation angle in radians
        qubit_index: Index of qubit to rotate (0-based)
        operators: Dictionary of operators from generate_system_operators()
        axis: Rotation axis ('x', 'y', or 'z')

    Returns:
        Rotated density matrix

    Example:
        >>> # For a two-qubit system (1 cavity, 1 field)
        >>> ops = generate_system_operators(1, 1, 2, 2, 2, 2)
        >>> rho_rotated = apply_qubit_rotation(rho, np.pi/4, qubit_index=0, operators=ops, axis='y')
        >>> # Rotates the first qubit by π/4 around the Y-axis
    """
    # Per-mode identities ordered as in generate_system_operators: cavity ⊗ field ⊗ qubits
    I_c = operators["I_c"]
    I_f = operators["I_f"]
    I_q = operators["I_q"]
    n_qubits = len(I_q)

    if not (0 <= qubit_index < n_qubits):
        raise ValueError(f"qubit_index must be in [0, {n_qubits}), got {qubit_index}")

    # Build single-qubit rotation matrix
    with qt.CoreOptions(default_dtype="jax"):
        if axis.lower() == "x":
            pauli = qt.sigmax()
        elif axis.lower() == "y":
            pauli = qt.sigmay()
        elif axis.lower() == "z":
            pauli = qt.sigmaz()
        else:
            raise ValueError(f"Invalid rotation axis: {axis}. Must be 'x', 'y', or 'z'.")

        rotation_single = (-1j * pauli * theta / 2).expm()

        # Replace the identity acting on the target qubit with the rotation,
        # then embed in the full composite space.
        qubit_ops = list(I_q)
        qubit_ops[qubit_index] = rotation_single
        rotation_gate = qt.tensor(list(I_c) + list(I_f) + qubit_ops)

    return rotation_gate * rho * rotation_gate.dag()  # type: ignore


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
            # For other states, construct the joint projector from the cached
            # per-mode identities (cavity ⊗ field ⊗ qubits ordering).
            I_c = operators["I_c"]
            I_f = operators["I_f"]
            if q_levels is None:
                q_levels = [2]*n_qubits
            elif isinstance(q_levels, int):
                q_levels = [q_levels] * n_qubits

            # Build projector for each qubit based on state string
            qubit_projectors = []
            for i, s in enumerate(state):
                l = q_levels[i]
                if s == '0':
                    qubit_projectors.append(qt.Qobj([[1, 0] + [0]*(l-2)] + [[0]*l]*(l-1)))
                elif s == '1':
                    qubit_projectors.append(qt.Qobj([[0]*l] + [[0, 1] + [0]*(l-2)] + [[0]*l]*(l-2)))
                else:
                    raise ValueError(f"Invalid state character '{s}', must be '0' or '1'")

            P = qt.tensor(list(I_c) + list(I_f) + qubit_projectors)

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
        if len(state) != len(qubit_indices):
            raise ValueError(f"Lenght of state string ({len(state)}) must match lenght of qubit_indices ({len(qubit_indices)})")
        I_c = operators["I_c"]
        I_f = operators["I_f"]
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
        P = qt.tensor(list(I_c) + list(I_f) + qubit_projector)

    else:
        raise ValueError(f'qubit indices must either be:\n \
        - int or list of int with lenght 1 ->   single qubit measurement in the interval [0,{n_qubits})\n\
        - a string = "all" ->                   joint measurement on all qubits\n \
        - a list of lenght between [2,{n_qubits}] ->     non cached measurement of non fixed number of qubits')

    probability = jnp.real((P * rho * P.dag()).tr())  # type: ignore
    return float(probability)


def embed_circuit_unitary(
    circuit_unitary: jnp.ndarray,
    field_levels: int,
    cavity_levels: int
) -> jnp.ndarray:
    """
    Embed an n-qubit circuit unitary into the full composite Hilbert space using JAX.

    The composite space is: input_field ⊗ resonator_cavity ⊗ qubit1 ⊗ qubit2 ⊗ ... ⊗ qubitn
    The circuit acts only on the qubit subspace (qubit1 ⊗ qubit2 ⊗ ... ⊗ qubitn).

    Args:
        circuit_unitary: (∏ qubit_levels)×(∏ qubit_levels) unitary matrix for n-qubit circuit (JAX array)
        field_levels: Number of levels in the input field subsystem
        cavity_levels: Number of levels in the resonator cavity subsystem

    Returns:
        Full-space unitary as JAX array: I_field ⊗ I_cavity ⊗ circuit_unitary

    Example:
        >>> # 2-qubit circuit unitary (4x4 for 2-level qubits)
        >>> U_circuit = jnp.eye(4, dtype=jnp.complex128)
        >>> U_full = embed_circuit_unitary(U_circuit, field_levels=2, cavity_levels=3)
        >>> # U_full shape: (2*3*4)×(2*3*4) = 24×24
    """
    # Build full operator using JAX Kronecker products
    # I_field ⊗ I_cavity ⊗ U_circuit
    I_field = jnp.eye(field_levels, dtype=jnp.complex128)
    I_cavity = jnp.eye(cavity_levels, dtype=jnp.complex128)

    # Kronecker product: I_field ⊗ I_cavity ⊗ U_circuit
    U_full_jax = jnp.kron(jnp.kron(I_field, I_cavity), circuit_unitary)

    return U_full_jax


def embed_operator(
    operator: qt.Qobj,
    positions: List[int],
    identities: List[qt.Qobj],
) -> qt.Qobj:
    """
    Embed an operator acting on one or two subsystems into the full composite space.

    The composite Hilbert space is described by ``identities``: one identity operator
    per subsystem, given in the canonical tensor order (cavity ⊗ field ⊗ qubits, as
    produced by :func:`generate_system_operators`). The ``operator`` acts jointly on
    the subsystems located at ``positions``; its tensor legs are assumed to be in the
    same order as ``positions`` (which must be ascending, matching the canonical order).

    The operator is tensored with identities on every other subsystem and then permuted
    back into the canonical order, so the returned qutip ``Qobj`` keeps the correct
    per-subsystem ``dims`` (and therefore composes cleanly with the system operators
    and states).

    Args:
        operator: Operator acting on the subsystems at ``positions``. Only its overall
            matrix size must equal ``prod(d_p for p in positions)``; the correct
            per-subsystem ``dims`` are (re)assigned here from ``positions`` and
            ``identities``. If the operator is supplied with flat or otherwise incorrect
            subsystem ``dims`` metadata (but the right size) the dims are reassigned and
            a warning is emitted.
        positions: Composite indices (0-based, ascending) of the subsystems the operator
            acts on. Length 1 (single subsystem) or 2 (two subsystems). The operator's
            tensor legs are assumed to follow this order.
        identities: Identity operators for every subsystem in canonical order
            (e.g. ``operators["I_c"] + operators["I_f"] + operators["I_q"]``).

    Returns:
        The operator embedded in the full composite Hilbert space.

    Example:
        >>> ops = generate_system_operators(1, 1, 2, 2, 2, 2)
        >>> identities = list(ops["I_c"]) + list(ops["I_f"]) + list(ops["I_q"])
        >>> # Embed a custom cavity(0)-qubit(0) operator: composite positions 0 and 2
        >>> H_full = embed_operator(custom_matrix, [0, 2], identities)
    """
    n_subsystems = len(identities)

    if any(not (0 <= p < n_subsystems) for p in positions):
        raise ValueError(f"positions {positions} out of range [0, {n_subsystems})")
    if len(set(positions)) != len(positions):
        raise ValueError(f"positions must be distinct, got {positions}")

    # Expected per-subsystem leg dimensions, taken from the target subsystems. We only
    # require the overall matrix size to match; the correct per-subsystem dims are
    # (re)assigned below so an operator carrying flat/absent/incorrect subsystem dims is
    # still embedded correctly.
    expected_legs = [identities[p].dims[0][0] for p in positions]
    expected_dims = [expected_legs, list(expected_legs)]
    expected_size = int(np.prod(expected_legs))

    mat = np.asarray(operator.full())
    if mat.shape != (expected_size, expected_size):
        raise ValueError(
            f"operator has matrix size {mat.shape} but the subsystems at positions {positions} "
            f"require a ({expected_size}, {expected_size}) operator (per-subsystem dims {expected_legs})"
        )

    # Warn if the supplied dims metadata had to be reinterpreted.
    given_dims = [list(operator.dims[0]), list(operator.dims[1])]
    if given_dims != expected_dims:
        warnings.warn(
            f"embed_operator: operator dims {given_dims} do not match the expected per-subsystem "
            f"dims {expected_dims} for positions {positions}. The matrix size is correct, so the "
            f"dims are being reassigned to {expected_dims} (legs assumed in the order of positions).",
            UserWarning,
        )

    others = [p for p in range(n_subsystems) if p not in positions]

    with qt.CoreOptions(default_dtype="jax"):
        # Attach the correct per-subsystem dims as metadata, under the JAX backend so the
        # operator tensors cleanly with the JAX-backed identities.
        op = qt.Qobj(mat, dims=[expected_legs, list(expected_legs)])
        combined = qt.tensor([op] + [identities[p] for p in others])

        # `combined`'s tensor factors are ordered as (positions + others); permute so
        # that factor k of the result is the subsystem at composite position k.
        source_order = list(positions) + others
        permutation = [source_order.index(k) for k in range(n_subsystems)]
        return combined.permute(permutation)

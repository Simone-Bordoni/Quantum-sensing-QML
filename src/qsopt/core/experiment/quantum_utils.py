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


def generate_single_qubit_operators(
    field_levels: int, cavity_levels: int, qubit_levels: int
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
        a_field = qt.destroy(field_levels)  # Field annihilation
        a_cavity = qt.destroy(cavity_levels)  # Cavity annihilation

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
            "a_in": qt.tensor(a_field, I_cavity, I_qubit),
            "a_in_dag": qt.tensor(a_field.dag(), I_cavity, I_qubit),
            # Resonator cavity operators
            "a": qt.tensor(I_field, a_cavity, I_qubit),
            "a_dag": qt.tensor(I_field, a_cavity.dag(), I_qubit),
            # Qubit operators
            "sigma_z": qt.tensor(I_field, I_cavity, sigma_z),
            "sigma_x": qt.tensor(I_field, I_cavity, sigma_x),
            "sigma_y": qt.tensor(I_field, I_cavity, sigma_y),
            "sigma_minus": qt.tensor(I_field, I_cavity, sigma_minus),
            "sigma_plus": qt.tensor(I_field, I_cavity, sigma_minus.dag()),
            # Qubit measurement projectors
            "P0": qt.tensor(I_field, I_cavity, P0),
            "P1": qt.tensor(I_field, I_cavity, P1),
            # Identity operators for reference
            "I_field": I_field,
            "I_cavity": I_cavity,
            "I_qubit": I_qubit,
        }

        return operators


def generate_two_qubit_operators(
    field_levels: int, cavity_levels: int, qubit_levels: Union[int, List[int]]
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
            raise ValueError(
                f"qubit_levels list must have at least 2 elements, got {len(qubit_levels)}"
            )
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
        a_field = qt.destroy(field_levels)  # Field annihilation
        a_cavity = qt.destroy(cavity_levels)  # Cavity annihilation

        # Qubit 1 operators
        sigma_z1 = qt.sigmaz() if q1_levels == 2 else qt.jmat(q1_levels - 1, "z")
        sigma_x1 = qt.sigmax() if q1_levels == 2 else qt.jmat(q1_levels - 1, "x")
        sigma_y1 = qt.sigmay() if q1_levels == 2 else qt.jmat(q1_levels - 1, "y")
        sigma_minus1 = qt.destroy(q1_levels)

        # Qubit 2 operators
        sigma_z2 = qt.sigmaz() if q2_levels == 2 else qt.jmat(q2_levels - 1, "z")
        sigma_x2 = qt.sigmax() if q2_levels == 2 else qt.jmat(q2_levels - 1, "x")
        sigma_y2 = qt.sigmay() if q2_levels == 2 else qt.jmat(q2_levels - 1, "y")
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
            "a_in": qt.tensor(a_field, I_cavity, I_q1, I_q2),
            "a_in_dag": qt.tensor(a_field.dag(), I_cavity, I_q1, I_q2),
            # Resonator cavity operators
            "a": qt.tensor(I_field, a_cavity, I_q1, I_q2),
            "a_dag": qt.tensor(I_field, a_cavity.dag(), I_q1, I_q2),
            # Qubit 1 operators
            "sigma_z1": qt.tensor(I_field, I_cavity, sigma_z1, I_q2),
            "sigma_x1": qt.tensor(I_field, I_cavity, sigma_x1, I_q2),
            "sigma_y1": qt.tensor(I_field, I_cavity, sigma_y1, I_q2),
            "sigma_minus1": qt.tensor(I_field, I_cavity, sigma_minus1, I_q2),
            "sigma_plus1": qt.tensor(I_field, I_cavity, sigma_minus1.dag(), I_q2),
            # Qubit 2 operators
            "sigma_z2": qt.tensor(I_field, I_cavity, I_q1, sigma_z2),
            "sigma_x2": qt.tensor(I_field, I_cavity, I_q1, sigma_x2),
            "sigma_y2": qt.tensor(I_field, I_cavity, I_q1, sigma_y2),
            "sigma_minus2": qt.tensor(I_field, I_cavity, I_q1, sigma_minus2),
            "sigma_plus2": qt.tensor(I_field, I_cavity, I_q1, sigma_minus2.dag()),
            # Joint measurement projectors (for 2-level qubits)
            "P00": qt.tensor(I_field, I_cavity, P0, P0),  # |00⟩⟨00|
            "P01": qt.tensor(I_field, I_cavity, P0, P1),  # |01⟩⟨01|
            "P10": qt.tensor(I_field, I_cavity, P1, P0),  # |10⟩⟨10|
            "P11": qt.tensor(I_field, I_cavity, P1, P1),  # |11⟩⟨11|
            # Individual qubit projectors
            "P0_q1": qt.tensor(I_field, I_cavity, P0, I_q2),  # |0⟩⟨0| ⊗ I for qubit1
            "P1_q1": qt.tensor(I_field, I_cavity, P1, I_q2),  # |1⟩⟨1| ⊗ I for qubit1
            "P0_q2": qt.tensor(I_field, I_cavity, I_q1, P0),  # I ⊗ |0⟩⟨0| for qubit2
            "P1_q2": qt.tensor(I_field, I_cavity, I_q1, P1),  # I ⊗ |1⟩⟨1| for qubit2
            # Rotation operators (Y-rotation by π/2, can be applied independently)
            "roty_q1": qt.tensor(I_field, I_cavity, rot_single, I_q2),  # Ry on qubit1 only
            "roty_q2": qt.tensor(I_field, I_cavity, I_q1, rot_single),  # Ry on qubit2 only
            "roty": qt.tensor(I_field, I_cavity, rot_single, rot_single),  # Simultaneous Ry on both
            # Identity operators for reference
            "I_field": I_field,
            "I_cavity": I_cavity,
            "I_q1": I_q1,
            "I_q2": I_q2,
        }

        return operators

def generate_n_qubit_operators(
    field_levels: int, cavity_levels: int, qubit_levels: Union[int, List[int]], n_qubits: int
) -> Dict[str, qt.Qobj]:
    """
    Generate operators for an n-qubit composite system.

    Creates operators for (field ⊗ cavity ⊗ qubit1 ⊗ ... ⊗ qubitn) composite space.
    Each qubit can have different level truncation for flexibility.

    Args:
        field_levels: Number of Fock levels for input field mode
        cavity_levels: Number of Fock levels for resonator cavity mode
        qubit_levels: Number of levels for each qubit. Can be:
                     - int: Same levels for all qubits (typically 2)
                     - List[int]: Individual levels [qubit1_levels, qubit2_levels, ..., qubitn_levels]

    Returns:
        Dictionary containing all operators in composite space:
        - Field operators: a_in, a_in_dag
        - Cavity operators: a, a_dag
        - Lists of qubits operators: sigma_z, sigma_x, sigma_y, sigma_minus, sigma_plus
        - Dictionary of all 2^n measurement projectors: P (matrix of joint qubit states)
        - Lists of individual qubit projectors: P0_q, P1_q
        - Rotation operators:
            * roty_q: list of Y-rotations on individual qubits
            * roty: Simultaneous Y-rotation on all qubits

    Example:
        >>> ops = generate_n_qubit_operators(2, 2, 2, 3)
        >>> # Access operators for each qubit
        >>> sz0 = ops['sigma_z'][0]
        >>> sz1 = ops['sigma_z'][1]
        >>> sz2 = ops['sigma_z'][2]
        >>> # Joint measurement projector |000⟩⟨000|
        >>> P000 = ops['P']['000']
    """
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
        # Identity operators for each subsystem
        I_field = qt.identity(field_levels)
        I_cavity = qt.identity(cavity_levels)

        # Individual subsystem operators
        a_field = qt.destroy(field_levels)  # Field annihilation
        a_cavity = qt.destroy(cavity_levels)  # Cavity annihilation

        # Lists of qubits operators
        I_q = [qt.identity(q_levels[i]) for i in range(n_qubits)]
        sigma_z = [qt.sigmaz() if q_levels[i] == 2 else qt.jmat(q_levels[i] - 1, "z") for i in range(n_qubits)]
        sigma_x = [qt.sigmax() if q_levels[i] == 2 else qt.jmat(q_levels[i] - 1, "x") for i in range(n_qubits)]
        sigma_y = [qt.sigmay() if q_levels[i] == 2 else qt.jmat(q_levels[i] - 1, "y") for i in range(n_qubits)]
        sigma_minus = [qt.destroy(q_levels[i]) for i in range(n_qubits)]

        # Measurement projectors for 2-level qubits
        P0 = qt.Qobj([[1, 0], [0, 0]])  # Ground state |0⟩⟨0|
        P1 = qt.Qobj([[0, 0], [0, 1]])  # Excited state |1⟩⟨1|
        
        Pbin[P0,P1]

        # Rotation operator (Y-rotation by π/2)
        # Ry(π/2) = [[cos(π/4), -sin(π/4)], [sin(π/4), cos(π/4)]]
        # = (1/√2)[[1, -1], [1, 1]]
        rot_single = qt.Qobj([[1, -1], [1, 1]]) / jnp.sqrt(2.0)


        # Embed operators in composite space (input_field ⊗ cavity ⊗ qubit1 ⊗ qubit2)
        operators = {
            # Input field operators
            "a_in": qt.tensor([a_field, I_cavity]+I_q),
            "a_in_dag": qt.tensor([a_field.dag(), I_cavity]+I_q),
            # Resonator cavity operators
            "a": qt.tensor([I_field, a_cavity]+I_q),
            "a_dag": qt.tensor([I_field, a_cavity.dag()]+I_q),
            # Lists of qubits operators
            "sigma_z": [qt.tensor([I_field, I_cavity] + I_q[:i] + sigma_z[i] + I_q[i+1:]) for i in range(n_qubits)],
            "sigma_x": [qt.tensor([I_field, I_cavity] + I_q[:i] + sigma_x[i] + I_q[i+1:]) for i in range(n_qubits)],
            "sigma_y": [qt.tensor([I_field, I_cavity] + I_q[:i] + sigma_y[i] + I_q[i+1:]) for i in range(n_qubits)],
            "sigma_minus": [qt.tensor([I_field, I_cavity] + I_q[:i] + sigma_minus[i] + I_q[i+1:]) for i in range(n_qubits)],
            "sigma_plus": [qt.tensor([I_field, I_cavity] + I_q[:i] + sigma_minus[i].dag() + I_q[i+1:]) for i in range(n_qubits)],
            # Dictionary of all 2^n measurement projectors for 2 level qubits: P
            "P": {bin(i)[2:]: qt.tensor([I_field, I_cavity] + [Pbin[int(bin(i)[2+j])] for j in range(n)]) for i in range(2^n) },
            # Individual qubit projectors
            "P0_q": [qt.tensor([I_field, I_cavity] + I_q[:i] +[P0] + I_q[i+1:]) for i in range(n_qubits)],  # |0⟩⟨0| ⊗ I for individual qubits
            "P1_q": [qt.tensor([I_field, I_cavity] + I_q[:i] +[P1] + I_q[i+1:]) for i in range(n_qubits)],  # |1⟩⟨1| ⊗ I for individual qubits
            # Rotation operators (Y-rotation by π/2, can be applied independently)
            "roty_q": [qt.tensor([I_field, I_cavity] + I_q[:i] + rot_single + I_q[i+1:]) for i in range(n)],  # Ry on only one qubit
            "roty": qt.tensor([I_field, I_cavity] + [rot_single]*n ),  # Simultaneous Ry on all qubits
            # Identity operators for reference
            "I_field": I_field,
            "I_cavity": I_cavity,
            "I_q": I_q
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

    For multi-qubit systems, qubits are initialized in equal superposition (|0⟩+|1⟩)/√2
    unless otherwise specified.

    Args:
        initial_config: InitialStateConfig object with state type and parameters
        field_levels: Number of Fock levels for input field
        cavity_levels: Number of Fock levels for resonator cavity
        qubit_levels: Number of levels for each qubit (int or list)
        n_qubits: Number of qubits in the system (1 or 2)

    Returns:
        Initial density matrix in composite Hilbert space

    Raises:
        ValueError: If required parameters are missing or invalid
        NotImplementedError: If n_qubits > 2

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
    custom_amplitudes: Dict[Tuple[int, int, int], complex],
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
    if n_qubits != 1:
        raise NotImplementedError("Custom states only supported for single qubit systems")

    # Initialize zero state vector
    total_dim = field_levels * cavity_levels * q_levels[0]
    psi_array = np.zeros((total_dim,), dtype=complex)

    # Fill in amplitudes from dictionary
    # Dictionary keys are tuples (field_n, cavity_n, qubit_n)
    for (n_field, n_cavity, n_qubit), amplitude in custom_amplitudes.items():
        # Validate indices
        if not (0 <= n_field < field_levels):
            raise ValueError(f"Field index {n_field} out of range [0, {field_levels})")
        if not (0 <= n_cavity < cavity_levels):
            raise ValueError(f"Cavity index {n_cavity} out of range [0, {cavity_levels})")
        if not (0 <= n_qubit < q_levels[0]):
            raise ValueError(f"Qubit index {n_qubit} out of range [0, {q_levels[0]})")

        # Compute flat index: field ⊗ cavity ⊗ qubit ordering
        idx = n_field * (cavity_levels * q_levels[0]) + n_cavity * q_levels[0] + n_qubit
        psi_array[idx] = amplitude

    # Normalize the state
    norm = np.linalg.norm(psi_array)
    if norm < 1e-10:
        raise ValueError("Custom state has zero norm")
    psi_array = psi_array / norm

    # Create QuTiP state
    psi = qt.Qobj(psi_array, dims=[[field_levels, cavity_levels, q_levels[0]], [1, 1, 1]])
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
    Apply rotation to a specific qubit in multi-qubit system.

    Generic function for rotating individual qubits in composite Hilbert space.
    Works for both single and multi-qubit systems.

    Args:
        rho: Density matrix in composite space
        theta: Rotation angle in radians
        qubit_index: Index of qubit to rotate (0-based)
        operators: Dictionary of operators from generate_*_operators()
        axis: Rotation axis ('x', 'y', or 'z')

    Returns:
        Rotated density matrix

    Example:
        >>> # For two-qubit system
        >>> ops = generate_two_qubit_operators(2, 2, [2, 2])
        >>> rho_rotated = apply_qubit_rotation(rho, np.pi/4, qubit_index=0, operators=ops, axis='y')
        >>> # Rotates first qubit by π/4 around Y-axis
    """
    # Get identity operators
    I_field = operators["I_field"]
    I_cavity = operators["I_cavity"]

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

    # Determine number of qubits from operators
    # Check if we have multi-qubit operators (e.g., 'I_q1', 'I_q2')
    if "I_q1" in operators:
        # Multi-qubit system
        n_qubits = 0
        i = 1
        while f"I_q{i}" in operators:
            n_qubits += 1
            i += 1

        # Build list of identity operators for each qubit
        qubit_identities = [operators[f"I_q{i+1}"] for i in range(n_qubits)]

        # Replace identity at qubit_index with rotation
        qubit_ops = qubit_identities.copy()
        qubit_ops[qubit_index] = rotation_single

        # Tensor product: field ⊗ cavity ⊗ q1 ⊗ q2 ⊗ ...
        rotation_gate = qt.tensor(I_field, I_cavity, *qubit_ops)
    else:
        # Single qubit system
        if qubit_index != 0:
            raise ValueError(f"qubit_index must be 0 for single-qubit system, got {qubit_index}")

        I_qubit = operators["I_qubit"]
        # For single qubit, just embed rotation
        rotation_gate = qt.tensor(I_field, I_cavity, rotation_single)

    return rotation_gate * rho * rotation_gate.dag()  # type: ignore


def measure_qubits_probability(
    rho: qt.Qobj, qubit_indices: List[int], operators: Dict[str, qt.Qobj], state: str = "0"
) -> float:
    """
    Measure probability for specific qubits in multi-qubit system.

    Generic measurement function supporting:
    - Single qubit measurement: qubit_indices=[0] measures qubit 0
    - Multi-qubit measurement: qubit_indices=[0,1] measures both qubits

    Args:
        rho: Density matrix in composite space
        qubit_indices: List of qubit indices to measure (0-based)
        operators: Dictionary of operators from generate_*_operators()
        state: State string to project onto
               - For single qubit: '0' or '1'
               - For multiple qubits: '00', '01', '10', '11', etc.

    Returns:
        Measurement probability ∈ [0,1]

    Example:
        >>> # Measure qubit 0 only
        >>> p0 = measure_qubits_probability(rho, [0], ops, state='0')
        >>>
        >>> # Measure both qubits jointly
        >>> p00 = measure_qubits_probability(rho, [0, 1], ops, state='00')
    """
    import jax.numpy as jnp

    # Determine if single or multi-qubit system
    if "I_q1" in operators:
        # Multi-qubit system
        if len(qubit_indices) == 1:
            # Single qubit measurement
            qubit_idx = qubit_indices[0]
            projector_key = f"P{state}_q{qubit_idx + 1}"
            if projector_key not in operators:
                raise ValueError(f"Projector {projector_key} not found in operators")
            P = operators[projector_key]
        else:
            # Joint measurement
            if len(state) != len(qubit_indices):
                raise ValueError(
                    f"State string length ({len(state)}) must match number of qubits ({len(qubit_indices)})"
                )
            projector_key = f"P{state}"
            if projector_key not in operators:
                raise ValueError(f"Joint projector {projector_key} not found in operators")
            P = operators[projector_key]
    else:
        # Single qubit system
        if len(qubit_indices) != 1 or qubit_indices[0] != 0:
            raise ValueError("Single-qubit system only supports qubit_indices=[0]")
        P = operators[f"P{state}"]

    probability = jnp.real((P * rho * P.dag()).tr())  # type: ignore
    return float(probability)

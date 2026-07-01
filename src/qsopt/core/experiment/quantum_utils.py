"""
Quantum System Utilities
=========================

Utility functions for quantum system setup, operator generation, and initial state preparation.
Designed to support both single and multi-qubit quantum sensing experiments.

This module provides reusable components that can be composed for different experiment types.
"""

from typing import Any, Callable, Dict, List, NamedTuple, Union
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt

from qsopt.core.experimental_parameters import (
    Interaction,
    InteractionType,
    State,
    SystemConfiguration,
)

# Types one parameter can be factored out of into an args-coefficient, so a sweep varies it
# via args without rebuilding (kappa enters as √kappa via the per-contribution transform).
PROMOTABLE_TYPES = frozenset({
    InteractionType.DISPERSIVE, InteractionType.DETUNING, InteractionType.COUPLING,
    InteractionType.XX, InteractionType.YY, InteractionType.ZZ,
    InteractionType.DISSIPATION, InteractionType.INPUT_OUTPUT,
})

_ID = lambda x: x  # noqa: E731  (linear parameter -> identity transform)


class InteractionTerm(NamedTuple):
    """One additive term of an interaction's Hamiltonian or Lindblad operator.

    Value of the term is ``const * Π transform(value[name]) * operator * (g(t) if modulated)``.

    Attributes:
        kind (str): 'H' for a Hamiltonian term, 'L' for a Lindblad collapse-operator term.
        operator (qt.Qobj | Callable): bare operator matrix, or a callable ``values -> Qobj`` for a
            structural (non-multiplicative) parameter that reshapes the matrix.
        const (complex): constant numeric prefactor.
        params (dict[str, Callable]): multiplicative parameter name -> transform f (identity, sqrt, ...).
        modulated (bool): whether the time-modulation pulse g(t) multiplies this term.
    """
    kind: str
    operator: Union[qt.Qobj, Callable]
    const: complex
    params: Dict[str, Callable]
    modulated: bool

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
            
        # Reset operators, individual qubits and global reset.
        # The l×l matrix [[1,...,1],[0,...,0],...] equals Σ_k |0⟩⟨k|: it maps every
        # Fock level to the ground state, so L ρ L† unconditionally resets the qubit to |0⟩.
        reset_q = [qt.Qobj([[1]*l] + [[0]*l]*(l-1)) for l in q_levels]
        reset_all = qt.tensor(I_c + I_f + reset_q)
        # measure_reset[k] = reset_all * P_all[k]: Kraus operator that first projects
        # the register onto computational basis state k, then resets all qubits to |0⟩.
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
            # Pre-zipped (op, op_dag) pairs, materialised as a list so they can be
            # iterated once per measurement without exhausting a single-use zip iterator.
            "measure_reset_pairs": list(zip(measure_reset, measure_reset_dag)),
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
    system_configuration: SystemConfiguration,
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
        system_configuration: SystemConfiguration with dictionaries of subsystem states
            (init_cavity_states, init_field_states) or a custom initial density_matrix
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
        >>> from qsopt.core.experimental_parameters import SystemConfiguration, SubsystemState, State
        >>> config = SystemConfiguration(
        ...     name="example",
        ...     init_field_states={0: SubsystemState(state_type=State.FOCK, parameters={"n": 1})},
        ... )
        >>> rho0 = generate_initial_state(
        ...     config, cavity_levels=2, field_levels=2, qubit_levels=2,
        ...     n_cavities=1, n_fields=1, n_qubits=1,
        ... )
    """

    cavity_states = system_configuration.init_cavity_states
    field_states = system_configuration.init_field_states
    density_matrix = system_configuration.density_matrix
    if (cavity_states is None or field_states is None) and density_matrix is None:
        raise ValueError("Either init_cavity_states and init_field_states dictionaries or a custom density_matrix must be provided in the system configuration.")

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
                if state.state_type == State.THERMAL:
                    n_avg = state.parameters["n_avg"]
                    state_matrix = _create_thermal(c_levels[i], n_avg)
                elif state.state_type == State.FOCK:
                    n = state.parameters["n"]
                    state_matrix = _create_fock(c_levels[i], n)
                elif state.state_type == State.COHERENT:
                    alpha = state.parameters["alpha"]
                    state_matrix = _create_coherent(c_levels[i], alpha)
                elif state.state_type == State.CUSTOM:
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
                if state.state_type == State.THERMAL:
                    n_avg = state.parameters["n_avg"]
                    state_matrix = _create_thermal(f_levels[i], n_avg)
                elif state.state_type == State.FOCK:
                    n = state.parameters["n"]
                    state_matrix = _create_fock(f_levels[i], n)
                elif state.state_type == State.COHERENT:
                    alpha = state.parameters["alpha"]
                    state_matrix = _create_coherent(f_levels[i], alpha)
                elif state.state_type == State.CUSTOM:
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


def build_hamiltonians(operators, experimental_params, n_cavities, n_fields, n_qubits,
                       overrides=None, dynamic_keys=None):
    """Build the per-configuration Hamiltonians and Lindblad operators (pure).

    Each interaction is decomposed into :class:`InteractionTerm` terms and assembled by
    ``_interaction_terms``: parameters are baked into the operators, or, for keys in
    ``dynamic_keys``, factored into an args-coefficient so a sweep varies them on a prebuilt
    solver. Qubit-noise Lindblads are added per configuration. Pure, so safe inside
    ``jax.vmap``/``jax.jit``.

    Args:
        operators (dict): system operators from ``generate_system_operators``.
        experimental_params (ExperimentalParameters): interactions, configurations and noise model.
        n_cavities (int): number of cavity modes.
        n_fields (int): number of field modes.
        n_qubits (int): number of qubits.
        overrides (dict[str, Any] | None): ``{global_args_key: value}`` replacing configured
            parameter values; may be traced JAX arrays. None uses the configured values.
        dynamic_keys (set | None): global keys factored into an args-coefficient instead of baked,
            so a sweep varies them via ``solver.run(args=...)``. Only :data:`PROMOTABLE_TYPES`.
    Returns:
        tuple: ``(hamiltonians, lindblad_operators, global_args)`` where
          - hamiltonians (dict[str, qt.QobjEvo]): 'base' plus each configuration name -> H.
          - lindblad_operators (dict[str, list[qt.Qobj | qt.QobjEvo]]): same keys -> collapse operators.
          - global_args (dict[str, Any]): resolved parameter values keyed by global key.
    """
    if operators is None:
        raise RuntimeError("Operators must be generated before Hamiltonian")

    H_const, H_time_dependent, L_interaction = [], [], []
    global_args: Dict[str, Any] = {}

    # base model: interactions common to every configuration
    for interaction in experimental_params.interactions:
        const_H, timedep_H, L_term = _interaction_terms(
            interaction, "BaseModel_", operators, n_cavities, n_fields, overrides, global_args, dynamic_keys)
        H_const.extend(const_H)
        H_time_dependent.extend(timedep_H)
        if L_term is not None:
            L_interaction.append(L_term)

    H_static = operators["identity"] if not H_const else sum(H_const)
    H_base = qt.QobjEvo([H_static] + H_time_dependent, args=global_args)

    L_base_noise = _qubit_noise(experimental_params.noise_model, operators, n_qubits)
    hamiltonians = {'base': H_base}
    lindblad_operators = {'base': L_interaction + L_base_noise}

    # per-configuration extras on top of the base model
    for configuration in experimental_params.configuration_set:
        const_terms, time_dependent_terms, lindblad_terms = [], [], []
        for interaction in configuration.interactions:
            const_H, timedep_H, L_term = _interaction_terms(
                interaction, f"Conf:{configuration.name}_", operators, n_cavities, n_fields, overrides, global_args, dynamic_keys)
            const_terms.extend(const_H)
            time_dependent_terms.extend(timedep_H)
            if L_term is not None:
                lindblad_terms.append(L_term)

        noise_terms = _qubit_noise(configuration.noise_model, operators, n_qubits) if configuration.noise_model is not None else L_base_noise
        conf_H_static = H_static.copy() if not const_terms else H_static + sum(const_terms)
        conf_H_time = H_time_dependent + time_dependent_terms

        hamiltonians[configuration.name] = qt.QobjEvo([conf_H_static] + conf_H_time, args=global_args)
        lindblad_operators[configuration.name] = L_interaction + lindblad_terms + noise_terms

    return hamiltonians, lindblad_operators, global_args


def build_hamiltonian_term(interaction, operators, n_cavities, n_fields):
    """Decompose an interaction into a list of :class:`InteractionTerm` (bare-operator terms).

    Operators carry no scalar prefactor; the assembler applies the parameter values, baking them
    or factoring a promoted one into an args-coefficient. All 'L' contributions form one collapse
    operator. kappa enters as √kappa, others linearly. A structural (non-multiplicative) parameter,
    e.g. inside a matrix exponential, is passed as a callable ``operator(values) -> Qobj`` and must
    stay out of :data:`PROMOTABLE_TYPES`.

    Args:
        interaction (Interaction): the interaction with its type, subsystems, parameters, time_modulation.
        operators (dict): system operators from ``generate_system_operators``.
        n_cavities (int): number of cavity modes (for custom-matrix tensor offsets).
        n_fields (int): number of field modes (for custom-matrix tensor offsets).
    Returns:
        list[InteractionTerm]: one per additive H/Lindblad term of the interaction.
    """
    int_type = interaction.interaction_type
    system1, index1 = interaction.subsystem1
    # flag only; the pulse is jitted in wrap_time_modulation (qutip-jax needs a jitted fn)
    has_mod = interaction.time_modulation is not None
    terms = []

    def add(kind, operator, const=1.0, params=None, modulated=has_mod):
        terms.append(InteractionTerm(kind, operator, complex(const), params or {}, modulated))

    if int_type == InteractionType.DETUNING:
        if system1 == 'cavity':
            a, a_dag = operators["a_c"][index1], operators["a_c_dag"][index1]
            add('H', a_dag * a, params={'delta': _ID})
        elif system1 == 'field':
            a, a_dag = operators["a_f"][index1], operators["a_f_dag"][index1]
            add('H', a_dag * a, params={'delta': _ID})
        elif system1 == 'qubit':
            add('H', operators["sigma_z"][index1], const=0.5, params={'delta': _ID})

    elif int_type == InteractionType.DRIVE:
        if system1 == 'cavity':
            a, a_dag = operators["a_c"][index1], operators["a_c_dag"][index1]
        elif system1 == 'field':
            a, a_dag = operators["a_f"][index1], operators["a_f_dag"][index1]
        # 1j*(eps*a_dag - conj(eps)*a)
        add('H', a_dag, const=1j, params={'amplitude': _ID})
        add('H', a, const=-1j, params={'amplitude': jnp.conj})

    elif int_type == InteractionType.DISSIPATION:
        add('L', operators["a_c"][index1], params={'kappa': jnp.sqrt})

    if interaction.subsystem2 is not None:
        system2, index2 = interaction.subsystem2

    if int_type == InteractionType.COUPLING:
        a1, a1_dag = operators["a_c"][index1], operators["a_c_dag"][index1]
        a2, a2_dag = operators["a_c"][index2], operators["a_c_dag"][index2]
        add('H', a1_dag * a2 + a1 * a2_dag, params={'gamma': _ID})

    if int_type == InteractionType.INPUT_OUTPUT:
        if system2 == 'cavity':
            system1, index1, system2, index2 = system2, index2, system1, index1
        ac, ac_dag = operators["a_c"][index1], operators["a_c_dag"][index1]
        af, af_dag = operators["a_f"][index2], operators["a_f_dag"][index2]
        add('H', af_dag * ac - af * ac_dag, const=1j/2, params={'kappa': jnp.sqrt, 'gamma': _ID})
        # one collapse op gamma*g(t)*a_f + sqrt(kappa)*a_c; g(t) modulates only a_f
        add('L', af, params={'gamma': _ID})
        add('L', ac, params={'kappa': jnp.sqrt}, modulated=False)

    if int_type == InteractionType.DISPERSIVE:
        if system2 == 'cavity':
            system1, index1, system2, index2 = system2, index2, system1, index1
        ac, ac_dag = operators["a_c"][index1], operators["a_c_dag"][index1]
        sz = operators["sigma_z"][index2]
        add('H', ac_dag * ac * sz, const=-1.0, params={'chi': _ID})

    if int_type in [InteractionType.XX, InteractionType.YY, InteractionType.ZZ]:
        if system1 != 'qubit' or system2 != 'qubit':
            raise ValueError(f"Qubit-qubit interactions must be between qubits, got {system1} and {system2}")
        pauli = {InteractionType.XX: "sigma_x", InteractionType.YY: "sigma_y", InteractionType.ZZ: "sigma_z"}[int_type]
        add('H', operators[pauli][index1] * operators[pauli][index2], const=0.5, params={'chi': _ID})

    if int_type in (InteractionType.CUSTOM_HAMILTONIAN, InteractionType.CUSTOM_LINDBLAD):
        # tensor positions of the subsystem(s) in canonical (cavity⊗field⊗qubits) order
        offsets = {'cavity': 0, 'field': n_cavities, 'qubit': n_cavities + n_fields}
        positions = [offsets[interaction.subsystem1[0]] + interaction.subsystem1[1]]
        if interaction.subsystem2 is not None:
            positions.append(offsets[interaction.subsystem2[0]] + interaction.subsystem2[1])
        # embed the custom matrix at those positions
        identities = list(operators["I_c"]) + list(operators["I_f"]) + list(operators["I_q"])
        embedded = embed_operator(interaction.custom_matrix, positions, identities)
        add('H' if int_type == InteractionType.CUSTOM_HAMILTONIAN else 'L', embedded)

    return terms


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


def _map_to_global_args(interaction, prefix, overrides, global_args):
    """Remap an interaction's local parameters to globally-unique keys.

    Builds ``f"{prefix}{context}__{name}"`` per local name and writes its value into
    ``global_args`` (override if given, else configured).

    Args:
        interaction (Interaction): the interaction whose parameters to remap.
        prefix (str): global-args key prefix ("BaseModel_" or "Conf:<name>_").
        overrides (dict[str, Any] | None): values replacing configured ones, keyed by global key.
        global_args (dict[str, Any]): accumulator, updated in place with each global key's value.
    Returns:
        tuple: ``(merged, key_map)`` with merged values and local name -> global key; a non-dict
            ``parameters`` (bare scalar) is returned unchanged with an empty key_map.
    """
    params = interaction.parameters
    if not isinstance(params, dict):
        return params, {}
    merged = dict(params)
    key_map = {}
    for key in params:
        global_key = f"{prefix}{interaction._interaction_context()}__{key}"
        key_map[key] = global_key
        if overrides is not None and global_key in overrides:
            merged[key] = overrides[global_key]
        global_args[global_key] = merged[key]
    return merged, key_map


def _wrap_time_modulation(func, key_map):
    """Adapt a user pulse ``func(t, **local_params)`` to a jitted ``coeff(t, **global_args)``.

    Picks this interaction's params out of the global args dict and calls ``func``; jitted because
    qutip-jax only accepts JAX-valued coefficients from jitted functions.

    Args:
        func (callable): user time-modulation g(t, **local_params).
        key_map (dict[str, str]): local parameter name -> global-args key.
    Returns:
        callable: jitted coefficient g(t, **global_args).
    """
    def wrapped(t, **all_args):
        local = {name: all_args[key] for name, key in key_map.items()}
        return func(t, **local)
    return jax.jit(wrapped)


def _make_coefficient(base, promoted_factors, time_modulation):
    """Build a jitted args-coefficient ``base * Π transform(args[key]) * g(t, **args)``.

    Args:
        base (complex): constant prefactor with baked (non-promoted) params folded in.
        promoted_factors (list[tuple[callable, str]]): (transform, global key) per promoted param.
        time_modulation (callable | None): wrapped pulse g(t, **args), or None if not modulated.
    Returns:
        callable: jitted coefficient; base times each promoted factor, times the pulse when present.
    """
    def coefficient(t, **args):
        value = base
        for transform, global_key in promoted_factors:   # one factor per promoted parameter
            value = value * transform(args[global_key])
        return value * time_modulation(t, **args) if time_modulation is not None else value
    return jax.jit(coefficient)


def _interaction_terms(interaction, prefix, operators, n_cavities, n_fields, overrides, global_args, dynamic_keys):
    """Assemble an interaction from its terms, baking or promoting each parameter.

    Parameters in ``dynamic_keys`` are factored into the coefficient as ``f(args[key])`` so they
    can be swept via ``args``; others are baked (overrides applied), keeping ``g(t)``.

    Args:
        interaction (Interaction): the interaction to assemble.
        prefix (str): global-args key prefix ("BaseModel_" or "Conf:<name>_").
        operators (dict): system operators from ``generate_system_operators``.
        n_cavities (int): number of cavity modes.
        n_fields (int): number of field modes.
        overrides (dict[str, Any] | None): values replacing configured ones, keyed by global key.
        global_args (dict[str, Any]): accumulator of resolved values by global key.
        dynamic_keys (set | None): global keys to promote into args-coefficients.
    Returns:
        tuple: ``(const_H, timedep_H, L_term)`` where
          - const_H (list[qt.Qobj]): constant Hamiltonian terms.
          - timedep_H (list[[qt.Qobj, callable]]): ``[operator, coeff(t, **args)]`` pairs.
          - L_term (qt.Qobj | qt.QobjEvo | None): the single Lindblad collapse operator.
    """
    # resolve local params to global keys; find which ones this sweep promotes
    merged, key_map = _map_to_global_args(interaction, prefix, overrides, global_args)
    promoted = {name for name, key in key_map.items() if dynamic_keys and key in dynamic_keys}
    # promotion is only supported for the curated multiplicative types
    if promoted and interaction.interaction_type not in PROMOTABLE_TYPES:
        raise NotImplementedError(
            f"Cannot factor {sorted(promoted)} of {interaction._interaction_context()} into an args-coefficient; "
            f"only {sorted(t.name for t in PROMOTABLE_TYPES)} support it. Sweep it via the rebuild path instead.")
    # the interaction's pulse g(t), wrapped to read global args (None if not modulated)
    time_modulation = _wrap_time_modulation(interaction.time_modulation, key_map) if interaction.time_modulation is not None else None

    const_H, timedep_H, L_const, L_timedep = [], [], [], []
    for term in build_hamiltonian_term(interaction, operators, n_cavities, n_fields):
        # structural (callable) operators are rebuilt from the resolved values
        operator = term.operator(merged) if callable(term.operator) else term.operator
        # bake non-promoted params into the prefactor; collect promoted ones as coeff factors
        base = term.const
        promoted_factors = []
        for name, transform in term.params.items():
            if name in promoted:
                promoted_factors.append((transform, key_map[name]))
            else:
                base *= complex(transform(merged[name]))
        term_modulation = time_modulation if term.modulated else None
        # constant term, or an [operator, coeff] pair when time- or args-dependent
        if not promoted_factors and term_modulation is None:
            part = base * operator
        else:
            part = [operator, _make_coefficient(base, promoted_factors, term_modulation)]
        # route H vs L, constant vs time-dependent
        is_pair = isinstance(part, list)
        if term.kind == 'H':
            (timedep_H if is_pair else const_H).append(part)
        else:
            (L_timedep if is_pair else L_const).append(part)

    # all 'L' pieces are ONE collapse operator: keep them in a single QobjEvo (or Qobj)
    if not L_const and not L_timedep:
        L_term = None
    elif not L_timedep:
        L_term = sum(L_const)
    else:
        L_term = qt.QobjEvo(([sum(L_const)] if L_const else []) + L_timedep, args=global_args)
    return const_H, timedep_H, L_term


def _qubit_noise(noise_model, operators, n_qubits):
    """Flatten the per-qubit Lindblad noise operators for a noise model.

    Args:
        noise_model (NoiseModel): rates + optional custom collapse operators.
        operators (dict): system operators (for the qubit Pauli/lowering operators).
        n_qubits (int): number of qubits.
    Returns:
        list[qt.Qobj | qt.QobjEvo]: noise collapse operators for all qubits.
    """
    per_qubit = [build_qubit_noise_operators(
        sigma_x=operators["sigma_x"][i], sigma_y=operators["sigma_y"][i],
        sigma_z=operators["sigma_z"][i], sigma_minus=operators["sigma_minus"][i],
        depolarizing_rate=noise_model.depolarizing[i],
        dephasing_rate=noise_model.dephasing[i],
        relaxation_rate=noise_model.relaxation[i],
    ) for i in range(n_qubits)]
    noise_operators: List[Union[qt.Qobj, qt.QobjEvo]] = [op for qubit in per_qubit for op in qubit]
    if noise_model.custom_operators is not None:
        noise_operators.extend(noise_model.custom_operators)
    return noise_operators


def embed_circuit_unitary(
    circuit_unitary: jnp.ndarray,
    field_levels: List[int],
    cavity_levels: List[int],
) -> jnp.ndarray:
    """
    Embed an n-qubit circuit unitary into the full composite Hilbert space using JAX.

    The composite space follows the canonical ordering
    (cavity_1 ⊗ ... ⊗ cavity_C ⊗ field_1 ⊗ ... ⊗ field_F ⊗ qubit_1 ⊗ ... ⊗ qubit_n),
    matching ``total_dims = cavity_levels + field_levels + qubit_levels``. The circuit
    acts only on the trailing qubit subspace, so the embedding is
    ``I_bosonic ⊗ circuit_unitary`` with the qubits as the least-significant factor.

    Because every cavity and field factor is an identity, their tensor product collapses
    to a single identity of size ``∏ cavity_levels · ∏ field_levels`` (identities commute,
    so the cavity/field interleaving is irrelevant). The embedding is therefore a single
    Kronecker product, which keeps the per-call cost minimal when circuits are recomputed.

    Args:
        circuit_unitary: (∏ qubit_levels)×(∏ qubit_levels) unitary matrix (JAX array).
        field_levels: Per-mode levels of the field subsystems, one entry per field mode.
        cavity_levels: Per-mode levels of the cavity subsystems, one entry per cavity mode.

    Returns:
        Full-space unitary as JAX array: ``I_{∏cavity·∏field} ⊗ circuit_unitary``.

    Example:
        >>> # 2-qubit circuit unitary (4x4) with 2 cavities (2,3) and 1 field (2)
        >>> U_circuit = jnp.eye(4, dtype=jnp.complex128)
        >>> U_full = embed_circuit_unitary(U_circuit, field_levels=[2], cavity_levels=[2, 3])
        >>> # U_full shape: (2*3 * 2 * 4)×(...) = 48×48
    """
    # Total dimension of all bosonic (cavity + field) modes; np.prod over the per-mode
    # level lists (returns 1 for an empty list, i.e. no such modes).
    bosonic_dim = int(np.prod(cavity_levels)) * int(np.prod(field_levels))

    # I_cavity ⊗ I_field == I_bosonic, so a single identity plus one Kronecker product
    # suffices regardless of the number of cavity/field modes. Under jit the identity is
    # a compile-time constant (folded by XLA), so it adds no per-call cost.
    I_bosonic = jnp.eye(bosonic_dim, dtype=jnp.complex128)
    U_full_jax = jnp.kron(I_bosonic, circuit_unitary)

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

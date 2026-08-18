"""
Experimental Parameters Class
============================

System configuration parameters for quantum sensing experiments including
physical model dimensions, interactions, noise models, measurement protocols,
and initial states.
"""

import copy
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Set, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp
import qutip as qt
import math
import inspect

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import

# Default measurement-window slope factor: slope = _WINDOW_SLOPE_FACTOR / window_width when
# window_slope is not given, so smaller windows get sharper edges (constant slope*width).
_WINDOW_SLOPE_FACTOR = 30.0


class State(Enum):
    """Enumeration of supported initial state configurations (for input field).
    
    Possible values:
    - VACUUM
    - FOCK
    - COHERENT
    - THERMAL
    - CUSTOM
    """

    VACUUM = "vacuum"
    FOCK = "fock"
    COHERENT = "coherent"
    THERMAL = "thermal"
    CUSTOM = "custom"


class InteractionType(Enum):
    """Enumeration of supported interaction types.
    
    Possible values:
    - DETUNING
    - DRIVE
    - DISSIPATION
    - COUPLING
    - INPUT_OUTPUT
    - DISPERSIVE
    - JAYNES_CUMMINGS
    - ZZ
    - XX
    - YY
    - CUSTOM_HAMILTONIAN
    - CUSTOM_LINDBLAD
    """

    # single subsystem interactions
    DETUNING = "detuning" # detuning of cavity, input field or qubit frequencies from reference frequency
    DRIVE = 'drive' # external drive on cavity or field modes
    DISSIPATION = 'dissipation' # dissipation processes on cavity subsystems

    # cavity-field, cavity-cavity
    COUPLING = "coupling" # coupling between cavities
    INPUT_OUTPUT = "input_output" # input-output coupling between a cavity and an input field (e.g., for open system dynamics)

    # qubit-cavity, qubit-field
    DISPERSIVE = "dispersive" # dispersive coupling between qubits and cavities/fields
    JAYNES_CUMMINGS = "jaynes-cummings" # Jaynes-Cummings interaction between qubits and cavities/fields (not implemented yet)

    # qubit-qubit interactions
    ZZ = "sz-sz"  # σz ⊗ σz interaction
    XX = "sx-sx"  # σx ⊗ σx interaction
    YY = "sy-sy"  # σy ⊗ σy interaction

    # single-qubit Lindblad noise channels (a NoiseModel is expanded into these before build,
    # so their rates are swept exactly like any other interaction parameter)
    DEPOLARIZING = "depolarizing"  # √(rate/3) σx, √(rate/3) σy, √(rate/3) σz (three collapse operators)
    DEPHASING = "dephasing"        # √rate σz
    RELAXATION = "relaxation"      # √rate σ₋

    # custom interactions
    CUSTOM_HAMILTONIAN= "custom_hamiltonian"  # Custom coherent interaction defined by user-provided matrix
    CUSTOM_LINDBLAD = "custom_lindblad"  # Custom dissipation defined by user-provided Lindblad operator


# Single-qubit Lindblad noise channels. A NoiseModel is expanded into interactions of these types at
# ExperimentalParameters init, so noise is stored and handled as ordinary interactions. They are the
# only types a configuration may re-declare over the base model (a per-channel override, see
# _validate_experimental_parameters and build_hamiltonians).
NOISE_INTERACTION_TYPES = frozenset({
    InteractionType.DEPOLARIZING, InteractionType.DEPHASING, InteractionType.RELAXATION,
})


class Interaction:
    """
    Configuration for interaction between subsystems.

    Attributes:
        interaction_type: InteractionType (ZZ, jaynes_cummings, etc.)
        subsystem1: Tuple of type (string) and index (int) of the first subsystem involved in the interaction (e.g., ('qubit', 3))
        subsystem2: Optional[Tuple[str,int]] of type (string) and index (int) of the second subsystem involved in the interaction (e.g., ('cavity', 1))
        parameters: Dict[str, Any] of interaction parameters or a numeric value (int, float, complex)
        time_modulation: Optional[Callable[[float, Dict[str, Any]], float]] function of time that modulates the interaction strength (e.g., for time-dependent interactions). Must return a non-negative value.
        custom_matrix: Optional[qt.Qobj] operator for CUSTOM_HAMILTONIAN/CUSTOM_LINDBLAD interactions,
            acting only on the involved subsystem(s). For a two-subsystem custom interaction the
            matrix legs are given in the (subsystem1, subsystem2) order you pass; if the subsystems
            are reordered into canonical order the legs are permuted automatically (which requires
            the matrix to carry structured per-subsystem dims [[d1, d2], [d1, d2]]).

    Note:
        Subsystems are given as (type, index) tuples, where type is 'qubit', 'cavity' or
        'field' and index is a non-negative int. Two-subsystem interactions are stored in
        canonical order (sorted by type then index: cavity < field < qubit); pass them
        already ordered to avoid surprises with custom_matrix leg ordering.

        For numeric-keyed types, parameters may be a dict with the named key OR a bare
        numeric value wrapped into that key automatically (parameters=0.5 == {'delta': 0.5}
        for DETUNING). After construction, parameters is always a dict. A zero value for any
        strength/rate parameter is allowed but emits a warning.

        Interaction types and their parameters:

    ```
    abbrev.: cav=cavity, fld=field, qb=qubit
    +-------------------+---------------+------------------------+
    | type              | subsystem(s)  | parameter(s)           |
    +-------------------+---------------+------------------------+
    | Single-subsystem  (subsystem2 = None)                      |
    +-------------------+---------------+------------------------+
    | detuning          | cav/fld/qb    | delta : float          |
    | drive             | cav / fld     | amplitude : complex    |
    | dissipation       | cav           | kappa : float >= 0     |
    +-------------------+---------------+------------------------+
    | Two-subsystem  (subsystem2 required)                       |
    +-------------------+---------------+------------------------+
    | coupling          | cav + cav     | gamma : complex        |
    | input_output      | cav + fld     | kappa : float >= 0     |
    |                   |               | gamma : float >= 0 (*1)|
    | dispersive        | qb + cav      | chi : float            |
    | jaynes-cummings   | qb + cav/fld  | (not implemented)      |
    +-------------------+---------------+------------------------+
    | Qubit-qubit  (both qubit)                                  |
    +-------------------+---------------+------------------------+
    | sz-sz/sx-sx/sy-sy | qb + qb       | chi : float >= 0 (*2)  |
    +-------------------+---------------+------------------------+
    | Custom  (subsystem2 optional, any types)                   |
    +-------------------+---------------+------------------------+
    | custom_hamiltonian| 1 or 2 subsys | custom_matrix (*3)     |
    | custom_lindblad   | 1 or 2 subsys | custom_matrix (*4)     |
    +-------------------+---------------+------------------------+
    (*1) gamma defaults to 1.0 if time_modulation is set
    (*2) 'strength' is a deprecated alias for chi
    (*3) Qobj, square, Hermitian
    (*4) Qobj, square (need not be Hermitian)
    ```

        For non-custom types a supplied custom_matrix is ignored (with a warning). A
        custom_matrix must be square with equal-axis dims, and its overall size must match
        the product of the involved subsystem truncation levels (checked by PhysicalModel).
    """

    def __init__(
        self,
        interaction_type: InteractionType,
        subsystem1: Optional[Tuple[str, int]] = None,
        subsystem2: Optional[Tuple[str, int]] = None,
        parameters: Optional[Union[Dict[str, Any], int, float, complex]] = 1.0,
        time_modulation: Optional[Callable[[float, Dict[str, Any]], float]] = None,
        custom_matrix: Optional[qt.Qobj] = None,
    ):

        self.interaction_type = interaction_type
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.parameters = parameters
        self.time_modulation = time_modulation
        self.custom_matrix = custom_matrix
        # True for interactions synthesized from a NoiseModel (see NoiseModel.to_interactions). Such
        # interactions are owned by ExperimentalParameters._absorb_noise_into_interactions, which
        # strips and regenerates them, so re-distributing noise (e.g. on copy) never doubles it.
        self.generated_noise = False

        self.__post_init__()

    def __post_init__(self):
        """Validate interaction parameters."""
        if not isinstance(self.interaction_type, InteractionType):
            raise TypeError(
                "interaction_type must be an InteractionType enum value"
            )

        # A custom interaction may omit its subsystem (subsystem1=None): the custom_matrix is then
        # taken to act on the full composite space and is used directly, without embedding. Its size
        # is checked against the total Hilbert dimension where that is known (PhysicalModel / build).
        # Only custom types may be subsystem-less; every other type is placed on specific subsystems.
        custom_types = {InteractionType.CUSTOM_HAMILTONIAN, InteractionType.CUSTOM_LINDBLAD}
        if self.subsystem1 is None:
            if self.interaction_type not in custom_types:
                raise ValueError(
                    f"subsystem1 is required for {self.interaction_type.value} interactions"
                )
            if self.subsystem2 is not None:
                raise ValueError(
                    "a full-space custom interaction (subsystem1=None) cannot also specify subsystem2"
                )
        else:
            # Validate subsystem specifications before using them in diagnostics
            if not isinstance(self.subsystem1, tuple) or len(self.subsystem1) != 2:
                raise ValueError(
                    f"subsystem1 must be a tuple of (type, index), but got {self.subsystem1}"
                )
            if self.subsystem1[0] is None or self.subsystem1[1] is None:
                raise ValueError(
                    f"subsystem1 must be a tuple of (type, index), but got {self.subsystem1}"
                )
            if self.subsystem2 is not None:
                if not isinstance(self.subsystem2, tuple) or len(self.subsystem2) != 2:
                    raise ValueError(
                        f"subsystem2 must be a tuple of (type, index) if provided, but got {self.subsystem2}"
                    )
                if self.subsystem2[0] is None or self.subsystem2[1] is None:
                    raise ValueError(
                        f"subsystem2 must be a tuple of (type, index) if provided, but got {self.subsystem2}"
                    )

            if not isinstance(self.subsystem1[1], int) or (
                self.subsystem2 is not None and not isinstance(self.subsystem2[1], int)
            ):
                raise TypeError(
                    "Subsystem indices must be integers"
                )
            if not isinstance(self.subsystem1[0], str) or (
                self.subsystem2 is not None and not isinstance(self.subsystem2[0], str)
            ):
                raise TypeError(
                    "Subsystem types must be strings like 'qubit', 'cavity', or 'field'"
                )
            if self.subsystem1 == self.subsystem2:
                raise ValueError("subsystem1 and subsystem2 must refer to different subsystems")
            if self.subsystem1[1] < 0 or (self.subsystem2 is not None and self.subsystem2[1] < 0):
                raise ValueError("Subsystem indices must be non-negative")
            if not (
                self.subsystem1[0] in ['qubit', 'cavity', 'field']
                and (self.subsystem2 is None or self.subsystem2[0] in ['qubit', 'cavity', 'field'])
            ):
                raise ValueError("Subsystem types must be 'qubit', 'cavity', or 'field'")

            # Ensure canonical ordering (sort by type and then index)
            if self.subsystem2 is not None and (self.subsystem1[0], self.subsystem1[1]) > (self.subsystem2[0], self.subsystem2[1]):
                self.subsystem1, self.subsystem2 = self.subsystem2, self.subsystem1

                # The matrix of a two-subsystem custom interaction is given with its tensor
                # legs in the user's (subsystem1, subsystem2) order. Now that the subsystems
                # have been swapped into canonical order, permute the matrix legs to match so
                # the stored (subsystem1, subsystem2) order and the matrix stay aligned. This
                # needs structured per-subsystem dims [[d1, d2], [d1, d2]]; a flat matrix
                # cannot be split into its two legs, so it must be supplied with structured dims.
                if isinstance(self.custom_matrix, qt.Qobj) and self.interaction_type in (
                    InteractionType.CUSTOM_HAMILTONIAN,
                    InteractionType.CUSTOM_LINDBLAD,
                ):
                    if len(self.custom_matrix.dims[0]) == 2:
                        self.custom_matrix = self.custom_matrix.permute([1, 0])
                    else:
                        raise ValueError(
                            self._with_context(
                                "The interaction's subsystems were reordered into canonical order "
                                "(cavity < field < qubit, then by index), so the custom_matrix tensor "
                                "legs must be permuted to match. This requires the matrix to carry "
                                "structured per-subsystem dims [[d1, d2], [d1, d2]] (in your original "
                                "subsystem1, subsystem2 order). Provide it with structured dims, or pass "
                                "the subsystems already in canonical order."
                            )
                        )

        # Validate time modulation function
        if self.time_modulation is not None and not callable(self.time_modulation):
            raise TypeError(
                self._with_context(
                    "time_modulation must be callable as time_modulation(time, **parameters)"
                    f". Got type: {type(self.time_modulation)}"
                )
            )
        elif self.time_modulation is not None:
            # Test the modulation function to catch signature/return-type issues early.
            try:
                params = self.parameters if isinstance(self.parameters, dict) else {}
                for _ in range(100):
                    random_time = np.random.uniform(-10, 10)
                    test_value = self.time_modulation(random_time, **params)
                    # Accept Python/NumPy/JAX scalars (0-d arrays); reject non-scalars.
                    if np.ndim(test_value) != 0:
                        raise ValueError(
                            self._with_context(
                                "time_modulation must return a scalar numeric value"
                                f". Got type: {type(test_value)} with ndim {np.ndim(test_value)}"
                            )
                        )
                    test_value = float(test_value)
                    if test_value < 0:
                        raise ValueError(
                            self._with_context(
                                "time_modulation must return a non-negative value, "
                                f"but got {test_value} for input time {random_time}"
                            )
                        )
            except Exception as e:
                raise ValueError(
                    self._with_context(
                        "time_modulation must be callable as time_modulation(time, **parameters) and return a non-negative scalar numeric value. "
                    )
                ) from e
             
        # Validate different interaction types

        if self.interaction_type in {InteractionType.DETUNING, InteractionType.DRIVE, InteractionType.DISSIPATION}:
            if self.subsystem2 is not None:
                raise ValueError(
                    self._with_context(
                        f"Interaction type is defined for single subsystems, "
                        f"but subsystem2 was provided: {self.subsystem2}"
                    )
                )
            if self.interaction_type == InteractionType.DETUNING:
                self._validate_detuning()

            if self.interaction_type == InteractionType.DRIVE:
                self._validate_drive()

            if self.interaction_type == InteractionType.DISSIPATION:
                self._validate_dissipation()

        elif self.interaction_type == InteractionType.COUPLING:
           self._validate_coupling()
        
        elif self.interaction_type == InteractionType.INPUT_OUTPUT:
            self._validate_input_output()

        elif self.interaction_type == InteractionType.DISPERSIVE:
            self._validate_dispersive()
        
        elif self.interaction_type == InteractionType.JAYNES_CUMMINGS:
            raise NotImplementedError(
                self._with_context(
                    "Jaynes-Cummings interaction is not implemented yet. "
                    "Please use supported interactions or implement JC interaction validation and parameter handling."
                )
            )
        
        elif self.interaction_type in {InteractionType.ZZ, InteractionType.XX, InteractionType.YY}:
            self._validate_qubit_qubit()

        elif self.interaction_type in {InteractionType.DEPOLARIZING, InteractionType.DEPHASING, InteractionType.RELAXATION}:
            if self.subsystem2 is not None:
                raise ValueError(
                    self._with_context(
                        f"Qubit-noise interaction is defined for a single qubit subsystem, "
                        f"but subsystem2 was provided: {self.subsystem2}"
                    )
                )
            self._validate_qubit_noise()

        elif self.interaction_type == InteractionType.CUSTOM_HAMILTONIAN:
            self._validate_custom_hamiltonian()

        elif self.interaction_type == InteractionType.CUSTOM_LINDBLAD:
            self._validate_custom_lindblad()

        else:
            raise NotImplementedError(
                self._with_context(
                    f"Interaction type is not supported yet. The following interactions are implemented:\n"
                    f"{[f'{interaction.value}' + f'\n' for interaction in InteractionType]}"
                )
            )

        if self.interaction_type not in {InteractionType.CUSTOM_HAMILTONIAN, InteractionType.CUSTOM_LINDBLAD} and self.custom_matrix is not None:
            warnings.warn(
                self._with_context(
                    "Custom matrix is provided but the interaction type is not a custom Hamiltonian or Lindblad. "
                    "The custom matrix will be ignored."
                ),
                UserWarning,
            )
            self.custom_matrix = None

        if self.parameters is None:
            self.parameters = {}

    def _validate_detuning(self):
        """Validate parameters for detuning interactions."""
        if self.subsystem1[0] not in {'cavity', 'field', 'qubit'}:
            raise ValueError(
                self._with_context(
                    f"Detuning interaction must be on a cavity, field or qubit subsystem, but got {self.subsystem1}"
                )
            )
        elif isinstance(self.parameters, (int, float)):
            self.parameters = {"delta": self.parameters}
        elif not isinstance(self.parameters, dict):
            raise TypeError(
                self._with_context(
                    "Parameters for detuning interaction must be a float or a dict with 'delta' key for detuning value"
                )
            )
        elif "delta" not in self.parameters:
            raise ValueError(
                self._with_context(
                    "Parameters for detuning interaction must include 'delta' key for detuning value"
                )
            )
        elif not isinstance(self.parameters["delta"], (int,float)):
            raise TypeError(
                self._with_context("Detuning value ('delta') must be a numeric value, float or int")
            )
        
        self.parameters["delta"] = float(self.parameters["delta"])
        
        if self.parameters["delta"] == 0:
            warnings.warn(
                self._with_context(
                    "Detuning value (delta) is zero. This means the subsystem frequency is at "
                    "the reference frequency, which may be intentional but should be double-checked."
                ),
                UserWarning,
            )

    def _validate_drive(self):
        """Validate parameters for drive interactions."""
        if self.subsystem1[0] not in {'cavity', 'field'}:
            raise ValueError(
                self._with_context(
                    f"Drive interaction must be on a cavity or field subsystem, but got {self.subsystem1}"
                )
            )
        elif isinstance(self.parameters, (int, float, complex)):
            self.parameters = {"amplitude": self.parameters}
        elif not isinstance(self.parameters, dict):
            raise TypeError(
                self._with_context(
                    "Parameters for drive interaction must be a float, a complex or a dict with 'amplitude' key"
                )
            )
        elif "amplitude" not in self.parameters:
            raise ValueError(
                self._with_context(
                    "Parameters for drive interaction must include 'amplitude' key for drive strength"
                )
            )
        elif not isinstance(self.parameters["amplitude"], (int, float, complex)):
            raise TypeError(
                self._with_context("Drive amplitude must be a numeric value, float, int or complex")
            )
        
        self.parameters["amplitude"] = complex(self.parameters["amplitude"])

        if self.parameters["amplitude"] == 0:
            warnings.warn(
                self._with_context(
                    "Drive amplitude is zero. This means no drive is applied, "
                    "which may be intentional but should be double-checked."
                ),
                UserWarning,
            )

    def _validate_dissipation(self):
        """Validate parameters for dissipation interactions."""
        if self.subsystem1[0] != 'cavity':
            raise ValueError(
                self._with_context(
                    f"Dissipation interaction must be on a cavity subsystem, but got {self.subsystem1}"
                )
            )
        elif isinstance(self.parameters, (int, float)):
            self.parameters = {"kappa": self.parameters}
        elif not isinstance(self.parameters, dict):
            raise TypeError(
                self._with_context(
                    "Parameters for dissipation interaction must be a float or a dict with 'kappa' key"
                )
            )
        elif "kappa" not in self.parameters:
            raise ValueError(
                self._with_context(
                    "Parameters for dissipation interaction must include 'kappa' key for dissipation rate"
                )
            )
        elif not isinstance(self.parameters["kappa"], (int, float)):
            raise TypeError(
                self._with_context("Dissipation rate (kappa) must be a numeric value, float or int")
            )
        
        self.parameters["kappa"] = float(self.parameters["kappa"])

        if self.parameters["kappa"] == 0:
            warnings.warn(
                self._with_context(
                    "Dissipation rate (kappa) is zero. This means no dissipation is applied, "
                    "which may be intentional but should be double-checked."
                ),
                UserWarning,
            )
        
        if self.parameters["kappa"] < 0:
            raise ValueError(
                self._with_context(
                    f"Dissipation rate (kappa) must be >= 0, got {self.parameters['kappa']}"
                )
            )

    def _validate_coupling(self):
        """Validate parameters for coupling interactions (cavity-cavity)."""

        if self.subsystem1 is None or self.subsystem2 is None:
            raise ValueError(
                self._with_context(
                    "Coupling interaction must involve two cavity subsystems"
                )
            )
        elif self.subsystem1[0] != 'cavity' or self.subsystem2[0] != 'cavity':
            raise ValueError(
                self._with_context(
                    "Coupling interaction must involve two cavity subsystems"
                )
            )
        
        if isinstance(self.parameters, (int, float, complex)):
            self.parameters = {"gamma": self.parameters}
        elif not isinstance(self.parameters, dict):
            raise TypeError(
                self._with_context(
                    "Parameters for coupling interaction must be a float, a complex or a dict with 'gamma' key"
                )
            )
        elif "gamma" not in self.parameters:
            raise ValueError(
                self._with_context(
                    "Parameters for coupling interaction must include 'gamma' key for coupling strength"
                )
            )
        elif not isinstance(self.parameters["gamma"], (int, float, complex)):
            raise TypeError(
                self._with_context("Coupling strength (gamma) must be a numeric value, float or complex")
            )
        
        self.parameters["gamma"] = complex(self.parameters["gamma"])

        if self.parameters["gamma"] == 0:
            warnings.warn(
                self._with_context(
                    "Coupling strength (gamma) is zero. This means no coupling between these subsystems, "
                    "which may be intentional but should be double-checked."
                ),
                UserWarning,
            )

    def _validate_input_output(self):
        """Validate parameters for input-output interactions (cavity-field)."""

        if self.subsystem1 is None or self.subsystem2 is None:
            raise ValueError(
                self._with_context(
                    "Input-output interaction must involve a cavity and a field subsystem"
                )
            )
        elif set([self.subsystem1[0], self.subsystem2[0]]) != {'cavity', 'field'}:
            raise ValueError(
                self._with_context(
                    "Input-output interaction must involve a cavity subsystem and a field subsystem"
                )
            )

        if isinstance(self.parameters, (int, float)):
            self.parameters = {"kappa": self.parameters}
        elif not isinstance(self.parameters, dict):
            raise TypeError(
                self._with_context(
                    "Parameters for input-output interaction must be a float or a dict with 'kappa' key for loss rate"
                )
            )
        elif "kappa" not in self.parameters:
            raise ValueError(
                self._with_context(
                    "Parameters dictionary for input-output interaction must include 'kappa' key for loss rate"
                )
            )
        elif not isinstance(self.parameters["kappa"], (int, float)):
            raise TypeError(
                self._with_context("Input-output loss rate (kappa) must be a numeric value, float")
            )
        
        if "gamma" not in self.parameters and self.time_modulation is None:
            raise ValueError(
                self._with_context(
                    "Input-output interaction must provide a 'gamma' coupling or a time-modulated coupling "
                    "via time_modulation"
                )
            )
        elif "gamma" in self.parameters and not isinstance(self.parameters["gamma"], (int, float)):
            raise TypeError(
                self._with_context("Input-output coupling strength (gamma) must be a numeric value (float)")
            )
        elif "gamma" not in self.parameters and self.time_modulation is not None:
            self.parameters["gamma"] = 1.0  # Default coupling strength for time-modulated input-output interaction if gamma not provided

        self.parameters["kappa"] = float(self.parameters["kappa"])
        self.parameters["gamma"] = float(self.parameters["gamma"])

        if self.parameters["kappa"] == 0:
            warnings.warn(
                self._with_context(
                    "Input-output loss rate (kappa) is zero. This means no loss through this input-output channel, "
                    "which may be intentional but should be double-checked."
                ),
                UserWarning,
            )
        elif self.parameters["kappa"] < 0:
            raise ValueError(
                self._with_context(
                    f"Input-output loss rate (kappa) must be >= 0, got {self.parameters['kappa']}"
                )
            )
        
        if self.parameters["gamma"] == 0:
            warnings.warn(
                self._with_context(
                    "Input-output coupling strength (gamma) is zero. This means no coupling between these subsystems, "
                    "which may be intentional but should be double-checked."
                ),
                UserWarning,
            )
        elif self.parameters["gamma"] < 0:  
            raise ValueError(
                self._with_context(
                    f"Input-output coupling strength (gamma) must be >= 0, got {self.parameters['gamma']}"
                )
            )

    def _validate_dispersive(self):
        """Validate parameters for dispersive interactions (qubit-cavity)."""
        
        if self.subsystem1 is None or self.subsystem2 is None:
            raise ValueError(
                self._with_context(
                    "Dispersive interaction must involve a qubit and a cavity"
                )
            )
        elif set([self.subsystem1[0], self.subsystem2[0]]) != {'cavity', 'qubit'}:
            raise ValueError(
                self._with_context(
                    "Dispersive interaction must involve a qubit and a cavity"
                )
            )

        if isinstance(self.parameters, (int, float)):
            self.parameters = {"chi": self.parameters}
        elif not isinstance(self.parameters, dict):
            raise TypeError(
                self._with_context(
                    "Parameters for dispersive interaction must be a float or a dict with 'chi' key for dispersive shift"
                )
            )
        elif "chi" not in self.parameters:
            raise ValueError(
                self._with_context(
                    "Parameters for dispersive interaction must include 'chi' key for dispersive shift value"
                )
            )
        elif not isinstance(self.parameters["chi"], (int, float)):
            raise TypeError(
                self._with_context("Dispersive shift (chi) must be a float")
            )
        
        self.parameters["chi"] = float(self.parameters["chi"])

        if self.parameters["chi"] == 0:
            warnings.warn(
                self._with_context(
                    "Dispersive shift (chi) is zero. This means no dispersive coupling between these subsystems, "
                    "which may be intentional but should be double-checked."
                ),
                UserWarning,
            )

    def _validate_qubit_qubit(self):
        """Validate parameters for qubit-qubit interactions."""
        
        if self.subsystem1 is None or self.subsystem2 is None:
            raise ValueError(
                self._with_context(
                    "Interaction must specify both subsystems"
                )
            )
        elif self.subsystem1[0] != 'qubit' or self.subsystem2[0] != 'qubit':
            raise ValueError(
                self._with_context(
                    "Interaction must be between two qubits"
                )
            )
        elif isinstance(self.parameters, (int, float)):
            self.parameters = {"chi": self.parameters}
        elif not isinstance(self.parameters, dict):
            raise TypeError(
                self._with_context(
                    "Parameters for interaction must be a float or a dict with 'chi' key"
                )
            )
        elif "chi" not in self.parameters and "strength" not in self.parameters:
            raise ValueError(
                self._with_context(
                    "Parameters for interaction must include 'chi' or 'strength' key for interaction strength"
                )
            )
        
        if "strength" in self.parameters:
            warnings.warn(
                self._with_context("Using 'strength' value for 'chi'."),
                UserWarning,
            )
            self.parameters["chi"] = self.parameters.pop("strength")

        if not isinstance(self.parameters["chi"], (int, float)):
            raise TypeError(
                self._with_context(
                    "Qubit interaction strength ('chi') must be a numeric value, float or int"
                )
            )
        elif self.parameters["chi"] < 0:
            raise ValueError(
                self._with_context(
                    f"Qubit interaction strength (chi) must be >= 0, got {self.parameters['chi']}"
                )
            )
        elif self.parameters["chi"] == 0:
            warnings.warn(
                self._with_context(
                    "Qubit-qubit interaction strength (chi) is zero. "
                    "This means no direct qubit-qubit coupling, which may be intentional for uncoupled qubit experiments."
                ),
                UserWarning,
                )
            
        self.parameters["chi"] = float(self.parameters["chi"])

    def _validate_qubit_noise(self):
        """Validate parameters for a single-qubit Lindblad noise channel.

        The rate key matches the interaction type ('depolarizing', 'dephasing' or 'relaxation');
        a bare numeric value is wrapped into that key.
        """
        rate_key = self.interaction_type.value  # 'depolarizing' | 'dephasing' | 'relaxation'
        if self.subsystem1[0] != 'qubit':
            raise ValueError(
                self._with_context(
                    f"{rate_key} noise interaction must be on a qubit subsystem, but got {self.subsystem1}"
                )
            )
        elif isinstance(self.parameters, (int, float)):
            self.parameters = {rate_key: self.parameters}
        elif not isinstance(self.parameters, dict):
            raise TypeError(
                self._with_context(
                    f"Parameters for {rate_key} noise interaction must be a float or a dict with '{rate_key}' key"
                )
            )
        elif rate_key not in self.parameters:
            raise ValueError(
                self._with_context(
                    f"Parameters for {rate_key} noise interaction must include '{rate_key}' key for the noise rate"
                )
            )
        elif not isinstance(self.parameters[rate_key], (int, float)):
            raise TypeError(
                self._with_context(f"{rate_key} noise rate must be a numeric value, float or int")
            )

        self.parameters[rate_key] = float(self.parameters[rate_key])

        if self.parameters[rate_key] < 0:
            raise ValueError(
                self._with_context(
                    f"{rate_key} noise rate must be >= 0, got {self.parameters[rate_key]}"
                )
            )
        elif self.parameters[rate_key] == 0:
            warnings.warn(
                self._with_context(
                    f"{rate_key} noise rate is zero. This means no {rate_key} noise is applied, "
                    "which may be intentional but should be double-checked."
                ),
                UserWarning,
            )

    def _validate_custom_hamiltonian(self):
        """Validate parameters for custom Hamiltonian interactions."""
        if self.custom_matrix is None:
            raise ValueError(
                self._with_context(
                    "custom_matrix must be provided for custom Hamiltonian interactions"
                )
            )
        elif not isinstance(self.custom_matrix, qt.Qobj):
            raise TypeError(
                self._with_context(
                    "custom_matrix must be a qutip.Qobj instance representing the interaction Hamiltonian"
                )
            )
        if self.custom_matrix.isherm == False:
            raise ValueError(
                self._with_context(
                    "custom_matrix for custom Hamiltonian interactions must be Hermitian"
                )
            )
        if self.custom_matrix.dims[0] != self.custom_matrix.dims[1]:
            raise ValueError(
                self._with_context(
                    f"custom_matrix for custom Hamiltonian interactions must have equal dimensions for both axes, got dims: {self.custom_matrix.dims}"
                )
            )

        if self.time_modulation is not None and self.parameters is not None and not isinstance(self.parameters, dict):
            raise TypeError(
                self._with_context(
                    "Parameters for time-modulated custom Hamiltonian interaction must be provided as a dict"
                )
            )
    
    def _validate_custom_lindblad(self):
        """Validate parameters for custom Lindblad interactions."""
        if self.custom_matrix is None:
            raise ValueError(
                self._with_context(
                    "custom_matrix must be provided for custom Lindblad interactions"
                )
            )
        elif not isinstance(self.custom_matrix, qt.Qobj):
            raise TypeError(
                self._with_context(
                    "custom_matrix must be a qutip.Qobj instance representing the Lindblad operator"
                )
            )
        if self.custom_matrix.dims[0] != self.custom_matrix.dims[1]:
            raise ValueError(
                self._with_context(
                    f"custom_matrix for custom Lindblad interactions must have equal dimensions for both axes, got dims: {self.custom_matrix.dims}"
                )
            )

        if self.time_modulation is not None and self.parameters is not None and not isinstance(self.parameters, dict):
            raise TypeError(
                self._with_context(
                    "Parameters for time-modulated custom Lindblad interaction must be provided as a dict"
                )
            )
        

        

    def copy(self) -> "Interaction":
        """Return a copy with independent ``parameters`` storage.

        Returns:
            - ``copy`` (Interaction): New Interaction with deep-copied ``parameters`` and ``custom_matrix``.
        """
        new = Interaction(
            subsystem1=self.subsystem1,
            subsystem2=self.subsystem2,
            interaction_type=self.interaction_type,
            parameters=copy.deepcopy(self.parameters),
            time_modulation=self.time_modulation,
            custom_matrix=copy.deepcopy(self.custom_matrix) if self.custom_matrix is not None else None,
        )
        new.generated_noise = self.generated_noise  # preserve so noise stays copy-idempotent
        return new

    def _interaction_context(self, include_parameters: bool = False) -> str:
        """Return a compact interaction label like 'dispersive(cavity1,qubit3)'.

        A subsystem-less (full-space) custom interaction is labelled '<type>(full)'.

        Args:
            ``include_parameters`` (bool): if True, append the parameter values and whether time
                modulation is on, e.g. "dispersive(cavity1,qubit3): {'chi': 0.5}, time_modulation:
                off" (default: False).

        Returns:
            - ``label`` (str): Interaction type and its subsystem(s), optionally with parameters
              and time-modulation state.
        """
        if self.subsystem1 is None:
            label = f"{self.interaction_type.value}(full)"
        else:
            parts = [f"{self.subsystem1[0]}{self.subsystem1[1]}"]
            if self.subsystem2 is not None:
                parts.append(f"{self.subsystem2[0]}{self.subsystem2[1]}")
            label = f"{self.interaction_type.value}({','.join(parts)})"

        if include_parameters:
            if self.parameters:
                label += f": {self.parameters}"
            label += f", time_modulation: {'on' if self.time_modulation is not None else 'off'}"
        return label

    def _with_context(self, message: str) -> str:
        """Prefix a diagnostic message with the interaction context label.

        Args:
            ``message`` (str): Diagnostic message to prefix.

        Returns:
            - ``labelled`` (str): Message prefixed with the interaction context.
        """
        return f"{self._interaction_context()}: {message}"


@dataclass
class PhysicalModel:
    """
    Physical model dimensions and interactions for the quantum system.

    Attributes:
        perturbation_type: str
        'transient' (event localized in time; measurements accumulate) or 'persistent'
        (always-present perturbation; count-invariant detection rate).
        n_cavities: Number of resonator cavities (typically 1 for single-mode systems)
        n_fields: Number of input field modes
        n_qubits: Number of qubits in the system
        cavity_levels: Number of levels for cavity modes (cavity truncation level)
        field_levels: Number of levels for input field modes (field truncation level)
        qubit_levels: Number of levels for qubits (typically 2 for two-level systems)
        interactions: List of interactions between subsystems (e.g., qubit-qubit, cavity-field, qubit-cavity etc.)
            Only interactions common to all configurations should be included here.
            Interactions for specific configurations should be specified in the SystemConfiguration class.
    """

    perturbation_type: str # Type of perturbation: transient or persistent
    n_cavities: int = 1  # Number of resonator cavities
    n_fields: int = 1  # Number of input field modes
    n_qubits: int = 1  # Number of qubits
    cavity_levels: int = 2  # Cavity truncation level
    field_levels: int = 2  # Field mode truncation level
    qubit_levels: int = 2  # Qubit truncation level
    interactions: Optional[List[Interaction]] = None  # Subsystems' interactions

    def __post_init__(self):
        """Convert levels to list format if necessary and set default interactions."""
        
        # Normalizing levels to list format for consistency
        self.cavity_levels = self._normalize_levels(self.cavity_levels, self.n_cavities, "cavity")
        self.field_levels = self._normalize_levels(self.field_levels, self.n_fields, "field")
        self.qubit_levels = self._normalize_levels(self.qubit_levels, self.n_qubits, "qubit")

        # Set empty list of interactions if None
        if self.interactions is None:
            self.interactions = []

        if self.n_qubits < 1:
            raise ValueError("There must be at least one qubit in the system (n_qubits >= 1)")

        # Validate interactions
        for interaction in self.interactions:
            if not isinstance(interaction, Interaction):
                raise TypeError("All interactions must be Interaction instances")
            # Check that interactions are between valid subsystems
            for subsystem in [interaction.subsystem1, interaction.subsystem2]:
                
                if subsystem is not None and subsystem[0] == 'cavity' and subsystem[1] >= self.n_cavities:
                    raise ValueError(
                        f"Interaction involves cavity {subsystem[1]}, but only {self.n_cavities} cavities in system"
                    )
                if subsystem is not None and subsystem[0] == 'field' and subsystem[1] >= self.n_fields:
                    raise ValueError(
                        f"Interaction involves field mode {subsystem[1]}, but only {self.n_fields} field modes in system"
                    )
                if subsystem is not None and subsystem[0] == 'qubit' and subsystem[1] >= self.n_qubits:
                    raise ValueError(
                        f"Interaction involves qubit {subsystem[1]}, but only {self.n_qubits} qubits in system"
                    )

            if interaction.interaction_type in (InteractionType.CUSTOM_HAMILTONIAN, InteractionType.CUSTOM_LINDBLAD):
                kind = "Hamiltonian" if interaction.interaction_type == InteractionType.CUSTOM_HAMILTONIAN else "Lindblad"
                shape = tuple(interaction.custom_matrix.shape)
                if interaction.subsystem1 is None:
                    # Full-space custom operator: its size must match the total composite Hilbert
                    # dimension. It is used directly (no embedding); the builder only (re)assigns
                    # the per-subsystem dims metadata.
                    total = math.prod(list(self.cavity_levels) + list(self.field_levels) + list(self.qubit_levels))
                    if shape != (total, total):
                        raise ValueError(
                            f"Full-space custom {kind} matrix for {interaction._interaction_context()} has size {shape}, "
                            f"but the full composite space requires a ({total}, {total}) operator."
                        )
                else:
                    # The custom operator acts only on the involved subsystem(s). We validate
                    # its overall matrix size against the product of those subsystem
                    # dimensions; the per-subsystem dims metadata and the embedding into the
                    # full composite space are (re)assigned later by quantum_utils.embed_operator.
                    #
                    # Interaction.__post_init__ has already permuted the matrix legs into the
                    # stored (canonical) (subsystem1, subsystem2) order, so embed_operator can
                    # place them at ascending composite positions.
                    subsystem_dims = [self._subsystem_dimension(interaction.subsystem1)]
                    if interaction.subsystem2 is not None:
                        subsystem_dims.append(self._subsystem_dimension(interaction.subsystem2))
                    expected_size = math.prod(subsystem_dims)

                    if shape != (expected_size, expected_size):
                        raise ValueError(
                            f"Custom {kind} matrix for {interaction._interaction_context()} has size {shape}, "
                            f"but the involved subsystem(s) require a ({expected_size}, {expected_size}) operator "
                            f"(per-subsystem dimensions {subsystem_dims})."
                        )
                    

        
        # Forbid duplicate interactions: two non-custom interactions with the same type
        # and the same (subsystem1, subsystem2) pair are rejected even if their parameters
        # differ. Custom Hamiltonian/Lindblad terms are exempt (they may legitimately repeat).
        duplicate_check = [((int1.interaction_type == int2.interaction_type) and \
                            (int1.subsystem1 == int2.subsystem1) and \
                            (int1.subsystem2 == int2.subsystem2)) \
                                for i, int1 in enumerate(self.interactions) for int2 in self.interactions[i+1:] \
                                    if not (int1.interaction_type in [InteractionType.CUSTOM_HAMILTONIAN, InteractionType.CUSTOM_LINDBLAD])]
        interaction_list = [int1._interaction_context() for i, int1 in enumerate(self.interactions) for int2 in self.interactions[i+1:] \
                                    if not (int1.interaction_type in [InteractionType.CUSTOM_HAMILTONIAN, InteractionType.CUSTOM_LINDBLAD])]

        if any(duplicate_check):
            duplicates = [int_summary for int_summary, check in zip(interaction_list, duplicate_check) if check]
            raise ValueError(f"PhysicalModel interactions must be unique, interactions between the same two subsystems and of the same kind cannot be repeated (even if with different parameters).\n\
                             Found interactions: {duplicates}")

    def _normalize_levels(self, levels: Union[int, List[int]], count: int, label: str) -> List[int]:
        """Normalize subsystem levels to a per-subsystem list.

        Args:
            ``levels`` (Union[int, List[int]]): Shared level, or one level per subsystem.
            ``count`` (int): Number of subsystems of this kind.
            ``label`` (str): Subsystem kind ('cavity', 'field' or 'qubit'), used in error messages.

        Returns:
            - ``levels`` (List[int]): One level per subsystem (empty list when ``count`` is 0).
        """
        if label == 'cavity':
            plural = 'cavities'
        else:
            plural = label + 's'

        if count == 0:
            warnings.warn(f"n_{plural} is set to 0. Setting {label}_levels to an empty list.", UserWarning)
            return []
        elif isinstance(levels, int):
            return [levels] * count
        elif isinstance(levels, list):
            if len(levels) != count:
                raise ValueError(
                    f"{label.capitalize()} levels list length ({len(levels)}) must match n_{plural} ({count})"
                )
            return levels
        else:
            raise TypeError(f"{label.capitalize()} levels must be an integer or a list of integers")

    def _subsystem_dimension(self, subsystem: Tuple[str, int]) -> int:
        """Return the Hilbert-space dimension (truncation level) of a subsystem.

        Args:
            ``subsystem`` (Tuple[str, int]): Subsystem as (type, index).

        Returns:
            - ``dimension`` (int): Truncation level of that subsystem.
        """
        stype, idx = subsystem
        if stype == 'cavity':
            return self.cavity_levels[idx]
        if stype == 'field':
            return self.field_levels[idx]
        if stype == 'qubit':
            return self.qubit_levels[idx]
        raise ValueError(f"Unknown subsystem type: {stype}")


    def copy(self, **updates) -> "PhysicalModel":
        """Create a copy of PhysicalModel with optional parameter updates.

        Args:
            **updates (Any): Attributes to override. Valid keys: perturbation_type,
                n_cavities, n_fields, n_qubits, cavity_levels, field_levels, qubit_levels, interactions.

        Returns:
            - copy (PhysicalModel): New instance with the updates applied.

        Example:
            >>> original = PhysicalModel(n_cavities=2, n_fields=2, n_qubits=2)
            >>> modified = original.copy(n_cavities=3)
            >>> modified.n_cavities
            3
            >>> modified.n_fields
            2
        """
        # Start with current values
        params = {
            "perturbation_type": self.perturbation_type,
            "n_cavities": self.n_cavities,
            "n_fields": self.n_fields,
            "n_qubits": self.n_qubits,
            "cavity_levels": (
                self.cavity_levels.copy()
                if isinstance(self.cavity_levels, list)
                else self.cavity_levels
            ),
            "qubit_levels": (
                self.qubit_levels.copy()
                if isinstance(self.qubit_levels, list)
                else self.qubit_levels
            ),
            "field_levels": (
                self.field_levels.copy()
                if isinstance(self.field_levels, list)
                else self.field_levels
            ),
            "interactions": (
                [interaction.copy() for interaction in self.interactions]
            ),
        }

        # Apply updates
        params.update(updates)

        return PhysicalModel(**params)


@dataclass
class TimeProtocol:
    """
    Time protocol configuration: simulation start and measurement timing.

    Two modes of operation:
    1. Mode A (explicit): give ``measurement_times`` (the measured times only; the unmeasured
       simulation start is ``t_simulation_start``).
    2. Mode B (interval): give ``n_measurements`` and ``time_interval`` (measurements run from
       ``t_simulation_start``).

    All time parameters are stored and used as absolute times (no normalization).

    Attributes:
        t_simulation_start: First, unmeasured, simulation timestamp (absolute time). Required.
        measurement_times: Mode A explicit list of measured times (absolute, ascending).
        n_measurements: Mode B number of measurements.
        time_interval: Mode B spacing between measurements (absolute time).
        random_measurements_offset: Collective per-realization timing shift added to all measurements.
                 True: uniform(-Δt, 0) with Δt = time_interval (interval mode only; raises with explicit
                 measurement_times, where the interval is ill-defined). A positive number w: uniform(-w, 0).
                 A callable f() -> float: custom shift sampled once per realization. False disables it.
        per_measurement_jitter: Independent per-measurement timing jitter: a Gaussian std (float) or a
                 sampler callable f() / f(t). None disables it.
        window_start: Start of the measurement window for the double-sigmoid weight (absolute time).
                     Must be given together with window_end (or both omitted for uniform weights).
        window_end: End of the measurement window (absolute time, must be > window_start).
        window_slope: Slope of the sigmoid edges; defaults to 30.0 / (window_end - window_start).
        noisy_simulation_start: apply the timing noise (offset + jitter) also to the simulation start.
                 On by default.
    """

    t_simulation_start: float                              # simulation start (required, absolute time)
    measurement_times: Optional[List[float]] = None        # mode A: explicit times
    n_measurements: Optional[int] = None                   # mode B: count ...
    time_interval: Optional[float] = None                  # ... and spacing
    random_measurements_offset: Union[bool, int, float, Callable] = False  # True: uniform(-Δt, 0); number w: uniform(-w, 0); f() -> float
    per_measurement_jitter: Optional[Union[Callable, float]] = None  # Gaussian std (float) or sampler f() / f(t)
    window_start: Optional[float] = None
    window_end: Optional[float] = None
    window_slope: Optional[float] = None
    noisy_simulation_start: bool = True

    def __post_init__(self):
        """Validate the measurement-time spec (mode A or B), uncertainty and window settings."""
        explicit = self.measurement_times is not None
        interval = self.n_measurements is not None or self.time_interval is not None
        if explicit == interval:
            raise ValueError("Specify either measurement_times, or n_measurements with time_interval (not both)")
        if explicit:
            if len(self.measurement_times) < 1:
                raise ValueError("measurement_times must contain at least one measurement")
            if list(self.measurement_times) != sorted(self.measurement_times):
                raise ValueError("measurement_times must be in ascending order")
        else:
            if self.n_measurements is None or self.time_interval is None:
                raise ValueError("interval mode requires both n_measurements and time_interval")
            if self.n_measurements < 1:
                raise ValueError("n_measurements must be >= 1")
            if self.time_interval <= 0:
                raise ValueError("time_interval must be positive")

        # Collective offset: bool, a positive width, or a callable f() -> float. True needs a
        # well-defined interval, so it is only allowed in interval mode.
        offset = self.random_measurements_offset
        if offset is False or offset is None:
            pass
        elif offset is True:
            if explicit:
                raise ValueError("random_measurements_offset=True requires interval mode (n_measurements + "
                                 "time_interval); with explicit measurement_times give a width or a callable")
        elif callable(offset):
            if len(inspect.signature(offset).parameters) != 0:
                raise TypeError("random_measurements_offset callable must take no argument (f())")
            try:
                float(offset())
            except Exception as e:
                raise TypeError("random_measurements_offset callable must be callable as f() and return a float") from e
        elif isinstance(offset, (int, float)):
            if offset <= 0:
                raise ValueError("random_measurements_offset width must be positive")
            self.random_measurements_offset = float(offset)
        else:
            raise TypeError("random_measurements_offset must be a bool, a positive number (distribution "
                            "width), or a callable f() -> float")

        jitter = self.per_measurement_jitter
        if jitter is not None:
            if isinstance(jitter, (int, float)):
                jitter = self.per_measurement_jitter = float(jitter)
            if callable(jitter):
                n_args = len(inspect.signature(jitter).parameters)
                if n_args not in (0, 1):
                    raise TypeError("per_measurement_jitter callable must take no argument (f()) or a single time (f(t))")
                try:
                    float(jitter() if n_args == 0 else jitter(0.0))
                except Exception as e:
                    raise TypeError("per_measurement_jitter callable must be callable as f()/f(t) and return a float") from e
            elif not isinstance(jitter, float) or jitter < 0:
                raise TypeError("per_measurement_jitter must be a non-negative float (Gaussian std) or a callable f()/f(t)")

        # Measurement window: double-sigmoid weight over measurement time. window_start/window_end
        # must be given together (or not at all -> uniform weights); window_slope defaults to
        # _WINDOW_SLOPE_FACTOR / (window_end - window_start).
        if (self.window_start is None) != (self.window_end is None):
            raise ValueError("window_start and window_end must be given together (or both omitted)")
        if self.window_start is not None and self.window_end <= self.window_start:
            raise ValueError("window_end must be greater than window_start")
        if self.window_slope is not None and self.window_slope <= 0:
            raise ValueError("window_slope must be positive")

        # Compile per_measurement_jitter into a single sampler + a time-dependence flag.
        self._build_jitter_sampler()

    def _build_jitter_sampler(self) -> None:
        """Compile ``per_measurement_jitter`` into a single sampler used by :meth:`sample_jitter`.

        Dispatches on the jitter kind once, at construction, instead of on every draw. Sets:
        - ``_jitter_sampler``: fn(batch_size, times, shift) -> (batch_size, len(times)) offsets, where
          ``shift`` is a per-realization collective time shift (length batch_size). No jitter gives an
          all-zeros draw, so the sampler is always callable. The float case stays a vectorized numpy
          draw; f()/f(t) are looped element-wise. Only a time-dependent f(t) uses ``shift``: it is
          evaluated at the actually-shifted measurement time f(t + shift), so the collective offset
          propagates into the jitter. The float/f() draws ignore ``shift`` (they don't depend on time).
        - ``_jitter_time_dependent`` (bool): True only for an f(t) jitter. Read by sweeps via
          :attr:`jitter_is_time_dependent` (a pre-drawn noise batch is invalid under swept times).
        """
        jitter = self.per_measurement_jitter
        if jitter is None or (isinstance(jitter, float) and jitter == 0):
            self._jitter_sampler = lambda batch_size, times, shift: np.zeros((batch_size, len(times)))
            self._jitter_time_dependent = False
        elif isinstance(jitter, float):
            self._jitter_sampler = lambda batch_size, times, shift: np.random.normal(0.0, jitter, (batch_size, len(times)))
            self._jitter_time_dependent = False
        elif len(inspect.signature(jitter).parameters) == 0:
            self._jitter_sampler = lambda batch_size, times, shift: np.array(
                [[float(jitter()) for _ in range(len(times))] for _ in range(batch_size)]
            )
            self._jitter_time_dependent = False
        else:
            self._jitter_sampler = lambda batch_size, times, shift: np.array(
                [[float(jitter(float(t) + float(s))) for t in times] for s in shift]
            )
            self._jitter_time_dependent = True

    @property
    def jitter_is_time_dependent(self) -> bool:
        """Whether ``per_measurement_jitter`` varies with measurement time (an f(t) callable).

        Returns:
            - ``time_dependent`` (bool): True only for a time-dependent jitter.
        """
        return self._jitter_time_dependent

    def copy(self) -> "TimeProtocol":
        """Return an independent copy (the mutable measurement-time list is deep-copied).

        Returns:
            - ``protocol`` (TimeProtocol): New instance with the same configuration.
        """
        return copy.deepcopy(self)

    @property
    def resolved_window_slope(self) -> Optional[float]:
        """Slope of the double-sigmoid window: ``window_slope`` if set, else auto from width.

        Returns:
            - ``slope`` (Optional[float]): Slope value, or None when no window is configured.
        """
        if self.window_start is None:
            return None
        if self.window_slope is not None:
            return self.window_slope
        return _WINDOW_SLOPE_FACTOR / (self.window_end - self.window_start)

    def measurement_weights(self, times):
        """Per-measurement window weight w(t) in [0, 1] at the given ``times``.

        Double sigmoid w(t) = sigmoid(s(t - a)) * sigmoid(-s(t - b)) with a=``window_start``,
        b=``window_end``, s=``resolved_window_slope``: ~1 inside [a, b], ->0 outside. All ones when
        no window is configured.

        Args:
            ``times`` (array-like): Times at which to evaluate the weight.

        Returns:
            - ``weights`` (jnp.ndarray): Weight in [0, 1] for each time.
        """
        t = jnp.asarray(times, float)
        if self.window_start is None:
            return jnp.ones_like(t)
        a, b, s = self.window_start, self.window_end, self.resolved_window_slope
        return 1.0 / (1.0 + jnp.exp(-s * (t - a))) * 1.0 / (1.0 + jnp.exp(s * (t - b)))

    def sample_jitter(self, batch_size: int, times: np.ndarray,
                      shift: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample per-measurement timing offsets from ``per_measurement_jitter``.

        Delegates to the sampler compiled at construction (see :meth:`_build_jitter_sampler`),
        so no jitter yields an all-zeros draw. Returns a (``batch_size``, M) array for the M given
        times only; the simulation-start jitter (when ``noisy_simulation_start`` is True) is handled
        separately in :meth:`ExperimentalParameters.get_measurement_uncertainties`.

        ``shift`` is a per-realization collective time offset added to the times before a
        time-dependent f(t) jitter is evaluated (so f sees the actually-shifted time). It defaults
        to zeros and is ignored by the float/f() samplers, which do not depend on time.

        Args:
            ``batch_size`` (int): Number of independent realizations.
            ``times`` (np.ndarray): Nominal measurement times (length M).
            ``shift`` (Optional[np.ndarray]): Per-realization collective time shift (length batch_size);
                None means no shift.

        Returns:
            - ``offsets`` (np.ndarray): Shape (``batch_size``, M) timing offsets.
        """
        shift = np.zeros(batch_size) if shift is None else np.asarray(shift, dtype=float)
        return self._jitter_sampler(batch_size, times, shift)

    def cap_jitter(self, jit, interval: Optional[float] = None, xp=np):
        """Clamp per-measurement jitter to ±interval/2 so the measurement train stays ordered.

        Interval mode only: a jitter larger than half the spacing could push a measurement past its
        neighbour, giving non-monotonic timestamps the ODE solver rejects. Clamping each offset to just
        inside ±interval/2 keeps the train strictly increasing (the tiny 1e-6 shave stops neighbours
        clipped to opposite bounds from landing on the exact same time). Explicit mode (no single
        interval) is returned unchanged.

        Args:
            ``jit`` (array): Jitter offsets to clamp.
            ``interval`` (Optional[float]): Spacing to cap against; defaults to ``time_interval``.
            ``xp``: Array module (np or jnp).

        Returns:
            - ``jit`` (array): Clamped jitter (unchanged in explicit mode).
        """
        if self.time_interval is None:
            return jit
        half = 0.5 * (1.0 - 1e-6) * (float(self.time_interval) if interval is None else interval)
        return xp.clip(jit, -half, half)

    def sample_collective_offset(self, batch_size: int) -> Optional[np.ndarray]:
        """Draw the per-realization collective time shift, or None when disabled.

        Dispatches on ``random_measurements_offset``: True -> uniform(-time_interval, 0), tracking the
        current interval spacing; a positive number w -> uniform(-w, 0) with a fixed absolute width; a
        callable f() -> float sampled once per realization. False/None returns None.

        Args:
            ``batch_size`` (int): Number of independent realizations.

        Returns:
            - ``shift`` (Optional[np.ndarray]): Shape (``batch_size``,) offsets, or None if disabled.
        """
        off = self.random_measurements_offset
        if off is False or off is None:
            return None
        if callable(off):
            return np.array([float(off()) for _ in range(batch_size)])
        width = float(self.time_interval) if off is True else float(off)  # True tracks time_interval; numeric fixed
        return np.random.uniform(-width, 0.0, size=(batch_size,))

    def resolve_measurement_times(self) -> np.ndarray:
        """Resolve the measured-only times (excludes the unmeasured ``t_simulation_start``).

        Mode A returns the explicit ``measurement_times``; mode B generates
        ``t_simulation_start`` + k*``time_interval`` for k=1..``n_measurements``.

        Returns:
            - ``times`` (np.ndarray): 1-D array of shape (M,) of absolute measured times.
        """
        t_start = float(self.t_simulation_start)
        if self.measurement_times is not None:      # mode A
            times = [float(t) for t in self.measurement_times]
        else:                                        # mode B
            times = [t_start + k * self.time_interval for k in range(1, self.n_measurements + 1)]
        if times and times[0] < t_start:
            raise ValueError(f"first measurement ({times[0]:.3g}) is before t_simulation_start ({t_start:.3g})")
        return np.array(times)

    @property
    def timestamps(self) -> np.ndarray:
        """Full solver sequence [``t_simulation_start``, *``measurement_times``] (start is unmeasured).

        Returns:
            - ``timestamps`` (np.ndarray): 1-D array of shape (M+1,).
        """
        return np.concatenate([[float(self.t_simulation_start)], self.resolve_measurement_times()])

    def _before_start_correction(self, base_v, shift, meas_jit, start_jit, t0, interval, eps, xp=np):
        """Before-start guard: bump measurements so none precede the (noisy) start; round onto the grid.

        Split out so the sweep can recompute it at a swept interval (the collective offset scales, so the
        guard must be re-derived rather than reused). ``xp`` is numpy or jax.numpy.

        Args:
            ``base_v`` (array): Measured times (length M) for the target interval.
            ``shift`` (array): Per-realization collective offset (batch,), already scaled.
            ``meas_jit`` (array): Per-measurement jitter (batch, M).
            ``start_jit`` (array): Start jitter (batch,).
            ``t0`` (float): Simulation start time.
            ``interval`` (float): Current measurement spacing (for grid rounding).
            ``eps`` (float): Small lower-bound tolerance.
            ``xp``: Array module (np or jnp).

        Returns:
            - ``correction`` (array): Shape (batch,) forward bump for the measured columns.
        """
        start_noise = (shift + start_jit) if self.noisy_simulation_start else xp.zeros_like(shift)
        out = shift[:, None] + meas_jit
        corr = xp.maximum(0.0, (t0 + start_noise) + eps - xp.min(base_v[None, :] + out, axis=1))
        if self.time_interval is not None:
            corr = xp.ceil(corr / interval) * interval
        return corr

    def sample_noise_components(self, batch_size: int = 1, offset: bool = True, jitter: bool = True):
        """Draw the timing-noise pieces: the collective shift, the before-start correction and the jitter.

        These are the three additive components :meth:`combine_noise` sums into the (batch, M+1)
        uncertainty. The jitter vector is (batch, M+1) with the start jitter in column 0. The correction
        is computed at this protocol's own interval.

        Args:
            ``batch_size`` (int): Number of independent realizations.
            ``offset`` (bool): Include the collective offset (see ``random_measurements_offset``).
            ``jitter`` (bool): Include per-measurement jitter from ``per_measurement_jitter``.

        Returns:
            - ``shift`` (np.ndarray): Collective offset (batch,), zeros when disabled.
            - ``correction`` (np.ndarray): Before-start bump (batch,).
            - ``jitter_vec`` (np.ndarray): Per-timestamp jitter (batch, M+1), column 0 is the start jitter.
        """
        base = self.resolve_measurement_times()
        t0 = float(self.t_simulation_start)
        shift = self.sample_collective_offset(batch_size) if offset else None
        shift = np.zeros(batch_size) if shift is None else shift
        # shift is folded into the time argument so a time-dependent f(t) jitter sees the shifted time.
        # Cap jitter at ±interval/2 (interval mode) so measurements keep their order.
        meas_jit = self.cap_jitter(self.sample_jitter(batch_size, base, shift)) if jitter \
            else np.zeros((batch_size, base.size))
        start_jit = (self.cap_jitter(self.sample_jitter(batch_size, np.array([t0]), shift)[:, 0])
                     if (jitter and self.noisy_simulation_start) else np.zeros(batch_size))
        # eps is a small guard scaled to the measurement grid (interval spacing, else the first gap).
        if self.time_interval is not None:
            interval = float(self.time_interval)
            eps = interval * 1e-3
        else:
            interval = 0.0
            eps = ((float(base[1] - base[0]) if base.size >= 2 else 0.0) or 1.0) * 1e-3
        correction = self._before_start_correction(base, shift, meas_jit, start_jit, t0, interval, eps)
        jitter_vec = np.concatenate([start_jit[:, None], meas_jit], axis=1)
        return shift, correction, jitter_vec

    def combine_noise(self, shift, correction, jitter_vec, xp=np) -> np.ndarray:
        """Combine the shift, correction and jitter into the (batch, M+1) timing uncertainty.

        Column 0 is the start offset (the shift is applied there only when ``noisy_simulation_start``);
        columns 1..M add shift + jitter + correction. ``xp`` is numpy or jax.numpy.

        Args:
            ``shift`` (array): Collective offset (batch,).
            ``correction`` (array): Before-start bump (batch,).
            ``jitter_vec`` (array): Per-timestamp jitter (batch, M+1), column 0 is the start jitter.
            ``xp``: Array module (np or jnp).

        Returns:
            - ``uncertainties`` (array): Shape (batch, M+1) timing offsets.
        """
        start_shift = shift if self.noisy_simulation_start else xp.zeros_like(shift)
        start_col = start_shift + jitter_vec[:, 0]
        meas = shift[:, None] + jitter_vec[:, 1:] + correction[:, None]
        return xp.concatenate([start_col[:, None], meas], axis=1)

    def get_measurement_uncertainties(self, batch_size: int = 1, offset: bool = True, jitter: bool = True) -> np.ndarray:
        """Sample stochastic timing uncertainties to add to ``timestamps``.

        Returns a (``batch_size``, M+1) array: column 0 is the start-time offset (zero unless
        ``noisy_simulation_start`` is True), columns 1..M are per-measurement offsets. Thin wrapper over
        :meth:`sample_noise_components` + :meth:`combine_noise`.

        Args:
            ``batch_size`` (int): Number of independent realizations.
            ``offset`` (bool): Apply the collective offset (see ``random_measurements_offset``).
            ``jitter`` (bool): Apply per-measurement jitter from ``per_measurement_jitter``.

        Returns:
            - ``uncertainties`` (np.ndarray): Shape (``batch_size``, M+1) timing offsets.
        """
        shift, correction, jitter_vec = self.sample_noise_components(batch_size, offset, jitter)
        return self.combine_noise(shift, correction, jitter_vec)

    def get_timestamps(self, batch_size: int = 1, offset: bool = True, jitter: bool = True) -> np.ndarray:
        """Return ``timestamps`` with stochastic uncertainties applied.

        Shape (``batch_size``, M+1). ``offset=jitter=False`` gives deterministic timestamps.
        Uncertainty (including before-start guard) comes from ``get_measurement_uncertainties``.

        Args:
            ``batch_size`` (int): Number of independent realizations.
            ``offset`` (bool): Apply the collective offset.
            ``jitter`` (bool): Apply per-measurement jitter.

        Returns:
            - ``timestamps`` (np.ndarray): Shape (``batch_size``, M+1).
        """
        return self.timestamps[None, :] + self.get_measurement_uncertainties(batch_size, offset, jitter)


@dataclass
class SubsystemState:
    """
    State for a single subsystem.

    Attributes:
        state_type: Type of initial state (VACUUM, FOCK, COHERENT, THERMAL, CUSTOM)
        parameters: Dictionary of parameters for the initial state
    """

    state_type: State = State.VACUUM
    parameters: Dict[str, Any] = None  # Parameters depend on state_type, e.g., {'n': 1} for Fock state

    def __post_init__(self):
        """Validate and normalize the parameters required by the chosen state_type."""
        if self.state_type == State.FOCK:
            if self.parameters is None or "n" not in self.parameters:
                raise ValueError("FOCK state requires 'n' parameter for photon number")
            n = self.parameters["n"]
            if not isinstance(n, int) or n < 0:
                raise ValueError("FOCK state 'n' parameter must be a non-negative integer")
            
        elif self.state_type == State.COHERENT:
            if self.parameters is None or "alpha" not in self.parameters:
                raise ValueError("COHERENT state requires 'alpha' parameter for coherent amplitude")
            alpha = self.parameters["alpha"]
            if not isinstance(alpha, (int, float, complex)):
                raise ValueError("COHERENT state 'alpha' parameter must be a numeric value (real or complex)")
            self.parameters["alpha"] = complex(alpha)  # Ensure alpha is stored as a complex number

        elif self.state_type == State.THERMAL:
            if self.parameters is None or "n_avg" not in self.parameters:
                raise ValueError("THERMAL state requires 'n_avg' parameter for mean photon number")
            n_avg = self.parameters["n_avg"]
            if not isinstance(n_avg, (int, float)) or n_avg < 0:
                raise ValueError("THERMAL state 'n_avg' parameter must be a non-negative number")
            self.parameters["n_avg"] = float(n_avg)  # Ensure n_avg is stored as a float

        elif self.state_type == State.CUSTOM:
            # A custom single-mode state is a pure state given by its Fock-basis amplitudes:
            # |ψ⟩ = Σ aₙ|n⟩, with parameters={'amplitudes': [a0, a1, ...]}.
            if self.parameters is None or "amplitudes" not in self.parameters:
                raise ValueError(
                    "CUSTOM state requires an 'amplitudes' parameter: a 1D sequence of Fock-basis "
                    "amplitudes [a0, a1, ...] defining the pure state |psi> = sum_n a_n |n>"
                )
            amplitudes = np.asarray(self.parameters["amplitudes"], dtype=complex).reshape(-1)
            if amplitudes.size == 0:
                raise ValueError("CUSTOM state 'amplitudes' must be a non-empty 1D sequence")
            norm = np.linalg.norm(amplitudes)
            if norm < 1e-12:
                raise ValueError("CUSTOM state 'amplitudes' must have non-zero norm")
            # Store the normalized amplitudes as a plain list of complex numbers.
            self.parameters["amplitudes"] = (amplitudes / norm).tolist()

    def copy(self) -> "SubsystemState":
        """Return a copy with independent ``parameters`` storage.

        Returns:
            - ``copy`` (SubsystemState): New instance with deep-copied ``parameters``.
        """
        parameters = copy.deepcopy(self.parameters) if self.parameters is not None else None
        return SubsystemState(state_type=self.state_type, parameters=parameters)

    # custom_amplitudes: Optional[Dict[Tuple[int, int, int], complex]] = None


@dataclass
class NoiseModel:
    """
    Noise model configuration.

    The qubit channels can be specified either as Lindblad rates (``relaxation``, ``dephasing``)
    or as coherence times (``t1``, ``t2``); when times are given the matching rates are derived
    at construction. Times are in the model's time units, i.e. the same axis as ``TimeProtocol``
    (divide a physical T1/T2 by the time unit, so a T1 in seconds becomes ``chi * T1`` with
    ``chi`` the anchoring angular frequency in rad/s).

    Attributes:
        relaxation: Relaxation rate. Can be a float (same for all qubits)
                   or a list of floats (individual rate per qubit).
        dephasing: Dephasing rate. Can be a float (same for all qubits)
                  or a list of floats (individual rate per qubit).
        depolarizing: Depolarization rate. Can be a float (same for all qubits)
                     or a list of floats (individual rate per qubit).
        t1: Energy relaxation time, as an alternative to ``relaxation``. Float or one per qubit.
            default None
        t2: Coherence time, as an alternative to ``dephasing``. Float or one per qubit.
            Bounded by ``t2 <= 2 * t1``; omitting ``t1`` treats relaxation as absent.
            default None
        custom_operators: Custom Lindblad operators
    """

    relaxation: Union[float, List[float]] = 0.0  # Relaxation rate
    dephasing: Union[float, List[float]] = 0.0  # Dephasing rate
    depolarizing: Union[float, List[float]] = 0.0  # Depolarization rate
    t1: Union[float, List[float]] = None # Longitudinal relaxation time
    t2: Union[float, List[float]] = None # Trasversal relaxation time
    custom_operators: Optional[List[Any]] = None  # Custom Lindblad operators

    def __post_init__(self) -> None:
        """Derive ``relaxation`` and ``dephasing`` from ``t1``/``t2`` when those are given.

        The collapse operators are ``sqrt(relaxation) * sigma_minus`` and
        ``sqrt(dephasing) * sigma_z``, so populations decay at ``relaxation`` while coherences
        decay at ``relaxation / 2 + 2 * dephasing``. Matching that to ``1/t1`` and ``1/t2`` gives
        ``relaxation = 1/t1`` and ``dephasing = (1/t2 - 1/(2*t1)) / 2``.

        Times broadcast against each other, and a scalar time gives a scalar rate. A missing
        ``t1`` means no relaxation, a missing ``t2`` no pure dephasing.

        Raises:
            ValueError: If a time is paired with a non-zero rate for the same channel, if a time
                is non-positive, if the ``t1``/``t2`` shapes do not broadcast, or if
                ``t2 > 2 * t1`` (which would need a negative dephasing rate).
        """
        # Reject a channel given twice, and unphysical times. None means "use the rate".
        for attr, rate_attr in (("t1", "relaxation"), ("t2", "dephasing")):
            time = getattr(self, attr)
            if time is None:
                continue
            rate = getattr(self, rate_attr)
            if np.any(np.asarray(rate, dtype=float) != 0.0):
                raise ValueError(
                    f"NoiseModel takes either {attr} or {rate_attr}, not both "
                    f"(got {attr}={time}, {rate_attr}={rate})"
                )
            if np.any(np.asarray(time, dtype=float) <= 0.0):
                raise ValueError(f"{attr} must be > 0, got {time}")

        # An infinite t1 stands in for an absent one: zeroes gamma_1 and its share of gamma_phi.
        t1 = np.asarray(np.inf if self.t1 is None else self.t1, dtype=float)

        # sigma_minus decays populations at its rate, so gamma_1 = 1/t1 goes in unchanged.
        if self.t1 is not None:
            self.relaxation = (1.0 / t1).tolist()

        # gamma_phi = 1/t2 - gamma_1/2 drops the relaxation share; sigma_z decays coherences at
        # twice its rate, hence the /2. Negative gamma_phi means t2 broke the 2*t1 ceiling.
        if self.t2 is not None:
            gamma_phi = 1.0 / np.asarray(self.t2, dtype=float) - 0.5 / t1
            if np.any(gamma_phi < 0.0):
                raise ValueError(f"t2 must be <= 2*t1, got t1={self.t1}, t2={self.t2}")
            self.dephasing = (gamma_phi / 2.0).tolist()

    def _normalize_noise_rates(self, n_qubits: int):
        """Normalize ``depolarizing``, ``dephasing``, ``relaxation`` to per-qubit lists.

        Args:
            ``n_qubits`` (int): Number of qubits from the ``PhysicalModel``.
        """
        for attr in ["depolarizing", "dephasing", "relaxation"]:
            value = getattr(self, attr)
            if isinstance(value, (int, float)):
                values = [float(value)] * n_qubits
            elif isinstance(value, list):
                if len(value) != n_qubits:
                    raise ValueError(
                        f"{attr} list length ({len(value)}) must match n_qubits ({n_qubits})"
                    )
                value_list = list(value)  # narrow type for pylint
                values = [float(v) for v in value_list]
            else:
                raise TypeError(f"{attr} must be a float or a list of floats")

            for i, rate in enumerate(values):
                if rate < 0:
                    raise ValueError(f"{attr} rate for qubit {i} must be >= 0, got {rate}")

            setattr(self, attr, values)

    def to_interactions(self, n_qubits: int) -> List["Interaction"]:
        """Expand the noise model into a list of interactions, built like any other interaction.

        Each non-zero per-qubit rate becomes an :class:`Interaction` of the matching noise type on
        that qubit (rate keys look like ``...<type>(qubitI)__<type>``), so noise rates are assembled
        and swept through the exact same pipeline as every other interaction parameter. Zero rates are
        skipped (no collapse operator and nothing to sweep). Each custom collapse operator becomes a
        subsystem-less (full-space) ``custom_lindblad`` interaction, so they too flow through the
        interaction pipeline instead of a separate noise route.

        Args:
            ``n_qubits`` (int): Number of qubits in the system.

        Returns:
            - ``interactions`` (List[Interaction]): One interaction per non-zero rate, plus one
              full-space ``custom_lindblad`` per custom collapse operator.
        """
        specs = [
            ("depolarizing", InteractionType.DEPOLARIZING),
            ("dephasing", InteractionType.DEPHASING),
            ("relaxation", InteractionType.RELAXATION),
        ]
        interactions: List["Interaction"] = []
        for attr, interaction_type in specs:
            rates = getattr(self, attr)
            rate_list = rates if isinstance(rates, list) else [rates] * n_qubits
            for i in range(n_qubits):
                rate = rate_list[i] if i < len(rate_list) else 0.0
                if rate != 0.0:
                    interactions.append(
                        Interaction(interaction_type, ("qubit", i), parameters={attr: float(rate)})
                    )
        for operator in (self.custom_operators or []):
            interactions.append(
                Interaction(InteractionType.CUSTOM_LINDBLAD, custom_matrix=operator)
            )
        for interaction in interactions:
            interaction.generated_noise = True  # owned by _absorb_noise_into_interactions
        return interactions

    def copy(self) -> "NoiseModel":
        """Return a copy with independent rate storage.

        Returns:
            - ``copy`` (NoiseModel): New instance with deep-copied rates and ``custom_operators``.
        """
        return NoiseModel(
            depolarizing=copy.deepcopy(self.depolarizing),
            dephasing=copy.deepcopy(self.dephasing),
            relaxation=copy.deepcopy(self.relaxation),
            custom_operators=copy.deepcopy(self.custom_operators),
        )


@dataclass
class SystemConfiguration:
    """
    System configuration composed of the initial state specification, noise model
    and configuration-specific interactions.

    The initial state of the {cavities} ⊗ {fields} subsystem is specified directly
    through this configuration. Any non explicited subsystem will be initialized in
    the vacuum state.

    Attributes:
        name: Unique name identifying this configuration
        init_cavity_states: Dict of cavity states, keyed by cavity index (0-based)
        init_field_states: Dict of input field states, keyed by field mode index (0-based)
        density_matrix: Optional density matrix for the {cavities} ⊗ {fields} subsystem.
            Overrides init_cavity_states and init_field_states when provided.
        noise_model: Optional configuration-specific noise model
        interactions: Optional configuration-specific interactions
    """

    name: str
    init_cavity_states: Optional[Dict[int, SubsystemState]] = field(default_factory=dict)
    init_field_states: Optional[Dict[int, SubsystemState]] = field(default_factory=dict)
    density_matrix: Optional[qt.Qobj] = None  # Overrides cavity/field states when provided
    noise_model: Optional[NoiseModel] = None
    interactions: Optional[List[Interaction]] = None
    is_ground: bool = False

    def __post_init__(self):
        """Validate configuration data, including the initial-state specification."""
        if not self.name:
            raise ValueError("System configuration must have a non-empty name")

        # ---- Initial state specification ----
        if self.init_cavity_states is not None:
            if len(list(self.init_cavity_states.keys())) != len(set(self.init_cavity_states.keys())):
                raise ValueError("Cavity states got duplicate indices. Each cavity accepts only a single state.")
            for index, state in self.init_cavity_states.items():
                if not isinstance(state, SubsystemState):
                    raise TypeError(f"Cavity state for cavity {index} must be a SubsystemState instance")
        if self.init_field_states is not None:
            if len(list(self.init_field_states.keys())) != len(set(self.init_field_states.keys())):
                raise ValueError("Field states got duplicate indices. Each field mode accepts only a single state.")
            for index, state in self.init_field_states.items():
                if not isinstance(state, SubsystemState):
                    raise TypeError(f"Field state for field mode {index} must be a SubsystemState instance")

        if self.density_matrix is not None:
            if (self.init_cavity_states and len(self.init_cavity_states) > 0) or (
                self.init_field_states and len(self.init_field_states) > 0
            ):
                warnings.warn(
                    "density_matrix is provided; init_cavity_states and init_field_states will be ignored.",
                    UserWarning,
                )
            if not isinstance(self.density_matrix, qt.Qobj):
                raise ValueError("density_matrix must be a Qobj representing the density matrix")
            if not self.density_matrix.isherm or (self.density_matrix.eigenenergies().min() < 0) or not np.isclose(self.density_matrix.tr(), 1.0):
                raise ValueError("density_matrix must be a valid density matrix (Hermitian, positive semidefinite, trace 1)")
        if self.init_cavity_states == {} and self.init_field_states == {} and self.density_matrix is None:
            warnings.warn("No subsystem state was initialized nor a density_matrix was provided. The initial state will be the vacuum state.", UserWarning)

        # ---- Noise model ----
        if self.noise_model is not None and not isinstance(self.noise_model, NoiseModel):
            raise TypeError("noise_model must be a NoiseModel instance or None")

        # ---- Configuration-specific interactions ----
        if self.interactions is not None:
            for interaction in self.interactions:
                if not isinstance(interaction, Interaction):
                    raise TypeError("All interactions must be Interaction instances")
                
            
            # Forbid duplicate interactions within this configuration (same rule as the
            # physical model): identical type and (subsystem1, subsystem2) pair, parameters aside.
            duplicate_check = [((int1.interaction_type == int2.interaction_type) and \
                                (int1.subsystem1 == int2.subsystem1) and \
                                (int1.subsystem2 == int2.subsystem2)) \
                                    for i, int1 in enumerate(self.interactions) for int2 in self.interactions[i+1:] \
                                    if not (int1.interaction_type in [InteractionType.CUSTOM_HAMILTONIAN, InteractionType.CUSTOM_LINDBLAD])]
            interaction_list = [int1._interaction_context() for i, int1 in enumerate(self.interactions) for int2 in self.interactions[i+1:] \
                                    if not (int1.interaction_type in [InteractionType.CUSTOM_HAMILTONIAN, InteractionType.CUSTOM_LINDBLAD])]

            if any(duplicate_check):
                duplicates = [int_summary for int_summary, check in zip(interaction_list, duplicate_check) if check]
                raise ValueError(f"SystemConfiguration interactions must be unique, interactions between the same two subsystems and of the same kind cannot be repeated (even if with different parameters).\n\
                                Found interactions: {duplicates}")
        else:
            self.interactions = []

    def copy(self) -> "SystemConfiguration":
        """Return a copy with independent nested state storage.

        Returns:
            - ``copy`` (SystemConfiguration): New instance with deep-copied states, ``noise_model``, and ``interactions``.
        """
        init_cavity_states = None
        if self.init_cavity_states is not None:
            init_cavity_states = {idx: state.copy() for idx, state in self.init_cavity_states.items()}
        init_field_states = None
        if self.init_field_states is not None:
            init_field_states = {idx: state.copy() for idx, state in self.init_field_states.items()}
        return SystemConfiguration(
            name=self.name,
            init_cavity_states=init_cavity_states,
            init_field_states=init_field_states,
            density_matrix=self.density_matrix.copy() if self.density_matrix is not None else None,
            noise_model=self.noise_model.copy() if self.noise_model is not None else None,
            interactions=[interaction.copy() for interaction in (self.interactions or [])],
            is_ground=self.is_ground,
        )
        


class ExperimentalParameters:
    """
    Complete experimental configuration for quantum sensing protocols.

    This class contains all the system parameters that define the physical
    quantum sensing setup including Hilbert space dimensions, interactions,
    measurement protocols, noise models, and initial state preparation.

    The parameters are organized into logical groups and provide validation
    and consistency checking for the experimental configuration.

    Args:
            physical_model: Physical model
            noise_model: Noise model
            time_protocol: Time protocol (simulation start + measurement timing)
            configuration_set: Set or list of system configurations to be simulated (e.g., different initial states, noise levels, interactions)
            random_seed: Random seed for reproducibility of uncertainty calculations
    """

    def __init__(
        self,
        physical_model: Optional[PhysicalModel] = None,
        noise_model: Optional[NoiseModel] = None,
        time_protocol: Optional[TimeProtocol] = None,
        configuration_set: Optional[Union[Set[SystemConfiguration], List[SystemConfiguration]]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize experimental parameters.
        """
        self.physical_model = physical_model or PhysicalModel()
        self.noise_model = noise_model or NoiseModel()
        self.time_protocol = time_protocol or TimeProtocol()

        if configuration_set is None:
            raise NotImplementedError("Please provide a set or list of SystemConfiguration instances (Preset configuration set is not implemented yet).")
        self.configuration_set = configuration_set

        # Normalize multi-qubit parameters based on n_qubits
        n_qubits = self.n_qubits
        self.noise_model._normalize_noise_rates(n_qubits)

        # Expand the noise models into ordinary interactions so noise is stored, reported and swept
        # like any other interaction (must run before validation so the overlap rules see them).
        self._absorb_noise_into_interactions()

        # Random seed for uncertainty calculations
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Computed measurement times list (denormalized)
        self._measurement_times_list: Optional[List[float]] = None
        self._update_measurement_times()

        # Validation
        self._validate_experimental_parameters()

    def _absorb_noise_into_interactions(self) -> None:
        """Translate the noise models into interactions so noise is handled like any other interaction.

        If no configuration overrides the noise, the base noise applies uniformly and goes once into
        ``physical_model.interactions``. Otherwise each configuration gets its effective noise (its own
        ``noise_model``, else the base) in its own ``interactions``. Either way a noise channel is never
        in both base and a configuration. The ``noise_model`` objects are kept (for representation);
        previously generated noise interactions are stripped first, so re-running (e.g. on
        :meth:`copy`) does not double the noise.
        """
        n_qubits = self.n_qubits

        # drop any noise interactions from a previous distribution so this is idempotent
        self.physical_model.interactions = [i for i in self.physical_model.interactions if not i.generated_noise]
        for config in self.configuration_set:
            config.interactions = [i for i in config.interactions if not i.generated_noise]
            if config.noise_model is not None:
                config.noise_model._normalize_noise_rates(n_qubits)

        if not any(config.noise_model is not None for config in self.configuration_set):
            self.physical_model.interactions.extend(self.noise_model.to_interactions(n_qubits))
        else:
            for config in self.configuration_set:
                effective_noise = config.noise_model if config.noise_model is not None else self.noise_model
                config.interactions.extend(effective_noise.to_interactions(n_qubits))

    def _update_measurement_times(self) -> None:
        """Resolve and cache ``_t_start`` and ``_measurement_times_list``.

        ``t_start`` is the required ``TimeProtocol.t_simulation_start``. ``measurement_times`` are
        measured-only; the full solver sequence is ``timestamps`` = [t_start] + measurement_times.
        Mode A uses the explicit list; mode B generates t_start + k*``time_interval`` for
        k=1..``n_measurements``.
        """
        tp = self.time_protocol
        self._t_start = float(tp.t_simulation_start)
        self._measurement_times_list = [float(t) for t in tp.resolve_measurement_times()]

    @property
    def t_simulation_start(self) -> float:
        """Simulation start time (the first, unmeasured, timestamp).

        Returns:
            - ``t_start`` (float): Start time of the simulation.
        """
        if self._measurement_times_list is None:
            self._update_measurement_times()
        return self._t_start

    @t_simulation_start.setter
    def t_simulation_start(self, value: float) -> None:
        """Set the simulation start on the time protocol and recompute measurement times.

        Args:
            ``value`` (float): New simulation start time (absolute time).
        """
        self.time_protocol.t_simulation_start = float(value)
        self._update_measurement_times()

    @property
    def timestamps(self) -> np.ndarray:
        """Full solver sequence [``t_simulation_start``, *``measurement_times``] (start is unmeasured).

        Delegates to :attr:`TimeProtocol.timestamps`.

        Returns:
            - ``timestamps`` (np.ndarray): 1-D array of shape (M+1,).
        """
        return self.time_protocol.timestamps

    def get_measurement_uncertainties(self, batch_size: int = 1, offset: bool = True, jitter: bool = True) -> np.ndarray:
        """Sample stochastic timing uncertainties to add to ``timestamps``.

        Delegates to :meth:`TimeProtocol.get_measurement_uncertainties`.

        Args:
            ``batch_size`` (int): Number of independent realizations.
            ``offset`` (bool): Apply the collective offset (see ``random_measurements_offset``).
            ``jitter`` (bool): Apply per-measurement jitter from ``per_measurement_jitter``.

        Returns:
            - ``uncertainties`` (np.ndarray): Shape (``batch_size``, M+1) timing offsets.
        """
        return self.time_protocol.get_measurement_uncertainties(batch_size, offset, jitter)

    def get_timestamps(self, batch_size: int = 1, offset: bool = True, jitter: bool = True) -> np.ndarray:
        """Return ``timestamps`` with stochastic uncertainties applied.

        Delegates to :meth:`TimeProtocol.get_timestamps`.

        Args:
            ``batch_size`` (int): Number of independent realizations.
            ``offset`` (bool): Apply the collective offset.
            ``jitter`` (bool): Apply per-measurement jitter.

        Returns:
            - ``timestamps`` (np.ndarray): Shape (``batch_size``, M+1).
        """
        return self.time_protocol.get_timestamps(batch_size, offset, jitter)

    def collective_offset_desc(self) -> str:
        """Human-readable description of the collective offset distribution ('off' when disabled).

        Returns:
            - ``desc`` (str): 'off', 'custom' (callable), or 'uniform(-w, 0)' (True uses time_interval,
              a numeric offset uses its fixed width).
        """
        off = self.time_protocol.random_measurements_offset
        if not off:
            return "off"
        if callable(off):
            return "custom"
        width = float(self.time_protocol.time_interval) if off is True else float(off)
        return f"uniform(-{width:.3g}, 0)"


    def _validate_experimental_parameters(self) -> None:
        """Validate parameter consistency and physical constraints."""
        # Validate subsystem levels
        for i, level in enumerate(self.cavity_levels):
            if level < 2:
                raise ValueError(f"Every cavity must have at least 2 levels. Cavity_{i} got {level}")
        for i, level in enumerate(self.field_levels):
            if level < 2:
                raise ValueError(f"Every field must have at least 2 levels. Field_{i} got {level}")
        for i, level in enumerate(self.qubit_levels):
            if level < 2:
                raise ValueError(f"Every qubit must have at least 2 levels. Qubit_{i} got {level}")

        # Ensure measurement times are computed
        if self._measurement_times_list is None:
            self._update_measurement_times()
        if self._measurement_times_list is None:
            raise ValueError("Measurement times could not be computed")

        # Validate configuration set
        if not isinstance(self.configuration_set, (list, set)) or len(self.configuration_set) <= 1:
            raise NotImplementedError(
                "Please provide a list or set of SystemConfiguration instances with at least 2 elements "
                "(Preset configuration list is not implemented yet)."
            )

        self.configuration_set = list(self.configuration_set)

        for config in self.configuration_set:
            if not isinstance(config, SystemConfiguration):
                raise TypeError("All items in configuration_set must be SystemConfiguration instances")
            if config.noise_model is not None:
                config.noise_model._normalize_noise_rates(self.n_qubits)
            if config.interactions:
                for interaction in config.interactions:
                    for subsystem in [interaction.subsystem1, interaction.subsystem2]:
                        if subsystem is not None and subsystem[0] == 'cavity' and subsystem[1] >= self.n_cavities:
                            raise ValueError(
                                f"Custom interaction of {config.name} involves cavity {subsystem[1]}, but only {self.n_cavities} cavities in system"
                            )
                        if subsystem is not None and subsystem[0] == 'field' and subsystem[1] >= self.n_fields:
                            raise ValueError(
                                f"Custom interaction of {config.name} involves field mode {subsystem[1]}, but only {self.n_fields} field modes in system"
                            )
                        if subsystem is not None and subsystem[0] == 'qubit' and subsystem[1] >= self.n_qubits:
                            raise ValueError(
                                f"Custom interaction of {config.name} involves qubit {subsystem[1]}, but only {self.n_qubits} qubits in system"
                            )
                        
                
                # A configuration may not re-declare an interaction already present in the
                # shared physical model (matched by type and subsystem pair, parameters aside).
                duplicate_check = [((int1.interaction_type == int2.interaction_type) and \
                                    (int1.subsystem1 == int2.subsystem1) and \
                                    (int1.subsystem2 == int2.subsystem2)) \
                                        for int1 in self.interactions for int2 in config.interactions \
                                    if not (int1.interaction_type in [InteractionType.CUSTOM_HAMILTONIAN, InteractionType.CUSTOM_LINDBLAD])]
                interaction_list = [int1._interaction_context() for int1 in self.interactions for int2 in config.interactions \
                                    if not (int1.interaction_type in [InteractionType.CUSTOM_HAMILTONIAN, InteractionType.CUSTOM_LINDBLAD])]

                if any(duplicate_check):
                    duplicates = [int_summary for int_summary, check in zip(interaction_list, duplicate_check) if check]
                    raise ValueError(f"PhysicalModel interactions cannot be used inside systemconfigurations (even if with different parameters), found duplicates in {config.name} configuration.\n\
                                    Found interactions: {duplicates}")

            if config.init_cavity_states is not None:
                for (index, state) in config.init_cavity_states.items():
                    if not 0 <= index < self.n_cavities:
                        raise ValueError(
                            f"Initial state of {config.name} specifies cavity state for cavity {index}, but only {self.n_cavities} cavities in system. Indexing starts from 0."
                        )
                    if state.state_type == State.FOCK:
                        n = state.parameters["n"]
                        if self.cavity_levels[index] < n:
                            raise ValueError(
                                f"Initial state of {config.name} specifies FOCK state with n={state.parameters['n']} for cavity {index},\n\
                                but cavity truncation level is {self.cavity_levels[index]}.\n\
                                Valid n values are 0 to {self.cavity_levels[index]-1}, or raise the cavity_levels truncation parameter."
                            )

                    if state.state_type == State.COHERENT:
                        # Warn when the truncation is below mean + 6 std dev of the photon
                        # number (a coherent state has mean and variance both |alpha|^2).
                        n_avg = pow(abs(state.parameters["alpha"]), 2)
                        if self.cavity_levels[index] < (n_avg + 6*np.sqrt(n_avg)):
                            warnings.warn(
                                f"Initial state of {config.name} specifies COHERENT state with alpha={state.parameters['alpha']} for cavity {index},\n\
                                which may lead to significant population in levels above the cavity truncation level {self.cavity_levels[index]}.\n\
                                Consider increasing cavity_levels or reducing alpha for more accurate simulations.",
                                UserWarning,
                            )
                    if state.state_type == State.THERMAL:
                        # Same mean + 6 std dev check; a thermal state has variance
                        # n_avg*(n_avg + 1), hence the different square-root term.
                        n_avg = state.parameters["n_avg"]
                        if self.cavity_levels[index] < (n_avg + 6*np.sqrt(n_avg*(n_avg + 1))):
                            warnings.warn(
                                f"Initial state of {config.name} specifies THERMAL state with n_avg={n_avg} for cavity {index},\n\
                                which may lead to significant population in levels above the cavity truncation level {self.cavity_levels[index]}.\n\
                                Consider increasing cavity_levels or reducing n_avg for more accurate simulations.",
                                UserWarning,
                            )
                    if state.state_type == State.CUSTOM:
                        n_amplitudes = len(state.parameters["amplitudes"])
                        if self.cavity_levels[index] < n_amplitudes:
                            raise ValueError(
                                f"Initial state of {config.name} specifies a CUSTOM state with {n_amplitudes} "
                                f"Fock amplitudes for cavity {index}, but the cavity truncation level is "
                                f"{self.cavity_levels[index]}. Provide at most {self.cavity_levels[index]} "
                                f"amplitudes or raise the cavity_levels truncation parameter."
                            )

            if config.init_field_states is not None:
                for (index, state) in config.init_field_states.items():
                    if not 0 <= index < self.n_fields:
                        raise ValueError(
                            f"Initial state of {config.name} specifies field state for field mode {index}, but only {self.n_fields} field modes in system. Indexing starts from 0."
                        )
                    if state.state_type == State.FOCK:
                        n = state.parameters["n"]
                        if self.field_levels[index] < n:
                            raise ValueError(
                                f"Initial state of {config.name} specifies FOCK state with n={state.parameters['n']} for field mode {index},\n\
                                but field truncation level is {self.field_levels[index]}.\n\
                                Valid n values are 0 to {self.field_levels[index]-1}, or raise the field_levels truncation parameter."
                            )
                    if state.state_type == State.COHERENT:
                        # Mean + 6 std dev truncation heuristic (see cavity states above).
                        n_avg = pow(abs(state.parameters["alpha"]), 2)
                        if self.field_levels[index] < (n_avg + 6*np.sqrt(n_avg)):
                            warnings.warn(
                                f"Initial state of {config.name} specifies COHERENT state with alpha={state.parameters['alpha']} for field mode {index},\n\
                                which may lead to significant population in levels above the field truncation level {self.field_levels[index]}.\n\
                                Consider increasing field_levels or reducing alpha for more accurate simulations.",
                                UserWarning,
                            )
                    if state.state_type == State.THERMAL:
                        # Mean + 6 std dev truncation heuristic (see cavity states above).
                        n_avg = state.parameters["n_avg"]
                        if self.field_levels[index] < (n_avg + 6*np.sqrt(n_avg*(n_avg + 1))):
                            warnings.warn(
                                f"Initial state of {config.name} specifies THERMAL state with n_avg={n_avg} for field mode {index},\n\
                                which may lead to significant population in levels above the field truncation level {self.field_levels[index]}.\n\
                                Consider increasing field_levels or reducing n_avg for more accurate simulations.",
                                UserWarning,
                            )
                    if state.state_type == State.CUSTOM:
                        n_amplitudes = len(state.parameters["amplitudes"])
                        if self.field_levels[index] < n_amplitudes:
                            raise ValueError(
                                f"Initial state of {config.name} specifies a CUSTOM state with {n_amplitudes} "
                                f"Fock amplitudes for field mode {index}, but the field truncation level is "
                                f"{self.field_levels[index]}. Provide at most {self.field_levels[index]} "
                                f"amplitudes or raise the field_levels truncation parameter."
                            )

            if config.density_matrix is not None:
                dim = math.prod(self.cavity_levels) * math.prod(self.field_levels)
                if config.density_matrix.shape != (dim, dim):
                    raise ValueError(
                        f"Initial state of {config.name} specifies a density matrix with shape {config.density_matrix.shape}, but expected shape is ({dim}, {dim}) based on the physical model dimensions (only considers cavity and field modes)."
                    )
            else:
                # No density matrix: fill every cavity/field left unspecified with the
                # vacuum state, so all subsystems have an explicit initial state downstream.
                if config.init_cavity_states is not None:
                    vacuum_cavity_idx = list(
                        set(range(self.n_cavities))
                        - set(config.init_cavity_states.keys())
                    )
                else:
                    vacuum_cavity_idx = list(range(self.n_cavities))
                    config.init_cavity_states = {}
                if config.init_field_states is not None:
                    vacuum_field_idx = list(
                        set(range(self.n_fields))
                        - set(config.init_field_states.keys())
                    )
                else:
                    vacuum_field_idx = list(range(self.n_fields))
                    config.init_field_states = {}

                for idx in vacuum_cavity_idx:
                    config.init_cavity_states[idx] = SubsystemState(
                        state_type=State.VACUUM
                    )
                for idx in vacuum_field_idx:
                    config.init_field_states[idx] = SubsystemState(
                        state_type=State.VACUUM
                    )

        names = [config.name for config in self.configuration_set]
        if len(names) != len(set(names)):
            raise ValueError("All SystemConfiguration instances in configuration_set must have unique names")

        # Exactly one configuration must be flagged as the ground (is_ground) configuration.
        ground_configs = [config.name for config in self.configuration_set if config.is_ground]
        if len(ground_configs) != 1:
            raise ValueError(
                f"Exactly one SystemConfiguration in configuration_set must have is_ground=True, "
                f"but {len(ground_configs)} do: {ground_configs}"
            )

        self.ground = ground_configs[0]

    # Direct access to commonly used parameters for easier integration

    @property
    def perturbation_type(self) -> str:
        """Direct access to the type of perturbation"""
        return self.physical_model.perturbation_type

    @property
    def n_cavities(self) -> int:
        """Direct access to number of cavities."""
        return self.physical_model.n_cavities

    @property
    def n_fields(self) -> int:
        """Direct access to number of fields."""
        return self.physical_model.n_fields

    @property
    def n_qubits(self) -> int:
        """Direct access to number of qubits."""
        return self.physical_model.n_qubits

    @property
    def cavity_levels(self) -> Union[int, List[int]]:
        """Direct access to cavity levels."""
        return self.physical_model.cavity_levels

    @cavity_levels.setter
    def cavity_levels(self, value: Union[int, List[int]]) -> None:
        """Set cavity levels."""
        self.physical_model.cavity_levels = self.physical_model._normalize_levels(value, self.n_cavities,"cavity")

    @property
    def field_levels(self) -> Union[int, List[int]]:
        """Direct access to field levels."""
        return self.physical_model.field_levels

    @field_levels.setter
    def field_levels(self, value: Union[int, List[int]]) -> None:
        """Set field levels."""
        self.physical_model.field_levels = self.physical_model._normalize_levels(value, self.n_fields,"field")

    @property
    def qubit_levels(self) -> Union[int, List[int]]:
        """Direct access to qubit levels (returns list after normalization)."""
        return self.physical_model.qubit_levels

    @qubit_levels.setter
    def qubit_levels(self, value: Union[int, List[int]]) -> None:
        """Set qubit levels."""
        self.physical_model.qubit_levels = self.physical_model._normalize_levels(value, self.n_qubits,"qubit")
    
    @property
    def interactions(self) -> List[Interaction]:
        """Return the list of interactions from the physical model."""
        return self.physical_model.interactions


    @property
    def measurement_times(self) -> np.ndarray:
        """Absolute measurement times with no normalization applied.

        Returns:
            - ``measurement_times`` (np.ndarray): 1-D array of shape (M,).
        """
        if self._measurement_times_list is None:
            self._update_measurement_times()
        return np.array(self._measurement_times_list)

    @measurement_times.setter
    def measurement_times(self, value: Union[List[float], np.ndarray]) -> None:
        """Update the explicit measurement times (explicit mode only; no mode switching).

        Args:
            ``value`` (Union[List[float], np.ndarray]): Absolute measurement times (ascending).

        Raises:
            ValueError: If the protocol uses interval mode (rebuild it to change timing spec).
        """
        tp = self.time_protocol
        if tp.measurement_times is None:
            raise ValueError("measurement_times is only settable in explicit mode; this protocol uses "
                             "n_measurements + time_interval. Rebuild the protocol to change the timing spec.")
        tp.measurement_times = [float(t) for t in np.array(value)]
        tp.__post_init__()  # re-validate (ascending, before-start) and rebuild samplers
        self._update_measurement_times()

    @property
    def time_interval(self) -> float:
        """Direct access to time interval (absolute time)."""
        return self.time_protocol.time_interval

    @time_interval.setter
    def time_interval(self, value: float) -> None:
        """Update the measurement spacing (interval mode only; no mode switching).

        Args:
            ``value`` (float): New time interval (absolute time).

        Raises:
            ValueError: If the protocol uses explicit measurement_times (rebuild it to change timing spec).
        """
        tp = self.time_protocol
        if tp.measurement_times is not None:
            raise ValueError("time_interval is only settable in interval mode; this protocol uses explicit "
                             "measurement_times. Rebuild the protocol to change the timing spec.")
        tp.time_interval = float(value)
        tp.__post_init__()  # re-validate (positive interval) and rebuild samplers
        self._update_measurement_times()

    @property
    def seed(self) -> Optional[int]:
        """Direct access to random seed."""
        return self.random_seed

    @seed.setter
    def seed(self, value: Optional[int]) -> None:
        """Set ``random_seed`` and reinitialize the numpy RNG.

        Args:
            ``value`` (Optional[int]): New seed, or None for non-deterministic behavior.
        """
        self.random_seed = value
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def add_interaction(self, interaction: Interaction) -> None:
        """Append an ``Interaction`` to ``physical_model.interactions``.

        Args:
            ``interaction`` (Interaction): Interaction to add.
        """
        if not isinstance(interaction, Interaction):
            raise TypeError("interaction must be an Interaction instance")
        self.physical_model.interactions.append(interaction)

    def get_configuration(self, name: str) -> Optional[SystemConfiguration]:
        """Retrieve a ``SystemConfiguration`` by ``name`` from ``configuration_set``.

        Args:
            ``name`` (str): Name of the configuration to find.

        Returns:
            - ``config`` (Optional[SystemConfiguration]): Matching configuration, or None if not found.
        """

        for config in self.configuration_set:
            if config.name == name:
                return config
    
    def get_configuration_names(self) -> List[str]:
        """Return the names of all configurations in ``configuration_set``.

        Returns:
            - ``names`` (List[str]): Ordered list of configuration names.
        """
        return [config.name for config in self.configuration_set]

    def add_configuration(self, config: SystemConfiguration) -> None:
        """Append a ``SystemConfiguration`` to ``configuration_set``.

        Args:
            ``config`` (SystemConfiguration): Configuration to add.

        Raises:
            ValueError: If a configuration with the same ``name`` already exists.
        """
        if config.name in [cfg.name for cfg in self.configuration_set]:
            raise ValueError(f"Configuration with name '{config.name}' already exists")
        self.configuration_set.append(config)

    def remove_configuration(self, name: str) -> None:
        """Remove the ``SystemConfiguration`` with the given ``name`` from ``configuration_set``.

        Args:
            ``name`` (str): Name of the configuration to remove.
        """
        configuration_set = [cfg for cfg in self.configuration_set if cfg.name != name]
        if configuration_set == self.configuration_set:
            raise ValueError(f"No configuration with name '{name}' found to remove") 
        else:
            self.configuration_set = configuration_set

    def copy(self, **updates) -> "ExperimentalParameters":
        """Create a copy of ``ExperimentalParameters`` with optional updates.

        All nested objects (``physical_model``, ``time_protocol``, ``noise_model``,
        ``configuration_set``) are deep-copied to avoid sharing mutable state.

        Args:
            **updates (Any): Attributes to override. Valid keys: ``physical_model``
                (``PhysicalModel`` instance or dict of kwargs), ``noise_model``,
                ``time_protocol``, ``configuration_set``, ``random_seed``.

        Returns:
            - ``copy`` (ExperimentalParameters): New instance with the updates applied.

        Example:
            >>> # Pass a PhysicalModel instance
            >>> new_params = exp_params.copy(
            ...     physical_model=exp_params.physical_model.copy(n_cavities=2)
            ... )
            >>>
            >>> # Or pass a dict of kwargs (convenience shorthand)
            >>> new_params = exp_params.copy(
            ...     physical_model={'n_cavities': 2, 'n_fields': 2}
            ... )
        """
        # Deep copy nested configurations
        new_phys_model = self.physical_model
        new_noise_model = self.noise_model
        new_time_protocol = self.time_protocol
        new_random_seed = self.random_seed
        new_config_set = self.configuration_set 

        # Handle updates
        if "physical_model" in updates:
            pm_update = updates["physical_model"]
            if isinstance(pm_update, dict):
                # If dict, use copy method with updates
                new_phys_model = self.physical_model.copy(**pm_update)
            else:
                # If PhysicalModel instance, use directly
                new_phys_model = pm_update
        else:
            # Deep copy existing physical model
            new_phys_model = self.physical_model.copy()
            
        if "noise_model" in updates:
            new_noise_model = updates["noise_model"]
        else:
            # Create new instance with same values
            new_noise_model = self.noise_model.copy()

        if "time_protocol" in updates:
            new_time_protocol = updates["time_protocol"]
        else:
            new_time_protocol = self.time_protocol.copy()

        if "configuration_set" in updates:
            new_config_set = updates["configuration_set"]
        else:
            # Create new instance with same values
            new_config_set = [config.copy() for config in self.configuration_set]


        if "random_seed" in updates:
            new_random_seed = updates["random_seed"]

        return ExperimentalParameters(
            physical_model=new_phys_model,
            noise_model=new_noise_model,
            time_protocol=new_time_protocol,
            configuration_set=new_config_set,
            random_seed=new_random_seed,
        )

    def __repr__(self) -> str:
        """Human-readable summary of all parameters grouped by system, measurement, noise, and configurations.

        Returns:
            - ``repr`` (str): Multi-line string with validation status at the end.
        """
        lines = []
        # System Dimensions Group
        lines.append("SYSTEM DIMENSIONS")

        # Calculate total dimension
        n_cavities = self.n_cavities
        n_fields = self.n_fields
        n_qubits = self.n_qubits
        qubit_levels_list = self.qubit_levels
        cavity_levels_list = self.cavity_levels
        field_levels_list = self.field_levels

        qubit_dim = int(np.prod(qubit_levels_list))
        cavity_dim = int(np.prod(cavity_levels_list))
        field_dim = int(np.prod(field_levels_list))

        total_dim = cavity_dim * qubit_dim * field_dim
        lines.append(f"  Cavities: {n_cavities}   levels {cavity_levels_list}")
        lines.append(f"  Fields:   {n_fields}   levels {field_levels_list}")
        lines.append(f"  Qubits:   {n_qubits}   levels {qubit_levels_list}")
        lines.append(f"  Total dimension: {total_dim}")

        def format_subsystem(subsystem: Tuple[str, int]) -> str:
            return f"{subsystem[0]}{subsystem[1]}"

        def format_params(parameters: Any) -> str:
            if isinstance(parameters, dict):
                if not parameters:
                    return "no parameters"
                items = sorted(parameters.items(), key=lambda item: str(item[0]))
                return ", ".join(f"{key}={value}" for key, value in items)
            return f"value={parameters}"

        def format_interaction_name(interaction: Interaction) -> str:
            subsystems = format_subsystem(interaction.subsystem1)
            if interaction.subsystem2 is not None:
                subsystems += f"-{format_subsystem(interaction.subsystem2)}"
            return f"{interaction.interaction_type.value}({subsystems})"

        def format_interaction(interaction: Interaction) -> str:
            return f"{format_interaction_name(interaction)}: {format_params(interaction.parameters)}"

        def format_subsystem_state(state: Any) -> str:
            if state is None:
                return "None"
            if not isinstance(state, SubsystemState):
                return f"invalid_state={state}"
            # Parameter-less states (e.g. vacuum) are shown by name only, no "no parameters".
            if not isinstance(state.parameters, dict) or not state.parameters:
                return f"{state.state_type.value}"
            return f"{state.state_type.value}({format_params(state.parameters)})"

        def describe_uncertainty(func: Any) -> str:
            # Give the standard, representable uncertainties a readable name instead of
            # printing a raw function object; fall back to "custom" for anything else.
            if func is None:
                return "none"
            if not callable(func):
                return str(func)
            try:
                samples = [float(func(t)) for t in np.linspace(-10, 10, 11)]
            except Exception:
                return "custom"
            if all(abs(s) < 1e-12 for s in samples):
                return "none"
            if all(abs(s - samples[0]) < 1e-12 for s in samples):
                return f"constant ({samples[0]:g})"
            return "custom"

        # Identity of an interaction (type + subsystems, ignoring parameters); lets the
        # repr recognize the same interaction across different configurations.
        def interaction_key(interaction: Interaction) -> Tuple[str, Tuple[str, int], Tuple[str, int]]:
            return (
                interaction.interaction_type.value,
                interaction.subsystem1,
                interaction.subsystem2,
            )

        # Hashable snapshot of an interaction's parameters, so the repr can tell when the
        # same interaction carries different parameters in different configurations.
        def parameter_signature(parameters: Any) -> Tuple[Any, ...]:
            if isinstance(parameters, dict):
                items = sorted(parameters.items(), key=lambda item: str(item[0]))
                return tuple((str(key), repr(value)) for key, value in items)
            return ("value", repr(parameters))

        def append_grouped_interactions(grouped_interactions: List[Interaction], base_indent: str) -> None:
            # Display interactions grouped under a single subsystem, with priority
            # cavity > qubit > field: an interaction touching a cavity is shown under that
            # cavity, else under a qubit, else under a field. Ties between subsystems of the
            # same type are broken by the lowest index.
            by_cavity: Dict[int, List[Tuple[int, Interaction]]] = {}
            by_qubit: Dict[int, List[Tuple[int, Interaction]]] = {}
            by_field: Dict[int, List[Tuple[int, Interaction]]] = {}
            no_cavity: List[Tuple[int, Interaction]] = []
            indexed_interactions = list(enumerate(grouped_interactions))
            assigned: Set[int] = set()

            # First pass: assign every cavity-touching interaction to a cavity bucket.
            for index, interaction in indexed_interactions:
                subsystems = [s for s in (interaction.subsystem1, interaction.subsystem2) if s is not None]
                types = {subsystem_type for subsystem_type, _ in subsystems}
                if "cavity" in types:
                    cavity_indices = [
                        idx
                        for subsystem_type, idx in subsystems
                        if subsystem_type == "cavity"
                    ]
                    cavity_index = min(cavity_indices)
                    by_cavity.setdefault(cavity_index, []).append((index, interaction))
                    assigned.add(index)
                else:
                    no_cavity.append((index, interaction))

            # Second pass: assign the rest to a qubit bucket, otherwise a field bucket.
            for index, interaction in no_cavity:
                if index in assigned:
                    continue
                subsystems = [s for s in (interaction.subsystem1, interaction.subsystem2) if s is not None]
                types = {subsystem_type for subsystem_type, _ in subsystems}
                if "qubit" in types:
                    qubit_indices = [
                        idx
                        for subsystem_type, idx in subsystems
                        if subsystem_type == "qubit"
                    ]
                    qubit_index = min(qubit_indices)
                    by_qubit.setdefault(qubit_index, []).append((index, interaction))
                    assigned.add(index)
                else:
                    field_indices = [
                        idx
                        for subsystem_type, idx in subsystems
                        if subsystem_type == "field"
                    ]
                    if field_indices:
                        field_index = min(field_indices)
                        by_field.setdefault(field_index, []).append((index, interaction))
                        assigned.add(index)

            # Emit interactions in the cavity > qubit > field order (each bucket sorted by index),
            # but without printing the grouping headers/subsystem labels -- the order alone reflects it.
            for bucket in (by_cavity, by_qubit, by_field):
                for index in sorted(bucket):
                    for _, interaction in bucket[index]:
                        lines.append(f"{base_indent}{format_interaction(interaction)}")

        # Physical Model Interactions
        lines.append("")
        lines.append("PHYSICAL MODEL")
        lines.append(f"  Perturbation type:    {self.perturbation_type}")
        interactions = list(self.physical_model.interactions)
        if not interactions:
            lines.append("  Interactions:         None")
        else:
            lines.append(f"  Interactions:         {len(interactions)} interaction(s)")
            static_interactions = [
                interaction for interaction in interactions if interaction.time_modulation is None
            ]
            time_dependent_interactions = [
                interaction for interaction in interactions if interaction.time_modulation is not None
            ]

            if static_interactions:
                lines.append("  Static interactions:")
                append_grouped_interactions(static_interactions, "    ")

            if time_dependent_interactions:
                lines.append("  Time-dependent interactions:")
                append_grouped_interactions(time_dependent_interactions, "    ")

        # Measurement Protocol Group
        lines.append("")
        lines.append("TIME PROTOCOL")

        lines.append(f"  Start time:           {self.t_simulation_start:>8.4f}")
        # Determine mode (list or interval)
        times_list = (
            self._measurement_times_list if self._measurement_times_list is not None else []
        )
        if self.time_protocol.measurement_times is not None:
            lines.append("  Mode:                 Explicit list")
            n_measurements = len(times_list)
            lines.append(f"  Number of measurements: {n_measurements:>6}")
            lines.append(f"  Measurement times:    {times_list}")
        else:
            lines.append("  Mode:                 Interval-based")
            lines.append(f"  Time interval:        {self.time_protocol.time_interval:>8.4f}")
            n_measurements = len(times_list)
            lines.append(f"  Number of measurements: {n_measurements:>6}")
            lines.append(f"  Computed times:       {times_list}")

        lines.append(f"  Collective offset:    {self.collective_offset_desc()}")
        lines.append(f"  Per-measurement jitter: {describe_uncertainty(self.time_protocol.per_measurement_jitter)}")
        lines.append(f"  Noisy simulation start: {'on' if self.time_protocol.noisy_simulation_start else 'off'}")
        if self.time_protocol.window_start is not None:
            slope = self.time_protocol.resolved_window_slope
            lines.append(f"  Measurement window:   [{self.time_protocol.window_start}, {self.time_protocol.window_end}]  slope={slope:.4g}")
        else:
            lines.append("  Measurement window:   None (uniform weights)")
        # Noise Configuration Group
        lines.append("")
        lines.append("NOISE MODEL")
        lines.append(f"  Depolarizing rate:    {self.noise_model.depolarizing}")
        lines.append(f"  Dephasing rate:       {self.noise_model.dephasing}")
        lines.append(f"  Relaxation rate:      {self.noise_model.relaxation}")

        if self.noise_model.custom_operators is not None:
            lines.append(f"  Custom operators:     {len(self.noise_model.custom_operators):>6}")
        else:
            lines.append("  Custom operators:     None")

        # Configuration Set Group
        lines.append("")
        lines.append("CONFIGURATIONS")
        lines.append(f"  Count:                {len(self.configuration_set)}")
        lines.append(f"  Ground configuration: {self.ground}")
        # Collect the set of distinct parameter signatures for each interaction type
        # across all configurations. Used below to show parameters only when different
        # configurations parameterize the same interaction type differently.
        interaction_param_variants: Dict[Tuple[str, Tuple[str, int], Tuple[str, int]], Set[Tuple[Any, ...]]] = {}
        for config in self.configuration_set:
            for interaction in config.interactions or []:
                key = interaction_key(interaction)
                interaction_param_variants.setdefault(key, set()).add(
                    parameter_signature(interaction.parameters)
                )

        for i, config in enumerate(self.configuration_set):
            has_noise = "yes" if config.noise_model is not None else "no"
            interactions = config.interactions or []
            ground_tag = " (ground)" if config.is_ground else ""

            lines.append(
                f"    [{i}] {config.name}{ground_tag}: noise_override={has_noise}, interactions={len(interactions)}"
            )

            # A configuration is initialised either from a custom density matrix or from
            # per-subsystem states, never both.
            if config.density_matrix is not None:
                lines.append(
                    f"      Initial state: custom density matrix (shape={config.density_matrix.shape})"
                )
            elif config.init_cavity_states or config.init_field_states:
                for index in sorted(config.init_cavity_states or {}):
                    state = config.init_cavity_states[index]
                    lines.append(f"      Cavity {index}: {format_subsystem_state(state)}")
                for index in sorted(config.init_field_states or {}):
                    state = config.init_field_states[index]
                    lines.append(f"      Field {index}: {format_subsystem_state(state)}")
            else:
                lines.append("      Initial state: None")

            if interactions:
                lines.append("      Interactions:")
                for interaction in interactions:
                    name = format_interaction_name(interaction)
                    key = interaction_key(interaction)
                    param_variants = interaction_param_variants.get(key, set())
                    if len(param_variants) > 1:
                        lines.append(f"        {name}: {format_params(interaction.parameters)}")
                    else:
                        lines.append(f"        {name}")

        # Overall System Status
        lines.append("")
        lines.append("SYSTEM STATUS")

        try:
            self._validate_experimental_parameters()
            lines.append("  Configuration:        VALID")
        except Exception as exc:
            lines.append("  Configuration:        INVALID")
            lines.append(f"  Error:                {str(exc)}")

        lines.append("")
        lines.append("RANDOM SEED")
        lines.append(f"  Seed:                 {self.random_seed}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation (delegates to ``__repr__``)."""
        return self.__repr__()

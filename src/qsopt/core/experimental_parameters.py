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
import qutip as qt
import math

# Import qutip_jax to enable JAX backend
import qutip_jax  # pylint: disable=unused-import


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

    # custom interactions
    CUSTOM_HAMILTONIAN= "custom_hamiltonian"  # Custom coherent interaction defined by user-provided matrix
    CUSTOM_LINDBLAD = "custom_lindblad"  # Custom dissipation defined by user-provided Lindblad operator


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
        subsystem1: Tuple[str, int],
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

        self.__post_init__()

    def __post_init__(self):
        """Validate interaction parameters."""
        if not isinstance(self.interaction_type, InteractionType):
            raise TypeError(
                "interaction_type must be an InteractionType enum value"
            )

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
        """Return a copy with independent parameter storage."""
        return Interaction(
            subsystem1=self.subsystem1,
            subsystem2=self.subsystem2,
            interaction_type=self.interaction_type,
            parameters=copy.deepcopy(self.parameters),
            time_modulation=self.time_modulation,
            custom_matrix=copy.deepcopy(self.custom_matrix) if self.custom_matrix is not None else None,
        )

    def _interaction_context(self) -> str:
        """Return a compact interaction label like 'dispersive(cavity1,qubit3)'."""
        parts = [f"{self.subsystem1[0]}{self.subsystem1[1]}"]
        if self.subsystem2 is not None:
            parts.append(f"{self.subsystem2[0]}{self.subsystem2[1]}")
        return f"{self.interaction_type.value}({','.join(parts)})"

    def _with_context(self, message: str) -> str:
        """Prefix diagnostic messages with the interaction context label."""
        return f"{self._interaction_context()}: {message}"


@dataclass
class PhysicalModel:
    """
    Physical model dimensions and interactions for the quantum system.

    Attributes:
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

                kind = "Hamiltonian" if interaction.interaction_type == InteractionType.CUSTOM_HAMILTONIAN else "Lindblad"
                shape = tuple(interaction.custom_matrix.shape)
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
        """Normalize levels to list format."""
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
        """Return the Hilbert-space dimension (truncation level) of a subsystem."""
        stype, idx = subsystem
        if stype == 'cavity':
            return self.cavity_levels[idx]
        if stype == 'field':
            return self.field_levels[idx]
        if stype == 'qubit':
            return self.qubit_levels[idx]
        raise ValueError(f"Unknown subsystem type: {stype}")


    def copy(self, **updates) -> "PhysicalModel":
        """
        Create a copy of PhysicalModel with optional parameter updates.

        This method creates a new PhysicalModel instance with all attributes
        copied from the current instance. You can override specific attributes
        by passing them as keyword arguments.

        Args:
            **updates: Keyword arguments for attributes to update in the copy.
                      Valid keys: n_cavities, n_fields, n_qubits, cavity_levels, qubit_levels, field_levels, interactions

        Returns:
            New PhysicalModel instance with updated values

        Example:
            >>> original = PhysicalModel(n_cavities=2, n_fields=2, n_qubits=2)
            >>> modified = original.copy(n_cavities=3)  # Keep all other params, change n_cavities
            >>> modified.n_cavities
            3
            >>> modified.n_fields            
            2
        """
        # Start with current values
        params = {
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
class MeasurementProtocol:
    """
    Measurement protocol configuration.

    Two modes of operation:
    1. List mode: Provide explicit list of measurement times
    2. Interval mode: Provide initial_time, final_time, and time_interval

    All time parameters are stored and used as absolute times (no normalization).

    Attributes:
        measurement_times: Explicit list of measurement times (absolute time values).
                          If None, will be computed from initial_time, final_time, time_interval.
        initial_time: Initial time for interval mode (absolute time)
        final_time: Final time for interval mode (absolute time)
        time_interval: Time interval between measurements for interval mode (absolute time)
        initial_time_uncertainty: Initial time uncertainty specification (absolute time).
                 Can be a float or special string keywords (e.g., 'max_interval') that
                 will be resolved dynamically. When numeric, represents the half-width of
                 a uniform distribution [-initial_time_uncertainty, initial_time_uncertainty].
        single_measurement_uncertainty: Function defining uncertainty at each measurement time
    """

    measurement_times: Optional[List[float]] = None
    initial_time: float = -5.0
    final_time: float = 5.0
    time_interval: float = 1.0
    initial_time_uncertainty: Union[float, str] = 0.0
    single_measurement_uncertainty: Callable[[float], float] = lambda t: 0.0

    def __post_init__(self):
        """Validate the measurement-time specification and uncertainty settings."""
        if self.measurement_times is not None:
            if len(self.measurement_times) < 2:
                raise ValueError("At least two measurement times must be specified in measurement_times list")
            if sorted(self.measurement_times) != self.measurement_times:
                raise ValueError("Measurement times in measurement_times list must be in ascending order")
        else:
            if self.time_interval <= 0:
                raise ValueError("Time interval must be positive for interval mode")
            if self.final_time <= self.initial_time:
                raise ValueError("Final time must be greater than initial time for interval mode")
        if self.initial_time_uncertainty is not None:
            if isinstance(self.initial_time_uncertainty, (int, float)):
                if self.initial_time_uncertainty < 0:
                    raise ValueError("Initial time uncertainty must be >= 0")
            elif isinstance(self.initial_time_uncertainty, str):
                if not self.initial_time_uncertainty.strip():
                    raise ValueError("initial_time_uncertainty string specification cannot be empty")
                # Supported keywords will be validated and resolved in ExperimentalParameters
            else:
                raise TypeError("initial_time_uncertainty must be a float or a supported string keyword")
        if self.single_measurement_uncertainty is not None:
            if not callable(self.single_measurement_uncertainty):
                raise TypeError("single_measurement_uncertainty must be a callable function of time")
            # Test the single_measurement_uncertainty function with a sample time value
            try:
                for _ in range(100):
                    random_time = np.random.uniform(-10, 10)
                    test_value = self.single_measurement_uncertainty(random_time)
                    if not isinstance(test_value, (int, float)):
                        raise ValueError(f"single_measurement_uncertainty function must return a numeric value. Got type: {type(test_value)}")
            except Exception as e:
                raise ValueError("single_measurement_uncertainty function got an error during testing:") from e


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
        """Return a copy with independent parameter storage."""
        parameters = copy.deepcopy(self.parameters) if self.parameters is not None else None
        return SubsystemState(state_type=self.state_type, parameters=parameters)

    # custom_amplitudes: Optional[Dict[Tuple[int, int, int], complex]] = None

@dataclass
class NoiseModel:
    """
    Noise model configuration.

    Attributes:
        depolarizing: Depolarization rate. Can be a float (same for all qubits)
                     or a list of floats (individual rate per qubit).
        dephasing: Dephasing rate. Can be a float (same for all qubits)
                  or a list of floats (individual rate per qubit).
        relaxation: Relaxation rate. Can be a float (same for all qubits)
                   or a list of floats (individual rate per qubit).
        custom_operators: Custom Lindblad operators
    """

    depolarizing: Union[float, List[float]] = 0.0  # Depolarization rate
    dephasing: Union[float, List[float]] = 0.0  # Dephasing rate
    relaxation: Union[float, List[float]] = 0.0  # Relaxation rate
    custom_operators: Optional[List[Any]] = None  # Custom Lindblad operators

    def _normalize_noise_rates(self, n_qubits: int):
        """
        Normalize noise rates to list format.

        Args:
            n_qubits: Number of qubits from PhysicalModel
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

    def copy(self) -> "NoiseModel":
        """Return a copy with independent parameter storage."""
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
        """Return a copy with independent nested state."""
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
        )
        



class ExperimentalParameters:
    """
    Complete experimental configuration for quantum sensing protocols.

    This class contains all the system parameters that define the physical
    quantum sensing setup including Hilbert space dimensions, interactions,
    measurement protocols, noise models, and initial state preparation.

    The parameters are organized into logical groups and provide validation
    and consistency checking for the experimental configuration.
    """

    def __init__(
        self,
        physical_model: Optional[PhysicalModel] = None,
        noise_model: Optional[NoiseModel] = None,
        measurement: Optional[MeasurementProtocol] = None,
        configuration_set: Optional[Union[Set[SystemConfiguration], List[SystemConfiguration]]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize experimental parameters.

        Args:
            physical_model: Physical model configuration
            noise_model: Noise model configuration
            measurement: Measurement protocol configuration
            configuration_set: Set or list of system configurations to be simulated (e.g., different initial states, noise levels, interactions)
            random_seed: Random seed for reproducibility of uncertainty calculations
        """
        self.physical_model = physical_model or PhysicalModel()
        self.noise_model = noise_model or NoiseModel()
        self.measurement = measurement or MeasurementProtocol()

        if configuration_set is None:
            raise NotImplementedError("Please provide a set or list of SystemConfiguration instances (Preset configuration set is not implemented yet).")
        self.configuration_set = configuration_set

        # Normalize multi-qubit parameters based on n_qubits
        n_qubits = self.n_qubits
        self.noise_model._normalize_noise_rates(n_qubits)

        # Random seed for uncertainty calculations
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Computed measurement times list (denormalized)
        self._measurement_times_list: Optional[List[float]] = None
        self._update_measurement_times()

        # Validation
        self._validate_experimental_parameters()

    def _compute_measurement_times_from_interval(self) -> List[float]:
        """
        Compute measurement times from initial_time, final_time, and time_interval.

        Returns:
            List of measurement times (absolute time values)
        """
        initial = self.measurement.initial_time
        final = self.measurement.final_time
        interval = self.measurement.time_interval

        # Generate times using arange and ensure final time is included
        grid = np.arange(initial, final + interval / 2, interval, dtype=float)
        times = [float(t) for t in grid]

        return times

    def _update_measurement_times(self) -> None:
        """
        Update measurement times based on the measurement protocol.

        If measurement_times is provided as a list, use it directly.
        Otherwise, compute from initial_time, final_time, and time_interval.
        """
        if self.measurement.measurement_times is not None:
            self._measurement_times_list = [float(t) for t in self.measurement.measurement_times]
            self.measurement.measurement_times = self._measurement_times_list
        else:
            self._measurement_times_list = self._compute_measurement_times_from_interval()
            self.measurement.measurement_times = self._measurement_times_list

    def _resolve_initial_time_uncertainty(self) -> float:
        """Resolve the initial time uncertainty specification to a numeric value."""
        spec = self.measurement.initial_time_uncertainty

        if isinstance(spec, str):
            spec_stripped = spec.strip()
            key = spec_stripped.lower()
            if key in {"max_interval", "max_measurement_interval"}:
                times = self.measurement_times
                if times.size < 2:
                    raise ValueError(
                        "Cannot compute 'max_interval' initial_time_uncertainty with fewer than two measurement times"
                    )
                diffs = np.diff(np.sort(times))
                if diffs.size == 0:
                    return 0.0
                value = float(np.max(diffs))
            elif key in {"total_duration", "full_window"}:
                times = self.measurement_times
                if times.size == 0:
                    return 0.0
                value = float(np.max(times) - np.min(times))
            else:
                try:
                    value = float(spec_stripped)
                except ValueError as exc:
                    raise ValueError(
                        "Unsupported initial_time_uncertainty specification "
                        f"'{spec}'. Supported keywords: 'max_interval', 'max_measurement_interval', "
                        "'total_duration', 'full_window'."
                    ) from exc
        else:
            try:
                value = float(spec)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "initial_time_uncertainty must be a float or supported string keyword"
                ) from exc

        return value

    def get_measurement_times_with_uncertainty(self, batch_size: int = 1, base_times: np.ndarray = None) -> np.ndarray:
        """
        Get measurement times with random shift due to initial time uncertainty.

        The entire measurement sequence is shifted by a random value uniformly
        distributed in [-initial_time_uncertainty, initial_time_uncertainty].

        Uses the random_seed set during initialization for reproducibility.

        Args:
            batch_size: Number of independent realizations to generate (default: 1).
                       If batch_size=1, returns 1D array of shape (n_times,).
                       If batch_size>1, returns 2D array of shape (batch_size, n_times).

        Returns:
            Array of measurement times with uncertainty shift applied:
            - batch_size=1: 1D array of shape (n_times,)
            - batch_size>1: 2D array of shape (batch_size, n_times)
        """
        # Get base measurement times (absolute time values)
        if base_times is None:
            base_times = self.measurement_times
        uncertainty = self._resolve_initial_time_uncertainty()

        if batch_size == 1:
            # Single realization: return 1D array
            if uncertainty > 0:
                shift = np.random.uniform(-uncertainty, uncertainty)
                return base_times + shift
            return base_times

        # Multiple realizations: return 2D array (batch_size, n_times)
        if uncertainty > 0:
            # Generate batch_size random shifts
            shifts = np.random.uniform(-uncertainty, uncertainty, size=batch_size)
            # Broadcasting: (batch_size, 1) + (n_times,) -> (batch_size, n_times)
            return base_times[np.newaxis, :] + shifts[:, np.newaxis]

        # No uncertainty: tile the same times batch_size times
        return np.tile(base_times, (batch_size, 1))
        

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

        # Resolve initial time uncertainty to ensure specification is valid
        _ = self._resolve_initial_time_uncertainty()

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

    # Direct access to commonly used parameters for easier integration

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
        """
        Direct access to measurement times.

        Returns absolute time values with no normalization applied.

        Returns:
            Array of measurement times (absolute time values)
        """
        if self._measurement_times_list is None:
            self._update_measurement_times()
        return np.array(self._measurement_times_list)

    @measurement_times.setter
    def measurement_times(self, value: Union[List[float], np.ndarray]) -> None:
        """
        Set measurement times explicitly (overrides interval mode).

        Args:
            value: List or array of measurement times (absolute time values)
        """
        self.measurement.measurement_times = list(np.array(value))
        self._update_measurement_times()

    @property
    def time_interval(self) -> float:
        """Direct access to time interval (absolute time)."""
        return self.measurement.time_interval

    @time_interval.setter
    def time_interval(self, value: float) -> None:
        """
        Set time interval and recompute measurement times.

        Args:
            value: New time interval (absolute time)
        """
        self.measurement.time_interval = value
        # Clear explicit measurement times to use interval mode
        self.measurement.measurement_times = None
        self._update_measurement_times()

    @property
    def initial_time(self) -> float:
        """Direct access to initial time (absolute time)."""
        return self.measurement.initial_time

    @initial_time.setter
    def initial_time(self, value: float) -> None:
        """
        Set initial time and recompute measurement times.

        Args:
            value: New initial time (absolute time)
        """
        self.measurement.initial_time = value
        # Clear explicit measurement times to use interval mode
        self.measurement.measurement_times = None
        self._update_measurement_times()

    @property
    def final_time(self) -> float:
        """Direct access to final time (absolute time)."""
        return self.measurement.final_time

    @final_time.setter
    def final_time(self, value: float) -> None:
        """
        Set final time and recompute measurement times.

        Args:
            value: New final time (absolute time)
        """
        self.measurement.final_time = value
        # Clear explicit measurement times to use interval mode
        self.measurement.measurement_times = None
        self._update_measurement_times()

    @property
    def initial_time_uncertainty(self) -> float:
        """Resolved initial time uncertainty (absolute time)."""
        return self._resolve_initial_time_uncertainty()

    @initial_time_uncertainty.setter
    def initial_time_uncertainty(self, value: Union[float, str]) -> None:
        """
        Set initial time uncertainty specification.

        Args:
            value: Initial time uncertainty specification (absolute time or keyword)
        """
        self.measurement.initial_time_uncertainty = value

    @property
    def initial_time_uncertainty_spec(self) -> Union[float, str]:
        """Return the raw initial time uncertainty specification."""
        return self.measurement.initial_time_uncertainty

    @property
    def seed(self) -> Optional[int]:
        """Direct access to random seed."""
        return self.random_seed

    @seed.setter
    def seed(self, value: Optional[int]) -> None:
        """
        Set random seed and reinitialize random number generator.

        Args:
            value: New random seed (None for non-deterministic behavior)
        """
        self.random_seed = value
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def add_interaction(self, interaction: Interaction) -> None:
        """
        Add a new interaction to the physical model.

        Args:
            interaction: Interaction instance to add
        """
        if not isinstance(interaction, Interaction):
            raise TypeError("interaction must be an Interaction instance")
        self.physical_model.interactions.append(interaction)

    def get_configuration(self, name: str) -> Optional[SystemConfiguration]:
        """
        Retrieve a system configuration by name from the configuration set.

        Args:
            name: Name of the system configuration to retrieve

        Returns:
            SystemConfiguration instance or None if not found
        """

        for config in self.configuration_set:
            if config.name == name:
                return config
    
    def get_configuration_names(self) -> List[str]:
        """
        Get a list of all configuration names in the configuration set.

        Returns:
            List of configuration names
        """
        return [config.name for config in self.configuration_set]

    def add_configuration(self, config: SystemConfiguration) -> None:
        """
        Add a new system configuration to the configuration set.

        Args:
            config: SystemConfiguration instance to add

        Raises:
            ValueError: If a configuration with the same name already exists
        """
        if config.name in [cfg.name for cfg in self.configuration_set]:
            raise ValueError(f"Configuration with name '{config.name}' already exists")
        self.configuration_set.append(config)

    def remove_configuration(self, name: str) -> None:
        """
        Remove a system configuration from the configuration set by name.

        Args:
            name: Name of the system configuration to remove
        """
        configuration_set = [cfg for cfg in self.configuration_set if cfg.name != name]
        if configuration_set == self.configuration_set:
            raise ValueError(f"No configuration with name '{name}' found to remove") 
        else:
            self.configuration_set = configuration_set

    def copy(self, **updates) -> "ExperimentalParameters":
        """
        Create a copy of ExperimentalParameters with optional updates.

        This method creates a new ExperimentalParameters instance with all
        configuration copied. The nested objects (physical_model, measurement,
        noise_model, configuration_set) are copied to avoid unintended sharing of mutable state.

        Args:
            **updates: Keyword arguments for attributes to update. Can be:
                - physical_model: PhysicalModel instance or dict of updates
                - noise_model: NoiseModel instance
                - measurement: MeasurementProtocol instance
                - configuration_set: Set or list of SystemConfiguration instances
                - random_seed: int or None

        Returns:
            New ExperimentalParameters instance with updated values

        Example:
            >>> # Copy and update the physical model
            >>> new_params = exp_params.copy(
            ...     physical_model=exp_params.physical_model.copy(n_cavities=2)
            ... )
            >>>
            >>> # Or pass updates as dict (for convenience)
            >>> new_params = exp_params.copy(
            ...     physical_model={'n_cavities': 2, 'n_fields': 2}
            ... )
        """
        # Deep copy nested configurations
        new_phys_model = self.physical_model
        new_noise_model = self.noise_model
        new_measurement = self.measurement
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

        if "measurement" in updates:
            new_measurement = updates["measurement"]
        else:
            # Create new instance with same values
            new_measurement = MeasurementProtocol(
                measurement_times=(
                    self.measurement.measurement_times.copy()
                    if self.measurement.measurement_times
                    else None
                ),
                initial_time=self.measurement.initial_time,
                final_time=self.measurement.final_time,
                time_interval=self.measurement.time_interval,
                initial_time_uncertainty=self.measurement.initial_time_uncertainty,
                single_measurement_uncertainty=self.measurement.single_measurement_uncertainty,
            )

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
            measurement=new_measurement,
            configuration_set=new_config_set,
            random_seed=new_random_seed,
        )

    def __repr__(self) -> str:
        """
        Comprehensive string representation showing all parameters organized by groups.

        This method provides a detailed display of all experimental
        parameters, organized by their logical groups with validation status flags.
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

            if by_cavity:
                lines.append(f"{base_indent}By cavity:")
                for cavity_index in sorted(by_cavity):
                    lines.append(f"{base_indent}  Cavity {cavity_index}:")
                    for _, interaction in by_cavity[cavity_index]:
                        lines.append(f"{base_indent}    {format_interaction(interaction)}")

            if by_qubit:
                lines.append(f"{base_indent}By qubit:")
                for qubit_index in sorted(by_qubit):
                    lines.append(f"{base_indent}  Qubit {qubit_index}:")
                    for _, interaction in by_qubit[qubit_index]:
                        lines.append(f"{base_indent}    {format_interaction(interaction)}")

            if by_field:
                lines.append(f"{base_indent}By field:")
                for field_index in sorted(by_field):
                    lines.append(f"{base_indent}  Field {field_index}:")
                    for _, interaction in by_field[field_index]:
                        lines.append(f"{base_indent}    {format_interaction(interaction)}")

        # Physical Model Interactions
        lines.append("")
        lines.append("PHYSICAL MODEL")
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
        lines.append("MEASUREMENT PROTOCOL")

        # Determine mode (list or interval)
        times_list = (
            self._measurement_times_list if self._measurement_times_list is not None else []
        )
        if self.measurement.measurement_times is not None:
            lines.append("  Mode:                 Explicit list")
            n_measurements = len(times_list)
            lines.append(f"  Number of measurements: {n_measurements:>6}")
            lines.append(f"  Measurement times:    {times_list}")
        else:
            lines.append("  Mode:                 Interval-based")
            lines.append(f"  Initial time:         {self.measurement.initial_time:>8.4f}")
            lines.append(f"  Final time:           {self.measurement.final_time:>8.4f}")
            lines.append(f"  Time interval:        {self.measurement.time_interval:>8.4f}")
            n_measurements = len(times_list)
            lines.append(f"  Number of measurements: {n_measurements:>6}")
            lines.append(f"  Computed times:       {times_list}")

        uncertainty_value = self.initial_time_uncertainty
        if uncertainty_value > 0:
            lines.append(f"  Initial time uncertainty: {uncertainty_value:>8.4f}")
            if isinstance(self.measurement.initial_time_uncertainty, str):
                lines.append(f"    (specified as '{self.measurement.initial_time_uncertainty}')")
        lines.append(
            f"  Single measurement uncertainty: {describe_uncertainty(self.measurement.single_measurement_uncertainty)}"
        )
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

            lines.append(
                f"    [{i}] {config.name}: noise_override={has_noise}, interactions={len(interactions)}"
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
        """String representation (calls __repr__)."""
        return self.__repr__()

"""
Experimental Parameters Class
============================

System configuration parameters for quantum sensing experiments including
physical constants, system dimensions, measurement protocols, and initial states.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Set, Optional, Tuple, Union

import numpy as np


class InitialStateType(Enum):
    """Enumeration of supported initial state configurations (for input field)."""

    VACUUM = "vacuum"
    FOCK = "fock"
    COHERENT = "coherent"
    THERMAL = "thermal"
    CUSTOM = "custom"


class InteractionType(Enum):
    """Enumeration of supported qubit-qubit interaction types."""

    ZZ = "sz-sz"  # σz ⊗ σz interaction
    XX = "sx-sx"  # σx ⊗ σx interaction
    YY = "sy-sy"  # σy ⊗ σy interaction

@dataclass
class Interaction:
    """
    Configuration for interaction between subsystems.

    Attributes:
        subsystem1: Tuple of type (string) and index (int) of the first subsystem involved in the interaction (e.g., ('qubit', 3))
        subsystem2: Tuple[str,int] of type (string) and index (int) of the second subsystem involved in the interaction (e.g., ('cavity', 1))
        interaction_type: InteractionType (ZZ, jaynes_cummings, etc.)
        parameters: Dict[str, Any] of interaction parameters or float
    """

    def __init__(
        self,
        subsystem1: Tuple[str, int],
        subsystem2: Tuple[str, int],
        interaction_type: InteractionType,
        parameters: Optional[Union[Dict[str, Any], float, complex]] = 1.0,
        time_modulation: Optional[Callable[[float], float]] = None,
    ):

        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.interaction_type = interaction_type
        self.parameters = parameters
        self.time_modulation = time_modulation

        self.__post_init__()

    def __post_init__(self):
        """Validate interaction parameters."""
        
        # Validate subsystem specifications
        if self.subsystem1 == self.subsystem2:
            raise ValueError("subsystem1 and subsystem2 must refer to different subsystems")
        if self.subsystem1[1] < 0 or self.subsystem2[1] < 0:
            raise ValueError("Subsystem indices must be non-negative")
        if not (self.subsystem1[0] in ['qubit', 'cavity', 'field'] and self.subsystem2[0] in ['qubit', 'cavity', 'field']):
            raise ValueError("Subsystem types must be 'qubit', 'cavity', or 'field'")
        
        # Ensure canonical ordering (sort by type and then index)
        if (self.subsystem1[0], self.subsystem1[1]) > (self.subsystem2[0], self.subsystem2[1]):
            self.subsystem1, self.subsystem2 = self.subsystem2, self.subsystem1

        # Validate time modulation function
        if self.time_modulation is not None and not callable(self.time_modulation):
            raise TypeError("time_modulation must be a callable function of time")
        elif self.time_modulation is not None:
            # Test the time modulation function with a sample time value
            try:
                for _ in range(100):
                    random_time = np.random.uniform(-10, 10)
                    test_value = self.time_modulation(random_time)
                    if not isinstance(test_value, (int, float)):
                        raise ValueError(f"time_modulation function must return a numeric value. Got type: {type(test_value)}")
            except Exception as e:
                raise ValueError("time_modulation function is not callable with a float argument") from e
             
        # Validate different interaction types
        if self.interaction_type in {InteractionType.ZZ, InteractionType.XX, InteractionType.YY}:
            self._validate_qubit_qubit_interaction()

        else:
            raise NotImplementedError(f"Interaction type {self.interaction_type} is not supported yet. The following interactions are implemented:\n\
                                      {[f'{interaction.value}' + f'\n' for interaction in InteractionType]}")

    def _validate_qubit_qubit_interaction(self):
        """Validate parameters for qubit-qubit interactions."""
        
        if self.subsystem1[0] != 'qubit' or self.subsystem2[0] != 'qubit':
            raise ValueError(f"{self.interaction_type.value} interaction must be between two qubits")
        elif isinstance(self.parameters, (int, float)):
            self.parameters = {"chi": self.parameters}
        elif not isinstance(self.parameters, dict):
            raise TypeError("Parameters for ZZ, XX, YY interactions must be a float or a dict with 'chi' key")
        elif "chi" not in self.parameters or "strength" not in self.parameters:
            raise ValueError("Parameters for ZZ, XX, YY interactions must include 'chi' key for interaction strength")
        
        if "strenght" in self.parameters:
            warnings.warn("Using 'strenght' value for 'chi'.", UserWarning)
            self.parameters["chi"] = self.parameters.pop("strenght")

        if self.parameters["chi"] < 0:
            raise ValueError(f"Qubit interaction strength (chi) must be >= 0, got {self.parameters['chi']}")
        elif self.parameters["chi"] == 0:
            warnings.warn(
                f"Qubit-qubit interaction strength (chi) is zero for qubits {self.subsystem1[1]}, {self.subsystem2[1]}. "
                "This means no direct qubit-qubit coupling, which may be intentional for uncoupled qubit experiments.",
                UserWarning,
                )


@dataclass
class PhysicalModel:
    """
    Physical constants and interactions for the quantum system.

    Attributes:
        n_cavities: Number of resonator cavities (typically 1 for single-mode systems)
        n_fields: Number of input field modes
        n_qubits: Number of qubits in the system
        cavities_levels: Number of levels for cavity modes (cavity truncation level)
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
    interactions: Optional[List[Interaction]] = None  # Qubit-qubit interactions

    def __post_init__(self):
        """Convert levels to list format if necessary and set default interactions."""
        
        # Normalizing levels to list format for consistency
        self.cavity_levels = self._normalize_levels(self.cavity_levels, self.n_cavities, "cavity")
        self.field_levels = self._normalize_levels(self.field_levels, self.n_fields, "field")
        self.qubit_levels = self._normalize_levels(self.qubit_levels, self.n_qubits, "qubit")

        # Set empty list of interactions if None
        if self.interactions is None:
            self.interactions = []

        # Validate interactions
        for interaction in self.interactions:
            if not isinstance(interaction, Interaction):
                raise TypeError("All interactions must be Interaction instances")
            # Check that interactions are between valid subsystems
            for subsystem in [interaction.subsystem1, interaction.subsystem2]:
                if subsystem[0] == 'cavity' and subsystem[1] >= self.n_cavities:
                    raise ValueError(
                        f"Interaction involves cavity {subsystem[1]}, but only {self.n_cavities} cavities in system"
                    )
                if subsystem[0] == 'field' and subsystem[1] >= self.n_fields:
                    raise ValueError(
                        f"Interaction involves field mode {subsystem[1]}, but only {self.n_fields} field modes in system"
                    )
                if subsystem[0] == 'qubit' and subsystem[1] >= self.n_qubits:
                    raise ValueError(
                        f"Interaction involves qubit {subsystem[1]}, but only {self.n_qubits} qubits in system"
                    )
                
    def _normalize_levels(self, levels: Union[int, List[int]], count: int, label: str) -> List[int]:
        """Normalize levels to list format."""
        if label == 'cavity':
            plural = 'cavities'
        else:
            plural = label + 's'

        if count == 0:
            warnings.warn(f"n_{plural} is set to 0. Setting {label}_levels to an empty list.", UserWarning)
            return []
        if isinstance(levels, int):
            return [levels] * count
        if isinstance(levels, list):
            if len(levels) != count:
                raise ValueError(
                    f"{label.capitalize()} levels list length ({len(levels)}) must match n_{plural} ({count})"
                )
            return levels
        raise TypeError(f"{label.capitalize()} levels must be an integer or a list of integers")


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
                [
                    Interaction(
                        subsystem1=interaction.subsystem1,
                        subsystem2=interaction.subsystem2,
                        interaction_type=interaction.interaction_type,                        
                        parameters = interaction.parameters,
                        time_modulation = interaction.time_modulation
                    )
                    for interaction in self.interactions
                ] 
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


@dataclass
class InitialState:
    """
    Initial state configuration.

    Attributes:
        state_type: Type of initial state
        coherent_alpha: Coherent state parameter for coherent states
        thermal_n_bar: Average photon number for thermal states
        custom_amplitudes: Custom state amplitudes for CUSTOM type
    """

    state_type: InitialStateType = InitialStateType.FOCK
    coherent_alpha: Optional[complex] = None
    thermal_n_bar: Optional[float] = None
    custom_amplitudes: Optional[Dict[Tuple[int, int, int], complex]] = None


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
            n_qubits: Number of qubits from PhysicalConstants
        """
        for attr in ["depolarizing", "dephasing", "relaxation"]:
            value = getattr(self, attr)
            if isinstance(value, (int, float)):
                setattr(self, attr, [float(value)] * n_qubits)
            elif isinstance(value, list):
                if len(value) != n_qubits:
                    raise ValueError(
                        f"{attr} list length ({len(value)}) must match n_qubits ({n_qubits})"
                    )
                value_list = list(value)  # narrow type for pylint
                setattr(self, attr, [float(v) for v in value_list])
            else:
                raise TypeError(f"{attr} must be a float or a list of floats")

@dataclass
class SystemConfiguration:
    """
    System configuration composed of initial state, noise model and configuration-specific interactions."""

    name: str
    initial_state: InitialState = InitialState()
    noise_model: Optional[NoiseModel] = None
    interactions: Optional[List[Interaction]] = None

    def __post_init__(self):
        """Validate interactions."""
        if not self.name:
            raise ValueError("System configuration must have a non-empty name")
        
        if self.interactions is not None:
            for interaction in self.interactions:
                if not isinstance(interaction, Interaction):
                    raise TypeError("All interactions must be Interaction instances")
        



class ExperimentalParameters:
    """
    Complete experimental configuration for quantum sensing protocols.

    This class contains all the system parameters that define the physical
    quantum sensing setup including Hilbert space dimensions, coupling constants,
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
        n_qubits = self.physical_model.n_qubits
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

        if interval <= 0:
            raise ValueError("Time interval must be positive")
        if final <= initial:
            raise ValueError("Final time must be greater than initial time")

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
            if not spec_stripped:
                raise ValueError("initial_time_uncertainty string specification cannot be empty")

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
                        f"'{spec}'. Supported keywords: 'max_interval', 'total_duration'."
                    ) from exc
        else:
            try:
                value = float(spec)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "initial_time_uncertainty must be a float or supported string keyword"
                ) from exc

        if value < 0:
            raise ValueError("Initial time uncertainty must be >= 0")
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
        for i, level in enumerate(self.physical_model.cavity_levels):
            if level < 2:
                raise ValueError(f"Every cavity must have at least 2 levels. Cavity_{i} got {level}")
        for i, level in enumerate(self.physical_model.field_levels):
            if level < 2:
                raise ValueError(f"Every field must have at least 2 levels. Field_{i} got {level}")
        for i, level in enumerate(self.physical_model.qubit_levels):
            if level < 2:
                raise ValueError(f"Every qubit must have at least 2 levels. Qubit_{i} got {level}")

        # Validate noise rates (now lists)
        if not isinstance(self.noise_model.depolarizing, list):
            raise TypeError("depolarizing must be normalized to a list")
        if not isinstance(self.noise_model.dephasing, list):
            raise TypeError("dephasing must be normalized to a list")
        if not isinstance(self.noise_model.relaxation, list):
            raise TypeError("relaxation must be normalized to a list")

        for i, rate in enumerate(self.noise_model.depolarizing):
            if rate < 0:
                raise ValueError(f"Depolarization rate for qubit {i} must be >= 0, got {rate}")
        for i, rate in enumerate(self.noise_model.dephasing):
            if rate < 0:
                raise ValueError(f"Dephasing rate for qubit {i} must be >= 0, got {rate}")
        for i, rate in enumerate(self.noise_model.relaxation):
            if rate < 0:
                raise ValueError(f"Relaxation rate for qubit {i} must be >= 0, got {rate}")

        # Validate measurement protocol
        if self.measurement.time_interval <= 0:
            raise ValueError("Time interval must be positive")
        if self.measurement.final_time <= self.measurement.initial_time:
            raise ValueError("Final time must be greater than initial time")
        # Resolve initial time uncertainty to ensure specification is valid
        _ = self._resolve_initial_time_uncertainty()

        # Validate measurement times (len > 1 and sorted)
        if self._measurement_times_list is None:
            self._update_measurement_times()

        times_list = self._measurement_times_list
        if times_list is None:
            raise ValueError("Measurement times could not be computed")

        if len(times_list) < 2:
            raise ValueError("At least two measurement times must be specified")
        if sorted(times_list) != times_list:
            raise ValueError("Measurement times must be in ascending order")

        # Validate configuration set
        
        #validate configuration_set
        if not isinstance(self.configuration_set, (list, set)) or len(self.configuration_set) <= 1:
            raise NotImplementedError(
                "Please provide a list or set of SystemConfiguration instances with at least 2 elements "
                "(Preset configuration list is not implemented yet)."
            )

        self.configuration_set = list(self.configuration_set)

        for config in self.configuration_set:
            if not isinstance(config, SystemConfiguration):
                raise TypeError("All items in configuration_set must be SystemConfiguration instances")

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
        self.physical_model.cavity_levels = value
        # Re-normalize if necessary
        if hasattr(self, "physical_model"):
            self.physical_model._normalize_levels(self.physical_model.cavity_levels, self.n_cavities,"cavity")

    @property
    def field_levels(self) -> Union[int, List[int]]:
        """Direct access to field levels."""
        return self.physical_model.field_levels

    @field_levels.setter
    def field_levels(self, value: Union[int, List[int]]) -> None:
        """Set field levels."""
        self.physical_model.field_levels = value
        # Re-normalize if necessary
        if hasattr(self, "physical_model"):
            self.physical_model._normalize_levels(self.physical_model.field_levels, self.n_fields,"field")

    @property
    def qubit_levels(self) -> Union[int, List[int]]:
        """Direct access to qubit levels (returns list after normalization)."""
        return self.physical_model.qubit_levels

    @qubit_levels.setter
    def qubit_levels(self, value: Union[int, List[int]]) -> None:
        """Set qubit levels."""
        self.physical_model.qubit_levels = value
        # Re-normalize if necessary
        if hasattr(self, "physical_model"):
            self.physical_model._normalize_levels(self.physical_model.qubit_levels, self.n_qubits,"qubit")
    
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
    
    def get_all_configuration_names(self) -> List[str]:
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
        if name in self.configuration_set:
            del self.configuration_set[name]

    def copy(self, **updates) -> "ExperimentalParameters":
        """
        Create a copy of ExperimentalParameters with optional updates.

        This method creates a new ExperimentalParameters instance with all
        configuration copied. The nested objects (physical_constants, system_dims,
        measurement, initial_state, noise_model) are deep copied to avoid
        unintended sharing of mutable state.

        Args:
            **updates: Keyword arguments for attributes to update. Can be:
                - physical_model: PhysicalConstants instance or dict of updates
                - noise_model: NoiseConfiguration instance
                - measurement: MeasurementProtocol instance
                - configuration_set: Set or list of SystemConfiguration instances
                - random_seed: int or None

        Returns:
            New ExperimentalParameters instance with updated values

        Example:
            >>> # Copy and update physical constants
            >>> new_params = exp_params.copy(
            ...     physical_model=exp_params.physical_model.copy(chi=10.0)
            ... )
            >>>
            >>> # Or pass updates as dict (for convenience)
            >>> new_params = exp_params.copy(
            ...     physical_model={'chi': 10.0, 'photon_cavity_coupling': 20.0}
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
            depol = self.noise_model.depolarizing
            deph = self.noise_model.dephasing
            relax = self.noise_model.relaxation

            new_noise_model = NoiseModel(
                depolarizing=depol.copy() if isinstance(depol, list) else depol,
                dephasing=deph.copy() if isinstance(deph, list) else deph,
                relaxation=relax.copy() if isinstance(relax, list) else relax,
                custom_operators=(
                    self.noise_model.custom_operators.copy()
                    if self.noise_model.custom_operators
                    else None
                ),
            )

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
            new_config_set = [config for config in self.configuration_set]


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
        n_qubits = self.physical_constants.n_qubits
        qubit_levels_list = self.system_dims.qubit_levels
        if isinstance(qubit_levels_list, list):
            qubit_dim = np.prod(qubit_levels_list)
        else:
            qubit_dim = qubit_levels_list

        total_dim = self.system_dims.cavity_levels * qubit_dim * self.system_dims.field_levels
        lines.append(f"  Number of qubits:     {n_qubits:>6}")
        lines.append(f"  Cavity levels:        {self.system_dims.cavity_levels:>6}")
        lines.append(f"  Qubit levels:         {self.system_dims.qubit_levels}")
        lines.append(f"  Field levels:         {self.system_dims.field_levels:>6}")
        lines.append(f"  Total dimension:      {total_dim:>6}")

        # Physical Constants Group
        lines.append("PHYSICAL CONSTANTS")
        lines.append(f"  Chi:                  {self.physical_constants.chi}")
        lines.append(
            f"  Photon cavity coupling: {self.physical_constants.photon_cavity_coupling:>6.4f}"
        )
        lines.append(f"  Inverse pulse width:  {self.physical_constants.inverse_pulse_width:>8.4f}")

        # Qubit Interactions
        if self.physical_constants.qubit_interactions:
            lines.append(
                f"  Qubit interactions:   {len(self.physical_constants.qubit_interactions)} interaction(s)"
            )
            for i, interaction in enumerate(self.physical_constants.qubit_interactions):
                lines.append(
                    f"    [{i}] Qubits {interaction.qubit_indices}: "
                    f"{interaction.interaction_type.value}, χ={interaction.chi:.4f}"
                )
        else:
            lines.append("  Qubit interactions:   None")

        # Measurement Protocol Group
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
        # Initial State Configuration Group
        lines.append("INITIAL STATE")
        lines.append(f"  Type:                 {self.initial_state.state_type.value}")

        # Noise Configuration Group
        lines.append("NOISE MODEL")
        lines.append(f"  Depolarizing rate:    {self.noise_model.depolarizing}")
        lines.append(f"  Dephasing rate:       {self.noise_model.dephasing}")
        lines.append(f"  Relaxation rate:      {self.noise_model.relaxation}")

        if self.noise_model.custom_operators is not None:
            lines.append(f"  Custom operators:     {len(self.noise_model.custom_operators):>6}")
        else:
            lines.append("  Custom operators:     None")

        # Overall System Status
        lines.append("SYSTEM STATUS")

        try:
            self._validate_configuration()
            lines.append("  Configuration:        VALID")
        except ValueError as e:
            lines.append("  Configuration:        INVALID")
            lines.append(f"  Error:                {str(e)}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation (calls __repr__)."""
        return self.__repr__()

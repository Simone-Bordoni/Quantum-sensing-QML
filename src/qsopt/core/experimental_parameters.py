"""
Experimental Parameters Class
============================

System configuration parameters for quantum sensing experiments including
physical constants, system dimensions, measurement protocols, and initial states.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np


class InitialStateType(Enum):
    """Enumeration of supported initial state configurations."""

    VACUUM = "vacuum"  # |0,0,0⟩
    SINGLE_PHOTON = "single_photon"  # |1,0,0⟩
    COHERENT = "coherent"  # |α,0,0⟩ with qubit ground
    THERMAL = "thermal"  # Thermal state with qubit ground
    CUSTOM = "custom"  # User-defined state amplitudes


@dataclass
class PhysicalConstants:
    """
    Physical constants and coupling parameters for the quantum system.

    Attributes:
        chi: Dispersive coupling strength between resonator and qubit (Hz)
        photon_cavity_coupling: Photon-cavity coupling strength (Hz)
        inverse_pulse_width: Inverse of the pulse width (1/time units, typically 1/ns)
    """

    chi: float = 0.5  # Dispersive coupling in units of cavity decay rate
    photon_cavity_coupling: float = 1.0  # Photon-cavity coupling
    inverse_pulse_width: float = 0.1  # Inverse pulse width parameter


@dataclass
class SystemDimensions:
    """
    Hilbert space dimensions for the composite quantum system.

    The total system consists of three subsystems:
    - Field mode (field_levels)
    - Resonator cavity mode (cavity_levels)
    - Qubit (qubit_levels)

    Total Hilbert space dimension = field_levels × cavity_levels × qubit_levels

    Attributes:
        cavity_levels: Number of levels for cavity modes
        qubit_levels: Number of levels for qubit (typically 2)
        field_levels: Number of levels for field modes
    """

    cavity_levels: int = 2  # Cavity truncation level
    qubit_levels: int = 2  # Qubit levels
    field_levels: int = 2  # Field mode levels

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
class InitialStateConfig:
    """
    Initial state configuration.

    Attributes:
        state_type: Type of initial state
        coherent_alpha: Coherent state parameter for coherent states
        thermal_n_bar: Average photon number for thermal states
        custom_amplitudes: Custom state amplitudes for CUSTOM type
    """

    state_type: InitialStateType = InitialStateType.SINGLE_PHOTON
    coherent_alpha: Optional[complex] = None
    thermal_n_bar: Optional[float] = None
    custom_amplitudes: Optional[Dict[Tuple[int, int, int], complex]] = None


@dataclass
class NoiseConfiguration:
    """
    Noise model configuration.

    Attributes:
        depolarizing: Depolarization rate
        dephasing: Dephasing rate
        relaxation: Relaxation rate
        custom_operators: Custom Lindblad operators
    """

    depolarizing: float = 0.0  # Depolarization rate
    dephasing: float = 0.0  # Dephasing rate
    relaxation: float = 0.0  # Relaxation rate
    custom_operators: Optional[List[Any]] = None  # Custom Lindblad operators


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
        physical_constants: Optional[PhysicalConstants] = None,
        system_dims: Optional[SystemDimensions] = None,
        measurement: Optional[MeasurementProtocol] = None,
        initial_state: Optional[InitialStateConfig] = None,
        noise_config: Optional[NoiseConfiguration] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize experimental parameters.

        Args:
            physical_constants: Physical coupling constants and rates
            system_dims: Hilbert space dimensions
            measurement: Measurement protocol configuration
            initial_state: Initial state configuration
            noise_config: Noise model configuration
            random_seed: Random seed for reproducibility of uncertainty calculations
        """
        self.physical_constants = physical_constants or PhysicalConstants()
        self.system_dims = system_dims or SystemDimensions()
        self.measurement = measurement or MeasurementProtocol()
        self.noise_config = noise_config or NoiseConfiguration()
        self.initial_state = initial_state or InitialStateConfig()
        
        # Random seed for uncertainty calculations
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Computed measurement times list (denormalized)
        self._measurement_times_list: Optional[List[float]] = None
        self._update_measurement_times()

        # Validation
        self._validate_configuration()

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
        grid = np.arange(initial, final + interval/2, interval, dtype=float)
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
        else:
            self._measurement_times_list = self._compute_measurement_times_from_interval()

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

    def get_measurement_times_with_uncertainty(self, batch_size: int = 1) -> np.ndarray:
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

    def _validate_configuration(self) -> None:
        """Validate parameter consistency and physical constraints."""
        # Validate system dimensions
        if self.system_dims.cavity_levels < 2:
            raise ValueError("Cavity levels (cavity_levels) must be >= 2")
        if self.system_dims.field_levels < 2:
            raise ValueError("External field levels (field_levels) must be >= 2")
        if self.system_dims.qubit_levels < 2:
            raise ValueError("Qubit levels (qubit_levels) must be >= 2")

        # Validate coupling constants
        if self.physical_constants.chi <= 0:
            raise ValueError("Dispersive coupling (chi) must be > 0")
        if self.physical_constants.photon_cavity_coupling <= 0:
            raise ValueError("Photon-cavity coupling (photon_cavity_coupling) must be > 0")
        if self.physical_constants.inverse_pulse_width <= 0:
            raise ValueError("Pulse width parameter (inverse_pulse_width) must be > 0")

        # Validate noise rates
        if self.noise_config.depolarizing < 0:
            raise ValueError("Depolarization rate must be >= 0")
        if self.noise_config.dephasing < 0:
            raise ValueError("Dephasing rate must be >= 0")
        if self.noise_config.relaxation < 0:
            raise ValueError("Relaxation rate must be >= 0")

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

    # Direct access to commonly used parameters for easier integration

    @property
    def cavity_levels(self) -> int:
        """Direct access to cavity levels."""
        return self.system_dims.cavity_levels

    @cavity_levels.setter
    def cavity_levels(self, value: int) -> None:
        """Set cavity levels."""
        self.system_dims.cavity_levels = value

    @property
    def qubit_levels(self) -> int:
        """Direct access to qubit levels."""
        return self.system_dims.qubit_levels

    @qubit_levels.setter
    def qubit_levels(self, value: int) -> None:
        """Set qubit levels."""
        self.system_dims.qubit_levels = value

    @property
    def field_levels(self) -> int:
        """Direct access to field levels."""
        return self.system_dims.field_levels

    @field_levels.setter
    def field_levels(self, value: int) -> None:
        """Set field levels."""
        self.system_dims.field_levels = value

    @property
    def chi(self) -> float:
        """Direct access to dispersive coupling."""
        return self.physical_constants.chi

    @chi.setter
    def chi(self, value: float) -> None:
        """Set dispersive coupling."""
        self.physical_constants.chi = value

    @property
    def photon_cavity_coupling(self) -> float:
        """Direct access to photon-cavity coupling."""
        return self.physical_constants.photon_cavity_coupling

    @photon_cavity_coupling.setter
    def photon_cavity_coupling(self, value: float) -> None:
        """Set photon-cavity coupling."""
        self.physical_constants.photon_cavity_coupling = value

    @property
    def inverse_pulse_width(self) -> float:
        """Direct access to pulse width parameter."""
        return self.physical_constants.inverse_pulse_width

    @inverse_pulse_width.setter
    def inverse_pulse_width(self, value: float) -> None:
        """Set pulse width parameter."""
        self.physical_constants.inverse_pulse_width = value

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

    def __repr__(self) -> str:
        """
        Comprehensive string representation showing all parameters organized by groups.

        This method provides a detailed display of all experimental
        parameters, organized by their logical groups with validation status flags.
        """
        lines = []
        # System Dimensions Group
        lines.append("SYSTEM DIMENSIONS")
        total_dim = (
            self.system_dims.cavity_levels
            * self.system_dims.qubit_levels
            * self.system_dims.field_levels
        )
        lines.append(f"  Cavity levels:        {self.system_dims.cavity_levels:>6}")
        lines.append(f"  Qubit levels:         {self.system_dims.qubit_levels:>6}")
        lines.append(f"  Field levels:         {self.system_dims.field_levels:>6}")
        lines.append(f"  Total dimension:      {total_dim:>6}")

        # Validation flags for dimensions
        dim_valid = (
            self.system_dims.cavity_levels >= 2
            and self.system_dims.qubit_levels >= 2
            and self.system_dims.field_levels >= 2
        )

        # Physical Constants Group
        lines.append("PHYSICAL CONSTANTS")
        lines.append(f"  Chi:                  {self.physical_constants.chi:>8.4f}")
        lines.append(
            f"  Photon cavity coupling: {self.physical_constants.photon_cavity_coupling:>6.4f}"
        )
        lines.append(f"  Inverse pulse width:  {self.physical_constants.inverse_pulse_width:>8.4f}")

        # Validation flags for constants
        const_valid = (
            self.physical_constants.chi > 0
            and self.physical_constants.photon_cavity_coupling > 0
            and self.physical_constants.inverse_pulse_width > 0
        )

        # Measurement Protocol Group
        lines.append("MEASUREMENT PROTOCOL")
        
        # Determine mode (list or interval)
        times_list = self._measurement_times_list if self._measurement_times_list is not None else []
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
                lines.append(
                    f"    (specified as '{self.measurement.initial_time_uncertainty}')"
                )
        # Initial State Configuration Group
        lines.append("INITIAL STATE")
        lines.append(f"  Type:                 {self.initial_state.state_type.value}")

        # Noise Configuration Group
        lines.append("NOISE MODEL")
        lines.append(f"  Depolarizing rate:    {self.noise_config.depolarizing:>8.4f}")
        lines.append(f"  Dephasing rate:       {self.noise_config.dephasing:>8.4f}")
        lines.append(f"  Relaxation rate:      {self.noise_config.relaxation:>8.4f}")

        if self.noise_config.custom_operators is not None:
            lines.append(f"  Custom operators:     {len(self.noise_config.custom_operators):>6}")
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

        return '\n'.join(lines)
    
    def __str__(self) -> str:
        """String representation (calls __repr__)."""
        return self.__repr__()
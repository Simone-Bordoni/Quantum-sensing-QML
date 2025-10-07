"""
Experimental Parameters Class
============================

System configuration parameters for quantum sensing experiments including
physical constants, system dimensions, measurement protocols, and initial states.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class InitialStateType(Enum):
    """Enumeration of supported initial state configurations."""

    VACUUM_GROUND = "vacuum_ground"  # |0,0,0⟩
    SINGLE_PHOTON = "single_photon"  # |1,0,0⟩
    COHERENT_GROUND = "coherent_ground"  # |α,0,0⟩ with qubit ground
    THERMAL_GROUND = "thermal_ground"  # Thermal state with qubit ground
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

    Attributes:
        measurement_times: List of measurement times (in units of inverse_pulse_width)
    """

    measurement_times: List[float] = field(default_factory=lambda: [-5.0, 5.0])


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
    ):
        """
        Initialize experimental parameters.

        Args:
            physical_constants: Physical coupling constants and rates
            system_dims: Hilbert space dimensions
            measurement: Measurement protocol configuration
            initial_state: Initial state configuration
        """
        self.physical_constants = physical_constants or PhysicalConstants()
        self.system_dims = system_dims or SystemDimensions()
        self.measurement = measurement or MeasurementProtocol()
        self.noise_config = noise_config or NoiseConfiguration()
        self.initial_state = initial_state or InitialStateConfig()

        # Computed measurement times
        self._measurement_times = None
        self._update_measurement_times()

        # Additional attributes for backward compatibility
        self._measurement_results = [0, 1]  # Default measurement results

        # Validation
        self._validate_configuration()

    def _update_measurement_times(self) -> None:
        """Compute measurement times from protocol configuration."""
        self._measurement_times = (
            np.array(self.measurement.measurement_times)
            * self.physical_constants.inverse_pulse_width
        )

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

        # Validate measurement times (len > 1 and sorted)
        if len(self.measurement.measurement_times) < 2:
            raise ValueError("At least two measurement times must be specified")
        if sorted(self.measurement.measurement_times) != self.measurement.measurement_times:
            raise ValueError("Measurement times (measurement_times) must be in ascending order")

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
        """Direct access to measurement times."""
        if self._measurement_times is None:
            self._update_measurement_times()
        return self._measurement_times  # type: ignore

    @measurement_times.setter
    def measurement_times(self, value: np.ndarray) -> None:
        """Set measurement times."""
        self.measurement.measurement_times = list(
            value / self.physical_constants.inverse_pulse_width
        )
        self._update_measurement_times()

    def __str__(self) -> str:
        """
        Comprehensive string representation showing all parameters organized by groups.

        This method provides a detailed, human-readable display of all experimental
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
        status = "VALID" if dim_valid else "INVALID"
        lines.append(f"  Status:               {status}")

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
        status = "VALID" if const_valid else "INVALID"
        lines.append(f"  Status:               {status}")

        # Measurement Protocol Group
        lines.append("MEASUREMENT PROTOCOL") 
        n_measurements = len(self.measurement.measurement_times)
        lines.append(f"  Number of measurements: {n_measurements:>6}")
        lines.append(f"  Measurement times: {self.measurement.measurement_times}")

        # Validation flags for measurements
        meas_valid = (
            n_measurements >= 2
            and sorted(self.measurement.measurement_times) == self.measurement.measurement_times
        )
        status = "VALID" if meas_valid else "INVALID"
        lines.append(f"  Status:               {status}")

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

        # Validation flags for noise
        noise_valid = (
            self.noise_config.depolarizing >= 0
            and self.noise_config.dephasing >= 0
            and self.noise_config.relaxation >= 0
        )
        status = "VALID" if noise_valid else "INVALID"
        lines.append(f"  Status:               {status}")

        # Overall System Status
        lines.append("SYSTEM STATUS")

        try:
            self._validate_configuration()
            lines.append("  Configuration:        VALID")
        except ValueError as e:
            lines.append("  Configuration:        INVALID")
            lines.append(f"  Error:                {str(e)}")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        """Compact string representation of experimental parameters."""
        total_dim = (
            self.system_dims.cavity_levels
            * self.system_dims.field_levels
            * self.system_dims.qubit_levels
        )
        n_meas = len(self.measurement.measurement_times)
        chi_coupling_ratio = (
            self.physical_constants.chi / self.physical_constants.photon_cavity_coupling
        )
        dims_str = (
            f"{self.system_dims.field_levels}x{self.system_dims.cavity_levels}"
            f"x{self.system_dims.qubit_levels}={total_dim}"
        )
        return (
            f"ExperimentalParameters("
            f"dims={dims_str}, "
            f"chi / coupling_ratio={chi_coupling_ratio:.2f}, "
            f"number of measurements={n_meas}, "
            f"state={self.initial_state.state_type.value})"
        )

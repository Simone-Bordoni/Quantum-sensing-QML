"""
Experimental Parameters Class
============================

System configuration parameters for quantum sensing experiments including
physical constants, system dimensions, measurement protocols, and initial states.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class InitialStateType(Enum):
    """Enumeration of supported initial state configurations (for input field)."""

    VACUUM = "vacuum"
    SINGLE_PHOTON = "single_photon"
    COHERENT = "coherent"
    THERMAL = "thermal"
    CUSTOM = "custom"


class InteractionType(Enum):
    """Enumeration of supported qubit-qubit interaction types."""

    ZZ = "sz-sz"  # σz ⊗ σz interaction
    XX = "sx-sx"  # σx ⊗ σx interaction
    YY = "sy-sy"  # σy ⊗ σy interaction


@dataclass
class QubitInteraction:
    """
    Configuration for qubit-qubit interaction.

    Attributes:
        qubit_indices: Tuple of qubit indices involved in the interaction (e.g., (0, 1))
        chi: Interaction strength (coupling constant)
        interaction_type: Type of interaction (ZZ, XX, or YY)
    """

    qubit_indices: Tuple[int, int] = (0, 1)
    chi: float = 0.0
    interaction_type: InteractionType = InteractionType.ZZ

    def __post_init__(self):
        """Validate interaction parameters."""
        if len(self.qubit_indices) != 2:
            raise ValueError("qubit_indices must be a tuple of exactly 2 indices")
        if self.qubit_indices[0] == self.qubit_indices[1]:
            raise ValueError("qubit_indices must refer to different qubits")
        if self.qubit_indices[0] < 0 or self.qubit_indices[1] < 0:
            raise ValueError("qubit_indices must be non-negative")
        # Ensure canonical ordering (smaller index first)
        if self.qubit_indices[0] > self.qubit_indices[1]:
            self.qubit_indices = (self.qubit_indices[1], self.qubit_indices[0])

        # Validate interaction strength
        if self.chi < 0:
            raise ValueError(f"Qubit interaction strength (chi) must be >= 0, got {self.chi}")
        elif self.chi == 0:
            warnings.warn(
                f"Qubit-qubit interaction strength (chi) is zero for qubits {self.qubit_indices}. "
                "This means no direct qubit-qubit coupling, which may be intentional for uncoupled qubit experiments.",
                UserWarning,
            )


@dataclass
class PhysicalSetup:
    """
    Physical constants and coupling parameters for the quantum system.

    Attributes:
        n_qubits: Number of qubits in the system
        qubit_cavity_coupling: Dispersive coupling strength between resonator(s) and qubit(s) (Hz).
             Can be a float (same coupling for all qubits) 
             or a dictionary with keys as (qubit_index, cavity_index) tuples and values as float coupling strengths.
        cavity_cavity_coupling: Coupling strength between cavities (Hz).
             Can be a float (same coupling for all cavity pairs) 
             or a dictionary with keys as (cavity_index1, cavity_index2) tuples and values as float coupling strengths.
        qubit_interactions: List of qubit-qubit interactions. For two-qubit systems,
                           defaults to a single ZZ interaction between qubits 0 and 1.
                           For single-qubit systems, this is ignored.
        qubit_cavity_time_modulation: Time dependent modulation function of the coupling between qubits and cavities. 
             Dictionary with keys as (qubit_index, cavity_index) tuples and values as functions of time.
        cavity_cavity_time_modulation: Time dependent modulation function of the coupling between cavities.
             Dictionary with keys as (cavity_index1, cavity_index2) tuples and values as functions of time.
        inverse_pulse_width: Inverse of the pulse width (1/time units, typically 1/ns)
    """

    n_qubits: int = 1  # Number of qubits
    n_cavities: int = 2  # Number of cavities (default 2 for input field and resonator)
    qubit_cavity_coupling: Union[float, Dict[Tuple[int, int], float]] = 0.5  # Dispersive coupling in units of cavity decay rate
    cavity_cavity_coupling: Union[float, Dict[Tuple[int, int], float]] = 1.0  # Cavity-cavity coupling
    qubit_cavity_time_modulation: Optional[Dict[Tuple[int, int], Callable[[float], float]]] = None  # Optional time modulation functions for qubit-cavity coupling
    cavity_cavity_time_modulation: Optional[Dict[Tuple[int, int], Callable[[float], float]]] = None  # Optional time modulation functions for cavity-cavity coupling
    qubit_interactions: Optional[List[QubitInteraction]] = None  # Qubit-qubit interactions
    inverse_pulse_width: float = 0.1  # Inverse pulse width parameter

    def __post_init__(self):
        """Convert qubit_cavity_coupling to dictionary format if necessary and set default interactions."""
        if isinstance(self.qubit_cavity_coupling, (int, float)):
            self.qubit_cavity_coupling = { (i, j): float(self.qubit_cavity_coupling) for i in range(self.n_qubits) for j in range(self.n_cavities) }
        elif not isinstance(self.qubit_cavity_coupling, dict):
            raise TypeError("qubit_cavity_coupling must be a float or a dictionary")

        if isinstance(self.cavity_cavity_coupling, (int, float)):
            self.cavity_cavity_coupling = { (i, j): float(self.cavity_cavity_coupling) for i in range(self.n_cavities) for j in range(i) }
        elif not isinstance(self.cavity_cavity_coupling, dict):
            raise TypeError("cavity_cavity_coupling must be a float or a dictionary")
        
        # Set default time modulations and qubit interactions if not provided
        if self.cavity_cavity_time_modulation is None:
            self.cavity_cavity_time_modulation = {}
        if self.qubit_cavity_time_modulation is None:
            self.qubit_cavity_time_modulation = {}
        if self.qubit_interactions is None:
            self.qubit_interactions = []

        # Validate interactions
        for interaction in self.qubit_interactions:
            if not isinstance(interaction, QubitInteraction):
                raise TypeError("All qubit_interactions must be QubitInteraction instances")
            # Check that qubit indices are valid
            for idx in interaction.qubit_indices:
                if idx >= self.n_qubits:
                    raise ValueError(
                        f"Interaction involves qubit {idx}, but only {self.n_qubits} qubits in system"
                    )

    def copy(self, **updates) -> "PhysicalSetup":
        """
        Create a copy of PhysicalSetup with optional parameter updates.

        This method creates a new PhysicalSetup instance with all attributes
        copied from the current instance. You can override specific attributes
        by passing them as keyword arguments.

        Args:
            **updates: Keyword arguments for attributes to update in the copy.
                      Valid keys: n_qubits, n_cavities, qubit_cavity_coupling, cavity_cavity_coupling, 
                      qubit_interactions, qubit_cavity_time_modulation, cavity_cavity_time_modulation,
                      inverse_pulse_width

        Returns:
            New PhysicalSetup instance with updated values

        Example:
            >>> original = PhysicalSetup(qubit_cavity_coupling=5.0, cavity_cavity_coupling=10.0)
            >>> modified = original.copy(qubit_cavity_coupling=8.0)  # Keep all other params, change qubit_cavity_coupling
            >>> modified.qubit_cavity_coupling
            8.0
            >>> modified.cavity_cavity_coupling
            10.0
        """
        # Start with current values
        params = {
            "n_qubits": self.n_qubits,
            "n_cavities": self.n_cavities,
            "qubit_cavity_coupling": self.qubit_cavity_coupling.copy(),
            "cavity_cavity_coupling": self.cavity_cavity_coupling.copy(),
            "qubit_cavity_time_modulation": self.qubit_cavity_time_modulation.copy(),
            "cavity_cavity_time_modulation": self.cavity_cavity_time_modulation.copy(),
            "qubit_interactions": (
                [
                    QubitInteraction(
                        qubit_indices=interaction.qubit_indices,
                        chi=interaction.chi,
                        interaction_type=interaction.interaction_type,
                    )
                    for interaction in self.qubit_interactions
                ]
                if self.qubit_interactions
                else []
            ),
            "inverse_pulse_width": self.inverse_pulse_width,
        }

        # Apply updates
        params.update(updates)

        return PhysicalSetup(**params)


@dataclass
class SystemDimensions:
    """
    Hilbert space dimensions for the composite quantum system.

    The total system consists of subsystems:
    - Field mode (field_levels)
    - Resonator cavity mode (cavity_levels)
    - Qubit(s) (qubit_levels per qubit)

    Total Hilbert space dimension = field_levels × cavity_levels × q1_levels × ... × qn_levels

    Attributes:
        cavity_levels: Number of levels for cavity modes
        qubit_levels: Number of levels for qubit(s) (typically 2).
                     Can be an int (same for all qubits) or a list of ints
                     (individual levels per qubit).
        field_levels: Number of levels for field modes
    """

    cavity_levels: Union[int, List[int]] = 2  # Cavity truncation level
    qubit_levels: Union[int, List[int]] = 2  # Qubit levels

    def __post_init__(self):
        """Store original input and convert qubit_levels to list format if necessary."""
        # We need n_qubits from PhysicalSetup, but we can't access it here
        # So we'll handle this in the ExperimentalParameters.__init__

    def _normalize_cavity_levels(self, n_cavities: int):
        """
        Normalize cavity_levels to list format.

        Args:
            n_cavities: Number of cavities from PhysicalSetup
        """
        if isinstance(self.cavity_levels, int):
            self.cavity_levels = [self.cavity_levels] * n_cavities
        elif isinstance(self.cavity_levels, list):
            if len(self.cavity_levels) != n_cavities:
                raise ValueError(
                    f"cavity_levels list length ({len(self.cavity_levels)}) must match n_cavities ({n_cavities})"
                )
            self.cavity_levels = [int(c) for c in self.cavity_levels]
        else:
            raise TypeError("cavity_levels must be an int or a list of ints")

    def _normalize_qubit_levels(self, n_qubits: int):
        """
        Normalize qubit_levels to list format.

        Args:
            n_qubits: Number of qubits from PhysicalSetup
        """
        if isinstance(self.qubit_levels, int):
            self.qubit_levels = [self.qubit_levels] * n_qubits
        elif isinstance(self.qubit_levels, list):
            if len(self.qubit_levels) != n_qubits:
                raise ValueError(
                    f"qubit_levels list length ({len(self.qubit_levels)}) must match n_qubits ({n_qubits})"
                )
            self.qubit_levels = [int(q) for q in self.qubit_levels]
        else:
            raise TypeError("qubit_levels must be an int or a list of ints")


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
            n_qubits: Number of qubits from PhysicalSetup
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
        physical_setup: Optional[PhysicalSetup] = None,
        system_dims: Optional[SystemDimensions] = None,
        measurement: Optional[MeasurementProtocol] = None,
        initial_state: Optional[InitialStateConfig] = None,
        noise_config: Optional[NoiseConfiguration] = None,
        random_seed: Optional[int] = None,        
    ):
        """
        Initialize experimental parameters.

        Args:
            physical_setup: Physical coupling constants and rates
            system_dims: Hilbert space dimensions
            measurement: Measurement protocol configuration
            initial_state: Initial state configuration
            noise_config: Noise model configuration
            random_seed: Random seed for reproducibility of uncertainty calculations
        """
        self.physical_setup = physical_setup or PhysicalSetup()
        self.system_dims = system_dims or SystemDimensions()
        self.measurement = measurement or MeasurementProtocol()
        self.noise_config = noise_config or NoiseConfiguration()
        self.initial_state = initial_state or InitialStateConfig()

        # Normalize multi-qubit parameters based on n_qubits
        n_qubits = self.physical_setup.n_qubits
        self.system_dims._normalize_qubit_levels(n_qubits)
        self.noise_config._normalize_noise_rates(n_qubits)

        # Normalize multi-cavity parameters based on n_cavities
        n_cavities = self.physical_setup.n_cavities
        self.system_dims._normalize_cavity_levels(n_cavities)

        # Compute total system dimensions
        qubit_dim = np.prod(self.system_dims.qubit_levels)
        cavity_dim = np.prod(self.system_dims.cavity_levels)
        self.system_dims.total_dim = cavity_dim * qubit_dim

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
        if not isinstance(self.system_dims.cavity_levels, list):
            raise TypeError("cavity_levels must be normalized to a list")
        for i, levels in enumerate(self.system_dims.cavity_levels):
            if levels < 2:
                raise ValueError(f"Cavity {i} levels must be >= 2, got {levels}")

        # Validate qubit levels (now a list)
        if not isinstance(self.system_dims.qubit_levels, list):
            raise TypeError("qubit_levels must be normalized to a list")
        for i, levels in enumerate(self.system_dims.qubit_levels):
            if levels < 2:
                raise ValueError(f"Qubit {i} levels must be >= 2, got {levels}")

        # Validate coupling constants (qubit_cavity_coupling is now a list)
        if not isinstance(self.physical_setup.qubit_cavity_coupling, Dict[Tuple[int,int], float]):
            raise TypeError("qubit_cavity_coupling must be normalized to a dictionary")
        for (qubit, cavity), chi_val in self.physical_setup.qubit_cavity_coupling.items():
            if chi_val < 0:
                raise ValueError(
                    f"Dispersive coupling (qubit_cavity_coupling) for qubit {qubit} with cavity {cavity} must be >= 0, got {chi_val}"
                )
            elif chi_val == 0:
                warnings.warn(
                    f"Dispersive coupling (qubit_cavity_coupling) for qubit {qubit} with cavity {cavity} is zero. "
                    "This means no qubit-cavity interaction for this qubit, "
                    "which may not produce meaningful sensing results."
                    "Removing this coupling altogheter might be advisable if intentional.",
                    UserWarning,
                )
            
            if not isinstance(qubit, int):
                raise TypeError(f"Qubit indices in qubit_cavity_coupling must be integers, got {qubit}")
            elif not (0 <= qubit < self.physical_setup.n_qubits):
                raise ValueError(f"Qubit index ({qubit}) in qubit_cavity_coupling ({qubit}, {cavity}) is out of range for n_qubits={self.physical_setup.n_qubits}")

            if not isinstance(cavity, int):
                raise TypeError(f"Cavity indices in qubit_cavity_coupling must be integers, got {cavity}")
            elif not (0 <= cavity < self.physical_setup.n_cavities):
                raise ValueError(f"Cavity index ({cavity}) in qubit_cavity_coupling ({qubit}, {cavity}) is out of range for n_cavities={self.physical_setup.n_cavities}")


        for (cavity1, cavity2), coupling_val in self.physical_setup.cavity_cavity_coupling.items():
            if coupling_val < 0:
                raise ValueError(f"Cavity-cavity coupling (cavity_cavity_coupling) for cavities {cavity1} and {cavity2} must be >= 0")
            elif coupling_val == 0:
                warnings.warn(
                    f"Cavity-cavity coupling (cavity_cavity_coupling) for cavities {cavity1} and {cavity2} is zero. This means no "
                    "coupling between cavities, which may not produce meaningful sensing results."
                    "Removing this coupling altogheter might be advisable if intentional.",
                    UserWarning,
                )
        
            if not isinstance(cavity1, int):
                raise TypeError(f"Cavity indices in cavity_cavity_coupling must be integers, got {cavity1}")
            elif not (0 <= cavity1 < self.physical_setup.n_cavities):
                raise ValueError(f"Cavity index 1 in cavity_cavity_coupling ({cavity1}, {cavity2}) is out of range for n_cavities={self.physical_setup.n_cavities}")
            if not isinstance(cavity2, int):
                raise TypeError(f"Cavity indices in cavity_cavity_coupling must be integers, got {cavity2}")
            elif not (0 <= cavity2 < self.physical_setup.n_cavities):
                raise ValueError(f"Cavity index 2 in cavity_cavity_coupling ({cavity1}, {cavity2}) is out of range for n_cavities={self.physical_setup.n_cavities}")

        for (qubit,cavity), function in self.physical_setup.qubit_cavity_time_modulation.items():
            if not isinstance(qubit, int):
                raise TypeError(f"Qubit indices in qubit_cavity_time_modulation must be integers, got {qubit}")
            elif not (0 <= qubit < self.physical_setup.n_qubits):
                raise ValueError(f"Qubit index ({qubit}) in qubit_cavity_time_modulation ({qubit}, {cavity}) is out of range for n_qubits={self.physical_setup.n_qubits}")
            if not isinstance(cavity, int):
                raise TypeError(f"Cavity indices in qubit_cavity_time_modulation must be integers, got {cavity}")
            elif not (0 <= cavity < self.physical_setup.n_cavities):
                raise ValueError(f"Cavity index ({cavity}) in qubit_cavity_time_modulation ({qubit}, {cavity}) is out of range for n_cavities={self.physical_setup.n_cavities}")
            if not isinstance(function, Callable[[float], float]):
                raise TypeError(f"The values of the qubit_cavity_time_modulation dictionary must be a callable function"
                                f" that takes a single float argument (time) and returns a float. Got {type(function)} for key ({qubit}, {cavity})")

        for (cavity1,cavity2), function in self.physical_setup.cavity_cavity_time_modulation.items():
            if not isinstance(cavity1, int):
                raise TypeError(f"Cavity indices in cavity_cavity_time_modulation must be integers, got {cavity1}")
            elif not (0 <= cavity1 < self.physical_setup.n_cavities):
                raise ValueError(f"Cavity index 1 in cavity_cavity_time_modulation ({cavity1}, {cavity2}) is out of range for n_cavities={self.physical_setup.n_cavities}")
            if not isinstance(cavity2, int):
                raise TypeError(f"Cavity indices in cavity_cavity_time_modulation must be integers, got {cavity2}")
            elif not (0 <= cavity2 < self.physical_setup.n_cavities):
                raise ValueError(f"Cavity index 2 in cavity_cavity_time_modulation ({cavity1}, {cavity2}) is out of range for n_cavities={self.physical_setup.n_cavities}")
            if not isinstance(function, Callable[[float], float]):
                raise TypeError(f"The values of the cavity_cavity_time_modulation dictionary must be a callable function"
                                f" that takes a single float argument (time) and returns a float. Got {type(function)} for key ({cavity1}, {cavity2})") 

        if self.physical_setup.inverse_pulse_width <= 0:
            raise ValueError("Pulse width parameter (inverse_pulse_width) must be > 0")

        # Validate noise rates (now lists)
        if not isinstance(self.noise_config.depolarizing, list):
            raise TypeError("depolarizing must be normalized to a list")
        if not isinstance(self.noise_config.dephasing, list):
            raise TypeError("dephasing must be normalized to a list")
        if not isinstance(self.noise_config.relaxation, list):
            raise TypeError("relaxation must be normalized to a list")

        for i, rate in enumerate(self.noise_config.depolarizing):
            if rate < 0:
                raise ValueError(f"Depolarization rate for qubit {i} must be >= 0, got {rate}")
        for i, rate in enumerate(self.noise_config.dephasing):
            if rate < 0:
                raise ValueError(f"Dephasing rate for qubit {i} must be >= 0, got {rate}")
        for i, rate in enumerate(self.noise_config.relaxation):
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

    # Direct access to commonly used parameters for easier integration

    @property
    def n_cavities(self) -> int:
        """Direct access to number of cavities."""
        return self.physical_setup.n_cavities

    @property
    def n_qubits(self) -> int:
        """Direct access to number of qubits."""
        return self.physical_setup.n_qubits

    @property
    def cavity_levels(self) -> Union[int, List[int]]:
        """Direct access to cavity levels."""
        return self.system_dims.cavity_levels

    @cavity_levels.setter
    def cavity_levels(self, value: Union[int, List[int]]) -> None:
        """Set cavity levels."""
        self.system_dims.cavity_levels = value
        # Re-normalize if necessary
        if hasattr(self, "physical_setup"):
            self.system_dims._normalize_cavity_levels(self.physical_setup.n_cavities)

    @property
    def qubit_levels(self) -> Union[int, List[int]]:
        """Direct access to qubit levels (returns list after normalization)."""
        return self.system_dims.qubit_levels

    @qubit_levels.setter
    def qubit_levels(self, value: Union[int, List[int]]) -> None:
        """Set qubit levels."""
        self.system_dims.qubit_levels = value
        # Re-normalize if necessary
        if hasattr(self, "physical_setup"):
            self.system_dims._normalize_qubit_levels(self.physical_setup.n_qubits)

    @property
    def qubit_cavity_coupling(self) -> Union[float, Dict[Tuple[int,int], float]]:
        """Direct access to qubit-cavity coupling (returns dictionary after normalization)."""
        return self.physical_setup.qubit_cavity_coupling

    @qubit_cavity_coupling.setter
    def qubit_cavity_coupling(self, value: Union[float, Dict[Tuple[int,int], float]]) -> None:
        """Set qubit-cavity coupling."""
        # Store the value and re-normalize through __post_init__
        self.physical_setup.qubit_cavity_coupling = value
        self.physical_setup.__post_init__()

    @property
    def cavity_cavity_coupling(self) -> Union[float, Dict[Tuple[int,int], float]]:
        """Direct access to cavity-cavity coupling."""
        return self.physical_setup.cavity_cavity_coupling

    @cavity_cavity_coupling.setter
    def cavity_cavity_coupling(self, value: Union[float, Dict[Tuple[int,int], float]]) -> None:
        """Set cavity-cavity coupling."""
        # Store the value and re-normalize through __post_init__
        self.physical_setup.cavity_cavity_coupling = value
        self.physical_setup.__post_init__()
        
    @property
    def qubit_cavity_time_modulation(self) -> Callable[[float], float]:
        """Direct access to qubit-cavity time modulation."""
        return self.physical_setup.qubit_cavity_time_modulation      

    @qubit_cavity_time_modulation.setter
    def qubit_cavity_time_modulation(self, value: Callable[[float], float]) -> None:
        """Set qubit-cavity time modulation."""
        self.physical_setup.qubit_cavity_time_modulation = value

    @property
    def cavity_cavity_time_modulation(self) -> Callable[[float], float]:
        """Direct access to cavity-cavity time modulation."""
        return self.physical_setup.cavity_cavity_time_modulation 

    @cavity_cavity_time_modulation.setter
    def cavity_cavity_time_modulation(self, value: Callable[[float], float]) -> None:
        """Set cavity-cavity time modulation."""
        self.physical_setup.cavity_cavity_time_modulation = value

    @property
    def inverse_pulse_width(self) -> float:
        """Direct access to pulse width parameter."""
        return self.physical_setup.inverse_pulse_width

    @inverse_pulse_width.setter
    def inverse_pulse_width(self, value: float) -> None:
        """Set pulse width parameter."""
        self.physical_setup.inverse_pulse_width = value

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

    def copy(self, **updates) -> "ExperimentalParameters":
        """
        Create a copy of ExperimentalParameters with optional updates.

        This method creates a new ExperimentalParameters instance with all
        configuration copied. The nested objects (physical_setup, system_dims,
        measurement, initial_state, noise_config) are deep copied to avoid
        unintended sharing of mutable state.

        Args:
            **updates: Keyword arguments for attributes to update. Can be:
                - physical_setup: PhysicalSetup instance or dict of updates
                - system_dims: SystemDimensions instance
                - measurement: MeasurementProtocol instance
                - initial_state: InitialStateConfig instance
                - noise_config: NoiseConfiguration instance
                - random_seed: int or None

        Returns:
            New ExperimentalParameters instance with updated values

        Example:
            >>> # Copy and update physical setup
            >>> new_params = exp_params.copy(
            ...     physical_setup=exp_params.physical_setup.copy(chi=10.0)
            ... )
            >>>
            >>> # Or pass updates as dict (for convenience)
            >>> new_params = exp_params.copy(
            ...     physical_setup={'chi': 10.0, 'photon_cavity_coupling': 20.0}
            ... )
        """
        # Deep copy nested configurations
        new_phys_setup = self.physical_setup
        new_system_dims = self.system_dims
        new_measurement = self.measurement
        new_initial_state = self.initial_state
        new_noise_config = self.noise_config
        new_random_seed = self.random_seed

        # Handle updates
        if "physical_setup" in updates:
            pc_update = updates["physical_setup"]
            if isinstance(pc_update, dict):
                # If dict, use copy method with updates
                new_phys_setup = self.physical_setup.copy(**pc_update)
            elif isinstance(pc_update, PhysicalSetup):
                # If PhysicalSetup instance, use directly
                new_phys_setup = pc_update
            else:
                raise TypeError("physical_setup update must be a PhysicalSetup instance or a dict of updates")
        else:
            # Deep copy existing physical setup
            new_phys_setup = self.physical_setup.copy()

        if "system_dims" in updates:
            new_system_dims = updates["system_dims"]
        else:
            # Create new instance with same values
            new_system_dims = SystemDimensions(
                cavity_levels=(
                    self.system_dims.cavity_levels.copy()
                    if isinstance(self.system_dims.cavity_levels, list)
                    else self.system_dims.cavity_levels
                ),
                qubit_levels=(
                    self.system_dims.qubit_levels.copy()
                    if isinstance(self.system_dims.qubit_levels, list)
                    else self.system_dims.qubit_levels
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

        if "initial_state" in updates:
            new_initial_state = updates["initial_state"]
        else:
            # Create new instance with same values
            new_initial_state = InitialStateConfig(
                state_type=self.initial_state.state_type,
                coherent_alpha=self.initial_state.coherent_alpha,
                thermal_n_bar=self.initial_state.thermal_n_bar,
                custom_amplitudes=(
                    self.initial_state.custom_amplitudes.copy()
                    if self.initial_state.custom_amplitudes
                    else None
                ),
            )

        if "noise_config" in updates:
            new_noise_config = updates["noise_config"]
        else:
            # Create new instance with same values
            depol = self.noise_config.depolarizing
            deph = self.noise_config.dephasing
            relax = self.noise_config.relaxation

            new_noise_config = NoiseConfiguration(
                depolarizing=depol.copy() if isinstance(depol, list) else depol,
                dephasing=deph.copy() if isinstance(deph, list) else deph,
                relaxation=relax.copy() if isinstance(relax, list) else relax,
                custom_operators=(
                    self.noise_config.custom_operators.copy()
                    if self.noise_config.custom_operators
                    else None
                ),
            )

        if "random_seed" in updates:
            new_random_seed = updates["random_seed"]

        return ExperimentalParameters(
            physical_setup=new_phys_setup,
            system_dims=new_system_dims,
            measurement=new_measurement,
            initial_state=new_initial_state,
            noise_config=new_noise_config,
            random_seed=new_random_seed,
        )

    def __repr__(self) -> str:
        """
        Comprehensive string representation showing all parameters organized by groups.

        This method provides a detailed display of all experimental
        parameters, organized by their logical groups with validation status flags.
        """
        lines = []

        def _callable_name(fn: Callable[[float], float]) -> str:
            return getattr(fn, "__name__", type(fn).__name__)

        # System dimensions group
        lines.append("SYSTEM DIMENSIONS")

        n_qubits = self.physical_setup.n_qubits
        n_cavities = self.physical_setup.n_cavities
        qubit_levels = self.system_dims.qubit_levels
        cavity_levels = self.system_dims.cavity_levels
        total_dim = self.system_dims.total_dim

        lines.append(f"  Number of cavities:   {n_cavities:>6}")
        lines.append(f"  Number of qubits:     {n_qubits:>6}")
        lines.append(f"  Cavity levels:        {cavity_levels}")
        lines.append(f"  Qubit levels:         {qubit_levels}")
        lines.append(f"  Total dimension:      {total_dim:>6}")

        # Physical Setup Group
        lines.append("PHYSICAL SETUP")
        lines.append(f"  Qubit-Cavity coupling:")
        for (qubit, cavity), coupling in self.physical_setup.qubit_cavity_coupling.items():
            lines.append(f"      Qubit {qubit}, Cavity {cavity}: {coupling}")
        lines.append(f"  Cavity-Cavity coupling:")
        for (cavity1, cavity2), coupling in self.physical_setup.cavity_cavity_coupling.items():
            lines.append(f"      Cavity {cavity1}, Cavity {cavity2}: {coupling}")
        lines.append(f"  Qubit-Cavity time modulation:")
        for (qubit, cavity), function in self.physical_setup.qubit_cavity_time_modulation.items():
            lines.append(f"      Qubit {qubit}, Cavity {cavity}: {function} (callable: {_callable_name(function)})")
        lines.append(f"  Cavity-Cavity time modulation:")
        for (cavity1, cavity2), function in self.physical_setup.cavity_cavity_time_modulation.items():
            lines.append(f"      Cavity {cavity1}, Cavity {cavity2}: {function} (callable: {_callable_name(function)})")
        lines.append(f"  Inverse pulse width:  {self.physical_setup.inverse_pulse_width:>8.4f}")

        # Qubit Interactions
        if self.physical_setup.qubit_interactions:
            lines.append(
                f"  Qubit interactions:   {len(self.physical_setup.qubit_interactions)} interaction(s)"
            )
            for i, interaction in enumerate(self.physical_setup.qubit_interactions):
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
        lines.append(f"  Depolarizing rate:    {self.noise_config.depolarizing}")
        lines.append(f"  Dephasing rate:       {self.noise_config.dephasing}")
        lines.append(f"  Relaxation rate:      {self.noise_config.relaxation}")

        if self.noise_config.custom_operators is not None:
            lines.append(f"  Custom operators:     {len(self.noise_config.custom_operators):>6}")
        else:
            lines.append("  Custom operators:     None")

        # Overall System Status
        lines.append("SYSTEM STATUS")

        lines.append(f"  Random seed:          {self.random_seed}")

        try:
            self._validate_configuration()
            lines.append("  Configuration:        VALID")
        except (TypeError, ValueError) as exc:
            lines.append("  Configuration:        INVALID")
            lines.append(f"  Error:                {str(exc)}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation (calls __repr__)."""
        return self.__repr__()

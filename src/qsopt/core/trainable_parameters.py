"""
Simplified Trainable Parameters for Quantum Sensing Optimization
==============================================================

A streamlined parameter management system with three supported parameter types:
- Rotation angles (periodic, 0 to 2π)  
- Measurement times (bounded, configurable range)
- Custom parameters (user-defined constraints)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import warnings

import jax.numpy as jnp
import numpy as np
import optax


class ParameterType(Enum):
    """Parameter types supported by the simplified system."""
    ROTATION_ANGLE = "rotation_angle"
    MEASUREMENT_TIME = "measurement_time"
    CUSTOM = "custom"


@dataclass
class ParameterConstraints:
    """Parameter constraints for optimization."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    periodic: bool = False
    period: Optional[float] = None


@dataclass
class Parameter:
    """Individual parameter with metadata."""
    index: int
    name: str
    param_type: ParameterType
    value: float
    trainable: bool = True  # Whether to compute gradients for this parameter


class TrainableParameters:
    """Simplified parameter manager for quantum sensing optimization."""
    
    def __init__(self):
        """Initialize empty parameter manager."""
        self.parameters: List[Parameter] = []
        self.constraints: Dict[int, ParameterConstraints] = {}
        self.rotation_optimizer: Optional[optax.GradientTransformation] = None
        self.measurement_interval_defaults: Dict[str, Dict[str, Optional[Union[float, int]]]] = {}
    
    def add_rotation_angles(
        self,
        names: Union[str, List[str]],
        initial_values: Union[float, List[float], np.ndarray],
        optimizer: Optional[optax.GradientTransformation] = None,
        trainable: Union[bool, List[bool]] = True,
    ) -> None:
        """Add rotation angle parameters (periodic, 0 to 2π).
        
        Args:
            names: Parameter name(s)
            initial_values: Initial angle value(s) in radians
            optimizer: Optimizer for trainable parameters (default: SGD with lr=0.01)
            trainable: Whether parameter(s) are trainable (default: True).
                      Can be a single bool or a list matching the number of parameters.
        """
        names_list = [names] if isinstance(names, str) else names
        values = np.atleast_1d(initial_values)
        
        if len(names_list) != len(values):
            raise ValueError(f"Number of names ({len(names_list)}) must match number of values ({len(values)})")
        
        # Handle trainable flag
        if isinstance(trainable, bool):
            trainable_list = [trainable] * len(names_list)
        else:
            trainable_list = list(trainable)
            if len(trainable_list) != len(names_list):
                raise ValueError(f"Number of trainable flags ({len(trainable_list)}) must match number of parameters ({len(names_list)})")
        
        # Set or enforce rotation optimizer
        self.rotation_optimizer = optimizer or self.rotation_optimizer or optax.sgd(0.01)
        
        for param_name, value, is_trainable in zip(names_list, values, trainable_list):
            idx = len(self.parameters)
            param = Parameter(idx, param_name, ParameterType.ROTATION_ANGLE, float(value), is_trainable)
            self.parameters.append(param)
            
            # Rotation angles are periodic with period 2π
            self.constraints[idx] = ParameterConstraints(
                periodic=True,
                period=2 * np.pi
            )

    def add_measurement_interval(
        self,
        name: Union[str, List[str]],
        initial_value: Union[float, List[float], np.ndarray],
        min_interval: float = 1e-6,
        trainable: Union[bool, List[bool]] = True,
        grid_min: Optional[float] = None,
        grid_max: Optional[float] = None,
        grid_resolution: Optional[int] = None,
    ) -> None:
        """Add measurement interval parameters (time_interval for measurement protocol).
        
    These parameters represent the time interval between consecutive measurements.
    They must be strictly positive (> 0). Measurement intervals are not optimized
    via gradient descent; instead, optional ``grid_min``, ``grid_max``, and
    ``grid_resolution`` values are stored for default grid-search sweeps.
        
        Args:
            names: Parameter name(s) (typically 'time_interval')
            initial_values: Initial interval value(s) (must be > 0)
            min_interval: Minimum allowed interval (default: 1e-6, must be > 0)
            trainable: Whether parameter(s) are marked as trainable (default: True).
                      Can be a single bool or a list matching the number of parameters.
            grid_min: Optional default lower bound for grid search
            grid_max: Optional default upper bound for grid search
            grid_resolution: Optional default resolution for grid search
        
        Raises:
            ValueError: If initial values are not positive, if min_interval is not positive,
                or if grid_resolution is provided but not positive
        """
        names_list = [name] if isinstance(name, str) else name
        values = np.atleast_1d(initial_value)

        if len(names_list) != len(values):
            raise ValueError(f"Number of names ({len(names_list)}) must match number of values ({len(values)})")
        
        # Validate that all values are positive
        if np.any(values <= 0):
            raise ValueError(f"Measurement interval values must be > 0, got {values}")
        
        if min_interval <= 0:
            raise ValueError(f"Minimum interval must be > 0, got {min_interval}")
        
        # Handle trainable flag
        if isinstance(trainable, bool):
            trainable_list = [trainable] * len(names_list)
        else:
            trainable_list = list(trainable)
            if len(trainable_list) != len(names_list):
                raise ValueError(f"Number of trainable flags ({len(trainable_list)}) must match number of parameters ({len(names_list)})")
        
        if grid_resolution is not None and grid_resolution <= 0:
            raise ValueError(f"Grid resolution must be positive when provided, got {grid_resolution}")
        
        for param_name, value, is_trainable in zip(names_list, values, trainable_list):
            idx = len(self.parameters)
            param = Parameter(idx, param_name, ParameterType.MEASUREMENT_TIME, float(value), is_trainable)
            self.parameters.append(param)
            
            # Measurement intervals must be strictly positive
            self.constraints[idx] = ParameterConstraints(
                min_value=min_interval,
                max_value=None,  # No upper bound
                periodic=False
            )
            self.measurement_interval_defaults[param_name] = {
                "grid_min": grid_min,
                "grid_max": grid_max,
                "grid_resolution": grid_resolution,
            }
    
    def add_custom_parameters(
        self,
        names: Union[str, List[str]],
        initial_values: Union[float, List[float], np.ndarray],
        constraints: Optional[ParameterConstraints] = None,
        trainable: Union[bool, List[bool]] = True,
    ) -> None:
        """Add custom parameters with user-defined constraints.
        
        Args:
            names: Parameter name(s)
            initial_values: Initial value(s)
            constraints: Parameter constraints (default: no constraints)
            optimizer: Optimizer for trainable parameters (default: SGD with lr=0.01)
            trainable: Whether parameter(s) are trainable (default: True).
                      Can be a single bool or a list matching the number of parameters.
        """
        names_list = [names] if isinstance(names, str) else names
        values = np.atleast_1d(initial_values)
        
        if len(names_list) != len(values):
            raise ValueError(f"Number of names ({len(names_list)}) must match number of values ({len(values)})")
        
        # Handle trainable flag
        if isinstance(trainable, bool):
            trainable_list = [trainable] * len(names_list)
        else:
            trainable_list = list(trainable)
            if len(trainable_list) != len(names_list):
                raise ValueError(f"Number of trainable flags ({len(trainable_list)}) must match number of parameters ({len(names_list)})")
        
        default_constraints = constraints or ParameterConstraints()
        
        for param_name, value, is_trainable in zip(names_list, values, trainable_list):
            if is_trainable:
                warnings.warn(
                    "Custom parameter optimization is not supported; marking parameter as fixed.",
                    UserWarning,
                    stacklevel=2,
                )
                is_trainable = False
            idx = len(self.parameters)
            param = Parameter(idx, param_name, ParameterType.CUSTOM, float(value), is_trainable)
            self.parameters.append(param)
            self.constraints[idx] = default_constraints
    
    def get_parameter_vector(self) -> np.ndarray:
        """Get all parameter values as a vector."""
        return np.array([p.value for p in self.parameters])
    
    def set_parameter_vector(self, values: np.ndarray) -> None:
        """Set all parameter values from a vector."""
        if len(values) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} values, got {len(values)}")
        
        constrained_values = self.apply_constraints(values)
        for i, val in enumerate(constrained_values):
            self.parameters[i].value = float(val)
    
    def get_rotation_angles(self) -> Dict[str, np.ndarray]:
        """Get all rotation angle parameters."""
        angles = {}
        for param in self.parameters:
            if param.param_type == ParameterType.ROTATION_ANGLE:
                angles[param.name] = np.array([param.value])
        return angles
    
    def set_rotation_angles(self, angles: Dict[str, float]) -> None:
        """Set rotation angle parameters."""
        for param in self.parameters:
            if param.param_type == ParameterType.ROTATION_ANGLE and param.name in angles:
                param.value = angles[param.name]
    
    def get_measurement_times(self) -> Dict[str, np.ndarray]:
        """Get all measurement time parameters."""
        times = {}
        for param in self.parameters:
            if param.param_type == ParameterType.MEASUREMENT_TIME:
                times[param.name] = np.array([param.value])
        return times
    
    def set_measurement_times(self, times: Dict[str, float]) -> None:
        """Set measurement time parameters."""
        for param in self.parameters:
            if param.param_type == ParameterType.MEASUREMENT_TIME and param.name in times:
                param.value = times[param.name]

    def get_measurement_interval_defaults(self) -> Dict[str, Optional[Union[float, int]]]:
        """Return stored default grid-search settings for measurement intervals."""
        if not self.measurement_interval_defaults:
            return {}
        first_defaults = next(iter(self.measurement_interval_defaults.values()))
        return dict(first_defaults)
    
    def get_optimizer(self, parameter_index: int) -> optax.GradientTransformation:
        """Get the optimizer for a specific parameter."""
        if parameter_index < 0 or parameter_index >= len(self.parameters):
            raise ValueError(f"Invalid parameter index {parameter_index}")

        param = self.parameters[parameter_index]
        if param.param_type != ParameterType.ROTATION_ANGLE:
            raise ValueError(
                "Optimizers are only defined for rotation angle parameters in this configuration."
            )

        if self.rotation_optimizer is None:
            raise ValueError("Rotation optimizer has not been configured.")
        return self.rotation_optimizer
    
    def get_trainable_indices(self) -> List[int]:
        """Get indices of trainable parameters."""
        return [i for i, param in enumerate(self.parameters) if param.trainable]
    
    def get_trainable_mask(self) -> np.ndarray:
        """Get boolean mask for trainable parameters."""
        return np.array([param.trainable for param in self.parameters])
    
    def apply_constraints(self, values: np.ndarray) -> np.ndarray:
        """Apply constraints to parameter values."""
        constrained = np.array(values)
        
        for i, constraints in self.constraints.items():
            if i >= len(constrained):
                continue
                
            # Apply bounds
            if constraints.min_value is not None:
                constrained[i] = max(constrained[i], constraints.min_value)
            if constraints.max_value is not None:
                constrained[i] = min(constrained[i], constraints.max_value)
            
            # Apply periodicity
            if constraints.periodic and constraints.period is not None:
                constrained[i] = constrained[i] % constraints.period
        
        return constrained
    
    def __len__(self) -> int:
        """Return number of parameters."""
        return len(self.parameters)
    
    def __repr__(self) -> str:
        """Detailed string representation with parameter values grouped by type."""
        if not self.parameters:
            return "TrainableParameters(empty)"
        
        # Group parameters by type
        rotation_angles = [p for p in self.parameters if p.param_type == ParameterType.ROTATION_ANGLE]
        measurement_times = [p for p in self.parameters if p.param_type == ParameterType.MEASUREMENT_TIME]
        custom_params = [p for p in self.parameters if p.param_type == ParameterType.CUSTOM]
        
        lines = [f"Trainable Parameters: {len(self.parameters)}"]
        
        # Rotation angles section
        if rotation_angles:
            lines.append("  Rotation Angles:")
            for param in rotation_angles:
                angle_deg = np.degrees(param.value)
                trainable_str = "" if param.trainable else " [FIXED]"
                lines.append(f"    {param.name}: {param.value:.4f} rad ({angle_deg:.2f}°){trainable_str}")
        
        # Measurement times section
        if measurement_times:
            lines.append("  Measurement Times:")
            for param in measurement_times:
                trainable_str = "" if param.trainable else " [FIXED]"
                lines.append(f"    {param.name}: {param.value:.4f}{trainable_str}")
        
        # Custom parameters section
        if custom_params:
            lines.append("  Custom Parameters:")
            for param in custom_params:
                trainable_str = "" if param.trainable else " [FIXED]"
                lines.append(f"    {param.name}: {param.value:.4f}{trainable_str}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """String representation (calls __repr__)."""
        return self.__repr__()

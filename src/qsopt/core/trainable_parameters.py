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


class TrainableParameters:
    """Simplified parameter manager for quantum sensing optimization."""
    
    def __init__(self):
        """Initialize empty parameter manager."""
        self.parameters: List[Parameter] = []
        self.constraints: Dict[int, ParameterConstraints] = {}
        self.optimizers: Dict[int, optax.GradientTransformation] = {}
    
    def add_rotation_angles(self, names: Union[str, List[str]], 
                           initial_values: Union[float, List[float], np.ndarray],
                           optimizer: Optional[optax.GradientTransformation] = None) -> None:
        """Add rotation angle parameters (periodic, 0 to 2π)."""
        names_list = [names] if isinstance(names, str) else names
        values = np.atleast_1d(initial_values)
        
        if len(names_list) != len(values):
            raise ValueError(f"Number of names ({len(names_list)}) must match number of values ({len(values)})")
        
        # Use default optimizer if not provided
        param_optimizer = optimizer or optax.adam(0.01)
        
        for param_name, value in zip(names_list, values):
            idx = len(self.parameters)
            param = Parameter(idx, param_name, ParameterType.ROTATION_ANGLE, float(value))
            self.parameters.append(param)
            
            # Rotation angles are periodic with period 2π
            self.constraints[idx] = ParameterConstraints(
                periodic=True,
                period=2 * np.pi
            )
            self.optimizers[idx] = param_optimizer
    
    def add_measurement_times(self, names: Union[str, List[str]], 
                             initial_values: Union[float, List[float], np.ndarray],
                             min_time: float = 0.0, max_time: float = 10.0,
                             optimizer: Optional[optax.GradientTransformation] = None) -> None:
        """Add measurement time parameters - NOT IMPLEMENTED YET."""
        raise NotImplementedError("Measurement time parameters are not implemented yet")
    
    def add_custom_parameters(self, names: Union[str, List[str]], 
                             initial_values: Union[float, List[float], np.ndarray],
                             constraints: Optional[ParameterConstraints] = None,
                             optimizer: Optional[optax.GradientTransformation] = None) -> None:
        """Add custom parameters with user-defined constraints."""
        names_list = [names] if isinstance(names, str) else names
        values = np.atleast_1d(initial_values)
        
        if len(names_list) != len(values):
            raise ValueError(f"Number of names ({len(names_list)}) must match number of values ({len(values)})")
        
        default_constraints = constraints or ParameterConstraints()
        param_optimizer = optimizer or optax.adam(0.01)
        
        for param_name, value in zip(names_list, values):
            idx = len(self.parameters)
            param = Parameter(idx, param_name, ParameterType.CUSTOM, float(value))
            self.parameters.append(param)
            self.constraints[idx] = default_constraints
            self.optimizers[idx] = param_optimizer
    
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
    
    def get_optimizer(self, parameter_index: int) -> optax.GradientTransformation:
        """Get the optimizer for a specific parameter."""
        if parameter_index not in self.optimizers:
            raise ValueError(f"No optimizer found for parameter index {parameter_index}")
        return self.optimizers[parameter_index]
    
    def get_all_optimizers(self) -> Dict[int, optax.GradientTransformation]:
        """Get all optimizers."""
        return self.optimizers.copy()
    
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
        """String representation."""
        rotation_count = sum(1 for p in self.parameters if p.param_type == ParameterType.ROTATION_ANGLE)
        time_count = sum(1 for p in self.parameters if p.param_type == ParameterType.MEASUREMENT_TIME)
        custom_count = sum(1 for p in self.parameters if p.param_type == ParameterType.CUSTOM)
        
        return (f"TrainableParameters(total={len(self.parameters)}, "
                f"rotation_angles={rotation_count}, measurement_times={time_count}, "
                f"custom={custom_count})")

"""
Trainable Parameters Class
=========================

Flexible parameter management for quantum sensing optimization with support
for different parameter types, constraints, and optimization strategies.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum


class ParameterType(Enum):
    """Enumeration of different parameter types for optimization."""
    ROTATION_ANGLE = "rotation_angle"
    MEASUREMENT_TIME = "measurement_time"
    CUSTOM = "custom"   # For future extensions


@dataclass
class ParameterConstraints:
    """
    Constraints for parameter optimization.
    
    Attributes:
        min_value: Minimum allowed value (can be array for vector parameters)
        max_value: Maximum allowed value (can be array for vector parameters)
        periodic: Whether parameter is periodic (e.g., angles)
        period: Period for periodic parameters (e.g., 2π for angles)
        fixed_indices: Indices of parameters that should remain fixed
    """
    min_value: Optional[Union[float, np.ndarray]] = None
    max_value: Optional[Union[float, np.ndarray]] = None
    periodic: bool = False
    period: Optional[float] = None


@dataclass
class OptimizationConfig:
    """
    Optimization configuration for specific parameter groups.
    
    Attributes:
        learning_rate: Learning rate for this parameter group
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop', etc.)
        update_frequency: How often to update these parameters (1 = every step)
        lr_scheduler: Optional learning rate scheduler function
    """
    learning_rate: float = 0.01
    optimizer_type: str = 'sgd'
    update_frequency: int = 1
    lr_scheduler: Optional[Callable[[jnp.ndarray, int], jnp.ndarray]] = None


class ParameterGroup:
    """
    A group of related parameters with common optimization characteristics.
    
    This class manages a collection of parameters that share similar properties
    and optimization strategies (e.g., all rotation angles).
    """
    
    def __init__(self, 
                 name: str,
                 param_type: ParameterType,
                 initial_values: Union[float, List[float], np.ndarray],
                 constraints: Optional[ParameterConstraints] = None,
                 optimization_config: Optional[OptimizationConfig] = None,
                 fixed_indices: Optional[List[int]] = None):
        """
        Initialize parameter group.
        
        Args:
            name: Descriptive name for this parameter group
            param_type: Type of parameters in this group
            initial_values: Initial parameter values
            constraints: Constraints for optimization
            optimization_config: Optimization configuration
            fixed_indices: Indices of parameters to keep fixed
        """
        self.name = name
        self.param_type = param_type
        self.values = jnp.array(initial_values, dtype=float)
        self.constraints = constraints or ParameterConstraints()
        self.optimization_config = optimization_config or OptimizationConfig()
        
        # Internal state for optimization
        self._optimizer_state = None
        self._update_count = 0
        self._history = []
        
        # Set fixed indices with validation (using the property setter)
        self.fixed_indices = fixed_indices
        
    def apply_constraints(self, new_values: jnp.ndarray) -> jnp.ndarray:
        """
        Apply constraints to parameter values.
        
        Args:
            new_values: Proposed new parameter values
            
        Returns:
            Constrained parameter values
        """
        constrained = new_values
        
        # Apply bounds
        if self.constraints.min_value is not None:
            constrained = jnp.maximum(constrained, self.constraints.min_value)
        if self.constraints.max_value is not None:
            constrained = jnp.minimum(constrained, self.constraints.max_value)
            
        # Apply periodic constraints
        if self.constraints.periodic and self.constraints.period is not None:
            constrained = constrained % self.constraints.period
            
        # Keep fixed indices unchanged
        if self.fixed_indices:
            for idx in self.fixed_indices:
                constrained = constrained.at[idx].set(self.values[idx])
                    
        return constrained
    
    def update_values(self, new_values: jnp.ndarray) -> None:
        """
        Update parameter values with constraint application.
        
        Args:
            new_values: New parameter values to set
        """
        self.values = self.apply_constraints(new_values)
        self._update_count += 1
        
        # Store complete parameter history (more informative than mean)
        # Each history entry is a copy of all parameter values at this update
        self._history.append(np.array(self.values).copy())
    
    def reset_history(self) -> None:
        """Clear the optimization history."""
        self._history = []
        self._update_count = 0
    
    def get_parameter_history(self, param_index: Optional[int] = None) -> Union[List[float], List[np.ndarray]]:
        """
        Get optimization history for parameters.
        
        Args:
            param_index: If specified, return history for single parameter at this index.
                        If None, return complete parameter vector history.
        
        Returns:
            List of parameter values over optimization history
        """
        if param_index is not None:
            if param_index < 0 or param_index >= len(self.values):
                raise ValueError(f"Parameter index {param_index} out of bounds")
            return [hist[param_index] for hist in self._history]
        else:
            return self._history.copy()
    
    def get_history_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about parameter evolution during optimization.
        
        Returns:
            Dictionary with statistics (mean, std, min, max) for each parameter
        """
        if not self._history:
            return {}
        
        history_array = np.array(self._history)  # Shape: (n_updates, n_params)
        
        stats = {}
        for i in range(len(self.values)):
            param_history = history_array[:, i]
            stats[f'param_{i}'] = {
                'mean': float(np.mean(param_history)),
                'std': float(np.std(param_history)),
                'min': float(np.min(param_history)),
                'max': float(np.max(param_history)),
                'current': float(self.values[i])
            }
        
        return stats
    
    def __len__(self) -> int:
        """Return number of parameters in this group."""
        return len(self.values)
    
    @property
    def fixed_indices(self) -> List[int]:
        """Get fixed parameter indices."""
        return self._fixed_indices
    
    @fixed_indices.setter
    def fixed_indices(self, indices: Optional[List[int]]) -> None:
        """
        Set fixed parameter indices with validation.
        
        Args:
            indices: List of parameter indices to fix, or None
            
        Raises:
            ValueError: If any index is out of bounds
        """
        if indices is None:
            self._fixed_indices = []
        else:
            # Validate indices are within bounds
            for idx in indices:
                if idx < 0 or idx >= len(self.values):
                    raise ValueError(f"Fixed index {idx} is out of bounds for parameter group "
                                   f"with {len(self.values)} parameters")
            self._fixed_indices = list(indices)
    
    def __str__(self) -> str:
        """Concise string representation with key information."""
        value_range = f"[{float(jnp.min(self.values)):.3f}, {float(jnp.max(self.values)):.3f}]"
        fixed_info = f", fixed={len(self.fixed_indices)}" if self.fixed_indices else ""
        constraint_info = ""
        if self.constraints.min_value is not None or self.constraints.max_value is not None:
            constraint_info = ", constrained"
        if self.constraints.periodic:
            constraint_info += ", periodic"
        
        return (f"{self.name}: {self.param_type.value}({len(self.values)}) = {value_range}"
                f"{fixed_info}{constraint_info}")
    
    def __repr__(self) -> str:
        """String representation of parameter group."""
        return (f"ParameterGroup(name='{self.name}', type={self.param_type.value}, "
                f"size={len(self.values)}, updates={self._update_count})")


class TrainableParameters:
    """
    Flexible container for trainable parameters in quantum sensing optimization.
    
    This class manages multiple parameter groups with different optimization
    characteristics, constraints, and update strategies. It provides a unified
    interface for parameter access while maintaining the flexibility to handle
    diverse parameter types.
    """
    
    def __init__(self):
        """Initialize empty trainable parameters container."""
        self._parameter_groups: Dict[str, ParameterGroup] = {}
        self._parameter_order: List[str] = []
        
    def add_parameter_group(self, 
                           name: str,
                           param_type: ParameterType,
                           values: Union[float, List[float], np.ndarray],
                           constraints: Optional[ParameterConstraints] = None,
                           optimization_config: Optional[OptimizationConfig] = None,
                           fixed_indices: Optional[List[int]] = None) -> None:
        """
        Add a new parameter group.
        
        Args:
            name: Unique name for this parameter group
            param_type: Type of parameters
            values: Initial parameter values
            constraints: Parameter constraints
            optimization_config: Optimization configuration
            fixed_indices: Indices of parameters to keep fixed
            
        Raises:
            ValueError: If parameter group name already exists
        """
        if name in self._parameter_groups:
            raise ValueError(f"Parameter group '{name}' already exists")
            
        group = ParameterGroup(name, param_type, values, constraints, optimization_config, fixed_indices)
        self._parameter_groups[name] = group
        self._parameter_order.append(name)
    
    def add_rotation_angles(self, 
                           name: str,
                           initial_angles: Union[List[float], np.ndarray],
                           optimization_config: Optional[OptimizationConfig] = None) -> None:
        """
        Convenience method to add rotation angle parameters.
        
        Args:
            name: Name for the rotation angle group
            initial_angles: Initial rotation angles in radians
            optimization_config: Optional optimization configuration
        """
        constraints = ParameterConstraints(periodic=True, period=2*np.pi)
        self.add_parameter_group(name, ParameterType.ROTATION_ANGLE, initial_angles, constraints, optimization_config)
    
    def add_measurement_times(self,
                             name: str,
                             times: Union[List[float], np.ndarray],
                             constraints: Optional[ParameterConstraints] = None) -> None:
        """
        Convenience method to add measurement timing parameters.
        """
        raise NotImplementedError("add_measurement_times method is not yet implemented.")

    
    def get_parameter_group(self, name: str) -> ParameterGroup:
        """
        Get a parameter group by name.
        
        Args:
            name: Name of the parameter group
            
        Returns:
            The requested parameter group
            
        Raises:
            KeyError: If parameter group doesn't exist
        """
        if name not in self._parameter_groups:
            raise KeyError(f"Parameter group '{name}' not found")
        return self._parameter_groups[name]
    
    def get_parameter_vector(self) -> jnp.ndarray:
        """
        Get all parameters as a single flattened vector.
        
        Returns:
            Flattened array of all parameter values
        """
        if not self._parameter_groups:
            return jnp.array([])
            
        vectors = []
        for name in self._parameter_order:
            vectors.append(self._parameter_groups[name].values.flatten())
        
        return jnp.concatenate(vectors)
    
    def set_parameter_vector(self, param_vector: jnp.ndarray) -> None:
        """
        Set all parameters from a flattened vector.
        
        Args:
            param_vector: Flattened parameter vector
            
        Raises:
            ValueError: If vector length doesn't match total parameter count
        """
        if len(param_vector) != self.get_total_parameter_count():
            raise ValueError(f"Parameter vector length {len(param_vector)} doesn't match "
                           f"expected length {self.get_total_parameter_count()}")
        
        start_idx = 0
        for name in self._parameter_order:
            group = self._parameter_groups[name]
            end_idx = start_idx + len(group.values)
            
            new_values = param_vector[start_idx:end_idx].reshape(group.values.shape)
            group.update_values(new_values)
            
            start_idx = end_idx
    
    def get_parameter_by_type(self, param_type: ParameterType) -> Dict[str, jnp.ndarray]:
        """
        Get all parameters of a specific type.
        
        Args:
            param_type: Type of parameters to retrieve
            
        Returns:
            Dictionary mapping parameter group names to their values
        """
        result = {}
        for name, group in self._parameter_groups.items():
            if group.param_type == param_type:
                result[name] = group.values
        return result
    
    def get_rotation_angles(self) -> Dict[str, jnp.ndarray]:
        """Get all rotation angle parameters."""
        return self.get_parameter_by_type(ParameterType.ROTATION_ANGLE)
    
    def get_measurement_times(self) -> Dict[str, jnp.ndarray]:
        """Get all measurement timing parameters."""
        return self.get_parameter_by_type(ParameterType.MEASUREMENT_TIME)
    
    def get_total_parameter_count(self) -> int:
        """
        Get total number of individual parameters.
        
        Returns:
            Total parameter count across all groups
        """
        return sum(len(group.values) for group in self._parameter_groups.values())
    
    def copy(self) -> 'TrainableParameters':
        """
        Create a deep copy of the trainable parameters.
        
        Returns:
            New TrainableParameters instance with copied values
        """
        new_params = TrainableParameters()
        
        for name in self._parameter_order:
            group = self._parameter_groups[name]
            new_params.add_parameter_group(
                name=name,
                param_type=group.param_type,
                values=np.array(group.values),
                constraints=group.constraints,
                optimization_config=group.optimization_config,
                fixed_indices=group.fixed_indices.copy() if group.fixed_indices else None
            )
        
        return new_params
    
    def __len__(self) -> int:
        """Return number of parameter groups."""
        return len(self._parameter_groups)
    
    def __contains__(self, name: str) -> bool:
        """Check if parameter group exists."""
        return name in self._parameter_groups
    
    def __getitem__(self, name: str) -> ParameterGroup:
        """Access parameter group by name."""
        return self.get_parameter_group(name)
    
    def __str__(self) -> str:
        """Concise string representation showing all parameter groups."""
        if not self._parameter_groups:
            return "TrainableParameters: No parameter groups defined"
        
        lines = [f"TrainableParameters ({len(self._parameter_groups)} groups, {self.get_total_parameter_count()} total params):"]
        for name in self._parameter_order:
            lines.append(f"  {self._parameter_groups[name]}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation of trainable parameters."""
        total_params = self.get_total_parameter_count()
        group_count = len(self._parameter_groups)
        return f"TrainableParameters(groups={group_count}, total_params={total_params})"

"""
Tests for Simplified Trainable Parameters Classes
================================================

Test suite for the simplified TrainableParameters class.
"""

import numpy as np
import optax
import pytest

from qsopt.core.trainable_parameters import (Parameter, ParameterConstraints,
                                             ParameterType, TrainableParameters)


class TestParameterConstraints:
    """Test ParameterConstraints dataclass."""

    def test_default_initialization(self):
        """Test default parameter constraints."""
        constraints = ParameterConstraints()
        assert constraints.min_value is None
        assert constraints.max_value is None
        assert constraints.periodic is False
        assert constraints.period is None

    def test_custom_initialization(self):
        """Test custom parameter constraints."""
        constraints = ParameterConstraints(
            min_value=-1.0, max_value=1.0, periodic=True, period=2.0
        )
        assert constraints.min_value == -1.0
        assert constraints.max_value == 1.0
        assert constraints.periodic is True
        assert constraints.period == 2.0


class TestParameter:
    """Test Parameter dataclass."""

    def test_initialization(self):
        """Test parameter initialization."""
        param = Parameter(0, "test_angle", ParameterType.ROTATION_ANGLE, 1.57)
        assert param.index == 0
        assert param.name == "test_angle"
        assert param.param_type == ParameterType.ROTATION_ANGLE
        assert param.value == 1.57


class TestTrainableParameters:
    """Test TrainableParameters class."""

    def test_empty_initialization(self):
        """Test empty initialization."""
        params = TrainableParameters()
        assert len(params) == 0
        assert len(params.get_parameter_vector()) == 0
        assert len(params.get_all_optimizers()) == 0

    def test_add_single_rotation_angle(self):
        """Test adding single rotation angle."""
        params = TrainableParameters()
        params.add_rotation_angles("theta", 1.57)
        
        assert len(params) == 1
        vector = params.get_parameter_vector()
        assert np.isclose(vector[0], 1.57)
        
        # Check optimizer was created
        optimizers = params.get_all_optimizers()
        assert len(optimizers) == 1
        assert 0 in optimizers

    def test_add_single_rotation_angle_with_optimizer(self):
        """Test adding single rotation angle with custom optimizer."""
        params = TrainableParameters()
        custom_optimizer = optax.sgd(0.1)
        params.add_rotation_angles("theta", 1.57, optimizer=custom_optimizer)
        
        assert len(params) == 1
        retrieved_optimizer = params.get_optimizer(0)
        assert retrieved_optimizer is custom_optimizer

    def test_add_multiple_rotation_angles(self):
        """Test adding multiple rotation angles."""
        params = TrainableParameters()
        params.add_rotation_angles(["theta1", "theta2"], [0.0, 1.57])
        
        assert len(params) == 2
        vector = params.get_parameter_vector()
        assert np.allclose(vector, [0.0, 1.57])
        
        # Check optimizers were created
        optimizers = params.get_all_optimizers()
        assert len(optimizers) == 2

    def test_add_measurement_times_not_implemented(self):
        """Test that measurement times raise NotImplementedError."""
        params = TrainableParameters()
        
        with pytest.raises(NotImplementedError, match="Measurement time parameters are not implemented yet"):
            params.add_measurement_times("t1", 1.0)

    def test_add_custom_parameters(self):
        """Test adding custom parameters."""
        constraints = ParameterConstraints(min_value=-1.0, max_value=1.0)
        params = TrainableParameters()
        params.add_custom_parameters(["x", "y"], [0.5, -0.3], constraints)
        
        assert len(params) == 2
        vector = params.get_parameter_vector()
        assert np.allclose(vector, [0.5, -0.3])
        
        # Check optimizers were created
        optimizers = params.get_all_optimizers()
        assert len(optimizers) == 2

    def test_add_custom_parameters_with_optimizer(self):
        """Test adding custom parameters with custom optimizer."""
        constraints = ParameterConstraints(min_value=-1.0, max_value=1.0)
        custom_optimizer = optax.sgd(0.05)
        params = TrainableParameters()
        params.add_custom_parameters("x", 0.5, constraints, optimizer=custom_optimizer)
        
        retrieved_optimizer = params.get_optimizer(0)
        assert retrieved_optimizer is custom_optimizer

    def test_parameter_vector_operations(self):
        """Test parameter vector get/set operations."""
        params = TrainableParameters()
        params.add_rotation_angles("theta", 1.0)
        params.add_custom_parameters("x", 3.0)
        
        vector = params.get_parameter_vector()
        assert np.allclose(vector, [1.0, 3.0])
        
        new_vector = np.array([1.5, 3.5])
        params.set_parameter_vector(new_vector)
        
        updated_vector = params.get_parameter_vector()
        assert np.allclose(updated_vector, new_vector)

    def test_apply_constraints(self):
        """Test constraint application."""
        params = TrainableParameters()
        
        # Add rotation angle (periodic)
        params.add_rotation_angles("theta", 0.0)
        
        # Add custom parameter (bounded)
        constraints = ParameterConstraints(min_value=-1.0, max_value=1.0)
        params.add_custom_parameters("x", 0.0, constraints)
        
        # Test constraint application
        test_values = np.array([7.0, 2.0])  # Should be [7%2π, 1.0]
        constrained = params.apply_constraints(test_values)
        
        expected = np.array([7.0 % (2 * np.pi), 1.0])
        assert np.allclose(constrained, expected)

    def test_get_rotation_angles(self):
        """Test getting rotation angles."""
        params = TrainableParameters()
        params.add_rotation_angles(["theta1", "theta2"], [0.5, 1.5])
        
        angles = params.get_rotation_angles()
        assert "theta1" in angles
        assert "theta2" in angles
        assert np.isclose(angles["theta1"][0], 0.5)
        assert np.isclose(angles["theta2"][0], 1.5)

    def test_get_measurement_times_empty(self):
        """Test getting measurement times returns empty dict when none added."""
        params = TrainableParameters()
        params.add_rotation_angles("theta", 1.0)
        
        times = params.get_measurement_times()
        assert len(times) == 0

    def test_get_optimizer_invalid_index(self):
        """Test getting optimizer with invalid index."""
        params = TrainableParameters()
        
        with pytest.raises(ValueError, match="No optimizer found for parameter index 0"):
            params.get_optimizer(0)

    def test_validation_errors(self):
        """Test validation errors."""
        params = TrainableParameters()
        
        # Test mismatched names/values for rotation angles
        with pytest.raises(ValueError, match="Number of names"):
            params.add_rotation_angles(["theta1", "theta2"], [1.0])
        
        # Test mismatched names/values for custom parameters
        with pytest.raises(ValueError, match="Number of names"):
            params.add_custom_parameters(["x", "y"], [1.0])
        
        # Test wrong vector size
        params.add_rotation_angles("theta", 0.0)
        with pytest.raises(ValueError, match="Expected 1 values, got 2"):
            params.set_parameter_vector(np.array([1.0, 2.0]))

    def test_repr(self):
        """Test string representation."""
        params = TrainableParameters()
        params.add_rotation_angles("theta", 0.0)
        params.add_custom_parameters("x", 2.0)
        
        repr_str = repr(params)
        assert "rotation_angles=1" in repr_str
        assert "measurement_times=0" in repr_str  # Should be 0 since not implemented
        assert "custom=1" in repr_str


class TestParameterType:
    """Test ParameterType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ParameterType.ROTATION_ANGLE.value == "rotation_angle"
        assert ParameterType.MEASUREMENT_TIME.value == "measurement_time"
        assert ParameterType.CUSTOM.value == "custom"
        assert ParameterType.MEASUREMENT_TIME.value == "measurement_time"
        assert ParameterType.CUSTOM.value == "custom"

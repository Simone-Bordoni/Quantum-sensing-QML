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

    def test_add_measurement_interval(self):
        """Test adding measurement interval parameters."""
        params = TrainableParameters()
        
        # Test adding a single interval parameter
        params.add_measurement_interval("time_interval", 0.5)
        assert len(params) == 1
        assert params.parameters[0].param_type == ParameterType.MEASUREMENT_TIME
        assert params.parameters[0].value == 0.5
        assert params.parameters[0].trainable is False
        defaults = params.get_measurement_interval_defaults()
        assert defaults == {"grid_min": None, "grid_max": None, "grid_resolution": None}
        
        # Test that min_value constraint is set
        assert params.constraints[0].min_value is not None
        assert params.constraints[0].min_value > 0
        
        # Test that negative values are rejected
        params2 = TrainableParameters()
        with pytest.raises(ValueError, match="Measurement interval values must be > 0"):
            params2.add_measurement_interval("time_interval", -0.1)
        
        # Test that zero values are rejected
        with pytest.raises(ValueError, match="Measurement interval values must be > 0"):
            params2.add_measurement_interval("time_interval", 0.0)

        # Test grid search defaults validation
        with pytest.raises(ValueError, match="Grid resolution must be positive"):
            params2.add_measurement_interval("time_interval", 0.5, grid_resolution=0)

    def test_measurement_interval_grid_defaults(self):
        """Test storing and retrieving grid-search defaults for intervals."""
        params = TrainableParameters()
        params.add_measurement_interval(
            "time_interval",
            0.5,
            grid_min=0.1,
            grid_max=1.0,
            grid_resolution=75,
        )

        defaults = params.get_measurement_interval_defaults()
        assert defaults["grid_min"] == pytest.approx(0.1)
        assert defaults["grid_max"] == pytest.approx(1.0)
        assert defaults["grid_resolution"] == 75

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
        # Check structure
        assert "Trainable Parameters: 2" in repr_str
        assert "Rotation Angles:" in repr_str
        assert "Custom Parameters:" in repr_str
        # Check parameter names and values
        assert "theta:" in repr_str
        assert "0.0000 rad" in repr_str
        assert "x:" in repr_str
        assert "2.0000" in repr_str


    def test_trainable_flag_single_parameter(self):
        """Test trainable flag with single parameter."""
        params = TrainableParameters()
        
        # Add trainable parameter
        params.add_rotation_angles("theta1", 1.0, trainable=True)
        assert params.parameters[0].trainable is True
        
        # Add non-trainable parameter
        params.add_rotation_angles("theta2", 2.0, trainable=False)
        assert params.parameters[1].trainable is False
        
    def test_trainable_flag_multiple_parameters_uniform(self):
        """Test trainable flag with multiple parameters (uniform)."""
        params = TrainableParameters()
        
        # All trainable
        params.add_rotation_angles(["theta1", "theta2"], [1.0, 2.0], trainable=True)
        assert all(p.trainable for p in params.parameters)
        
        # All non-trainable
        params2 = TrainableParameters()
        params2.add_rotation_angles(["theta1", "theta2"], [1.0, 2.0], trainable=False)
        assert all(not p.trainable for p in params2.parameters)
    
    def test_trainable_flag_multiple_parameters_mixed(self):
        """Test trainable flag with multiple parameters (mixed)."""
        params = TrainableParameters()
        
        # Mixed trainable flags
        params.add_rotation_angles(
            ["theta1", "theta2", "theta3"], 
            [1.0, 2.0, 3.0], 
            trainable=[True, False, True]
        )
        
        assert params.parameters[0].trainable is True
        assert params.parameters[1].trainable is False
        assert params.parameters[2].trainable is True
    
    def test_trainable_flag_custom_parameters(self):
        """Test trainable flag with custom parameters."""
        params = TrainableParameters()
        constraints = ParameterConstraints(min_value=-1.0, max_value=1.0)
        
        # Single with trainable=False
        params.add_custom_parameters("x", 0.5, constraints, trainable=False)
        assert params.parameters[0].trainable is False
        
        # Multiple with mixed flags
        params.add_custom_parameters(
            ["y", "z"], 
            [0.3, 0.7], 
            constraints, 
            trainable=[True, False]
        )
        assert params.parameters[1].trainable is True
        assert params.parameters[2].trainable is False
    
    def test_get_trainable_indices(self):
        """Test get_trainable_indices method."""
        params = TrainableParameters()
        
        params.add_rotation_angles(
            ["theta1", "theta2", "theta3"], 
            [1.0, 2.0, 3.0], 
            trainable=[True, False, True]
        )
        
        trainable_indices = params.get_trainable_indices()
        assert trainable_indices == [0, 2]
    
    def test_get_trainable_mask(self):
        """Test get_trainable_mask method."""
        params = TrainableParameters()
        
        params.add_rotation_angles(
            ["theta1", "theta2", "theta3", "theta4"], 
            [1.0, 2.0, 3.0, 4.0], 
            trainable=[True, False, True, False]
        )
        
        mask = params.get_trainable_mask()
        expected = np.array([True, False, True, False])
        assert np.array_equal(mask, expected)
    
    def test_repr_with_fixed_parameters(self):
        """Test string representation with fixed parameters."""
        params = TrainableParameters()
        
        params.add_rotation_angles(
            ["theta1", "theta2"], 
            [1.0, 2.0], 
            trainable=[True, False]
        )
        
        repr_str = repr(params)
        # Check that FIXED tag appears for non-trainable parameter
        assert "[FIXED]" in repr_str


class TestParameterType:
    """Test ParameterType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ParameterType.ROTATION_ANGLE.value == "rotation_angle"
        assert ParameterType.MEASUREMENT_TIME.value == "measurement_time"
        assert ParameterType.CUSTOM.value == "custom"
        assert ParameterType.MEASUREMENT_TIME.value == "measurement_time"
        assert ParameterType.CUSTOM.value == "custom"

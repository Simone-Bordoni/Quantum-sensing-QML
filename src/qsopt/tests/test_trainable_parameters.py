"""
Tests for Trainable Parameters Classes
=====================================

Comprehensive test suite for ParameterGroup and TrainableParameters classes.
"""

from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from qsopt.core.trainable_parameters import (ParameterConstraints,
                                             ParameterGroup, ParameterType,
                                             TrainableParameters)


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
            min_value=0.0, max_value=2 * np.pi, periodic=True, period=2 * np.pi
        )
        assert constraints.min_value == 0.0
        assert constraints.max_value == 2 * np.pi
        assert constraints.periodic is True
        assert constraints.period == 2 * np.pi


class TestParameterGroup:
    """Test ParameterGroup class."""

    def test_basic_initialization(self):
        """Test basic parameter group initialization."""
        group = ParameterGroup("test", ParameterType.ROTATION_ANGLE, [0.0, 1.0, 2.0])
        assert group.name == "test"
        assert group.param_type == ParameterType.ROTATION_ANGLE
        assert len(group.values) == 3
        assert jnp.allclose(group.values, jnp.array([0.0, 1.0, 2.0]))
        assert len(group.fixed_indices) == 0
        assert group._update_count == 0

    def test_initialization_with_constraints(self):
        """Test initialization with constraints."""
        constraints = ParameterConstraints(min_value=0.0, max_value=2 * np.pi)
        group = ParameterGroup("test", ParameterType.ROTATION_ANGLE, [1.0], constraints=constraints)
        assert group.constraints.min_value == 0.0
        assert group.constraints.max_value == 2 * np.pi

    def test_initialization_with_fixed_indices(self):
        """Test initialization with fixed indices."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0, 3.0], fixed_indices=[0, 2])
        assert group.fixed_indices == [0, 2]

    def test_fixed_indices_property(self):
        """Test fixed_indices property getter and setter."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0, 3.0])

        # Test setter with valid indices
        group.fixed_indices = [0, 2]
        assert group.fixed_indices == [0, 2]

        # Test setter with None
        group.fixed_indices = None
        assert group.fixed_indices == []

    def test_fixed_indices_validation(self):
        """Test fixed indices validation."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0])

        # Test invalid index (too large)
        with pytest.raises(ValueError, match="Fixed index 3 is out of bounds"):
            group.fixed_indices = [3]

        # Test invalid index (negative)
        with pytest.raises(ValueError, match="Fixed index -1 is out of bounds"):
            group.fixed_indices = [-1]

    def test_apply_constraints_bounds(self):
        """Test constraint application with bounds."""
        constraints = ParameterConstraints(min_value=0.0, max_value=2.0)
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0], constraints=constraints)

        # Test values within bounds
        result = group.apply_constraints(jnp.array([1.5]))
        assert jnp.allclose(result, jnp.array([1.5]))

        # Test values below minimum
        result = group.apply_constraints(jnp.array([-0.5]))
        assert jnp.allclose(result, jnp.array([0.0]))

        # Test values above maximum
        result = group.apply_constraints(jnp.array([3.0]))
        assert jnp.allclose(result, jnp.array([2.0]))

    def test_apply_constraints_periodic(self):
        """Test constraint application with periodicity."""
        constraints = ParameterConstraints(periodic=True, period=2 * np.pi)
        group = ParameterGroup("test", ParameterType.ROTATION_ANGLE, [0.0], constraints=constraints)

        # Test periodic wrapping
        result = group.apply_constraints(jnp.array([3 * np.pi]))
        expected = 3 * np.pi % (2 * np.pi)
        assert jnp.allclose(result, jnp.array([expected]))

    def test_apply_constraints_fixed_indices(self):
        """Test constraint application with fixed indices."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0, 3.0], fixed_indices=[1])

        # Try to change all values, but index 1 should remain fixed
        new_values = jnp.array([10.0, 20.0, 30.0])
        result = group.apply_constraints(new_values)

        assert result[0] == 10.0  # Should change
        assert result[1] == 2.0  # Should remain fixed
        assert result[2] == 30.0  # Should change

    def test_update_values(self):
        """Test parameter value updates."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0])

        initial_count = group._update_count
        group.update_values(jnp.array([3.0, 4.0]))

        assert jnp.allclose(group.values, jnp.array([3.0, 4.0]))
        assert group._update_count == initial_count + 1
        assert len(group._history) == 1
        # History should store complete parameter arrays, not just means
        assert np.allclose(group._history[0], np.array([3.0, 4.0]))

    def test_parameter_history(self):
        """Test parameter history tracking."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0])

        # Make some updates
        group.update_values(jnp.array([1.5, 2.5]))
        group.update_values(jnp.array([2.0, 3.0]))

        # Test full history
        full_history = group.get_parameter_history()
        assert len(full_history) == 2
        assert np.allclose(full_history[0], np.array([1.5, 2.5]))
        assert np.allclose(full_history[1], np.array([2.0, 3.0]))

        # Test single parameter history
        param_0_history = group.get_parameter_history(0)
        assert len(param_0_history) == 2
        assert param_0_history == [1.5, 2.0]

        param_1_history = group.get_parameter_history(1)
        assert param_1_history == [2.5, 3.0]

        # Test invalid index
        with pytest.raises(ValueError, match="Parameter index 5 out of bounds"):
            group.get_parameter_history(5)

    def test_history_statistics(self):
        """Test parameter history statistics."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [0.0, 0.0])

        # No history initially
        stats = group.get_history_statistics()
        assert not stats

        # Add some history
        group.update_values(jnp.array([1.0, 2.0]))
        group.update_values(jnp.array([2.0, 4.0]))
        group.update_values(jnp.array([3.0, 6.0]))

        stats = group.get_history_statistics()
        assert "param_0" in stats
        assert "param_1" in stats

        # Check param_0 stats (values: [1.0, 2.0, 3.0])
        assert abs(stats["param_0"]["mean"] - 2.0) < 1e-10
        assert abs(stats["param_0"]["min"] - 1.0) < 1e-10
        assert abs(stats["param_0"]["max"] - 3.0) < 1e-10
        assert abs(stats["param_0"]["current"] - 3.0) < 1e-10

        # Check param_1 stats (values: [2.0, 4.0, 6.0])
        assert abs(stats["param_1"]["mean"] - 4.0) < 1e-10
        assert abs(stats["param_1"]["min"] - 2.0) < 1e-10
        assert abs(stats["param_1"]["max"] - 6.0) < 1e-10
        assert abs(stats["param_1"]["current"] - 6.0) < 1e-10

    def test_reset_history(self):
        """Test history reset."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0])
        group.update_values(jnp.array([2.0]))

        assert group._update_count > 0
        assert len(group._history) > 0

        group.reset_history()
        assert group._update_count == 0
        assert len(group._history) == 0

    def test_len(self):
        """Test length method."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0, 3.0])
        assert len(group) == 3

    def test_str_representation(self):
        """Test string representation."""
        # Test basic representation
        group = ParameterGroup("angles", ParameterType.ROTATION_ANGLE, [0.0, 1.57])
        str_repr = str(group)
        assert "angles" in str_repr
        assert "rotation_angle" in str_repr
        assert "(2)" in str_repr

        # Test with constraints and fixed indices
        constraints = ParameterConstraints(min_value=0.0, periodic=True)
        group = ParameterGroup(
            "angles",
            ParameterType.ROTATION_ANGLE,
            [0.0, 1.57],
            constraints=constraints,
            fixed_indices=[0],
        )
        str_repr = str(group)
        assert "fixed=1" in str_repr
        assert "constrained" in str_repr
        assert "periodic" in str_repr

    def test_repr(self):
        """Test repr representation."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0])
        repr_str = repr(group)
        assert "ParameterGroup" in repr_str
        assert "name='test'" in repr_str
        assert "custom" in repr_str
        assert "size=2" in repr_str

    def test_default_optimizer(self):
        """Test default optimizer initialization."""
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0])
        assert group.optimizer is not None
        # Verify it's an Adam optimizer by checking if it has the expected structure
        assert hasattr(group.optimizer, 'init')
        assert hasattr(group.optimizer, 'update')

    def test_custom_optimizer(self):
        """Test custom optimizer initialization."""
        custom_optimizer = optax.sgd(0.1)
        group = ParameterGroup("test", ParameterType.CUSTOM, [1.0, 2.0], optimizer=custom_optimizer)
        assert group.optimizer is custom_optimizer


class TestTrainableParameters:
    """Test TrainableParameters class."""

    def test_empty_initialization(self):
        """Test empty trainable parameters initialization."""
        params = TrainableParameters()
        assert len(params) == 0
        assert params.get_total_parameter_count() == 0
        assert len(params.get_parameter_vector()) == 0

    def test_add_parameter_group(self):
        """Test adding parameter groups with individual arguments."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="group1",
            param_type=ParameterType.ROTATION_ANGLE,
            initial_values=[0.0, 1.0],
        )

        assert len(params) == 1
        assert "group1" in params
        assert params.get_total_parameter_count() == 2

    def test_add_parameter_group_object(self):
        """Test adding parameter groups using ParameterGroup object."""
        params = TrainableParameters()

        # Create a ParameterGroup object first
        group = ParameterGroup(
            name="group1", param_type=ParameterType.ROTATION_ANGLE, initial_values=[0.0, 1.0]
        )

        # Add it using the object
        params.add_parameter_group(group)

        assert len(params) == 1
        assert "group1" in params
        assert params.get_total_parameter_count() == 2

        # Verify the group is the same object
        retrieved_group = params.get_parameter_group("group1")
        assert retrieved_group.name == "group1"
        assert jnp.allclose(retrieved_group.values, jnp.array([0.0, 1.0]))

    def test_add_parameter_group_missing_args(self):
        """Test adding parameter group with missing required arguments."""
        params = TrainableParameters()

        # Missing param_type
        with pytest.raises(ValueError, match="param_type is required"):
            params.add_parameter_group(name_or_group="group1", initial_values=[1.0])

        # Missing initial_values
        with pytest.raises(ValueError, match="initial_values is required"):
            params.add_parameter_group(name_or_group="group1", param_type=ParameterType.CUSTOM)

    def test_add_duplicate_group(self):
        """Test adding duplicate parameter group raises error."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="group1", param_type=ParameterType.CUSTOM, initial_values=[1.0]
        )

        with pytest.raises(ValueError, match="Parameter group 'group1' already exists"):
            params.add_parameter_group(
                name_or_group="group1", param_type=ParameterType.CUSTOM, initial_values=[2.0]
            )

    def test_add_duplicate_group_object(self):
        """Test adding duplicate parameter group with object raises error."""
        params = TrainableParameters()

        group1 = ParameterGroup("group1", ParameterType.CUSTOM, [1.0])
        group2 = ParameterGroup("group1", ParameterType.CUSTOM, [2.0])  # Same name

        params.add_parameter_group(group1)

        with pytest.raises(ValueError, match="Parameter group 'group1' already exists"):
            params.add_parameter_group(group2)

    def test_add_parameter_group_with_fixed_indices(self):
        """Test adding parameter group with fixed indices."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="group1",
            param_type=ParameterType.CUSTOM,
            initial_values=[1.0, 2.0, 3.0],
            fixed_indices=[0, 2],
        )

        group = params.get_parameter_group("group1")
        assert group.fixed_indices == [0, 2]

    def test_add_rotation_angles(self):
        """Test convenience method for adding rotation angles."""
        params = TrainableParameters()
        params.add_rotation_angles("angles", [0.0, np.pi / 2, np.pi])

        group = params.get_parameter_group("angles")
        assert group.param_type == ParameterType.ROTATION_ANGLE
        assert group.constraints.periodic is True
        assert group.constraints.period == 2 * np.pi

    def test_get_parameter_group(self):
        """Test parameter group retrieval."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="test", param_type=ParameterType.CUSTOM, initial_values=[1.0]
        )

        group = params.get_parameter_group("test")
        assert group.name == "test"

        # Test non-existent group
        with pytest.raises(KeyError, match="Parameter group 'nonexistent' not found"):
            params.get_parameter_group("nonexistent")

    def test_parameter_vector_operations(self):
        """Test parameter vector get/set operations."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="group1", param_type=ParameterType.CUSTOM, initial_values=[1.0, 2.0]
        )
        params.add_parameter_group(
            name_or_group="group2", param_type=ParameterType.CUSTOM, initial_values=[3.0]
        )

        # Test get parameter vector
        vector = params.get_parameter_vector()
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(vector, expected)

        # Test set parameter vector
        new_vector = jnp.array([10.0, 20.0, 30.0])
        params.set_parameter_vector(new_vector)

        assert jnp.allclose(params.get_parameter_group("group1").values, jnp.array([10.0, 20.0]))
        assert jnp.allclose(params.get_parameter_group("group2").values, jnp.array([30.0]))

    def test_set_parameter_vector_wrong_size(self):
        """Test setting parameter vector with wrong size."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="test", param_type=ParameterType.CUSTOM, initial_values=[1.0, 2.0]
        )

        with pytest.raises(
            ValueError, match="Parameter vector length 3 doesn't match expected length 2"
        ):
            params.set_parameter_vector(jnp.array([1.0, 2.0, 3.0]))

    def test_get_parameter_by_type(self):
        """Test parameter retrieval by type."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="angles1",
            param_type=ParameterType.ROTATION_ANGLE,
            initial_values=[0.0, 1.0],
        )
        params.add_parameter_group(
            name_or_group="custom1", param_type=ParameterType.CUSTOM, initial_values=[2.0]
        )
        params.add_parameter_group(
            name_or_group="angles2", param_type=ParameterType.ROTATION_ANGLE, initial_values=[3.0]
        )

        rotation_params = params.get_parameter_by_type(ParameterType.ROTATION_ANGLE)
        assert len(rotation_params) == 2
        assert "angles1" in rotation_params
        assert "angles2" in rotation_params

        custom_params = params.get_parameter_by_type(ParameterType.CUSTOM)
        assert len(custom_params) == 1
        assert "custom1" in custom_params

    def test_get_rotation_angles(self):
        """Test get rotation angles convenience method."""
        params = TrainableParameters()
        params.add_rotation_angles("angles", [0.0, np.pi])

        rotation_angles = params.get_rotation_angles()
        assert "angles" in rotation_angles
        assert jnp.allclose(rotation_angles["angles"], jnp.array([0.0, np.pi]))

    def test_get_measurement_times(self):
        """Test get measurement times convenience method."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="times",
            param_type=ParameterType.MEASUREMENT_TIME,
            initial_values=[1.0, 2.0],
        )

        times = params.get_measurement_times()
        assert "times" in times
        assert jnp.allclose(times["times"], jnp.array([1.0, 2.0]))

    def test_copy(self):
        """Test parameter copy functionality."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="group1",
            param_type=ParameterType.CUSTOM,
            initial_values=[1.0, 2.0],
            fixed_indices=[0],
        )

        # Modify original
        params.get_parameter_group("group1").update_values(jnp.array([10.0, 20.0]))

        # Create copy
        params_copy = params.copy()

        # Verify copy independence
        assert len(params_copy) == 1
        assert "group1" in params_copy
        original_group = params.get_parameter_group("group1")
        copied_group = params_copy.get_parameter_group("group1")

        assert jnp.allclose(original_group.values, copied_group.values)
        assert copied_group.fixed_indices == original_group.fixed_indices

        # Modify original, verify copy unchanged
        params.get_parameter_group("group1").update_values(jnp.array([100.0, 200.0]))
        assert not jnp.allclose(copied_group.values, jnp.array([100.0, 200.0]))

    def test_contains(self):
        """Test __contains__ method."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="test", param_type=ParameterType.CUSTOM, initial_values=[1.0]
        )

        assert "test" in params
        assert "nonexistent" not in params

    def test_getitem(self):
        """Test __getitem__ method."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="test", param_type=ParameterType.CUSTOM, initial_values=[1.0]
        )

        group = params["test"]
        assert group.name == "test"

    def test_str_representation(self):
        """Test string representation."""
        # Test empty parameters
        params = TrainableParameters()
        str_repr = str(params)
        assert "No parameter groups defined" in str_repr

        # Test with parameter groups
        params.add_parameter_group(
            name_or_group="group1",
            param_type=ParameterType.ROTATION_ANGLE,
            initial_values=[0.0, 1.0],
        )
        params.add_parameter_group(
            name_or_group="group2", param_type=ParameterType.CUSTOM, initial_values=[2.0]
        )

        str_repr = str(params)
        assert "TrainableParameters" in str_repr
        assert "(2 groups, 3 total params)" in str_repr
        assert "group1" in str_repr
        assert "group2" in str_repr

    def test_repr(self):
        """Test repr representation."""
        params = TrainableParameters()
        params.add_parameter_group(
            name_or_group="test", param_type=ParameterType.CUSTOM, initial_values=[1.0, 2.0]
        )

        repr_str = repr(params)
        assert "TrainableParameters" in repr_str
        assert "groups=1" in repr_str
        assert "total_params=2" in repr_str


class TestParameterType:
    """Test ParameterType enum."""

    def test_enum_values(self):
        """Test parameter type enum values."""
        assert ParameterType.ROTATION_ANGLE.value == "rotation_angle"
        assert ParameterType.MEASUREMENT_TIME.value == "measurement_time"
        assert ParameterType.CUSTOM.value == "custom"

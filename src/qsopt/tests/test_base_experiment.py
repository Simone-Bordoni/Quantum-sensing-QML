"""
Tests for Base Experiment Abstract Class
========================================

Test suite for the abstract Experiment base class.
"""

import optax
import pytest

from qsopt.core.experiment import Experiment
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    InitialStateType,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalConstants,
    SystemDimensions,
)
from qsopt.core.trainable_parameters import TrainableParameters


class ConcreteExperiment(Experiment):
    """Concrete implementation of Experiment for testing."""

    def _generate_operators(self):
        """Minimal implementation."""
        self.operators = {}

    def _generate_hamiltonian(self):
        """Minimal implementation."""
        self.hamiltonians = {}
        self.lindblad_operators = {}

    def _initialize_caches(self):
        """Minimal implementation."""
        pass

    def get_initial_state(self):
        """Minimal implementation."""
        import qutip as qt

        return qt.basis(2, 0).proj()

    def simulation(self, *args, **kwargs):
        """Minimal implementation."""
        return 0.5

    def run_simulation(self, batch_size=1):
        """Minimal implementation."""
        from qsopt.core.callback import OptimizationCallback

        return OptimizationCallback()

    def optimize_rotations(
        self,
        num_steps=100,
        batch_size=1,
        tolerance=1e-6,
        verbose=True,
        verbose_step=10,
        callback=None,
        **kwargs,
    ):
        """Minimal implementation."""
        from qsopt.core.callback import OptimizationCallback

        return OptimizationCallback()


@pytest.fixture
def experimental_params():
    """Create experimental parameters for testing."""
    return ExperimentalParameters(
        physical_constants=PhysicalConstants(
            chi=0.01, photon_cavity_coupling=0.1, inverse_pulse_width=1.0
        ),
        system_dims=SystemDimensions(cavity_levels=3, qubit_levels=2, field_levels=3),
        measurement=MeasurementProtocol(initial_time=0.0, final_time=10.0, time_interval=1.0),
        noise_config=NoiseConfiguration(),
        initial_state=InitialStateConfig(state_type=InitialStateType.VACUUM),
    )


@pytest.fixture
def trainable_params():
    """Create trainable parameters for testing."""
    params = TrainableParameters()
    params.add_rotation_angles(
        ["theta1", "theta2"], [0.0, 0.0], optimizer=optax.adam(learning_rate=0.01)
    )
    return params


class TestExperimentAbstractClass:
    """Tests for the abstract Experiment class."""

    def test_cannot_instantiate_abstract_class(self, experimental_params, trainable_params):
        """Test that Experiment cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Experiment(experimental_params, trainable_params)

    def test_concrete_implementation_works(self, experimental_params, trainable_params):
        """Test that a concrete implementation can be instantiated."""
        exp = ConcreteExperiment(experimental_params, trainable_params)
        assert exp is not None
        assert exp.experimental_params is experimental_params
        assert exp.trainable_params is trainable_params

    def test_base_attributes_initialized(self, experimental_params, trainable_params):
        """Test that base class initializes required attributes."""
        exp = ConcreteExperiment(experimental_params, trainable_params)

        # Check that base attributes are set
        assert hasattr(exp, "experimental_params")
        assert hasattr(exp, "trainable_params")
        assert hasattr(exp, "operators")
        assert hasattr(exp, "hamiltonians")
        assert hasattr(exp, "lindblad_operators")
        assert hasattr(exp, "callback")

    def test_abstract_methods_must_be_implemented(self):
        """Test that all abstract methods must be implemented."""

        class IncompleteExperiment(Experiment):
            """Incomplete implementation missing abstract methods."""

            pass

        # Should raise TypeError for missing abstract methods
        with pytest.raises(TypeError):
            # This will fail because abstract methods are not implemented
            experimental_params = ExperimentalParameters(
                physical_constants=PhysicalConstants(),
                system_dims=SystemDimensions(),
                measurement=MeasurementProtocol(
                    initial_time=0.0, final_time=10.0, time_interval=1.0
                ),
                noise_config=NoiseConfiguration(),
                initial_state=InitialStateConfig(state_type=InitialStateType.VACUUM),
            )
            trainable_params = TrainableParameters()
            IncompleteExperiment(experimental_params, trainable_params)

    def test_rotation_angles_property_getter(self, experimental_params, trainable_params):
        """Test the rotation_angles property getter."""
        exp = ConcreteExperiment(experimental_params, trainable_params)
        angles = exp.rotation_angles

        assert isinstance(angles, dict)
        assert len(angles) == 2  # We added 2 rotation angles
        assert all(isinstance(v, float) for v in angles.values())

    def test_rotation_angles_property_setter(self, experimental_params, trainable_params):
        """Test the rotation_angles property setter."""
        exp = ConcreteExperiment(experimental_params, trainable_params)

        # Get parameter names
        angle_names = list(exp.rotation_angles.keys())

        # Set new values
        new_angles = {angle_names[0]: 1.0, angle_names[1]: 2.0}
        exp.rotation_angles = new_angles

        # Verify they were set
        updated_angles = exp.rotation_angles
        assert updated_angles[angle_names[0]] == 1.0
        assert updated_angles[angle_names[1]] == 2.0

    def test_default_optimize_measurement_times_not_implemented(
        self, experimental_params, trainable_params
    ):
        """Test that default optimize_measurement_times raises NotImplementedError."""
        exp = ConcreteExperiment(experimental_params, trainable_params)

        with pytest.raises(
            NotImplementedError, match="does not implement measurement time optimization"
        ):
            exp.optimize_measurement_times()

    def test_default_save_experiment_report_not_implemented(
        self, experimental_params, trainable_params
    ):
        """Test that default save_experiment_report raises NotImplementedError."""
        exp = ConcreteExperiment(experimental_params, trainable_params)

        with pytest.raises(
            NotImplementedError, match="does not implement experiment report saving"
        ):
            exp.save_experiment_report()

    def test_callback_initialized(self, experimental_params, trainable_params):
        """Test that callback is properly initialized."""
        exp = ConcreteExperiment(experimental_params, trainable_params)

        assert exp.callback is not None
        from qsopt.core.callback import OptimizationCallback

        assert isinstance(exp.callback, OptimizationCallback)

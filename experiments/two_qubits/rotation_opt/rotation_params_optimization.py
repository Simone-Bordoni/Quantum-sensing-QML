import numpy as np
import optax
import matplotlib.pyplot as plt
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalConstants,
    SystemDimensions,
    MeasurementProtocol,
    InteractionType,
    QubitInteraction,
    InitialStateConfig,
    InitialStateType,
    NoiseConfiguration
)
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.core.experiment.two_qubit_experiment import TwoQubitExperiment
from qsopt.utils.visualization import plot_optimization_dashboard

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

physical_constants = PhysicalConstants(
    n_qubits=2,
    chi=[30.0, 30.0],
    photon_cavity_coupling=15.0,
    inverse_pulse_width=1.0,
)

system_dims = SystemDimensions(
    field_levels=2,
    cavity_levels=2,
    qubit_levels=[2, 2]
)

measurement = MeasurementProtocol(
    measurement_times=[-5.0, 5.0] 
)

initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)

noise_config = NoiseConfiguration(
    depolarizing=[0.0, 0.0],
    dephasing=[0.0, 0.0],
    relaxation=[0.0, 0.0]
)

exp_params = ExperimentalParameters(
    physical_constants=physical_constants,
    system_dims=system_dims,
    measurement=measurement,
    initial_state=initial_state,
    noise_config=noise_config
)


trainable_params = TrainableParameters()
optimizer = optax.sgd(learning_rate=5.)

trainable_params.add_rotation_angles(
    names=["theta1_q1", "theta2_q1", "theta1_q2", "theta2_q2"],
    initial_values=[np.pi/2, -np.pi/2, np.pi/2, -np.pi/2],

)

experiment = TwoQubitExperiment(exp_params, trainable_params)

benchmark_results = experiment.run_simulation()

history = experiment.optimize_rotations(
    num_steps=500,
    batch_size=1,
    tolerance=1e-8,
    verbose=True,
    verbose_step=50,
    theta_init=[np.pi/3, -np.pi/3, np.pi/4, -np.pi]
)

fig = plot_optimization_dashboard(
    optimization_callback=history,
    reference_callback=benchmark_results,
    save_path='experiments/two_qubits/rotation_opt/results/opt_dashboard_1.png'
)

experiment.save_experiment_report(save_path='experiments/two_qubits/rotation_opt/results/experiment_report_1.json')
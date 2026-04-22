import numpy as np
import optax
import matplotlib.pyplot as plt
from pathlib import Path
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalSetup,
    SystemDimensions,
    MeasurementProtocol,
    InitialStateConfig,
    InitialStateType,
    NoiseConfiguration
)
from qsopt.core.circuit import create_ry_circuit
from qsopt.core.experiment import Experiment
from qsopt.core.loss_functions import DetectionMetric
from qsopt.utils.visualization import plot_optimization_dashboard

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

physical_setup = PhysicalSetup(
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
    physical_setup=physical_setup,
    system_dims=system_dims,
    measurement=measurement,
    initial_state=initial_state,
    noise_config=noise_config
)


initial_circuit = create_ry_circuit(n_qubits=2, theta_values=np.pi / 2)
final_circuit = create_ry_circuit(n_qubits=2, theta_values=-np.pi / 2)
detection_metric = DetectionMetric(n_qubits=2, detection_criterion='max computational distance')
optimizer = optax.sgd(learning_rate=5.0)

experiment = Experiment(exp_params, initial_circuit, final_circuit, detection_metric=detection_metric)

benchmark_results = experiment.run_simulation()

history = experiment.optimize_rotations(
    num_steps=500,
    batch_size=1,
    tolerance=1e-8,
    verbose=True,
    verbose_step=50,
    initial_values=[np.pi/3, -np.pi/3, np.pi/4, -np.pi],
    optimizer=optimizer,
)

fig = plot_optimization_dashboard(
    optimization_callback=history,
    reference_callback=benchmark_results,
    save_path='experiments/two_qubits/rotation_opt/results/opt_dashboard.png'
)

history.save(str(Path('experiments/two_qubits/rotation_opt/results/optimization_history.npz')))
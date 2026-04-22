import numpy as np
import optax
from pathlib import Path

from qsopt.core.circuit import create_ry_circuit
from qsopt.core.experiment import Experiment
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    InitialStateType,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalSetup,
    SystemDimensions,
)
from qsopt.utils.visualization import plot_optimization_dashboard

gm = 0.03 * 2 * np.pi

# Define custom physical setup
custom_setup = PhysicalSetup(
    chi = 0.5 * gm,                    # Dispersive coupling
    photon_cavity_coupling = gm,  # Photon-cavity coupling
    inverse_pulse_width = 0.1 * gm      # Inverse pulse width
)

# Define custom system dimensions
custom_dims = SystemDimensions(
    cavity_levels=2,
    qubit_levels=2,
    field_levels=2
)

# Define measurement protocol
custom_measurement = MeasurementProtocol(
    measurement_times = list(np.array([-5.0, 0.0, 5.0])/(0.1 * gm)) # Specific measurement times
)

# Define initial state configuration (SINGLE_PHOTON)
initial_state = InitialStateConfig(
    state_type=InitialStateType.SINGLE_PHOTON
)

# Define noise configuration
noise_config = NoiseConfiguration(
    depolarizing = 0.0005,  
    dephasing = 0.0005,      
    relaxation = 0.0005
)

# Create parameters with custom configuration
exp_parameters = ExperimentalParameters(
    physical_setup=custom_setup,
    system_dims=custom_dims,
    measurement=custom_measurement,
    initial_state=initial_state,
    noise_config=noise_config
)

initial_circuit = create_ry_circuit(n_qubits=1, theta_values=np.pi / 2)
final_circuit = create_ry_circuit(n_qubits=1, theta_values=-np.pi / 2)

experiment = Experiment(exp_parameters, initial_circuit, final_circuit)

benchmark_results = experiment.run_simulation()

history = experiment.optimize_rotations(
    initial_values=[1.4, -1.4],
    num_steps=200,
    batch_size=1,
    verbose=True,
    verbose_step=20,
    tolerance=1e-9,
    optimizer=optax.sgd(learning_rate=0.3),
)

fig = plot_optimization_dashboard(
    optimization_callback=history,
    reference_callback=benchmark_results,
    save_path='experiments/single_qubit/rotation_opt/results/opt_dashboard_1.pdf'
)

history.save(str(Path("experiments/single_qubit/rotation_opt/results/optimization_history_1.npz")))
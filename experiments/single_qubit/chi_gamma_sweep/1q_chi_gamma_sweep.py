import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalConstants,
    SystemDimensions,
    NoiseConfiguration,
    MeasurementProtocol,
    InitialStateConfig,
    InitialStateType
)
from qsopt.core.circuit import create_ry_circuit
from qsopt.core.experiment import Experiment
from qsopt.utils import plot_sweep_results

# Single Qubit Experiment Setup
phys_const_1q = PhysicalConstants(
    n_qubits=1,
    chi=5.0,  # Dispersive coupling
    photon_cavity_coupling=10.0,  # gamma
    inverse_pulse_width=1.0  # sigma (pulse width parameter)
)

sys_dims_1q = SystemDimensions(
    field_levels=2,
    cavity_levels=2,
    qubit_levels=2
)

noise = NoiseConfiguration(
    dephasing=0.01,
    relaxation=0.01,
    depolarizing=0.01
)

meas_protocol_1q = MeasurementProtocol(
    measurement_times=[-5., 5.],
    initial_time_uncertainty=0.0
)

initial_state_1q = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)

exp_params_1q = ExperimentalParameters(
    physical_constants=phys_const_1q,
    system_dims=sys_dims_1q,
    measurement=meas_protocol_1q,
    initial_state=initial_state_1q,
    noise_config=noise
)

initial_circuit = create_ry_circuit(n_qubits=1, theta_values=np.pi / 2)
final_circuit = create_ry_circuit(n_qubits=1, theta_values=-np.pi / 2)

# Create experiment
exp_1q = Experiment(exp_params_1q, initial_circuit, final_circuit)

results_1q_sweep = exp_1q.sweep_chi_gamma(
    chi_interval=[0.1, 15.0],
    gamma_interval=[0.1, 30.0],
    resolution_chi=30,
    resolution_gamma=30,
    chi_scale='linear',
    gamma_scale='linear',
    batch_size=1,
    verbose=True
)

plot_sweep_results(
    results_1q_sweep,
    results_to_plot=['contrast_map'],
    mark_optimal=True,
    save_path=str(Path("experiments/single_qubit/chi_gamma_sweep/contrast_map.png"))
)

plot_sweep_results(
    results_1q_sweep,
    results_to_plot=['detection_map', 'detection_without_map'],
    mark_optimal=True,
    save_path=str(Path("experiments/single_qubit/chi_gamma_sweep/detection_map.png"))
)

plt.show()
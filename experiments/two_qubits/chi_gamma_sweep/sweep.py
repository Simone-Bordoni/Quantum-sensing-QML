import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalConstants,
    QubitInteraction,
    NoiseConfiguration,
    SystemDimensions,
    MeasurementProtocol,
    InitialStateConfig,
    InteractionType,
    InitialStateType
)
from qsopt.core.circuit import create_ry_circuit
from qsopt.core.experiment import Experiment
from qsopt.utils import plot_sweep_results

np.random.seed(42)

interaction=QubitInteraction(
    qubit_indices=(0, 1),
    interaction_type=InteractionType.XX,
    chi=0.1
)

phys_const_2q = PhysicalConstants(
    n_qubits=2,
    chi=[40.0, 40.0],  # Equal dispersive coupling for both qubits
    photon_cavity_coupling=10.0,  # gamma
    inverse_pulse_width=1.0,
    qubit_interactions=[interaction]
)

sys_dims_2q = SystemDimensions(
    field_levels=2,
    cavity_levels=2,
    qubit_levels=[2, 2]
)

meas_protocol_2q = MeasurementProtocol(
    measurement_times=[-5, 5],
)

initial_state_2q = InitialStateConfig(
    state_type=InitialStateType.SINGLE_PHOTON
)

noise = NoiseConfiguration(
    dephasing=0.0,
    relaxation=0.0,
    depolarizing=0.0
)

exp_params_2q = ExperimentalParameters(
    physical_constants=phys_const_2q,
    system_dims=sys_dims_2q,
    measurement=meas_protocol_2q,
    initial_state=initial_state_2q,
    noise_config=noise
)

initial_circuit = create_ry_circuit(n_qubits=2, theta_values=np.pi / 2)
final_circuit = create_ry_circuit(n_qubits=2, theta_values=-np.pi / 2)

# Create experiment with default detector (1-P(00))
exp_2q = Experiment(exp_params_2q, initial_circuit, final_circuit)

results_2q_sweep = exp_2q.sweep_chi_gamma(
    chi_interval=[0.1, 40.0],
    gamma_interval=[0.1, 40.0],
    resolution_chi=40,
    resolution_gamma=40,
    chi_scale='linear',
    gamma_scale='linear',
    batch_size=1,
    verbose=True
)

plot_sweep_results(
    results_2q_sweep,
    results_to_plot=['p00', 'p01', 'p10', 'p11'],
    save_path=str(Path('experiments/two_qubits/chi_gamma_sweep/coupledxx_qubits_probability_maps.png'))
)

plot_sweep_results(
    results_2q_sweep,
    results_to_plot=['metric_map', 'detection_map', 'detection_without_map'],
    mark_optimal=True,
    save_path=str(Path('experiments/two_qubits/chi_gamma_sweep/coupledxx_qubits_detection_maps.png'))
)

plt.show()
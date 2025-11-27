import numpy as np
import matplotlib.pyplot as plt
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
from qsopt.core.trainable_parameters import TrainableParameters
from qsopt.utils import compute_asymmetry_coupling_sweep
from qsopt.core.experiment import TwoQubitExperiment
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

# Rotation parameters for both qubits (theta1_q1, theta2_q1, theta1_q2, theta2_q2)
train_params_2q = TrainableParameters()
train_params_2q.add_rotation_angles(
    ["theta1_q1", "theta2_q1", "theta1_q2", "theta2_q2"],
    [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]
)

# Create experiment with default detector (1-P(00))
exp_2q = TwoQubitExperiment(exp_params_2q, train_params_2q)

results = compute_asymmetry_coupling_sweep(
    exp_2q,
    asymmetry_interval=[-8.0, 8.0],
    coupling_interval=[0.0, 10.0],
    resolution_asymmetry=30,
    resolution_coupling=30,
    interaction_type=InteractionType.ZZ,
    chi_mean_factor=10.0,  # χ_mean = 10 * γ = 100
    gamma=20.0,
    verbose=True
)

plot_sweep_results(
    results,
    results_to_plot=['p00', 'p01', 'p10', 'p11'],
    save_path='experiments/two_qubits/asymmetry_coupling_sweep/coupledzz_qubits_probability_maps.png'
)

plot_sweep_results(
    results,
    results_to_plot=['contrast_map', 'detection_map', 'detection_without_map'],
    mark_optimal=True,
    save_path='experiments/two_qubits/asymmetry_coupling_sweep/coupledzz_qubits_detection_maps.png'
)
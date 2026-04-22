import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalSetup,
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
from qsopt.utils.results import SweepResults
from qsopt.utils import plot_sweep_results

np.random.seed(42)

interaction=QubitInteraction(
    qubit_indices=(0, 1),
    interaction_type=InteractionType.XX,
    chi=0.1
)

phys_setup_2q = PhysicalSetup(
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
    physical_setup=phys_setup_2q,
    system_dims=sys_dims_2q,
    measurement=meas_protocol_2q,
    initial_state=initial_state_2q,
    noise_config=noise
)

initial_circuit = create_ry_circuit(n_qubits=2, theta_values=np.pi / 2)
final_circuit = create_ry_circuit(n_qubits=2, theta_values=-np.pi / 2)

# Create experiment with default detector (1-P(00))
exp_2q = Experiment(exp_params_2q, initial_circuit, final_circuit)


def compute_asymmetry_coupling_sweep(
    experiment: Experiment,
    asymmetry_interval: list[float],
    coupling_interval: list[float],
    resolution_asymmetry: int,
    resolution_coupling: int,
    interaction_type: InteractionType,
    chi_mean_factor: float,
    gamma: float,
    verbose: bool = True,
) -> SweepResults:
    """Compute Δχ/γ vs χ12/γ sweep using the unified Experiment API."""
    asymmetry_vals = np.linspace(asymmetry_interval[0], asymmetry_interval[1], resolution_asymmetry)
    coupling_vals = np.linspace(coupling_interval[0], coupling_interval[1], resolution_coupling)

    shape = (resolution_coupling, resolution_asymmetry)
    metric_map = np.zeros(shape)
    detection_map = np.zeros(shape)
    detection_without_map = np.zeros(shape)
    p00 = np.zeros(shape)
    p01 = np.zeros(shape)
    p10 = np.zeros(shape)
    p11 = np.zeros(shape)

    chi_mean = chi_mean_factor * gamma
    total_points = resolution_asymmetry * resolution_coupling
    done = 0

    saved_state = experiment._save_sweep_state()
    try:
        for i, delta_ratio in enumerate(asymmetry_vals):
            delta_chi = delta_ratio * gamma
            chi1 = chi_mean + 0.5 * delta_chi
            chi2 = chi_mean - 0.5 * delta_chi

            for j, coupling_ratio in enumerate(coupling_vals):
                chi12 = coupling_ratio * gamma
                current_interaction = QubitInteraction(
                    qubit_indices=(0, 1),
                    interaction_type=interaction_type,
                    chi=chi12,
                )

                experiment._update_chi_gamma(
                    chi=[chi1, chi2],
                    gamma=gamma,
                    qubit_interactions=[current_interaction],
                )

                sim = experiment.run_simulation_with_probabilities()
                metric_map[j, i] = sim["metric"]
                detection_map[j, i] = sim["detection_with"]
                detection_without_map[j, i] = sim["detection_without"]
                p00[j, i] = sim["probs_with"]["00"]
                p01[j, i] = sim["probs_with"]["01"]
                p10[j, i] = sim["probs_with"]["10"]
                p11[j, i] = sim["probs_with"]["11"]

                done += 1
                if verbose and done % max(1, total_points // 10) == 0:
                    print(f"Progress: {100.0 * done / total_points:.1f}%")
    finally:
        experiment._restore_sweep_state(saved_state)

    max_idx = np.unravel_index(np.argmax(metric_map), metric_map.shape)
    metadata = {
        "optimal_idx": max_idx,
        "max_metric": float(metric_map[max_idx]),
        "n_qubits": experiment.experimental_params.n_qubits,
        "cavity_levels": experiment.experimental_params.system_dims.cavity_levels,
        "qubit_levels": experiment.experimental_params.system_dims.qubit_levels,
        "field_levels": experiment.experimental_params.system_dims.field_levels,
        "measurement_times": experiment.experimental_params.measurement.measurement_times,
        "initial_time_uncertainty": experiment.experimental_params.measurement.initial_time_uncertainty,
        "depolarizing_rate": experiment.experimental_params.noise_config.depolarizing,
        "dephasing_rate": experiment.experimental_params.noise_config.dephasing,
        "relaxation_rate": experiment.experimental_params.noise_config.relaxation,
        "initial_state": experiment.experimental_params.initial_state.state_type.name,
        "inverse_pulse_width": experiment.experimental_params.physical_setup.inverse_pulse_width,
    }

    return SweepResults(
        param1_name="chi12/gamma",
        param1_vals=coupling_vals,
        param1_scale="linear",
        param2_name="Delta chi/gamma",
        param2_vals=asymmetry_vals,
        param2_scale="linear",
        results={
            "metric_map": metric_map,
            "detection_map": detection_map,
            "detection_without_map": detection_without_map,
            "p00": p00,
            "p01": p01,
            "p10": p10,
            "p11": p11,
        },
        metadata=metadata,
    )

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
    save_path=str(Path('experiments/two_qubits/asymmetry_coupling_sweep/coupledzz_qubits_probability_maps.png'))
)

plot_sweep_results(
    results,
    results_to_plot=['metric_map', 'detection_map', 'detection_without_map'],
    mark_optimal=True,
    save_path=str(Path('experiments/two_qubits/asymmetry_coupling_sweep/coupledzz_qubits_detection_maps.png'))
)

plt.show()
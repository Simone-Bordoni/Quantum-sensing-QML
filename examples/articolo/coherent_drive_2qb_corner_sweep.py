"""Coherent-drive 2-qubit detector: 4D parameter sweep + corner plot.

Reproduces the `coherent_drive_2_qubits` tutorial setup and runs the high-dimensional
sweep over (chi, kappa, amplitude, time_interval). Saves the SweepResults to an .npz
and the corner-view figure to a .pdf.
"""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: only save figures to file
import numpy as np

from qsopt import (
    DetectionMetric,
    Experiment,
    ExperimentalParameters,
    Interaction,
    InteractionType,
    NoiseModel,
    PhysicalModel,
    SystemConfiguration,
    TimeProtocol,
)
from qsopt.core.circuit import create_ry_circuit
from qsopt.core.gates import CNOTGate
from qsopt.utils.results import save_results
from qsopt.utils.visualization import plot_sweep_corner

# Output location (next to this script).
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SWEEP_PATH = os.path.join(RESULTS_DIR, "coherent_drive_2qb_corner_sweep.npz")
CORNER_PATH = os.path.join(RESULTS_DIR, "coherent_drive_2qb_corner.pdf")


def build_experiment():
    """Build the coherent-drive 2-qubit Experiment from the tutorial.

    Returns:
        Experiment: The configured experiment ready to sweep.
    """
    chi = 7
    eps = chi * 2
    kappa = 0.01

    interactions = [
        Interaction(
            interaction_type=InteractionType.DISPERSIVE,
            subsystem1=("cavity", 0),
            subsystem2=("qubit", 0),
            parameters={"chi": chi},
        ),
        Interaction(
            interaction_type=InteractionType.DISSIPATION,
            subsystem1=("cavity", 0),
            parameters={"kappa": kappa},
        ),
    ]

    physical_model = PhysicalModel(
        perturbation_type="persistent",
        n_cavities=1,
        n_fields=0,
        n_qubits=2,
        cavity_levels=4,
        field_levels=1,
        qubit_levels=2,
        interactions=interactions,
    )

    noise = NoiseModel(depolarizing=0.01, dephasing=0.01, relaxation=0.01)

    custom_times = TimeProtocol(
        t_simulation_start=-6,
        n_measurements=4,
        time_interval=4,
        random_measurements_offset=True,
        noisy_simulation_start=True,
    )

    config_set = [
        SystemConfiguration(name="no drive", is_ground=True),
        SystemConfiguration(
            name="driven",
            interactions=[
                Interaction(
                    interaction_type=InteractionType.DRIVE,
                    subsystem1=("cavity", 0),
                    parameters={"amplitude": eps},
                ),
            ],
        ),
    ]

    exp_parameters = ExperimentalParameters(
        physical_model=physical_model,
        noise_model=noise,
        time_protocol=custom_times,
        configuration_set=config_set,
    )

    # Trainable RY + CNOT circuits applied before/after the evolution.
    initial_circuit = create_ry_circuit(n_qubits=2, theta_values=np.pi / 2)
    initial_circuit.add_entangling_layer(CNOTGate, pattern="circular")
    final_circuit = create_ry_circuit(n_qubits=2, theta_values=-np.pi / 2)
    final_circuit.add_entangling_layer(CNOTGate, pattern="circular")

    detection_metric = DetectionMetric(
        n_cavities=1,
        n_fields=0,
        n_qubits=2,
        config_names=["no drive", "driven"],
        detection_criterion="max trace distance",
        perturbation_type=physical_model.perturbation_type,
    )

    return Experiment(
        experimental_params=exp_parameters,
        initial_circuit=initial_circuit,
        final_circuit=final_circuit,
        detection_metric=detection_metric,
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    experiment = build_experiment()

    # A time_interval axis needs a fixed measurement window so the metric is scored
    # over the same region as the spacing changes.
    windowed_protocol = TimeProtocol(
        t_simulation_start=-4,
        n_measurements=20,
        time_interval=4,
        random_measurements_offset=True,
        per_measurement_jitter=0.01,
        noisy_simulation_start=True,
        window_start=-2,
        window_end=8,
    )

    # 4D sweep over the knobs of this model: chi (dispersive coupling), kappa (cavity
    # dissipation), amplitude (coherent drive of the 'driven' config) and the measurement
    # time_interval.
    results_hd = experiment.sweep(
        {
            "chi": np.linspace(1.0, 20.0, 4),
            "kappa": np.linspace(8.0, 20.0, 4),
            "amplitude": np.linspace(1.0, 20.0, 4),
            "time_interval": np.linspace(3.2, 4.0, 4),
        },
        time_protocol=windowed_protocol,
        batch_size=32,
        verbose=True,
    )
    print(results_hd)

    save_results(results_hd, SWEEP_PATH)
    print(f"Saved sweep to {SWEEP_PATH}")

    # Corner view: every axis pair at once (validation quantity, as in the tutorial).
    plot_sweep_corner(results_hd, quantity="validation", save_path=CORNER_PATH)
    print(f"Saved corner plot to {CORNER_PATH}")


if __name__ == "__main__":
    main()

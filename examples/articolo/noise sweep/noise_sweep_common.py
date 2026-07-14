"""Shared setup for the RY noise-sweep experiments (1-qubit and 2-qubit).

Values are taken from the ``two_qubit_ry_angle_sweep`` tutorial and are shared across both
qubit counts: only the circuits (and their qubit-count-driven consequences -- number of
dispersive interactions and the detection metric's ``n_qubits``) change between the 1-qubit
and 2-qubit experiments. The noise model and time protocol are always the tutorial's.

Each experiment runs ``len(NOISE_LEVELS)`` optimizations at increasing noise, then, for every
level, derives the deployable detection protocol and records the accuracy: the average along
the diagonal of the joint confusion matrix (the per-configuration true-positive rate),
excluding the prediction-only ``mixed`` column.
"""

import gc
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import erfc

import matplotlib
matplotlib.use("Agg")  # headless: scripts run without a display
import matplotlib.pyplot as plt

from qsopt import (
    DetectionMetric,
    Experiment,
    ExperimentalParameters,
    Interaction,
    InteractionType,
    NoiseModel,
    OptimizationCallback,
    PhysicalModel,
    State,
    SubsystemState,
    SystemConfiguration,
    TimeProtocol,
)
from qsopt.core.circuit import create_ry_circuit

# ------------------------------------------------------------------ shared values (from tutorial)
SIGMA = 1
K = 15 * SIGMA
# Per-qubit dispersive couplings; qubit 0 is identical across both experiments.
CHI_PER_QUBIT = [0.5 * K, 0.3 * K]

# Five optimizations at increasing noise (equal per-channel rate). Includes the tutorial's 0.01.
NOISE_LEVELS = np.linspace(0.0, 0.04, 5)

# Optimization hyper-parameters (shared).
NUM_STEPS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.1
TOLERANCE = 1e-7
ANNEAL_TOLERANCES = 1000

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def pulse(t, **kwargs):
    """Time-dependent input-cavity coupling g(t) (tutorial pulse).

    Args:
        t: float or JAX array, time variable.
        **kwargs: dict carrying 'sigma' (pulse bandwidth).

    Returns:
        JAX array: normalized coupling strength g(t).
    """
    sigma = kwargs.get("sigma", 1.0)
    # Clamp |x|<=8 so exp(-x^2)/erfc(x) stays finite far past the pulse (avoids a 0/0 NaN).
    dx = jnp.clip(sigma * t, -8.0, 8.0)
    coupling = jnp.sqrt(2 * sigma / jnp.sqrt(jnp.pi) * jnp.exp(-(dx**2)) / erfc(dx))
    return jnp.array(coupling, float)


def build_experiment(n_qubits, noise_level):
    """Build the transient single-photon RY experiment for a given qubit count and noise level.

    Everything except the circuits, the number of dispersive interactions and the detection
    metric's ``n_qubits`` is identical across qubit counts; the noise model and time protocol
    are the tutorial's.

    Args:
        n_qubits (int): number of qubits (1 or 2).
        noise_level (float): per-channel rate for depolarizing = dephasing = relaxation.

    Returns:
        Experiment: the configured experiment (default RY setup/decoding circuits).
    """
    chi_list = CHI_PER_QUBIT[:n_qubits]
    interactions = [
        Interaction(
            interaction_type=InteractionType.DISPERSIVE,
            subsystem1=("cavity", 0),
            subsystem2=("qubit", i),
            parameters={"chi": chi_list[i]},
        )
        for i in range(n_qubits)
    ]

    physical_model = PhysicalModel(
        perturbation_type="transient",
        n_cavities=1,
        n_fields=1,
        n_qubits=n_qubits,
        cavity_levels=2,
        field_levels=2,
        qubit_levels=2,
        interactions=interactions,
    )

    # Noise pattern from the tutorial (equal channels), scaled to this level.
    noise = NoiseModel(
        depolarizing=noise_level,
        dephasing=noise_level,
        relaxation=noise_level,
    )

    time_protocol = TimeProtocol(
        t_simulation_start=-6,
        n_measurements=4,
        time_interval=4,
        random_measurements_offset=True,
        noisy_simulation_start=True,
    )

    config_set = [
        SystemConfiguration(
            name="no interaction",
            init_field_states={0: SubsystemState(State.FOCK, {"n": 0})},
            is_ground=True,
        ),
        SystemConfiguration(
            name="with interaction",
            init_field_states={0: SubsystemState(State.FOCK, {"n": 1})},
            interactions=[
                Interaction(
                    interaction_type=InteractionType.INPUT_OUTPUT,
                    subsystem1=("cavity", 0),
                    subsystem2=("field", 0),
                    parameters={"gamma": 1, "kappa": K, "sigma": SIGMA},
                    time_modulation=pulse,
                ),
            ],
        ),
    ]

    exp_parameters = ExperimentalParameters(
        physical_model=physical_model,
        noise_model=noise,
        time_protocol=time_protocol,
        configuration_set=config_set,
    )

    # Circuits are the only genuine difference between the experiments: one RY per qubit.
    initial_circuit = create_ry_circuit(n_qubits=n_qubits, theta_values=[np.pi / 2] * n_qubits)
    final_circuit = create_ry_circuit(n_qubits=n_qubits, theta_values=[-np.pi / 2] * n_qubits)

    detection_metric = DetectionMetric(
        n_cavities=1,
        n_fields=1,
        n_qubits=n_qubits,
        config_names=["with interaction", "no interaction"],
        detection_criterion="max computational distance",
        perturbation_type=physical_model.perturbation_type,
    )

    return Experiment(
        experimental_params=exp_parameters,
        initial_circuit=initial_circuit,
        final_circuit=final_circuit,
        detection_metric=detection_metric,
    )


def diagonal_accuracy(confusion_matrix, config_names):
    """Average true-positive rate: mean of the confusion-matrix diagonal, excluding 'mixed'.

    The diagonal entries are ``(config, config)``; the ``mixed`` label is prediction-only (a
    column, never a true row), so restricting to the diagonal already excludes it.

    Args:
        confusion_matrix (Dict[Tuple[str, str], float]): joint confusion matrix from make_protocol.
        config_names (List[str]): the true-configuration names (matrix rows).

    Returns:
        - ``accuracy`` (float): mean of the per-configuration diagonal entries.
        - ``diagonal`` (List[float]): the per-configuration diagonal entries, in ``config_names`` order.
    """
    diagonal = [float(confusion_matrix.get((name, name), 0.0)) for name in config_names]
    return float(np.mean(diagonal)), diagonal


def run_noise_sweep(n_qubits, name):
    """Run one optimization per noise level, then save the accuracy data and plot.

    Args:
        n_qubits (int): number of qubits (1 or 2).
        name (str): short tag used in the output file names.

    Returns:
        - ``summary_path`` (str): path of the saved summary ``.npz``.
        - ``plot_path`` (str): path of the saved accuracy ``.pdf``.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Setup angles [0.4, 0.5, ...] then decoding angles [-1.1, -1.0, ...], trimmed to the qubit count.
    initial_values = [0.4, 0.5][:n_qubits] + [-1.1, -1.0][:n_qubits]

    accuracies = []
    validations = []
    metrics = []
    diagonals = []
    config_names = None

    for i, noise_level in enumerate(NOISE_LEVELS):
        print(f"\n{'=' * 80}\n[{name}] optimization {i + 1}/{len(NOISE_LEVELS)} "
              f"at noise level {noise_level:.4f}\n{'=' * 80}")

        experiment = build_experiment(n_qubits, noise_level)
        config_names = experiment.config_names

        callback = OptimizationCallback(save_every=1, save_best=True)
        callback = experiment.optimize_rotations(
            initial_values=initial_values,
            num_steps=NUM_STEPS,
            batch_size=BATCH_SIZE,
            tolerance=TOLERANCE,
            optimizer=optax.sgd(learning_rate=LEARNING_RATE),
            anneal_tolerances=ANNEAL_TOLERANCES,
            callback=callback,
            verbose=True,
            verbose_step=25,
        )

        # Derive the deployable protocol (joint confusion matrix) at the best parameters.
        callback = experiment.make_protocol(callback=callback, batch_size=BATCH_SIZE)

        accuracy, diagonal = diagonal_accuracy(callback.confusion_matrix, config_names)
        accuracies.append(accuracy)
        diagonals.append(diagonal)
        validations.append(float(callback.best_validation))
        metrics.append(float(callback.best_metrics["metric"]))
        print(f"[{name}] noise {noise_level:.4f} -> accuracy {accuracy:.4f} "
              f"(diagonal {['%.4f' % d for d in diagonal]})")

        # Persist the full per-level optimization + protocol callback.
        callback.save(os.path.join(RESULTS_DIR, f"noise_sweep_{name}_level{i}.npz"))

        # Release JAX/compilation memory between levels.
        jax.clear_caches()
        gc.collect()

    accuracies = np.array(accuracies)
    diagonals = np.array(diagonals)

    # Save the summary data.
    summary_path = os.path.join(RESULTS_DIR, f"noise_sweep_{name}.npz")
    np.savez(
        summary_path,
        noise_levels=NOISE_LEVELS,
        accuracies=accuracies,
        diagonals=diagonals,
        validations=np.array(validations),
        metrics=np.array(metrics),
        config_names=np.array(config_names, dtype=object),
        n_qubits=n_qubits,
        batch_size=BATCH_SIZE,
        num_steps=NUM_STEPS,
    )
    print(f"\n[{name}] saved summary data to {summary_path}")

    # Plot accuracy vs noise.
    plot_path = os.path.join(RESULTS_DIR, f"noise_sweep_{name}_accuracy.pdf")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(NOISE_LEVELS, accuracies, "o-", color="C0", lw=2, ms=8)
    ax.set_xlabel("noise level (per-channel rate)", fontsize=13)
    ax.set_ylabel("accuracy (mean confusion diagonal)", fontsize=13)
    ax.set_title(f"{n_qubits}-qubit RY detector: accuracy vs noise", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"[{name}] saved accuracy plot to {plot_path}")

    return summary_path, plot_path

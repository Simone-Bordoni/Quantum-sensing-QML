"""
Re-derive the detection protocol / confusion matrix for the multi-photon detector run stored
in ``multi-photon_data.npz`` under a new, three-measurement measurement protocol.

When launched it:
  1. loads the saved (optimization) callback ``multi-photon_data.npz``,
  2. rebuilds its experiment (2 qubits, entangling setup/decode circuits, dispersive chi =
     [7.5, 3.75], transient itinerant Fock photon 0/1/2) under a NEW protocol: three
     measurements at t = -2, 4, 10 (start fixed at -8, 6.0 spacing) with a collective
     uniform(-6, 0) measurement-timing offset,
  3. runs ``make_protocol`` on the callback's best parameters with ``batch_size=16``,
     processed in CHUNKS so peak memory stays bounded (identical result to a single
     batch=16 vmap; only the leaf-probability average is accumulated across chunks),
  4. saves the new modified callback to ``<OUTPUT_STEM>.npz``,
  5. writes the confusion matrix as a square heatmap with NO title (rows == columns == the
     true configurations; the prediction-only 'mixed' column and the state-map / false-signal
     summary are omitted).

Intended to run on a GPU: JAX will use the GPU automatically; ``CHUNK_SIZE`` caps how many of
the 16 timing realizations are held on the device at once, so raise it if the GPU has spare
memory (must divide ``BATCH_SIZE`` to avoid a recompile on the last chunk).

Usage:
    python make_protocol_3meas.py [input_npz]
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend: render to file, no GUI
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import qutip as qt
from jax.scipy.special import erfc

from qsopt import (
    Interaction, InteractionType, PhysicalModel, NoiseModel, TimeProtocol,
    SystemConfiguration, SubsystemState, State, ExperimentalParameters,
    DetectionMetric, Experiment, RYGate, RZGate, CNOTGate, create_ry_circuit,
)
from qsopt.core.callback import OptimizationCallback
from qsopt.core.core_utils import derive_detection_map

# --- Configuration --------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_NPZ = os.path.join(HERE, "multi-photon_data.npz")   # argv[1] overrides
OUTPUT_STEM = "multi-photon_data_3meas_b16"               # output npz + confusion pdf stem

BATCH_SIZE = 16     # timing realizations averaged over the measurement offset
CHUNK_SIZE = 4      # realizations per vmap chunk (raise on GPU; should divide BATCH_SIZE)

# Physics of the trained run (must match how the callback was optimized).
PERTURBATION_TYPE = "transient"
N_QUBITS = 2
MAX_PHOTON = 2
MAX_SEPARATED_PHOTON = 2
CHI_LIST = [7.5, 3.75]
KAPPA = 15.0
GAMMA = 1.0
SIGMA = 1.0
RANDOM_SEED = 0     # makes the offset sampling reproducible


def pulse(t, **kwargs):
    """Gaussian input-cavity coupling g(t) for the itinerant photon release."""
    sigma = kwargs.get("sigma", 0.1)
    dx = sigma * t
    coupling = jnp.sqrt(2 * sigma / jnp.sqrt(jnp.pi) * jnp.exp(-(dx ** 2)) / erfc(dx))
    return jnp.array(coupling, float)


def new_protocol():
    """The new three-measurement protocol with a collective offset (start fixed)."""
    return TimeProtocol(
        t_simulation_start=-8.0, n_measurements=3, time_interval=6.0,
        random_measurements_offset=True, noisy_simulation_start=False)


def build_circuits():
    """The entangling setup/decode circuits used by the trained run (structure only;
    make_protocol overwrites the angles with the callback's best parameters)."""
    init = create_ry_circuit(n_qubits=N_QUBITS, theta_values=np.pi / 4)
    init.add_entangling_layer(CNOTGate, pattern="circular")
    init.add_layer(RYGate, parameters=np.pi / 4)
    init.add_layer(RZGate, parameters=0.0)

    final = create_ry_circuit(n_qubits=N_QUBITS, theta_values=-np.pi / 4)
    final.add_entangling_layer(CNOTGate, pattern="circular")
    final.add_layer(RYGate, parameters=-np.pi / 4)
    return init, final


def gen_config_set():
    """Configurations 0/1/2-photons; the 0-photon baseline is the ground config."""
    def input_output_interaction():
        return Interaction(
            interaction_type=InteractionType.INPUT_OUTPUT,
            subsystem1=("cavity", 0), subsystem2=("field", 0),
            parameters={"gamma": GAMMA, "kappa": KAPPA, "sigma": SIGMA},
            time_modulation=pulse,
        )

    with qt.CoreOptions(default_dtype="jax"):
        config_set = [SystemConfiguration(
            name="0-photons",
            init_field_states={0: SubsystemState(State.FOCK, {"n": 0})},
            is_ground=True,
        )]
        config_set += [SystemConfiguration(
            name=f"{i}-photons",
            init_field_states={0: SubsystemState(State.FOCK, {"n": i})},
            interactions=[input_output_interaction()],
        ) for i in range(1, MAX_SEPARATED_PHOTON + 1)]
        if MAX_PHOTON > MAX_SEPARATED_PHOTON:
            dim = MAX_PHOTON + 1
            field_mixture = sum(qt.fock_dm(dim, n)
                                for n in range(MAX_SEPARATED_PHOTON, MAX_PHOTON + 1))
            field_mixture = field_mixture / field_mixture.tr()
            density_matrix = qt.tensor(qt.fock_dm(dim, 0), field_mixture)
            config_set[-1] = SystemConfiguration(
                name=f"{MAX_SEPARATED_PHOTON}+ photons",
                density_matrix=density_matrix,
                interactions=[input_output_interaction()],
            )
    return config_set


def build_experiment(init_circuit, final_circuit, protocol):
    """Rebuild the Experiment under the given protocol."""
    qubit_cavity = [Interaction(
        interaction_type=InteractionType.DISPERSIVE,
        subsystem1=("cavity", 0), subsystem2=("qubit", i),
        parameters={"chi": CHI_LIST[i]}) for i in range(N_QUBITS)]
    qubit_qubit = [Interaction(
        interaction_type=InteractionType.ZZ,
        subsystem1=("qubit", i), subsystem2=("qubit", j),
        parameters={"chi": 0.001})
        for i in range(N_QUBITS - 1) for j in range(i + 1, N_QUBITS)]

    physical_model = PhysicalModel(
        PERTURBATION_TYPE, n_cavities=1, n_fields=1, n_qubits=N_QUBITS,
        cavity_levels=MAX_PHOTON + 1, field_levels=MAX_PHOTON + 1, qubit_levels=2,
        interactions=qubit_cavity + qubit_qubit)
    noise = NoiseModel(depolarizing=0.001, dephasing=0.001, relaxation=0.001)
    exp_params = ExperimentalParameters(
        physical_model=physical_model, noise_model=noise,
        time_protocol=protocol, configuration_set=gen_config_set(),
        random_seed=RANDOM_SEED)
    detection_metric = DetectionMetric(
        n_cavities=1, n_fields=1, n_qubits=N_QUBITS,
        config_names=exp_params.get_configuration_names(),
        perturbation_type=PERTURBATION_TYPE,
        detection_criterion="max computational distance")
    return Experiment(
        experimental_params=exp_params,
        initial_circuit=init_circuit, final_circuit=final_circuit,
        detection_metric=detection_metric)


def make_protocol_chunked(experiment, callback, batch_size, chunk_size,
                          false_signal_weight=1.0, false_signal_constraint=None,
                          contested_threshold=1e-3):
    """Chunked equivalent of ``Experiment.make_protocol`` for an uncertain protocol.

    Averages the branching leaf probabilities over ``batch_size`` timing realizations, but
    processes them ``chunk_size`` at a time so only that many realizations are vmapped (and
    held in memory) at once. Since make_protocol averages leaf probabilities linearly before
    deriving one map, the accumulated result is identical to a single batch=``batch_size`` call.

    Args:
        - ``experiment`` (Experiment): the rebuilt experiment.
        - ``callback`` (OptimizationCallback): supplies best params, receives the protocol.
        - ``batch_size`` (int): total timing realizations to average over.
        - ``chunk_size`` (int): realizations vmapped per chunk (should divide batch_size).
        - ``false_signal_weight`` / ``false_signal_constraint`` / ``contested_threshold``:
          forwarded to :func:`derive_detection_map` (defaults match make_protocol).

    Returns:
        - ``callback`` (OptimizationCallback): with states_map, joint confusion_matrix,
          false_signal and the averaged probability_tree stored on it.
    """
    exp = experiment
    # Load the callback's best parameters onto the circuits (mirrors make_protocol).
    best = callback.get_best_trainable_params()
    if best is not None:
        best_initial, best_final = best
        exp.initial_circuit.set_trainable_parameters(list(best_initial))
        exp.final_circuit.set_trainable_parameters(list(best_final))

    init_states = exp._cached_initial_states
    if init_states is None:
        raise RuntimeError("Initial states cache is not initialized.")
    solvers = exp.get_solvers()
    circuit_unitaries = exp._prepare_circuit_unitaries()

    # (batch, M+1) stochastic timestamps: each row is one timing realization of the protocol.
    timestamps = jnp.asarray(exp.experimental_params.get_timestamps(batch_size), dtype=float)
    n = int(timestamps.shape[0])

    probability_tree = {}
    for config in exp.experimental_params.configuration_set:
        branch = lambda ts, s=solvers[config.name], r=init_states[config.name]: \
            exp.branching_simulation(s, r, ts, precomputed_unitaries=circuit_unitaries)

        # Accumulate the summed leaf probabilities over chunks, then divide by n for the mean.
        acc = None
        for start in range(0, n, chunk_size):
            batched = jax.vmap(branch)(timestamps[start:start + chunk_size])  # {path: (chunk,)}
            sums = {path: float(jnp.sum(prob)) for path, prob in batched.items()}
            if acc is None:
                acc = sums
            else:
                for path in acc:
                    acc[path] += sums[path]
        probability_tree[config.name] = {path: value / n for path, value in acc.items()}

    result = derive_detection_map(
        probability_tree, exp.config_names, exp.experimental_params.ground,
        exp.experimental_params.perturbation_type, 2 ** exp.n_qubits,
        false_signal_weight=false_signal_weight,
        false_signal_constraint=false_signal_constraint,
        contested_threshold=contested_threshold)
    callback.set_measurement_protocol(
        probability_tree=probability_tree, states_map=result["states_map"],
        confusion_matrix=result["confusion_matrix"], false_signal=result["false_signal"])
    return callback


def plot_confusion_no_title(callback, save_path):
    """Write the confusion matrix as a square, paper-ready heatmap (no title).

    Rows == columns == the true configurations (the prediction-only 'mixed' column and the
    summary overlay are omitted). Tick labels are the bare photon numbers, the axes are named
    'true photon number' (y) and 'predicted photon number' (x), cell values are 2-decimal, and
    the fonts are enlarged for print.
    """
    confusion_matrix = callback.confusion_matrix or {}
    states_map = callback.states_map or {}

    def short(name):
        # '0-photons' -> '0', '2+ photons' -> '2+'
        return name.replace("-photons", "").replace(" photons", "").replace("photons", "").strip()

    names = []
    for true_name, _ in confusion_matrix:
        if true_name not in names:
            names.append(true_name)
    for name in states_map.keys():
        if name not in names:
            names.append(name)
    labels = [short(n) for n in names]

    k = len(names)
    cm = np.array(
        [[float(confusion_matrix.get((t, p), 0.0)) for p in names] for t in names],
        dtype=float).reshape(k, k)

    # Fixed heatmap size (inches); the figure is sized around it so the grid stays 6x6 in
    # regardless of the number of configurations. (Do not crop on save or the size is lost.)
    MATRIX_IN = 6.0
    LEFT, BOTTOM, TOP = 1.15, 1.05, 0.22
    CBAR_GAP, CBAR_W, CBAR_LABELS = 0.12, 0.32, 0.80
    fig_w = LEFT + MATRIX_IN + CBAR_GAP + CBAR_W + CBAR_LABELS   # -> 8.39 in
    fig_h = BOTTOM + MATRIX_IN + TOP                             # -> 7.27 in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([LEFT / fig_w, BOTTOM / fig_h, MATRIX_IN / fig_w, MATRIX_IN / fig_h])
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=max(1.0, float(cm.max())), aspect="auto")
    cax = fig.add_axes([(LEFT + MATRIX_IN + CBAR_GAP) / fig_w, BOTTOM / fig_h,
                        CBAR_W / fig_w, MATRIX_IN / fig_h])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=0, fontsize=24)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, rotation=0, fontsize=24)
    # Bare photon-number ticks, adjacent to the axes; drop the tick marks.
    ax.tick_params(axis="both", length=0, pad=4)
    ax.set_xlabel("predicted photon number", fontsize=22, labelpad=10)
    ax.set_ylabel("true photon number", fontsize=22, labelpad=10)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=34, fontweight="bold")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main(argv):
    input_npz = argv[0] if argv else INPUT_NPZ
    print(f"Loading callback: {input_npz}")
    callback = OptimizationCallback.load_callback(input_npz)
    if callback.mode is None:
        callback.mode = "optimization"

    init_circuit, final_circuit = build_circuits()
    experiment = build_experiment(init_circuit, final_circuit, new_protocol())

    print(f"jax devices: {jax.devices()}")
    print(f"make_protocol chunked (batch_size={BATCH_SIZE}, chunk_size={CHUNK_SIZE}) ...",
          flush=True)
    t0 = time.time()
    callback = make_protocol_chunked(experiment, callback, BATCH_SIZE, CHUNK_SIZE)
    print(f"  done in {time.time() - t0:.0f}s")

    out_npz = os.path.join(HERE, OUTPUT_STEM + ".npz")
    out_pdf = os.path.join(HERE, OUTPUT_STEM + "_confusion.pdf")
    callback.save(out_npz)
    plot_confusion_no_title(callback, out_pdf)

    cm = callback.confusion_matrix or {}
    names = list(dict.fromkeys(t for t, _ in cm))
    print(f"Saved callback:  {out_npz}")
    print(f"Saved confusion: {out_pdf}")
    print(f"configs: {names}")
    print(f"confusion diagonal: {[round(float(cm[(c, c)]), 4) for c in names]}")
    print(f"states_map: {callback.states_map}")
    print(f"false_signal: "
          f"{ {k: round(float(v), 4) for k, v in (callback.false_signal or {}).items()} }")


if __name__ == "__main__":
    main(sys.argv[1:])

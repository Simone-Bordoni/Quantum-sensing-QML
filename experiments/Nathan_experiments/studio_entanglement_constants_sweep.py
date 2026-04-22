"""Sweep system constants for saved studio_entanglement runs.

This script loads saved OptimizationCallback files (NPZ content, even if extension is .pkl),
reconstructs the corresponding experiment topology, fixes circuit parameters to the saved
optimized values, and then sweeps chi and gamma to assess robustness.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qsopt.core.callback import OptimizationCallback
from qsopt.core.circuit import QuantumCircuit, create_ry_circuit
from qsopt.core.experiment.experiment import Experiment
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    InitialStateConfig,
    InitialStateType,
    InteractionType,
    MeasurementProtocol,
    NoiseConfiguration,
    PhysicalConstants,
    QubitInteraction,
    SystemDimensions,
)
from qsopt.core.gates import CNOTGate, RYGate, RZGate
from qsopt.core.loss_functions import DetectionMetric
from qsopt.utils.results import save_results
from qsopt.utils.visualization import plot_sweep_results

BASE_MEASUREMENT_TIMES = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=float)
# Edit this path for your cluster/user setup if needed.
DEFAULT_RESULTS_DIR = (Path("/raid/home/ncampioni/Quantum-sensing-QML") / "personal_results" / "studio_entanglement").resolve()


def _log_event(log_path: Path, message: str) -> None:
    """Append timestamped messages to a sweep log file."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load saved studio_entanglement histories, keep circuit parameters fixed, "
            "and sweep chi/gamma to assess if the operating region is robust."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing history_*.pkl or history_*.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sweep files, PNGs, and reports.",
    )
    parser.add_argument(
        "--gamma-nominal",
        type=float,
        default=15.0,
        help="Nominal gamma used in training setup.",
    )
    parser.add_argument(
        "--inverse-pulse-width",
        type=float,
        default=1.0,
        help="Inverse pulse width used in training setup.",
    )
    parser.add_argument(
        "--chi-1q-multiplier",
        type=float,
        default=0.5,
        help="chi = multiplier * gamma_nominal for 1-qubit experiments.",
    )
    parser.add_argument(
        "--chi-multiqubit-multiplier",
        type=float,
        default=2.0,
        help="chi = multiplier * gamma_nominal for n>=2 experiments.",
    )
    parser.add_argument(
        "--qq-interaction-chi",
        type=float,
        default=0.1,
        help="Chi used for qubit-qubit interactions in reconstruction.",
    )
    parser.add_argument(
        "--chi-factor-min",
        type=float,
        default=0.5,
        help="Lower multiplicative factor for chi sweep around nominal chi.",
    )
    parser.add_argument(
        "--chi-factor-max",
        type=float,
        default=2.0,
        help="Upper multiplicative factor for chi sweep around nominal chi.",
    )
    parser.add_argument(
        "--gamma-factor-min",
        type=float,
        default=0.5,
        help="Lower multiplicative factor for gamma sweep around nominal gamma.",
    )
    parser.add_argument(
        "--gamma-factor-max",
        type=float,
        default=2.0,
        help="Upper multiplicative factor for gamma sweep around nominal gamma.",
    )
    parser.add_argument(
        "--resolution-chi",
        type=int,
        default=30,
        help="Number of points on chi axis.",
    )
    parser.add_argument(
        "--resolution-gamma",
        type=int,
        default=30,
        help="Number of points on gamma axis.",
    )
    parser.add_argument(
        "--chi-scale",
        choices=["linear", "log"],
        default="linear",
        help="Scale for chi axis.",
    )
    parser.add_argument(
        "--gamma-scale",
        choices=["linear", "log"],
        default="linear",
        help="Scale for gamma axis.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used when evaluating detection in sweep.",
    )
    parser.add_argument(
        "--relative-quality-threshold",
        type=float,
        default=0.9,
        help="Quality threshold as fraction of max metric (for good-range mask).",
    )
    parser.add_argument(
        "--min-good-fraction",
        type=float,
        default=0.15,
        help="Minimum fraction of grid above threshold to label GOOD_RANGE.",
    )
    parser.add_argument(
        "--use-last-params",
        action="store_true",
        help="Use last saved parameters instead of best parameters from history.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose sweep progress output.",
    )
    return parser.parse_args()


def _nominal_chi(n_qubits: int, gamma_nominal: float, chi_1q_multiplier: float, chi_multiqubit_multiplier: float) -> float:
    if n_qubits == 1:
        return chi_1q_multiplier * gamma_nominal
    return chi_multiqubit_multiplier * gamma_nominal


def _measurement_protocol(inverse_pulse_width: float) -> MeasurementProtocol:
    times = list(BASE_MEASUREMENT_TIMES / inverse_pulse_width)
    return MeasurementProtocol(measurement_times=times)


def _interaction_list(n_qubits: int, qq_interaction_chi: float) -> List[QubitInteraction]:
    return [
        QubitInteraction(
            qubit_indices=(i, j), interaction_type=InteractionType.XX, chi=qq_interaction_chi
        )
        for i in range(n_qubits)
        for j in range(n_qubits)
        if i != j
    ]


def _create_std_experiment_setup(
    n_qubits: int,
    initial_circuit: QuantumCircuit,
    final_circuit: QuantumCircuit,
    gamma_nominal: float,
    inverse_pulse_width: float,
    chi_1q_multiplier: float,
    chi_multiqubit_multiplier: float,
    qq_interaction_chi: float,
) -> Experiment:
    chi_nominal = _nominal_chi(
        n_qubits,
        gamma_nominal=gamma_nominal,
        chi_1q_multiplier=chi_1q_multiplier,
        chi_multiqubit_multiplier=chi_multiqubit_multiplier,
    )

    physical_constants = PhysicalConstants(
        n_qubits=n_qubits,
        chi=chi_nominal,
        photon_cavity_coupling=gamma_nominal,
        inverse_pulse_width=inverse_pulse_width,
        qubit_interactions=_interaction_list(n_qubits, qq_interaction_chi),
    )

    exp_params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=SystemDimensions(),
        measurement=_measurement_protocol(inverse_pulse_width),
        initial_state=InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON),
        noise_config=NoiseConfiguration(
            depolarizing=0.0001,
            dephasing=0.0001,
            relaxation=0.0001,
        ),
    )

    return Experiment(
        experimental_params=exp_params,
        initial_circuit=initial_circuit,
        final_circuit=final_circuit,
        detection_metric=DetectionMetric(
            n_qubits=n_qubits,
            detection_criterion="max computational distance",
        ),
    )


def _build_initial_circuit(n_qubits: int, variant: str) -> QuantumCircuit:
    zeros = np.zeros(n_qubits, dtype=float)
    circuit = create_ry_circuit(n_qubits, zeros)

    if variant in {"ent", "zent"}:
        circuit.add_entangling_layer(CNOTGate, pattern="circular")
        circuit.add_layer(gate_type=RYGate, parameters=zeros)

    if variant in {"z", "zent"}:
        circuit.add_layer(gate_type=RZGate, parameters=zeros)

    return circuit


def _build_final_circuit(n_qubits: int, variant: str) -> QuantumCircuit:
    zeros = np.zeros(n_qubits, dtype=float)
    circuit = create_ry_circuit(n_qubits, zeros)

    if variant == "ent":
        circuit.add_entangling_layer(CNOTGate, pattern="circular")
        circuit.add_layer(gate_type=RYGate, parameters=zeros)
    elif variant != "no":
        raise ValueError(f"Unsupported final-circuit variant: {variant}")

    return circuit


def _parse_experiment_name(exp_name: str) -> Tuple[int, str, str]:
    if exp_name in {"ctrl", "control_1qb"}:
        return 1, "no", "no"

    match = re.fullmatch(r"(\d+)qb_([a-z]+)_([a-z]+)", exp_name)
    if not match:
        raise ValueError(
            f"Cannot parse experiment name '{exp_name}'. Expected formats like "
            "'2qb_ent_ent' or 'ctrl'."
        )

    n_qubits = int(match.group(1))
    initial_variant = match.group(2)
    final_variant = match.group(3)

    if initial_variant not in {"no", "z", "ent", "zent"}:
        raise ValueError(f"Unsupported initial-circuit variant in '{exp_name}': {initial_variant}")
    if final_variant not in {"no", "ent"}:
        raise ValueError(f"Unsupported final-circuit variant in '{exp_name}': {final_variant}")

    return n_qubits, initial_variant, final_variant


def _history_files(results_dir: Path) -> List[Path]:
    files = [
        path
        for path in results_dir.glob("history_*")
        if path.is_file() and path.suffix.lower() in {".pkl", ".npz"}
    ]
    return sorted(files)


def _exp_name_from_history_path(history_path: Path) -> str:
    stem = history_path.stem
    if not stem.startswith("history_"):
        raise ValueError(f"Not a history file name: {history_path.name}")
    exp_name = stem[len("history_") :]
    # OptimizationCallback.save uses np.savez; if path ends with .pkl, NumPy writes
    # a file named *.pkl.npz, so the stem still contains a trailing .pkl.
    if exp_name.endswith(".pkl"):
        exp_name = exp_name[:-4]
    return exp_name


def _load_fixed_params(
    history_path: Path,
    use_last_params: bool,
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    callback = OptimizationCallback.load_callback(str(history_path))

    if not use_last_params and callback.best_trainable_params is not None:
        initial_params, final_params = callback.best_trainable_params
        source = "best"
        metric = float(callback.best_metric)
    else:
        initial_params, final_params, _ = callback.get_params(epoch=-1)
        source = "last"
        metric_history = callback.history.get("metric", [])
        metric = float(metric_history[-1]) if metric_history else float("nan")

    return (
        np.asarray(initial_params, dtype=float).reshape(-1),
        np.asarray(final_params, dtype=float).reshape(-1),
        source,
        metric,
    )


def _apply_fixed_params(
    experiment: Experiment,
    initial_params: np.ndarray,
    final_params: np.ndarray,
    exp_name: str,
    history_path: Path,
) -> None:
    n_init = experiment.initial_circuit.count_trainable_parameters()
    n_final = experiment.final_circuit.count_trainable_parameters()

    if len(initial_params) != n_init:
        raise ValueError(
            f"{exp_name}: initial parameter count mismatch. "
            f"History has {len(initial_params)}, circuit expects {n_init}. "
            f"Source: {history_path}"
        )
    if len(final_params) != n_final:
        raise ValueError(
            f"{exp_name}: final parameter count mismatch. "
            f"History has {len(final_params)}, circuit expects {n_final}. "
            f"Source: {history_path}"
        )

    experiment.initial_circuit.set_trainable_parameters(initial_params.tolist())
    experiment.final_circuit.set_trainable_parameters(final_params.tolist())


def _evaluate_baseline_metric(experiment: Experiment, batch_size: int) -> float:
    callback = experiment.run_simulation(batch_size=batch_size)
    return float(callback.history["metric"][-1])


def _good_ranges(
    sweep_metric_map: np.ndarray,
    gamma_vals: np.ndarray,
    chi_vals: np.ndarray,
    threshold: float,
) -> Tuple[float, Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    good_mask = sweep_metric_map >= threshold
    good_fraction = float(np.mean(good_mask))

    if not np.any(good_mask):
        return good_fraction, None, None

    good_gamma_idx = np.where(np.any(good_mask, axis=1))[0]
    good_chi_idx = np.where(np.any(good_mask, axis=0))[0]

    gamma_range = (
        float(gamma_vals[good_gamma_idx[0]]),
        float(gamma_vals[good_gamma_idx[-1]]),
    )
    chi_range = (
        float(chi_vals[good_chi_idx[0]]),
        float(chi_vals[good_chi_idx[-1]]),
    )

    return good_fraction, gamma_range, chi_range


def _build_experiment_for_name(exp_name: str, args: argparse.Namespace) -> Tuple[Experiment, int]:
    n_qubits, initial_variant, final_variant = _parse_experiment_name(exp_name)

    initial_circuit = _build_initial_circuit(n_qubits=n_qubits, variant=initial_variant)
    final_circuit = _build_final_circuit(n_qubits=n_qubits, variant=final_variant)

    experiment = _create_std_experiment_setup(
        n_qubits=n_qubits,
        initial_circuit=initial_circuit,
        final_circuit=final_circuit,
        gamma_nominal=args.gamma_nominal,
        inverse_pulse_width=args.inverse_pulse_width,
        chi_1q_multiplier=args.chi_1q_multiplier,
        chi_multiqubit_multiplier=args.chi_multiqubit_multiplier,
        qq_interaction_chi=args.qq_interaction_chi,
    )
    return experiment, n_qubits


def _write_summary_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    if not rows:
        return

    fieldnames = [
        "experiment",
        "history_file",
        "n_qubits",
        "param_source",
        "history_metric",
        "gamma_nominal",
        "chi_nominal",
        "baseline_metric",
        "max_metric",
        "baseline_over_max",
        "quality_threshold",
        "good_fraction",
        "status",
        "optimal_gamma",
        "optimal_chi",
        "recommended_gamma_min",
        "recommended_gamma_max",
        "recommended_chi_min",
        "recommended_chi_max",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_txt(rows: List[Dict[str, object]], txt_path: Path) -> None:
    lines: List[str] = []
    lines.append("Studio entanglement constants sweep summary")
    lines.append("=" * 50)

    for row in rows:
        lines.append(
            f"- {row['experiment']}: {row['status']} | "
            f"baseline={row['baseline_metric']:.6f} | max={row['max_metric']:.6f} | "
            f"good_fraction={row['good_fraction']:.3f}"
        )
        if row["recommended_gamma_min"] is not None and row["recommended_chi_min"] is not None:
            lines.append(
                f"  recommended gamma range: [{row['recommended_gamma_min']:.6f}, "
                f"{row['recommended_gamma_max']:.6f}]"
            )
            lines.append(
                f"  recommended chi range:   [{row['recommended_chi_min']:.6f}, "
                f"{row['recommended_chi_max']:.6f}]"
            )
        else:
            lines.append("  no robust high-quality region found under current threshold")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}. "
            "Use --results-dir to point to your saved studio_entanglement outputs."
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (results_dir / "constant_sweep_analysis").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sweep_log.txt"

    _log_event(log_path, "START constants sweep analysis")
    _log_event(log_path, f"results_dir={results_dir}")
    _log_event(log_path, f"output_dir={output_dir}")
    _log_event(
        log_path,
        (
            f"sweep_config: chi_scale={args.chi_scale}, gamma_scale={args.gamma_scale}, "
            f"resolution_chi={args.resolution_chi}, resolution_gamma={args.resolution_gamma}, "
            f"batch_size={args.batch_size}, verbose={args.verbose}"
        ),
    )

    history_files = _history_files(results_dir)
    if not history_files:
        _log_event(log_path, "No history files found. Exiting with error.")
        raise FileNotFoundError(
            f"No history_*.pkl or history_*.npz files found in {results_dir}."
        )

    _log_event(log_path, f"Found {len(history_files)} history files")
    rows: List[Dict[str, object]] = []

    for history_path in history_files:
        exp_name = _exp_name_from_history_path(history_path)
        _log_event(log_path, f"Processing experiment={exp_name} history={history_path.name}")

        try:
            experiment, n_qubits = _build_experiment_for_name(exp_name=exp_name, args=args)

            initial_params, final_params, param_source, history_metric = _load_fixed_params(
                history_path=history_path,
                use_last_params=args.use_last_params,
            )
            _apply_fixed_params(
                experiment=experiment,
                initial_params=initial_params,
                final_params=final_params,
                exp_name=exp_name,
                history_path=history_path,
            )

            gamma_nominal = args.gamma_nominal
            chi_nominal = _nominal_chi(
                n_qubits=n_qubits,
                gamma_nominal=gamma_nominal,
                chi_1q_multiplier=args.chi_1q_multiplier,
                chi_multiqubit_multiplier=args.chi_multiqubit_multiplier,
            )

            chi_interval = [
                chi_nominal * args.chi_factor_min,
                chi_nominal * args.chi_factor_max,
            ]
            gamma_interval = [
                gamma_nominal * args.gamma_factor_min,
                gamma_nominal * args.gamma_factor_max,
            ]

            baseline_metric = _evaluate_baseline_metric(experiment=experiment, batch_size=args.batch_size)

            sweep = experiment.sweep_chi_gamma(
                chi_interval=chi_interval,
                gamma_interval=gamma_interval,
                resolution_chi=args.resolution_chi,
                resolution_gamma=args.resolution_gamma,
                chi_scale=args.chi_scale,
                gamma_scale=args.gamma_scale,
                batch_size=args.batch_size,
                verbose=args.verbose,
            )

            exp_out_dir = output_dir / exp_name
            exp_out_dir.mkdir(parents=True, exist_ok=True)

            sweep_npz_path = exp_out_dir / f"sweep_{exp_name}.npz"
            save_results(sweep, sweep_npz_path)

            sweep_png_path = exp_out_dir / f"sweep_{exp_name}.png"
            fig = plot_sweep_results(
                sweep,
                results_to_plot=["metric_map", "detection_map", "detection_without_map"],
                mark_optimal=True,
                save_path=str(sweep_png_path),
            )
            plt.close(fig)

            metric_map = np.asarray(sweep.results["metric_map"], dtype=float)
            max_metric = float(np.max(metric_map))
            quality_threshold = float(max_metric * args.relative_quality_threshold)

            gamma_vals = np.asarray(sweep.param1_vals, dtype=float)
            chi_vals = np.asarray(sweep.param2_vals, dtype=float)

            good_fraction, gamma_good_range, chi_good_range = _good_ranges(
                sweep_metric_map=metric_map,
                gamma_vals=gamma_vals,
                chi_vals=chi_vals,
                threshold=quality_threshold,
            )

            status = "GOOD_RANGE" if good_fraction >= args.min_good_fraction else "NARROW_RANGE"

            baseline_over_max = float("nan")
            if not np.isclose(max_metric, 0.0):
                baseline_over_max = baseline_metric / max_metric

            row = {
                "experiment": exp_name,
                "history_file": history_path.name,
                "n_qubits": n_qubits,
                "param_source": param_source,
                "history_metric": history_metric,
                "gamma_nominal": gamma_nominal,
                "chi_nominal": chi_nominal,
                "baseline_metric": baseline_metric,
                "max_metric": max_metric,
                "baseline_over_max": baseline_over_max,
                "quality_threshold": quality_threshold,
                "good_fraction": good_fraction,
                "status": status,
                "optimal_gamma": float(sweep.metadata["optimal_gamma"]),
                "optimal_chi": float(sweep.metadata["optimal_chi"]),
                "recommended_gamma_min": None if gamma_good_range is None else gamma_good_range[0],
                "recommended_gamma_max": None if gamma_good_range is None else gamma_good_range[1],
                "recommended_chi_min": None if chi_good_range is None else chi_good_range[0],
                "recommended_chi_max": None if chi_good_range is None else chi_good_range[1],
            }
            rows.append(row)

            _log_event(
                log_path,
                (
                    f"Completed experiment={exp_name} status={status} "
                    f"baseline={baseline_metric:.6f} max={max_metric:.6f} "
                    f"good_fraction={good_fraction:.3f} sweep_file={sweep_npz_path} "
                    f"sweep_png={sweep_png_path}"
                ),
            )
        except ValueError as exc:
            _log_event(log_path, f"Skipping experiment={exp_name}. reason={exc}")
            continue
        except Exception as exc:
            _log_event(log_path, f"ERROR experiment={exp_name}. reason={exc}")
            continue

    summary_csv = output_dir / "summary.csv"
    summary_txt = output_dir / "summary.txt"

    _write_summary_csv(rows=rows, csv_path=summary_csv)
    _write_summary_txt(rows=rows, txt_path=summary_txt)
    _log_event(log_path, f"Wrote summary CSV: {summary_csv}")
    _log_event(log_path, f"Wrote summary text: {summary_txt}")
    _log_event(log_path, "END constants sweep analysis")


if __name__ == "__main__":
    main()

# Import required libraries
import gc
import os
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import optax

from qsopt.core.callback import OptimizationCallback
from qsopt.core.circuit import create_ry_circuit
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
from qsopt.utils.visualization import plot_optimization_dashboard

# Set this manually to skip the input prompt.
# Example Windows: r"C:\Users\your_name\Desktop"
# Example Linux: "/home/your_name"
MANUAL_HOME_FOLDER = r"/raid/home/ncampioni/Quantum-sensing-QML"
override_default = True

# If True, runs continue from saved histories when available.
CONTINUE_SAVED_RUNS = False

# Optional deterministic run setup.
RANDOM_SEED = None


DEFAULT_SETUP_CONFIG = {
    "inverse_pulse_width": 1.0,
    "gm_factor": 15.0,
    "chi_factor": 2.0,
    "pair_chi": 0.1,
    "measurement_times_scaled": [-5.0, 0.0, 5.0],
    "noise": {
        "depolarizing": 1e-4,
        "dephasing": 1e-4,
        "relaxation": 1e-4,
    },
}

DEFAULT_TRAINING_CONFIG = {
    "tot_steps": 10000,
    "checkpoint_interval": 500,
    "tolerance": 1e-8,
    "optimizer_name": "sgd",
    "learning_rate": 0.05,
}


# Configure here per-qubit-group and per-experiment overrides.
# Each experiment can override setup variables, training options, and optional
# trainable circuit parameters (to replace random initialization).
#
# Example usage of trainable parameter overrides:
# - Without previous params (explicit values):
#   "trainable_params_overrides": {
#       "initial": [0.1] * 6,
#       "final": [-0.1] * 3,
#   }
#
# - With previous params (reuse best saved params):
#   "trainable_params_overrides": {
#       "initial": {"best_from": "2qb_ent_ent", "which": "initial"},
#       "final": {"best_from": "2qb_ent_ent", "which": "final"},
#   }
#
# - Mixing new params with previous best (append/prepend):
#   "trainable_params_overrides": {
#       "initial": [0.0, 0.0, {"best_from": "2qb_ent_ent", "which": "initial", "position": "append"}],
#   }
EXPERIMENT_GROUP_CONFIGS = [
    {
        "n_qubits": 2,
        "detection_criterion": "max computational distance",
        "detection_param": None,
        "default_setup_overrides": {"gm_factor": 15.0, "chi_factor": 2.0},
        "default_training_overrides": {},
        "experiments": [
            {"variant": "no_no"},
            {"variant": "ent_no",
                "trainable_params_overrides": {
                    "initial": [0,0, {"best_from": "2qb_no_no", "which": "initial", "position": "append"}],
                    "final": {"best_from": "2qb_no_no", "which": "final"},
                }
            },
            {"variant": "ent_ent",
                "training_overrides": {"tot_steps": 10000},
                "trainable_params_overrides": {
                    "initial": [0,0, {"best_from": "2qb_no_no", "which": "initial", "position": "append"}],
                    "final": [0,0, {"best_from": "2qb_no_no", "which": "final", "position": "append"}],
                }
            },
            {"variant": "z_no",
                "training_overrides": {"tot_steps": 10000},
                "trainable_params_overrides": {
                    "initial": [{"best_from": "2qb_no_no", "which": "initial", "position": "prepend"},0,0],
                    "final": {"best_from": "2qb_no_no", "which": "final"},
                }
            },
            {"variant": "zent_ent",
                "training_overrides": {"tot_steps": 10000},
                "trainable_params_overrides": {
                    "initial": [0,0, {"best_from": "2qb_z_no", "which": "initial", "position": "append"}],
                    "final": [0,0, {"best_from": "2qb_z_no", "which": "final", "position": "append"}],
                }
            },
                # Example: uncomment to tune one experiment only.
                # "training_overrides": {"learning_rate": 0.02, "tot_steps": 14000},
                # "training_overrides": {"optimizer": optax.adam(0.02)},
                # "training_overrides": {"optimizer_name": "adam", "learning_rate": 0.02},
                # "setup_overrides": {"noise": {"dephasing": 5e-4}},
                # "trainable_params_overrides": {
                #     "initial": [0.1] * 9,
                #     "final": [-0.1] * 6,
                # },
        ],
    },
    {
        "n_qubits": 3,
        "detection_criterion": "max computational distance",
        "detection_param": None,
        "default_setup_overrides": {},
        "default_training_overrides": {},
        "experiments": [
            {"variant": "no_no",
                "training_overrides": {"tot_steps": 10000},
                "trainable_params_overrides": {
                    "initial": [0, {"best_from": "2qb_no_no", "which": "initial", "position": "append"}],
                    "final": [0, {"best_from": "2qb_no_no", "which": "final", "position": "append"}],
                }
            },
            {"variant": "ent_no",
                "training_overrides": {"tot_steps": 10000},
                "trainable_params_overrides": {
                    "initial": [0,0,0, {"best_from": "3qb_no_no", "which": "initial", "position": "append"}],
                    "final": {"best_from": "3qb_no_no", "which": "final"},
                }
            },
            {"variant": "ent_ent",
                "training_overrides": {"tot_steps": 10000},
                "trainable_params_overrides": {
                    "initial": [0,0,0, {"best_from": "3qb_no_no", "which": "initial", "position": "append"}],
                    "final": [0,0,0, {"best_from": "3qb_no_no", "which": "final", "position": "append"}],
                }
            },
            {"variant": "z_no",
                "trainable_params_overrides": {
                    "initial": [{"best_from": "3qb_no_no", "which": "initial", "position": "prepend"}, 0,0,0],
                    "final": {"best_from": "3qb_no_no", "which": "final"},
                }
            },
            {"variant": "zent_ent",
                "training_overrides": {"tot_steps": 10000},
                "trainable_params_overrides": {
                    "initial": [0,0,0, {"best_from": "3qb_z_no", "which": "initial", "position": "append"}],
                    "final": [0,0,0, {"best_from": "3qb_z_no", "which": "final", "position": "append"}],
                }
            }
        ],
    },
]


def resolve_home_folder():
    default_home = os.path.expanduser("~")

    if MANUAL_HOME_FOLDER:
        if override_default:
            default_home = os.path.normpath(os.path.expanduser(MANUAL_HOME_FOLDER))
        else:
            return os.path.normpath(os.path.expanduser(MANUAL_HOME_FOLDER))

    try:
        user_input = input(f"Home folder [default is: {default_home}]: ").strip()
    except EOFError:
        user_input = ""

    print(f"Using home folder: {user_input if user_input else default_home}")
    selected_home = user_input if user_input else default_home
    return os.path.normpath(os.path.expanduser(selected_home))


home_path = resolve_home_folder()
save_folder = os.path.join(home_path, "personal_results", "studio_entanglement")
error_folder = os.path.join(save_folder, "errors")
log_file = os.path.join(save_folder, "log.txt")

os.makedirs(save_folder, exist_ok=True)
os.makedirs(error_folder, exist_ok=True)


def log_event(event, experiment_name, details=""):
    timestamp = datetime.now().isoformat(timespec="seconds")
    details_text = f" | {details}" if details else ""
    with open(log_file, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"[{timestamp}] {event} | {experiment_name}{details_text}\n")


log_event(
    "START PROGRAM",
    "",
    f"Initializing experiments and training runs | continue_saved_runs={CONTINUE_SAVED_RUNS}",
)


def merge_nested_dict(base, overrides):
    merged = deepcopy(base)
    if not overrides:
        return merged

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_optimizer(training_config):
    direct_optimizer = training_config.get("optimizer")
    if direct_optimizer is not None:
        if hasattr(direct_optimizer, "init") and hasattr(direct_optimizer, "update"):
            return direct_optimizer
        raise TypeError(
            "training_config['optimizer'] must expose both 'init' and 'update' methods"
        )

    optimizer_factory = training_config.get("optimizer_factory")
    if optimizer_factory is not None:
        if not callable(optimizer_factory):
            raise TypeError("training_config['optimizer_factory'] must be callable")
        built_optimizer = optimizer_factory(training_config)
        if hasattr(built_optimizer, "init") and hasattr(built_optimizer, "update"):
            return built_optimizer
        raise TypeError(
            "training_config['optimizer_factory'] must return an optimizer exposing 'init' and 'update'"
        )

    optimizer_name = str(training_config.get("optimizer_name", "sgd")).lower()
    learning_rate = float(training_config.get("learning_rate", 0.05))

    if optimizer_name == "sgd":
        return optax.sgd(learning_rate=learning_rate)
    if optimizer_name == "adam":
        return optax.adam(learning_rate=learning_rate)

    raise ValueError(f"Unsupported optimizer_name={optimizer_name}")


def get_last_gradient_info(callback):
    grads_history = callback.history.get("grads", [])
    if not grads_history:
        return "grad_norm=None"

    last_grads = grads_history[-1]
    if last_grads is None:
        return "grad_norm=None"

    try:
        grad_array = np.asarray(last_grads, dtype=float).reshape(-1)
        grad_norm = float(np.linalg.norm(grad_array))
        return f"grad_norm={grad_norm:.6e}"
    except Exception:
        return "grad_norm=unavailable"


def build_measurement_protocol(setup_config):
    inverse_pulse_width = float(setup_config["inverse_pulse_width"])
    measurement_times = list(
        np.asarray(setup_config["measurement_times_scaled"], dtype=float) / inverse_pulse_width
    )
    return MeasurementProtocol(measurement_times=measurement_times)


def create_std_experiment_setup(
    n_qubits,
    initial_circuit,
    final_circuit,
    detection_metric,
    setup_config,
):
    inverse_pulse_width = float(setup_config["inverse_pulse_width"])
    gm = float(setup_config["gm_factor"]) * inverse_pulse_width
    chi = float(setup_config["chi_factor"]) * gm

    interactions = [
        QubitInteraction(
            qubit_indices=(i, j),
            interaction_type=InteractionType.XX,
            chi=float(setup_config["pair_chi"]),
        )
        for i in range(n_qubits)
        for j in range(n_qubits)
        if i != j
    ]

    physical_constants = PhysicalConstants(
        n_qubits=n_qubits,
        chi=chi,
        photon_cavity_coupling=gm,
        inverse_pulse_width=inverse_pulse_width,
        qubit_interactions=interactions,
    )

    noise_values = setup_config.get("noise", {})
    noise_config = NoiseConfiguration(
        depolarizing=float(noise_values.get("depolarizing", 1e-4)),
        dephasing=float(noise_values.get("dephasing", 1e-4)),
        relaxation=float(noise_values.get("relaxation", 1e-4)),
    )

    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)

    exp_params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=SystemDimensions(),
        measurement=build_measurement_protocol(setup_config),
        initial_state=initial_state,
        noise_config=noise_config,
    )

    return Experiment(
        experimental_params=exp_params,
        initial_circuit=initial_circuit,
        final_circuit=final_circuit,
        detection_metric=detection_metric,
    )


def build_circuit_variant(n_qubits, variant):
    if variant == "no_no":
        input_circuit = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi / 2)
        final_circuit = create_ry_circuit(n_qubits, -np.random.rand(n_qubits) * np.pi / 2)
        return input_circuit, final_circuit

    if variant == "ent_no":
        input_circuit = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi / 4)
        input_circuit.add_entangling_layer(CNOTGate, pattern="circular")
        input_circuit.add_layer(gate_type=RYGate, parameters=np.random.rand(n_qubits) * np.pi / 4)
        final_circuit = create_ry_circuit(n_qubits, -np.random.rand(n_qubits) * np.pi / 2)
        return input_circuit, final_circuit

    if variant == "ent_ent":
        input_circuit = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi / 4)
        input_circuit.add_entangling_layer(CNOTGate, pattern="circular")
        input_circuit.add_layer(gate_type=RYGate, parameters=np.random.rand(n_qubits) * np.pi / 4)

        final_circuit = create_ry_circuit(n_qubits, -np.random.rand(n_qubits) * np.pi / 4)
        final_circuit.add_entangling_layer(CNOTGate, pattern="circular")
        final_circuit.add_layer(gate_type=RYGate, parameters=-np.random.rand(n_qubits) * np.pi / 4)
        return input_circuit, final_circuit

    if variant == "z_no":
        input_circuit = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi / 2)
        input_circuit.add_layer(gate_type=RZGate, parameters=np.random.rand(n_qubits) * np.pi / 3)
        final_circuit = create_ry_circuit(n_qubits, -np.random.rand(n_qubits) * np.pi / 2)
        return input_circuit, final_circuit

    if variant == "zent_ent":
        input_circuit = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi / 4)
        input_circuit.add_entangling_layer(CNOTGate, pattern="circular")
        input_circuit.add_layer(gate_type=RYGate, parameters=np.random.rand(n_qubits) * np.pi / 4)
        input_circuit.add_layer(gate_type=RZGate, parameters=np.random.rand(n_qubits) * np.pi / 3)

        final_circuit = create_ry_circuit(n_qubits, -np.random.rand(n_qubits) * np.pi / 4)
        final_circuit.add_entangling_layer(CNOTGate, pattern="circular")
        final_circuit.add_layer(gate_type=RYGate, parameters=-np.random.rand(n_qubits) * np.pi / 4)
        return input_circuit, final_circuit

    raise ValueError(f"Unknown experiment variant={variant}")


def _normalize_trainable_params(exp_name, param_group, values, expected_count):
    parsed = np.asarray(values, dtype=float).reshape(-1)
    if len(parsed) != expected_count:
        raise ValueError(
            f"{exp_name}: {param_group} trainable parameter count mismatch. "
            f"Provided {len(parsed)}, expected {expected_count}."
        )
    return parsed


def apply_trainable_params_overrides(
    experiment,
    exp_name,
    params_overrides,
    completed_best_params=None,
):
    if not params_overrides:
        return

    if not isinstance(params_overrides, dict):
        raise TypeError(f"{exp_name}: trainable_params_overrides must be a dict")

    if completed_best_params is None:
        completed_best_params = {}

    initial_override = params_overrides.get("initial", None)
    final_override = params_overrides.get("final", None)

    n_initial = experiment.initial_circuit.count_trainable_parameters()
    n_final = experiment.final_circuit.count_trainable_parameters()

    if initial_override is not None:
        resolved_initial = _resolve_and_merge_params(
            exp_name=exp_name,
            param_group="initial",
            values=initial_override,
            expected_count=n_initial,
            completed_best_params=completed_best_params,
        )
        experiment.initial_circuit.set_trainable_parameters(resolved_initial)

    if final_override is not None:
        resolved_final = _resolve_and_merge_params(
            exp_name=exp_name,
            param_group="final",
            values=final_override,
            expected_count=n_final,
            completed_best_params=completed_best_params,
        )
        experiment.final_circuit.set_trainable_parameters(resolved_final)


def _resolve_best_placeholder(item, which_group, completed_best_params):
    """
    Resolve a placeholder dict of form {"best_from": "exp_name", "which": "initial"|"final"}
    using best parameters of experiments already completed in the current program run
    or from saved histories on disk.

    Returns a list of floats for the requested parameter group, or raises if unavailable.
    """
    if not isinstance(item, dict):
        raise TypeError("best placeholder must be a dict with key 'best_from'")
    source = item.get("best_from")
    if source is None:
        raise ValueError("best placeholder missing 'best_from' key")

    if source in completed_best_params:
        best = completed_best_params[source]
    else:
        history, _ = try_load_saved_history(save_folder, source)
        if history is None:
            raise ValueError(
                f"best_from={source} is not available yet. "
                f"Complete it earlier in this run or ensure a saved history exists in {save_folder}."
            )
        best = history.get_best_trainable_params()
        if best is None:
            raise ValueError(f"No best parameters in history for {source}")

    best_initial, best_final = best
    if which_group == "initial":
        return list(best_initial)
    if which_group == "final":
        return list(best_final)
    raise ValueError("which_group must be 'initial' or 'final'")


def _resolve_and_merge_params(exp_name, param_group, values, expected_count, completed_best_params):
    """
    Resolve a values spec which may include numeric entries and dict placeholders
    referring to best parameters from completed runs or saved histories. Supported
    placeholder dicts:
      {"best_from": "other_exp", "which": "initial"|"final", "position": "append"|"prepend"}

    The `values` argument may be:
      - a list/iterable of numbers -> returned as-is (after length check)
      - a dict placeholder -> resolved to the referenced best list
      - a list mixing numbers and a single placeholder dict -> merges according to position
    """
    # If a plain numeric list, just validate and return
    try:
        arr = np.asarray(values, dtype=float).reshape(-1).tolist()
        if len(arr) == expected_count:
            return arr
    except Exception:
        pass

    # Handle dict placeholder directly
    if isinstance(values, dict) and "best_from" in values:
        resolved = _resolve_best_placeholder(values, param_group, completed_best_params)
        if len(resolved) != expected_count:
            raise ValueError(
                f"{exp_name}: {param_group} resolved best parameter count mismatch. "
                f"Resolved {len(resolved)}, expected {expected_count}."
            )
        return resolved

    # Handle list that may mix numbers and a placeholder dict
    if isinstance(values, (list, tuple)):
        placeholders = [v for v in values if isinstance(v, dict) and "best_from" in v]
        if len(placeholders) > 1:
            raise ValueError(f"{exp_name}: only a single best_from placeholder is supported in a list")

        numeric_parts = [v for v in values if not (isinstance(v, dict) and "best_from" in v)]
        if not placeholders:
            raise ValueError(
                f"{exp_name}: {param_group} trainable parameter count mismatch. "
                f"Provided {len(values)}, expected {expected_count}."
            )

        placeholder = placeholders[0]
        which = placeholder.get("which", param_group)
        position = placeholder.get("position", "append")
        resolved_best = _resolve_best_placeholder(placeholder, which, completed_best_params)

        if position == "append":
            merged = list(numeric_parts) + list(resolved_best)
        elif position == "prepend":
            merged = list(resolved_best) + list(numeric_parts)
        else:
            raise ValueError("Unsupported position value for best placeholder; use 'append' or 'prepend'")

        if len(merged) != expected_count:
            raise ValueError(
                f"{exp_name}: {param_group} merged parameter count mismatch. "
                f"Merged {len(merged)}, expected {expected_count}."
            )
        return merged

    raise TypeError(f"{exp_name}: unsupported type for {param_group} trainable parameter override")


def validate_experiment_parameter_counts(experiment_bundle):
    for exp_name, exp_info in experiment_bundle.items():
        experiment = exp_info["experiment"]

        expected_initial = experiment.initial_circuit.count_trainable_parameters()
        expected_final = experiment.final_circuit.count_trainable_parameters()

        current_initial = len(experiment.initial_circuit.get_trainable_parameters())
        current_final = len(experiment.final_circuit.get_trainable_parameters())

        if current_initial != expected_initial:
            raise ValueError(
                f"{exp_name}: initial trainable parameter count mismatch. "
                f"Current {current_initial}, expected {expected_initial}."
            )
        if current_final != expected_final:
            raise ValueError(
                f"{exp_name}: final trainable parameter count mismatch. "
                f"Current {current_final}, expected {expected_final}."
            )


def _placeholder_best_params_for_experiment(experiment):
    n_initial = experiment.initial_circuit.count_trainable_parameters()
    n_final = experiment.final_circuit.count_trainable_parameters()
    return ([0.0] * n_initial, [0.0] * n_final)


def preflight_validate_param_overrides(experiment_bundles):
    """
    Validate that all parameter overrides resolve to the correct trainable
    parameter counts before any experiment is launched.
    """
    available_best_params = {}

    for exp_bundle in experiment_bundles:
        for exp_name, exp_info in exp_bundle.items():
            experiment = exp_info["experiment"]
            params_overrides = exp_info.get("params_overrides", {})

            if params_overrides:
                n_initial = experiment.initial_circuit.count_trainable_parameters()
                n_final = experiment.final_circuit.count_trainable_parameters()

                initial_override = params_overrides.get("initial", None)
                final_override = params_overrides.get("final", None)

                try:
                    if initial_override is not None:
                        _resolve_and_merge_params(
                            exp_name=exp_name,
                            param_group="initial",
                            values=initial_override,
                            expected_count=n_initial,
                            completed_best_params=available_best_params,
                        )
                    if final_override is not None:
                        _resolve_and_merge_params(
                            exp_name=exp_name,
                            param_group="final",
                            values=final_override,
                            expected_count=n_final,
                            completed_best_params=available_best_params,
                        )
                except Exception as error:
                    raise ValueError(
                        f"Preflight validation failed for {exp_name}: {error}"
                    ) from error

            available_best_params[exp_name] = _placeholder_best_params_for_experiment(experiment)


def build_experiment_bundle(group_config):
    n_qubits = int(group_config["n_qubits"])
    detection_metric = DetectionMetric(
        n_qubits=n_qubits,
        detection_criterion=group_config.get("detection_criterion", "max computational distance"),
        detection_param=group_config.get("detection_param", None),
    )

    base_setup = merge_nested_dict(DEFAULT_SETUP_CONFIG, group_config.get("default_setup_overrides", {}))
    base_training = merge_nested_dict(
        DEFAULT_TRAINING_CONFIG,
        group_config.get("default_training_overrides", {}),
    )

    experiment_bundle = {}
    for experiment_cfg in group_config.get("experiments", []):
        variant = experiment_cfg["variant"]
        exp_name = experiment_cfg.get("name", f"{n_qubits}qb_{variant}")

        setup_config = merge_nested_dict(base_setup, experiment_cfg.get("setup_overrides", {}))
        training_config = merge_nested_dict(base_training, experiment_cfg.get("training_overrides", {}))
        params_overrides = experiment_cfg.get("trainable_params_overrides", {})

        initial_circuit, final_circuit = build_circuit_variant(n_qubits, variant)
        experiment = create_std_experiment_setup(
            n_qubits=n_qubits,
            initial_circuit=initial_circuit,
            final_circuit=final_circuit,
            detection_metric=detection_metric,
            setup_config=setup_config,
        )
        experiment_bundle[exp_name] = {
            "experiment": experiment,
            "setup": setup_config,
            "training": training_config,
            "params_overrides": params_overrides,
        }

    return experiment_bundle


def get_history_path(save_dir, exp_name):
    return os.path.join(save_dir, f"history_{exp_name}.npz")


def get_callback_text_path(save_dir, exp_name):
    return os.path.join(save_dir, f"results_{exp_name}.txt")


def save_callback_text(save_dir, exp_name, history):
    path = get_callback_text_path(save_dir, exp_name)
    if history is None:
        text = f"{exp_name}: no callback available (history is None)."
    else:
        text = str(history)
        if "°" in text:
            text = text.replace("°", " deg")

    if not text.endswith("\n"):
        text += "\n"

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def try_load_saved_history(save_dir, exp_name):
    candidates = [
        get_history_path(save_dir, exp_name),
        os.path.join(save_dir, f"history_{exp_name}.pkl"),
        os.path.join(save_dir, f"history_{exp_name}.pkl.npz"),
    ]

    for candidate_path in candidates:
        if not os.path.exists(candidate_path):
            continue
        try:
            loaded_history = OptimizationCallback.load_callback(candidate_path)
            return loaded_history, candidate_path
        except Exception as error:
            log_event("RESUME_LOAD_ERROR", exp_name, f"path={candidate_path} error={error}")

    return None, None


def run_experiment_with_checkpoints(
    experiment,
    tot_steps,
    checkpoint_interval,
    tolerance,
    optimizer,
    save_dir,
    exp_name,
    continue_saved_runs=False,
):
    tot_steps = int(tot_steps)
    checkpoint_interval = int(checkpoint_interval)
    tolerance = float(tolerance)

    if tot_steps < 0:
        raise ValueError("tot_steps must be >= 0")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be > 0")

    history = None
    start_epoch = 0
    history_path = get_history_path(save_dir, exp_name)

    if continue_saved_runs:
        history, loaded_path = try_load_saved_history(save_dir, exp_name)
        if history is not None:
            start_epoch = int(history.epoch)
            log_event(
                "RESUME_LOADED",
                exp_name,
                f"path={loaded_path} saved_epoch={start_epoch} converged={history.converged}",
            )

    missing_steps = max(tot_steps - start_epoch, 0)

    log_event(
        "TRAINING_START",
        exp_name,
        (
            f"tot_steps={tot_steps} start_epoch={start_epoch} missing_steps={missing_steps} "
            f"checkpoint_interval={checkpoint_interval} tolerance={tolerance} "
            f"resume={continue_saved_runs}"
        ),
    )

    if history is not None and history.converged:
        history.converged = False

    while missing_steps > 0:
        steps_to_run = min(checkpoint_interval, missing_steps)
        missing_steps -= steps_to_run

        history = experiment.optimize_rotations(
            num_steps=steps_to_run,
            tolerance=tolerance,
            callback=history,
            optimizer=optimizer,
            verbose=False,
            hot_start=history is not None,
            noisy_training=True
        )

        history.save(history_path)

        if history.converged:
            break

        log_event(
            "CHECKPOINT",
            exp_name,
            f"epoch={history.epoch} best_metric={history.best_metric} {get_last_gradient_info(history)}",
        )

    if history is None:
        log_event("TRAINING_END", exp_name, "status=skipped reason=tot_steps<=0")
        log_event("", "", "-" * 80)
        return None

    log_event(
        "TRAINING_END",
        exp_name,
        (
            f"converged={history.converged} epoch={history.epoch} "
            f"best_metric={history.best_metric} {get_last_gradient_info(history)}"
        ),
    )
    log_event("", "", "-" * 80)
    return history


def run_experiment_ensemble(
    experiment_bundle,
    save_dir,
    continue_saved_runs=False,
    completed_best_params=None,
):
    if completed_best_params is None:
        completed_best_params = {}

    for exp_name in list(experiment_bundle.keys()):
        
        history = None
        callback_written = False

        try:
            experiment_info = experiment_bundle[exp_name]
            experiment = experiment_info["experiment"]
            training_config = experiment_info["training"]
            params_overrides = experiment_info.get("params_overrides", {})

            apply_trainable_params_overrides(
                experiment=experiment,
                exp_name=exp_name,
                params_overrides=params_overrides,
                completed_best_params=completed_best_params,
            )

            optimizer = build_optimizer(training_config)

            history = run_experiment_with_checkpoints(
                experiment=experiment,
                tot_steps=training_config["tot_steps"],
                checkpoint_interval=training_config["checkpoint_interval"],
                tolerance=training_config["tolerance"],
                optimizer=optimizer,
                save_dir=save_dir,
                exp_name=exp_name,
                continue_saved_runs=continue_saved_runs,
            )

            if history is not None:
                best = history.get_best_trainable_params()
                if best is not None:
                    best_initial, best_final = best
                    completed_best_params[exp_name] = (
                        list(np.asarray(best_initial, dtype=float).reshape(-1)),
                        list(np.asarray(best_final, dtype=float).reshape(-1)),
                    )

                fig = plot_optimization_dashboard(
                    optimization_callback=history,
                    show_metric=True,
                    show_gradients=True,
                    show_parameters=True,
                    show_detection_measures=True,
                    show_trajectory=False,
                    save_path=os.path.join(save_dir, f"dashboard_{exp_name}.pdf"),
                )
                plt.close(fig)

                save_callback_text(save_dir, exp_name, history)
                callback_written = True

        except Exception as error:
            log_event("TRAINING_ERROR", exp_name, f"error={error}")
            log_event("", "", "-" * 80)
            with open(os.path.join(error_folder, "error_log.txt"), "a", encoding="utf-8") as handle:
                handle.write(f"Error in experiment {exp_name}: {str(error)}\n")

        finally:
            if not callback_written:
                save_callback_text(save_dir, exp_name, history)
            if history is not None:
                history.reset()
            if exp_name in experiment_bundle:
                del experiment_bundle[exp_name]
            if history is not None:
                del history
            gc.collect()

    return completed_best_params


if RANDOM_SEED is not None:
    np.random.seed(int(RANDOM_SEED))


all_experiment_bundles = []
for group_cfg in EXPERIMENT_GROUP_CONFIGS:
    exp_bundle = build_experiment_bundle(group_cfg)
    validate_experiment_parameter_counts(exp_bundle)
    all_experiment_bundles.append(exp_bundle)


preflight_validate_param_overrides(all_experiment_bundles)

completed_best_params = {}
for exp_bundle in all_experiment_bundles:
    completed_best_params = run_experiment_ensemble(
        experiment_bundle=exp_bundle,
        save_dir=save_folder,
        continue_saved_runs=CONTINUE_SAVED_RUNS,
        completed_best_params=completed_best_params,
    )


log_event("END PROGRAM", "All experiments completed", "*" * 70)
log_event("", "", "-" * 80)

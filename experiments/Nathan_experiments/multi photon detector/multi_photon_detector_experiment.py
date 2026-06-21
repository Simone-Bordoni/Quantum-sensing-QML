from qsopt import * 
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import qutip as qt
import qutip_jax
from jax.scipy.special import erfc
from typing import List, Optional
from qsopt.core.circuit import create_ry_circuit
from qsopt.core.gates import RZGate, CNOTGate
from qsopt.utils.visualization import plot_optimization_dashboard
import optax
import os
import sys
import gc
import traceback
from datetime import datetime



def pulse(t, **kwargs):
    """
    Time-dependent coupling function for input cavity transparency.

    Args:
        t: float or JAX array, time variable
        **kwargs: Dictionary containing 'sigma' parameter (pulse bandwidth)

    Returns:
        JAX array: Normalized coupling strength g(t)
    """
    sigma = kwargs.get("sigma", 0.1)
    dx = sigma * t
    coupling = jnp.sqrt(2 * sigma / jnp.sqrt(jnp.pi) * jnp.exp(-(dx**2)) / erfc(dx))
    return jnp.array(coupling, float)


def run_experiment(
        n_qubits: int,
        max_photon: int,
        max_separated_photon: int,
        chi_list: List[float],
        k: float,
        init_circuit: QuantumCircuit,
        final_circuit: QuantumCircuit,
        initial_values: Optional[List[float]] = None,
        gamma: float=1.0,
        sigma: float=1.0,
        num_steps: int=1000):
    """
    Run the multi-photon detector experiment with specified parameters.

    Args:
        n_qubits: int, number of qubits in the system
        max_photon: int, maximum photon number (the cavity/field Hilbert space uses
            max_photon + 1 levels so that the max_photon-photon state is representable)
        max_separated_photon: int, highest photon number resolved individually; if
            max_photon exceeds it, the remaining higher photon numbers
            (max_separated_photon .. max_photon) are grouped into one mixed config
        chi_list: List[float], list of dispersive coupling strengths for each qubit
        k: float, scaling factor for dispersive couplings
        gamma: float, decay rate parameter
        init_circuit: QuantumCircuit, initial circuit with trainable parameters
        final_circuit: QuantumCircuit, final circuit with trainable parameters
        initial_values: List[float], optional initial values for the trainable
            parameters in the circuits. If None, the circuits' current values are used.
        sigma: float, pulse bandwidth parameter (default: 1.0)
        num_steps: int, maximum number of optimization steps (default: 1000)
    Returns:
        callback: results of the experiment simulation
    """
    # Define interactions
    qubit_cavity_interactions = [
        Interaction(interaction_type = InteractionType.DISPERSIVE,
                    subsystem1 = ('cavity',0),
                    subsystem2 = ('qubit',i),
                    parameters = {'chi':chi_list[i]}
                    )
        for i in range(n_qubits)
        ]
    qubit_qubit_interactions = [
        Interaction(interaction_type = InteractionType.ZZ,
                subsystem1 = ('qubit',i),
                subsystem2 = ('qubit',j),
                parameters = {'chi':0.001}
                )
        for i in range(n_qubits-1)
        for j in range(i+1, n_qubits)
    ]

    interactions = qubit_cavity_interactions + qubit_qubit_interactions
    
    # Define custom physical model
    physical_model = PhysicalModel(
        n_cavities = 1,
        n_fields = 1,
        n_qubits = n_qubits,
        cavity_levels = max_photon+1,
        field_levels = max_photon+1,
        qubit_levels = 2,
        interactions = interactions
    )
    
    # Define noise model
    noise = NoiseModel(
        depolarizing=0.001,
        dephasing=0.001,
        relaxation=0.001
    )

    # Define measurement protocol
    custom_measurement = MeasurementProtocol(
        measurement_times=list(np.array([-8.0, 4.0])/sigma),
        initial_time_uncertainty=0/sigma
    )

    # Generate the system configurations for this experiment. The time-dependent
    # input-output interaction is attached per-configuration (see gen_config_set),
    # so the 0-photon configuration is built without it.
    config_set = gen_config_set(
        max_photon=max_photon,
        max_separated_photon=max_separated_photon,
        gamma=gamma,
        k=k,
        sigma=sigma,
    )

    # Create parameters with custom configuration
    exp_parameters = ExperimentalParameters(
        physical_model=physical_model,
        noise_model=noise,
        measurement=custom_measurement,
        configuration_set=config_set
    )

    detection_metric = DetectionMetric(n_cavities=1,
                                    n_fields=1,
                                    n_qubits=n_qubits,
                                    config_names=exp_parameters.get_all_configuration_names(),
                                    detection_criterion = 'max computational distance'
                                    )

    experiment = Experiment(
        experimental_params=exp_parameters,
        initial_circuit=init_circuit,
        final_circuit=final_circuit,
        detection_metric=detection_metric
    )

    history = OptimizationCallback(save_every=1, save_best=True)
    
    history = experiment.optimize_rotations(
        initial_values=initial_values,
        num_steps=num_steps,
        verbose=True,
        verbose_step=250,
        batch_size=1,
        tolerance=1e-9,
        callback=history,
        optimizer=optax.sgd(learning_rate=0.1)
    )

    return history
    


def gen_config_set(max_photon: int,
                   max_separated_photon: int,
                   gamma: float,
                   k: float,
                   sigma: float):
    """
    Generate the set of system configurations for the multi-photon detector experiment.

    The 0-photon configuration is built WITHOUT the time-dependent input-output
    interaction: with no incoming photon there is nothing to couple into the cavity,
    so the interaction is unnecessary and only slows the simulation down. Every
    configuration with one or more photons carries its own input-output interaction.

    Args:
        max_photon: Maximum photon number; the cavity/field Hilbert space uses
            ``max_photon + 1`` levels so that the ``max_photon``-photon state is
            representable.
        max_separated_photon: Highest photon number resolved individually. If
            ``max_photon`` exceeds it, photon numbers from ``max_separated_photon``
            up to and including ``max_photon`` are grouped into a single mixed
            configuration named ``'{max_separated_photon}+ photons'``.
        gamma: Decay rate parameter of the input-output interaction.
        k: Coupling (kappa) parameter of the input-output interaction.
        sigma: Pulse bandwidth parameter of the input-output interaction.

    Returns:
        List[SystemConfiguration] for the experiment.
    """
    if max_photon < max_separated_photon:
        raise ValueError("max_photon must be greater than or equal to max_separated_photon")

    def input_output_interaction():
        # A fresh Interaction per configuration (each carries its own state).
        return Interaction(
            interaction_type=InteractionType.INPUT_OUTPUT,
            subsystem1=('cavity', 0),
            subsystem2=('field', 0),
            parameters={'gamma': gamma, 'kappa': k, 'sigma': sigma},
            time_modulation=pulse,
        )

    with qt.CoreOptions(default_dtype="jax"):

        # 0-photon baseline configuration: no input-output interaction.
        config_set = [
            SystemConfiguration(
                name='0-photons',
                init_field_states={0: SubsystemState(State.FOCK, {'n': 0})},
            )
        ]

        # Individually resolved photon-number configurations (1 .. max_separated_photon),
        # each with its own input-output interaction.
        config_set += [
            SystemConfiguration(
                name=f'{i}-photons',
                init_field_states={0: SubsystemState(State.FOCK, {'n': i})},
                interactions=[input_output_interaction()],
            )
            for i in range(1, max_separated_photon+1)
        ]

        # Remaining higher photon numbers grouped into a single mixed configuration:
        # an equal mixture of the field Fock states with photon numbers from
        # max_separated_photon up to and including max_photon, with the cavity in
        # vacuum (consistent with the per-photon configurations above).
        if max_photon > max_separated_photon:
            dim = max_photon + 1
            field_mixture = sum(
                qt.fock_dm(dim, n) for n in range(max_separated_photon, max_photon + 1)
            )
            field_mixture = field_mixture / field_mixture.tr()
            density_matrix = qt.tensor(qt.fock_dm(dim, 0), field_mixture)
            config_set[-1] = SystemConfiguration(
                                name=f'{max_separated_photon}+ photons',
                                density_matrix=density_matrix,
                                interactions=[input_output_interaction()],
                            )

    return config_set

# Prepare cirucits
init_circuit_2qb = create_ry_circuit(n_qubits=2, theta_values=np.pi/2)
init_circuit_2qb.add_layer(RZGate, parameters=0.0)
final_circuit_2qb = create_ry_circuit(n_qubits=2, theta_values=-np.pi/2)

init_circuit_2qb_ent = create_ry_circuit(n_qubits=2, theta_values=np.pi/4)
init_circuit_2qb_ent.add_entangling_layer(CNOTGate, pattern='circular')
init_circuit_2qb_ent.add_layer(RYGate, parameters=np.pi/4)
init_circuit_2qb_ent.add_layer(RZGate, parameters=0.0)

final_circuit_2qb_ent = create_ry_circuit(n_qubits=2, theta_values=-np.pi/4)
final_circuit_2qb_ent.add_entangling_layer(CNOTGate, pattern='circular')
final_circuit_2qb_ent.add_layer(RYGate, parameters=-np.pi/4)

init_circuit_3qb = create_ry_circuit(n_qubits=3, theta_values=np.pi/2)
init_circuit_3qb.add_layer(RZGate, parameters=0.0)
final_circuit_3qb = create_ry_circuit(n_qubits=3, theta_values=-np.pi/2)

init_circuit_3qb_ent = create_ry_circuit(n_qubits=3, theta_values=np.pi/4)
init_circuit_3qb_ent.add_entangling_layer(CNOTGate, pattern='circular')
init_circuit_3qb_ent.add_layer(RYGate, parameters=np.pi/4)
init_circuit_3qb_ent.add_layer(RZGate, parameters=0.0)

final_circuit_3qb_ent = create_ry_circuit(n_qubits=3, theta_values=-np.pi/4)
final_circuit_3qb_ent.add_entangling_layer(CNOTGate, pattern='circular')
final_circuit_3qb_ent.add_layer(RYGate, parameters=-np.pi/4)


# Prepare experiments parameters
experiment_list = {
    # Fast smoke test (few optimization steps) to verify the pipeline runs
    # without errors before launching the full experiments.
    'test_fast': {
        'n_qubits': 2,
        'max_photon': 2,
        'chi_list': [1,1],
        'k': 15.0,
        'init_circuit': init_circuit_2qb,
        'final_circuit': final_circuit_2qb,
        'max_separated_photon': 2,
        'num_steps': 3
    },
    '2qb_2photons_chi_const_no_entanglement': {
        'n_qubits': 2,
        'max_photon': 2,
        'chi_list': [7.5, 7.5],
        'k': 15.0,
        'init_circuit': init_circuit_2qb,
        'final_circuit': final_circuit_2qb,
        'max_separated_photon': 2
    },
    '2qb_2photons_chi_const_entanglement': {
        'n_qubits': 2,
        'max_photon': 2,
        'chi_list': [7.5, 7.5],
        'k': 15.0,
        'init_circuit': init_circuit_2qb_ent,
        'final_circuit': final_circuit_2qb_ent,
        'max_separated_photon': 2
    },
    '2qb_2photons_chi_varied_no_entanglement': {
        'n_qubits': 2,
        'max_photon': 2,
        'chi_list': [7.5, 3.75],
        'k': 15.0,
        'init_circuit': init_circuit_2qb,
        'final_circuit': final_circuit_2qb,
        'max_separated_photon': 2
    },
    '2qb_2photons_chi_varied_entanglement': {
        'n_qubits': 2,
        'max_photon': 2,
        'chi_list': [7.5, 3.75],
        'k': 15.0,
        'init_circuit': init_circuit_2qb_ent,
        'final_circuit': final_circuit_2qb_ent,
        'max_separated_photon': 2
    },
    '3qb_2photons+_chi_const_no_entanglement': {
        'n_qubits': 3,
        'max_photon': 4,
        'chi_list': [7.5, 7.5, 7.5],
        'k': 15.0,
        'init_circuit': init_circuit_3qb,
        'final_circuit': final_circuit_3qb,
        'max_separated_photon': 2
    },
    '3qb_2photons+_chi_const_entanglement': {
        'n_qubits': 3,
        'max_photon': 4,
        'chi_list': [7.5, 7.5, 7.5],
        'k': 15.0,
        'init_circuit': init_circuit_3qb_ent,
        'final_circuit': final_circuit_3qb_ent,
        'max_separated_photon': 2
    },
    '3qb_2photons+_chi_varied_no_entanglement': {
        'n_qubits': 3,
        'max_photon': 4,
        'chi_list': [7.5, 3.75, 1.875],
        'k': 15.0,
        'init_circuit': init_circuit_3qb,
        'final_circuit': final_circuit_3qb,
        'max_separated_photon': 2
    },
    '3qb_2photons+_chi_varied_entanglement': {
        'n_qubits': 3,
        'max_photon': 4,
        'chi_list': [7.5, 3.75, 1.875],
        'k': 15.0,
        'init_circuit': init_circuit_3qb_ent,
        'final_circuit': final_circuit_3qb_ent,
        'max_separated_photon': 2
    },
    '3qb_3photons_chi_const_no_entanglement': {
        'n_qubits': 3,
        'max_photon': 3,
        'chi_list': [7.5, 7.5, 7.5],
        'k': 15.0,
        'init_circuit': init_circuit_3qb,
        'final_circuit': final_circuit_3qb,
        'max_separated_photon': 3
    },
    '3qb_3photons_chi_const_entanglement': {
        'n_qubits': 3,
        'max_photon': 3,
        'chi_list': [7.5, 7.5, 7.5],
        'k': 15.0,
        'init_circuit': init_circuit_3qb_ent,
        'final_circuit': final_circuit_3qb_ent,
        'max_separated_photon': 3
    },
    '3qb_3photons_chi_varied_no_entanglement': {
        'n_qubits': 3,
        'max_photon': 3,
        'chi_list': [7.5, 3.75, 1.875],
        'k': 15.0,
        'init_circuit': init_circuit_3qb,
        'final_circuit': final_circuit_3qb,
        'max_separated_photon': 3
    },
    '3qb_3photons_chi_varied_entanglement': {
        'n_qubits': 3,
        'max_photon': 3,
        'chi_list': [7.5, 3.75, 1.875],
        'k': 15.0,
        'init_circuit': init_circuit_3qb_ent,
        'final_circuit': final_circuit_3qb_ent,
        'max_separated_photon': 3
    }
}


class _Tee:
    """Write to several streams at once (e.g. terminal + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def random_initial_values(init_circuit, final_circuit, rng, low=-np.pi, high=np.pi):
    """
    Draw random initial trainable parameters for a pair of circuits.

    The number of values matches the total number of trainable parameters across
    the initial and final circuits, which is what ``optimize_rotations`` expects.

    Args:
        init_circuit: Initial QuantumCircuit.
        final_circuit: Final QuantumCircuit.
        rng: A ``numpy.random.Generator`` used to draw the values.
        low, high: Bounds (in radians) of the uniform distribution.

    Returns:
        List[float] of random initial parameter values.
    """
    n_params = (
        init_circuit.count_trainable_parameters()
        + final_circuit.count_trainable_parameters()
    )
    return rng.uniform(low, high, size=n_params).tolist()


def run_all_experiments(experiments=experiment_list, results_dir=None, n_runs=3, seed=0):
    """
    Run every experiment in ``experiments`` in series, with several randomized
    restarts per experiment.

    Each experiment is run ``n_runs`` times, each time with a fresh set of random
    initial trainable parameters. For every run:
    - the optimization results are saved to their own NPZ file
      (``<results_dir>/<name>_run{r}.npz``);
    - the optimization dashboard is saved to their own PDF file
      (``<results_dir>/<name>_run{r}_dashboard.pdf``);
    while all textual output (progress, summaries, errors) is mirrored to a
    single common log file (``<results_dir>/experiments_output.log``) as well
    as the terminal.

    Args:
        experiments: Mapping of experiment name -> kwargs for ``run_experiment``.
        results_dir: Directory where result files are written. Defaults to a
            ``results`` folder next to this script.
        n_runs: Number of randomized restarts per experiment (default: 3).
        seed: Seed for the random number generator, for reproducibility.

    Returns:
        Dict mapping each experiment name to a list of length ``n_runs`` holding
        each run's best validation value (``None`` for runs that failed). The full
        per-run results are written to disk as NPZ/PDF files; the callbacks are not
        retained in memory so they can be freed after every optimization.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    common_log = os.path.join(results_dir, "experiments_output.log")
    rng = np.random.default_rng(seed)
    results = {}

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(common_log, "w") as log_file:
        sys.stdout = _Tee(original_stdout, log_file)
        sys.stderr = _Tee(original_stderr, log_file)
        try:
            for i, (name, params) in enumerate(experiments.items(), start=1):
                print("\n" + "=" * 80)
                print(f" Experiment {i}/{len(experiments)}: {name} ".center(80, "="))
                print("=" * 80, flush=True)

                run_best_validations = []  # per-run best validation (None if it failed)
                for run in range(1, n_runs + 1):
                    initial_values = random_initial_values(
                        params["init_circuit"], params["final_circuit"], rng
                    )
                    print("\n" + "-" * 80)
                    print(f"Run {run}/{n_runs} of '{name}' "
                          f"started at {datetime.now():%Y-%m-%d %H:%M:%S}")
                    print(f"Random initial values: {initial_values}")
                    print("-" * 80, flush=True)

                    history = None
                    try:
                        history = run_experiment(initial_values=initial_values, **params)

                        npz_path = os.path.join(results_dir, f"{name}_run{run}.npz")
                        history.save(npz_path)
                        print(f"\nSaved results to {npz_path}")

                        plot_path = os.path.join(
                            results_dir, f"{name}_run{run}_dashboard.pdf"
                        )
                        plot_optimization_dashboard(
                            optimization_callback=history,
                            save_path=plot_path,
                            show_confusion_matrix_summary=True,
                        )
                        plt.close("all")
                        print(f"Saved dashboard to {plot_path}")

                        print("\n" + str(history))
                        run_best_validations.append(float(history.best_validation))
                    except Exception as exc:  # keep going if one run fails
                        print(f"\n!!! Run {run} of '{name}' FAILED: {exc}")
                        traceback.print_exc()
                        run_best_validations.append(None)
                    finally:
                        # Free memory after every optimization: the full per-step
                        # history is already saved to disk, so drop the callback,
                        # force garbage collection, and clear JAX's compilation
                        # cache so device (GPU) buffers are released before the
                        # next run instead of accumulating across the sweep.
                        history = None
                        gc.collect()
                        jax.clear_caches()

                # Report the best restart for this experiment.
                valid_runs = [
                    (r + 1, v) for r, v in enumerate(run_best_validations) if v is not None
                ]
                if valid_runs:
                    best_run, best_val = max(valid_runs, key=lambda rv: rv[1])
                    print(f"\n>>> Best run for '{name}': run {best_run} "
                          f"(validation = {best_val:.6f})", flush=True)
                results[name] = run_best_validations

            print("\n" + "=" * 80)
            print(" All experiments finished ".center(80, "="))
            print(f"Results written to: {results_dir}")
            print("=" * 80, flush=True)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    return results


if __name__ == "__main__":
    run_all_experiments()

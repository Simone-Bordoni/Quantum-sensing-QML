# Import required libraries
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import optax
from qsopt.core.experimental_parameters import (
    ExperimentalParameters,
    PhysicalConstants,
    SystemDimensions,
    MeasurementProtocol,
    InteractionType,
    QubitInteraction,
    InitialStateConfig,
    InitialStateType,
    NoiseConfiguration,
)
from qsopt.core.circuit import QuantumCircuit, create_ry_circuit
from qsopt.core.experiment.experiment import Experiment
from qsopt.core.loss_functions import DetectionMetric
from qsopt.core.gates import CNOTGate, RYGate, RZGate
from qsopt.utils.visualization import plot_optimization_dashboard
import time as t

# Set this manually to skip the input prompt.
# Example Windows: r"C:\Users\your_name\Desktop"
# Example Linux: "/home/your_name"
MANUAL_HOME_FOLDER = r'/raid/home/ncampioni'


def resolve_home_folder():
    default_home = os.path.expanduser("~")

    if MANUAL_HOME_FOLDER:
        return os.path.normpath(os.path.expanduser(MANUAL_HOME_FOLDER))

    try:
        user_input = input(f"Home folder [default is: {default_home}]: ").strip()
    except EOFError:
        user_input = ""
    print(f"Using home folder: {user_input if user_input else default_home}")
    selected_home = user_input if user_input else default_home
    return os.path.normpath(os.path.expanduser(selected_home))


home_path = resolve_home_folder()
save_folder = os.path.join(home_path, 'results', 'studio_entanglement')
error_folder = os.path.join(save_folder, 'errors')
training_log_file = os.path.join(save_folder, 'log.txt')

os.makedirs(save_folder, exist_ok=True)
os.makedirs(error_folder, exist_ok=True)


def log_training_event(event, experiment_name, details=""):
    timestamp = datetime.now().isoformat(timespec="seconds")
    details_text = f" | {details}" if details else ""
    with open(training_log_file, 'a', encoding='utf-8') as log_file:
        log_file.write(f"[{timestamp}] {event} | {experiment_name}{details_text}\n")



log_training_event('START PROGRAM', '', 'Initializing experiments and training runs')


def get_last_gradient_info(callback):
    grads_history = callback.history.get('grads', [])
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

inverse_pulse_width = 1
gm = 15 * inverse_pulse_width


measurement = MeasurementProtocol(
    measurement_times=list(np.array([-5.0, -2.5, 0.0, 2.5, 5.0])/inverse_pulse_width)
)


def create_std_experiment_setup(n_qubits, initial_circuit, final_circuit, detection_metric):

    interactions = [QubitInteraction(
        qubit_indices=(i, j),
        interaction_type=InteractionType.XX,
        chi=0.1
    ) for i in range(n_qubits) for j in range(n_qubits) if i!=j ]

    physical_constants = PhysicalConstants(
        n_qubits=n_qubits,
        chi= 2.0*gm,
        photon_cavity_coupling=gm,
        inverse_pulse_width=inverse_pulse_width,
        qubit_interactions=interactions
    )

    noise_config = NoiseConfiguration(
        depolarizing=0.0001,
        dephasing=0.0001,
        relaxation=0.0001
    )

    
    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)

    exp_params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=SystemDimensions(),
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config
    )

    return Experiment(
        experimental_params=exp_params,
        initial_circuit=initial_circuit,
        final_circuit=final_circuit,
        detection_metric=detection_metric
    )

#ESPERIMENTO DI CONTROLLO: 1 qubit, circuiti 1 layer (rotazione), metrica: max computational distance

n_qubits = 1
input_circ_1qb = create_ry_circuit(n_qubits, np.pi/2)
final_circ_1qb = create_ry_circuit(n_qubits, -np.pi/2)
computation_dist_1qb = DetectionMetric(n_qubits=1, detection_criterion = 'max computational distance')

experiment_1qb = create_std_experiment_setup(n_qubits, initial_circuit=input_circ_1qb, final_circuit=final_circ_1qb, detection_metric=computation_dist_1qb)


def build_experiment_dict(n_qubits, detection_metric):

    input_circ = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi/2)
    final_circ = create_ry_circuit(n_qubits, -np.random.rand(n_qubits) * np.pi/2)

    initial_circ_z = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi/2)
    initial_circ_z.add_layer(gate_type = RZGate, parameters = np.random.rand(n_qubits) * np.pi/3)

    input_circ_ent = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi/4)
    input_circ_ent.add_entangling_layer(CNOTGate, pattern='circular')
    input_circ_ent.add_layer(gate_type = RYGate, parameters = np.random.rand(n_qubits) * np.pi/4)

    final_circ_ent = create_ry_circuit(n_qubits, -np.random.rand(n_qubits) * np.pi/4)
    final_circ_ent.add_entangling_layer(CNOTGate, pattern='circular')
    final_circ_ent.add_layer(gate_type = RYGate, parameters = -np.random.rand(n_qubits) * np.pi/4)

    input_circ_zent = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi/4)
    input_circ_zent.add_entangling_layer(CNOTGate, pattern='circular')
    input_circ_zent.add_layer(gate_type = RYGate, parameters = np.random.rand(n_qubits) * np.pi/4)
    input_circ_zent.add_layer(gate_type = RZGate, parameters = np.random.rand(n_qubits) * np.pi/3)

    experiment_dict = {}
    experiment_dict[f'{n_qubits}qb_no_no'] = create_std_experiment_setup(n_qubits, initial_circuit=input_circ, final_circuit=final_circ, detection_metric=detection_metric)
    experiment_dict[f'{n_qubits}qb_ent_no'] = create_std_experiment_setup(n_qubits, initial_circuit=input_circ_ent, final_circuit=final_circ, detection_metric=detection_metric)
    experiment_dict[f'{n_qubits}qb_ent_ent'] = create_std_experiment_setup(n_qubits, initial_circuit=input_circ_ent, final_circuit=final_circ_ent, detection_metric=detection_metric)
    experiment_dict[f'{n_qubits}qb_z_no'] = create_std_experiment_setup(n_qubits, initial_circuit=initial_circ_z, final_circuit=final_circ, detection_metric=detection_metric)
    experiment_dict[f'{n_qubits}qb_zent_ent'] = create_std_experiment_setup(n_qubits, initial_circuit=input_circ_zent, final_circuit=final_circ_ent, detection_metric=detection_metric)

    return experiment_dict

#ESPERIMENTI DI TEST, 1 QUBIT:

n_qubits = 1
computation_dist_1qb = DetectionMetric(n_qubits=1, detection_criterion = 'max computational distance')

input_circ_1qb = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi/2)
final_circ_1qb = create_ry_circuit(n_qubits, -np.random.rand(n_qubits) * np.pi/2)

initial_circ_z_1qb = create_ry_circuit(n_qubits, np.random.rand(n_qubits) * np.pi/2)
initial_circ_z_1qb.add_layer(gate_type = RZGate, parameters = np.random.rand(n_qubits) * np.pi/3)

test_dict_1qb = {}
test_dict_1qb[f'{n_qubits}qb_no_no'] = create_std_experiment_setup(n_qubits, initial_circuit=input_circ_1qb, final_circuit=final_circ_1qb, detection_metric=computation_dist_1qb)
test_dict_1qb[f'{n_qubits}qb_z_no'] = create_std_experiment_setup(n_qubits, initial_circuit=initial_circ_z_1qb, final_circuit=final_circ_1qb, detection_metric=computation_dist_1qb)


#ESPERIMENTI A 2 QUBIT CON E SENZA ENTANGLEMENT: 

n_qubits = 2
computation_dist_2qb = DetectionMetric(n_qubits=2, detection_criterion = 'max computational distance')
exp_dict_2qb = build_experiment_dict(n_qubits=n_qubits, detection_metric=computation_dist_2qb)

#ESPERIMENTI A 3 QUBIT CON E SENZA ENTANGLEMENT:

n_qubits = 3
computation_dist_3qb = DetectionMetric(n_qubits=3, detection_criterion = 'max computational distance')
exp_dict_3qb = build_experiment_dict(n_qubits=n_qubits, detection_metric=computation_dist_3qb)

#ESPERIMENTI A 5 QUBIT CON E SENZA ENTANGLEMENT:

n_qubits = 5
computation_dist_5qb = DetectionMetric(n_qubits=5, detection_criterion = 'max computational distance')
exp_dict_5qb = build_experiment_dict(n_qubits=n_qubits, detection_metric=computation_dist_5qb)

#DEFINISCO ALLENAMENTO CON DELLE COSTANTI DI ALLENAMENTO

optimizer = optax.sgd(learning_rate=0.05)


def run_experiment_with_checkpoints(experiment, tot_steps, checkpoint_interval, tolerance, optimizer, save_folder, exp_name):

    missing_steps = tot_steps
    history = None

    log_training_event(
        'TRAINING_START',
        exp_name,
        f"tot_steps={tot_steps} checkpoint_interval={checkpoint_interval} tolerance={tolerance}"
    )

    while missing_steps>0:
        if missing_steps < checkpoint_interval:
            steps_to_run = missing_steps
            missing_steps = 0
        else:
            steps_to_run = checkpoint_interval
            missing_steps -= checkpoint_interval

        history = experiment.optimize_rotations(
            num_steps=steps_to_run,
            tolerance=tolerance,
            callback=history,
            optimizer=optimizer,
            verbose = False,
            hot_start = history is not None,
        )
        
        history.save(os.path.join(save_folder, f'history_{exp_name}.pkl'))

        if history.converged:
            break
        log_training_event(
            'CHECKPOINT',
            exp_name,
            f"epoch={history.epoch} best_metric={history.best_metric} {get_last_gradient_info(history)}"
        )



    if history is None:
        log_training_event('TRAINING_END', exp_name, 'status=skipped reason=tot_steps<=0')
        return None

    log_training_event(
        'TRAINING_END',
        exp_name,
        f"converged={history.converged} epoch={history.epoch} best_metric={history.best_metric} {get_last_gradient_info(history)}"
    )
        
    return history

def run_experiment_ensemble(experiment_dict, tot_steps=1000, checkpoint_interval=200, tolerance=1e-9):
    for exp_name, experiment in experiment_dict.items():

        try:
            
            history = run_experiment_with_checkpoints(experiment, tot_steps=tot_steps, checkpoint_interval=checkpoint_interval, tolerance=tolerance, optimizer=optimizer, save_folder=save_folder, exp_name=exp_name)

            _ = plot_optimization_dashboard(
                optimization_callback=history,
                show_metric=True,
                show_gradients=True,
                show_parameters=True,
                show_detection_measures=True,
                show_trajectory=True,
                save_path=os.path.join(save_folder, f'dashboard_{exp_name}.pdf')  # Save to file
            )

        except Exception as e:
            log_training_event('TRAINING_ERROR', exp_name, f'error={e}')
            #print errors to file
            with open(os.path.join(error_folder, 'error_log.txt'), 'a', encoding='utf-8') as f:
                f.write(f'Error in experiment {exp_name}: {str(e)}\n')


#GIRO ESPERIMENTO DI CONTROLLO: 1 QUBIT, CIRCUITI 1 LAYER (ROT), METRICA: MAX COMPUTATIONAL DISTANCE

log_training_event('TRAINING_START', 'control_1qb', 'num_steps=1000 tolerance=1e-9')
history_ctrl = experiment_1qb.optimize_rotations(num_steps=1000, tolerance=1e-9, optimizer=optimizer, verbose=False)
history_ctrl.save(os.path.join(save_folder, 'history_ctrl.pkl'))
log_training_event(
    'TRAINING_END',
    'control_1qb',
    f"converged={history_ctrl.converged} epoch={history_ctrl.epoch} best_metric={history_ctrl.best_metric} {get_last_gradient_info(history_ctrl)}"
)

#GIRO ESPERIMENTI

#run_experiment_ensemble(test_dict_1qb, tot_steps=500, checkpoint_interval=100, tolerance=1e-9)

run_experiment_ensemble(exp_dict_2qb, tot_steps=1000, checkpoint_interval=200, tolerance=1e-9)
run_experiment_ensemble(exp_dict_3qb, tot_steps=1000, checkpoint_interval=200, tolerance=1e-9)
run_experiment_ensemble(exp_dict_5qb, tot_steps=1000, checkpoint_interval=200, tolerance=1e-9)

log_training_event('END PROGRAM', '', 'All experiments completed')
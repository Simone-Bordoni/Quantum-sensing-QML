from qsopt import * 
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import qutip as qt
import qutip_jax
from jax.scipy.special import erfc
from typing import List
from qsopt.core.circuit import create_ry_circuit
from qsopt.core.gates import RZGate
import optax



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
        chi_list: List[float], 
        k: float, 
        gamma: float, 
        init_circuit: QuantumCircuit, 
        final_circuit: QuantumCircuit, 
        initial_values: List[float],
        config_set: List[SystemConfiguration],
        sigma: float=1.0):
    """
    Run the multi-photon detector experiment with specified parameters.

    Args:
        n_qubits: int, number of qubits in the system
        chi_list: List[float], list of dispersive coupling strengths for each qubit
        k: float, scaling factor for dispersive couplings
        gamma: float, decay rate parameter
        init_circuit: QuantumCircuit, initial circuit with trainable parameters
        final_circuit: QuantumCircuit, final circuit with trainable parameters
        initial_values: List[float], initial values for the trainable parameters in the circuits
        config_set: List[SystemConfiguration], list of system configurations
        sigma: float, pulse bandwidth parameter (default: 1.0)
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
    cavity_field_interactions = [
        Interaction(interaction_type = InteractionType.INPUT_OUTPUT,
                subsystem1 = ('cavity',0),
                subsystem2 = ('field',0),
                parameters = {'gamma':gamma, 'kappa':k, 'sigma': sigma},
                time_modulation = pulse
                )
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

    interactions = qubit_cavity_interactions + cavity_field_interactions + qubit_qubit_interactions
    
    # Define custom physical model
    physical_model = PhysicalModel(
        n_cavities = 1,
        n_fields = 1,
        n_qubits = n_qubits,
        cavity_levels = max_photon,
        field_levels = max_photon,
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
        num_steps=20000,
        verbose=True,
        verbose_step=250,
        batch_size=1,
        tolerance=1e-9,
        callback=history,
        optimizer=optax.sgd(learning_rate=0.1)
    )
    

init_circuit_2qb = create_ry_circuit(n_qubits=2, theta_values=np.pi/2)#, trainable=False)
initial_circuit.add_layer(RZGate, parameters=0.0)
final_circuit = create_ry_circuit(n_qubits=2, theta_values=-np.pi/2)


def gen_config_set(max_photon: int, 
                   max_separated_photon: int):
    """
    
    """
    if max_photon < max_separated_photon:
        raise ValueError("max_photon must be greater than or equal to max_separated_photon")

    with qt.CoreOptions(default_dtype="jax"):

        density_matrix = qt.tensor(sum([qt.fock_dm(max_photon, i) for i in range(max_photon)]), qt.eye(max_photon))

        config_set = [
            SystemConfiguration(
                name = '0-photons',
                init_field_states = {i: SubsystemState(State.FOCK, {'n':i})}) 
            for i in range(max_separated_photon)
            ]
        
        if max_photon > max_separated_photon:
            config_set.append(SystemConfiguration(
                                    name = '2-photons',
                                    density_matrix = density_matrix))
    
    return config_set
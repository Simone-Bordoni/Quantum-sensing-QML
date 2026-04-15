# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
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
from qsopt.core.gates import CNOTGate as CNOT

inverse_pulse_width = 1
gm = 15 * inverse_pulse_width

# Define custom physical constants
control_constants = PhysicalConstants(
    n_qubits=1,
    chi=2.0*gm,
    photon_cavity_coupling=gm,
    inverse_pulse_width=inverse_pulse_width
)

interactions_xx_2qb = QubitInteraction(
    qubit_indices=(0, 1),
    interaction_type=InteractionType.XX,
    chi=0.1
)

two_qubits_constants = PhysicalConstants(
    n_qubits=2,
    chi=2.0*gm,
    photon_cavity_coupling=gm,
    inverse_pulse_width=inverse_pulse_width,
    qubit_interactions=interactions_xx_2qb
)
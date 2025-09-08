#!/usr/bin/env python3
"""
Test script for the SingleQubitExperiment class implementation.
"""

import numpy as np
from src.qsopt.core.experimental_parameters import (
    ExperimentalParameters, 
    PhysicalConstants, 
    SystemDimensions, 
    NoiseConfiguration,
    MeasurementProtocol,
    InitialStateConfig,
    InitialStateType
)
from src.qsopt.core.trainable_parameters import TrainableParameters
from src.qsopt.core.experiment import SingleQubitExperiment

def test_experiment_initialization():
    """Test basic experiment initialization and operator generation."""
    print("Testing SingleQubitExperiment initialization...")
    
    # Create experimental parameters similar to notebook
    physical_constants = PhysicalConstants(
        chi=0.5 * 0.03 * 2 * np.pi,  # From notebook: chi = 0.5*gm
        photon_cavity_coupling=0.03 * 2 * np.pi,  # From notebook: gm = .03 * 2 * np.pi
        inverse_pulse_width=0.1 * 0.03 * 2 * np.pi  # From notebook: sigma = .1*gm
    )
    
    system_dims = SystemDimensions(
        cavity_levels=2,  # From notebook: nlev = 2
        qubit_levels=2,   # From notebook: qlev = 2  
        field_levels=2    # Input field levels
    )
    
    noise_config = NoiseConfiguration(
        relaxation=0.0001 * 2 * np.pi,    # From notebook: gamma_relax
        dephasing=0.0001 * 2 * np.pi,     # From notebook: gamma_dephasing
        depolarizing=0.000 * 2 * np.pi    # From notebook: gamma_depol
    )
    
    measurement = MeasurementProtocol(
        measurement_times=[-5.0, -2.5, 0, 2.5, 5.0]  # From notebook: tmeas
    )
    
    initial_state = InitialStateConfig(
        state_type=InitialStateType.SINGLE_PHOTON  # |0,1,0⟩ from notebook
    )
    
    exp_params = ExperimentalParameters(
        physical_constants=physical_constants,
        system_dims=system_dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config
    )
    
    # Create trainable parameters
    trainable_params = TrainableParameters()
    trainable_params.add_rotation_angles(
        names=['theta1', 'theta2'],
        initial_values=[np.pi/2, -np.pi/2]
    )
    
    # Create experiment
    experiment = SingleQubitExperiment(exp_params, trainable_params)
    
    # Test that operators were created
    assert experiment.operators is not None, "Operators should be generated"
    assert experiment.hamiltonians is not None, "Hamiltonians should be generated"
    assert experiment.lindblad_operators is not None, "Lindblad operators should be generated"
    
    # Check key operators exist
    expected_ops = ['a_in', 'a_in_dag', 'a', 'a_dag', 'sigma_z', 'sigma_x', 'sigma_y', 'P0', 'P1']
    for op_name in expected_ops:
        assert op_name in experiment.operators, f"Operator {op_name} should exist"
    
    # Check Hamiltonians exist
    expected_ham = ['total', 'dispersive', 'coupling']
    for ham_name in expected_ham:
        assert ham_name in experiment.hamiltonians, f"Hamiltonian {ham_name} should exist"
    
    # Check Lindblad operator sets exist
    expected_lindblad = ['interaction', 'no_interaction', 'noise_only']
    for lind_name in expected_lindblad:
        assert lind_name in experiment.lindblad_operators, f"Lindblad set {lind_name} should exist"
    
    # Test solvers can be created
    solver_with = experiment.get_solver_with_interaction()
    solver_without = experiment.get_solver_no_interaction()
    
    assert solver_with is not None, "Solver with interaction should be created"
    assert solver_without is not None, "Solver without interaction should be created"
    
    print("✓ All tests passed!")
    
    # Print some basic info
    print(f"\nSystem Information:")
    print(f"  Total Hilbert space dimension: {system_dims.field_levels * system_dims.cavity_levels * system_dims.qubit_levels}")
    print(f"  Physical constants: χ={physical_constants.chi:.6f}, gm={physical_constants.photon_cavity_coupling:.6f}")
    print(f"  Noise rates: γ_relax={noise_config.relaxation:.6f}, γ_deph={noise_config.dephasing:.6f}")
    print(f"  Operators generated: {len(experiment.operators)}")
    print(f"  Hamiltonians generated: {len(experiment.hamiltonians)}")
    
    return experiment

if __name__ == "__main__":
    test_experiment_initialization()

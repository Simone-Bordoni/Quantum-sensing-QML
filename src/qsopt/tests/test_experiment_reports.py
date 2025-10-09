"""
Test save and load experiment report functionality.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from qsopt import (
    SingleQubitExperiment,
    ExperimentalParameters,
    PhysicalConstants,
    SystemDimensions,
    MeasurementProtocol,
    InitialStateConfig,
    InitialStateType,
    NoiseConfiguration,
    TrainableParameters
)


class TestExperimentReports:
    """Test suite for save/load experiment report functionality."""
    
    def create_test_experiment(self):
        """Helper to create a test experiment."""
        gm = 0.03 * 2 * np.pi
        
        constants = PhysicalConstants(
            chi=0.5 * gm,
            photon_cavity_coupling=gm,
            inverse_pulse_width=0.1 * gm
        )
        
        dims = SystemDimensions(cavity_levels=2, qubit_levels=2, field_levels=2)
        
        measurement = MeasurementProtocol(
            measurement_times=list(np.array([-5.0, 0.0, 5.0]) / (0.1 * gm))
        )
        
        initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
        
        noise_config = NoiseConfiguration(
            depolarizing=0.001,
            dephasing=0.001,
            relaxation=0.001
        )
        
        exp_params = ExperimentalParameters(
            physical_constants=constants,
            system_dims=dims,
            measurement=measurement,
            initial_state=initial_state,
            noise_config=noise_config
        )
        
        params = TrainableParameters()
        params.add_rotation_angles(['ry1', 'ry2'], [np.pi/2, -np.pi/2])
        
        return SingleQubitExperiment(exp_params, params)
    
    def test_save_report_after_simulation(self):
        """Test saving report after running a simulation."""
        experiment = self.create_test_experiment()
        
        # Run simulation
        results = experiment.run_simulation()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'test_report.json'
            
            # Save report
            experiment.save_experiment_report(str(report_path))
            
            # Verify file exists
            assert report_path.exists()
            
            # Load and verify content
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            assert report['experiment_type'] == 'SingleQubitExperiment'
            assert 'experimental_parameters' in report
            assert 'trainable_parameters' in report
            # After simulation (not optimization), callback_info is None
            assert report['callback_info'] is None
    
    def test_save_report_after_optimization(self):
        """Test saving report after optimization."""
        experiment = self.create_test_experiment()
        
        # Run short optimization
        history = experiment.optimize(
            theta_init=[1.5, -1.3],
            num_steps=5,
            verbose=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'opt_report.json'
            
            # Save report
            experiment.save_experiment_report(str(report_path))
            
            # Verify files exist
            assert report_path.exists()
            
            # Check if callback NPZ was created
            callback_path = Path(tmpdir) / 'opt_report_callback.npz'
            assert callback_path.exists()
            
            # Load and verify JSON content
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            assert report['callback_info']['mode'] == 'optimization'
            assert 'callback_data_path' in report['callback_info']
            assert 'optimization_summary' in report['callback_info']
    
    def test_load_experiment_report(self):
        """Test loading experiment report."""
        experiment = self.create_test_experiment()
        
        # Run optimization
        history = experiment.optimize(
            theta_init=[1.5, -1.3],
            num_steps=5,
            verbose=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'load_test.json'
            
            # Save report
            experiment.save_experiment_report(str(report_path))
            
            # Load report
            loaded = SingleQubitExperiment.load_experiment_report(str(report_path))
            
            assert loaded['experiment_type'] == 'SingleQubitExperiment'
            assert 'experimental_params_dict' in loaded
            assert 'trainable_params_dict' in loaded
            assert 'callback_data' in loaded
            
            # Verify callback data was loaded
            assert 'epochs' in loaded['callback_data']
            assert 'contrast' in loaded['callback_data']
    
    def test_report_contains_all_parameters(self):
        """Test that report contains all experimental parameters."""
        experiment = self.create_test_experiment()
        experiment.run_simulation()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'full_params.json'
            experiment.save_experiment_report(str(report_path))
            
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            exp_params = report['experimental_parameters']
            
            # Check all sections exist
            assert 'physical_constants' in exp_params
            assert 'system_dimensions' in exp_params
            assert 'measurement_protocol' in exp_params
            assert 'initial_state' in exp_params
            assert 'noise_configuration' in exp_params
            
            # Verify specific values
            assert exp_params['system_dimensions']['cavity_levels'] == 2
            assert exp_params['initial_state']['state_type'] == 'single_photon'
    
    def test_default_save_path_creates_directory(self):
        """Test that default save path creates results directory."""
        experiment = self.create_test_experiment()
        experiment.run_simulation()
        
        # Save with default path
        default_path = 'results/test_default.json'
        
        try:
            experiment.save_experiment_report(default_path)
            
            # Verify file exists
            assert Path(default_path).exists()
            assert Path('results').is_dir()
            
        finally:
            # Cleanup
            if Path(default_path).exists():
                Path(default_path).unlink()
            if Path('results').exists() and not any(Path('results').iterdir()):
                Path('results').rmdir()

"""
Tests for optimization callback functionality.

Tests the OptimizationCallback class including history tracking,
best parameter tracking, saving/loading, and integration with experiments.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from qsopt import OptimizationCallback, TrainableParameters


def create_test_trainable_params(angle_dict):
    """Helper function to create TrainableParameters for testing."""
    params = TrainableParameters()
    for name, value in angle_dict.items():
        params.add_rotation_angles(name, value)
    return params


class TestOptimizationCallback:
    """Test suite for OptimizationCallback class."""
    
    def test_callback_initialization(self):
        """Test callback initializes with correct default values."""
        callback = OptimizationCallback(save_every=5, save_best=True)
        
        assert callback.save_every == 5
        assert callback.save_best is True
        assert callback.epoch == 0
        assert callback.best_trainable_params is None
        assert callback.best_contrast == -float('inf')
        assert callback.best_metrics is None
        assert len(callback.history['epochs']) == 0
    
    def test_callback_single_call(self):
        """Test callback records single optimization step."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        params = create_test_trainable_params({'theta1': 1.0, 'theta2': 2.0})
        callback(
            trainable_params=params,
            prob_with=0.8,
            prob_without=0.2,
            contrast=0.6
        )
        
        assert callback.epoch == 1
        assert len(callback.history['epochs']) == 1
        assert callback.history['contrast'][0] == 0.6
        assert callback.history['prob_with'][0] == 0.8
        assert callback.history['prob_without'][0] == 0.2
    
    def test_callback_save_every(self):
        """Test callback respects save_every parameter."""
        callback = OptimizationCallback(save_every=3, save_best=False)
        
        # Call 10 times
        for i in range(10):
            params = create_test_trainable_params({'theta': float(i)})
            callback(
                trainable_params=params,
                prob_with=0.5,
                prob_without=0.3,
                contrast=0.2
            )
        
        assert callback.epoch == 10
        # Should save at epochs 3, 6, 9 (3 times)
        assert len(callback.history['epochs']) == 3
        assert callback.history['epochs'] == [3, 6, 9]
    
    def test_callback_best_tracking(self):
        """Test callback tracks best parameters correctly (maximize contrast)."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # First call - should be best
        params1 = create_test_trainable_params({'theta1': 1.0, 'theta2': 2.0})
        callback(
            trainable_params=params1,
            prob_with=0.5,
            prob_without=0.3,
            contrast=0.2
        )
        
        assert callback.best_contrast == 0.2
        assert callback.best_trainable_params is not None
        
        # Second call - worse (lower contrast)
        params2 = create_test_trainable_params({'theta1': 2.0, 'theta2': 3.0})
        callback(
            trainable_params=params2,
            prob_with=0.4,
            prob_without=0.3,
            contrast=0.1
        )
        
        # Best should not change
        assert callback.best_contrast == 0.2
        best_angles = callback.best_trainable_params.get_rotation_angles()
        assert best_angles['theta1'][0] == 1.0
        
        # Third call - better (higher contrast)
        params3 = create_test_trainable_params({'theta1': 3.0, 'theta2': 4.0})
        callback(
            trainable_params=params3,
            prob_with=0.9,
            prob_without=0.1,
            contrast=0.8
        )
        
        # Best should update
        assert callback.best_contrast == 0.8
        best_angles = callback.best_trainable_params.get_rotation_angles()
        assert best_angles['theta1'][0] == 3.0
        assert callback.best_metrics['contrast'] == 0.8
    
    def test_get_best_trainable_params(self):
        """Test retrieving best trainable parameters."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Initially None
        assert callback.get_best_trainable_params() is None
        
        # After recording
        params = create_test_trainable_params({'theta': 1.5})
        callback(
            trainable_params=params,
            prob_with=0.7,
            prob_without=0.2,
            contrast=0.5
        )
        
        best = callback.get_best_trainable_params()
        assert best is not None
        best_angles = best.get_rotation_angles()
        assert best_angles['theta'][0] == 1.5
    
    def test_get_best_metrics(self):
        """Test retrieving metrics at best parameters."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Initially None
        assert callback.get_best_metrics() is None
        
        # Record some steps
        params = create_test_trainable_params({'theta1': 1.0, 'theta2': 2.0})
        callback(
            trainable_params=params,
            prob_with=0.75,
            prob_without=0.25,
            contrast=0.50
        )
        
        metrics = callback.get_best_metrics()
        assert metrics is not None
        assert metrics['epoch'] == 1
        assert metrics['contrast'] == 0.50
        assert metrics['prob_with'] == 0.75
        assert metrics['prob_without'] == 0.25
    
    def test_get_history(self):
        """Test retrieving full optimization history."""
        callback = OptimizationCallback(save_every=1, save_best=False)
        
        # Record 3 steps
        for i in range(3):
            params = create_test_trainable_params({'theta': float(i)})
            callback(
                trainable_params=params,
                prob_with=0.5 + i*0.1,
                prob_without=0.3,
                contrast=0.2 + i*0.1
            )
        
        history = callback.get_history()
        assert len(history['epochs']) == 3
        assert len(history['contrast']) == 3
        assert len(history['trainable_params']) == 3
        # Use numpy for floating point comparison
        np.testing.assert_allclose(history['contrast'], [0.2, 0.3, 0.4], rtol=1e-10)
        assert history['epochs'] == [1, 2, 3]
    
    def test_save_and_load(self):
        """Test saving and loading callback data."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Record some data with increasing contrast
        for i in range(5):
            params = create_test_trainable_params({'theta1': float(i), 'theta2': float(i+1)})
            callback(
                trainable_params=params,
                prob_with=0.5 + i*0.05,
                prob_without=0.3,
                contrast=0.2 + i*0.05
            )
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_callback.npz"
            callback.save(str(filepath))
            
            # Check file exists
            assert filepath.exists()
            
            # Load data
            loaded_data = OptimizationCallback.load(str(filepath))
            
            # Verify arrays
            assert 'epochs' in loaded_data
            assert 'contrast' in loaded_data
            assert 'prob_with' in loaded_data
            assert 'prob_without' in loaded_data
            assert 'parameters' in loaded_data
            assert 'best_parameters' in loaded_data
            assert 'best_contrast' in loaded_data
            
            # Check values
            assert len(loaded_data['epochs']) == 5
            assert np.array_equal(loaded_data['epochs'], np.array([1, 2, 3, 4, 5]))
            # Best should be at epoch 5 (highest contrast)
            assert loaded_data['best_contrast'] == 0.4
            assert loaded_data['best_epoch'] == 5
    
    def test_reset(self):
        """Test resetting callback state."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Record some data
        params = create_test_trainable_params({'theta': 1.0})
        callback(
            trainable_params=params,
            prob_with=0.7,
            prob_without=0.3,
            contrast=0.4
        )
        
        # Verify data exists
        assert callback.epoch == 1
        assert len(callback.history['epochs']) == 1
        assert callback.best_trainable_params is not None
        
        # Reset
        callback.reset()
        
        # Verify reset
        assert callback.epoch == 0
        assert len(callback.history['epochs']) == 0
        assert callback.best_trainable_params is None
        assert callback.best_contrast == -float('inf')
        assert callback.best_metrics is None
    
    def test_repr_simulation_mode(self):
        """Test string representation for simulation mode."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Single call (simulation mode)
        params = create_test_trainable_params({'ry1': 1.5, 'ry2': 2.3})
        callback(
            trainable_params=params,
            prob_with=0.7,
            prob_without=0.3,
            contrast=0.4
        )
        
        repr_str = repr(callback)
        assert 'MODE: Single Simulation' in repr_str
        assert 'ry1' in repr_str
        assert 'ry2' in repr_str
        assert '0.4' in repr_str  # contrast
    
    def test_repr_optimization_mode(self):
        """Test string representation for optimization mode."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Multiple calls (optimization mode)
        for i in range(3):
            params = create_test_trainable_params({'theta': float(i)})
            callback(
                trainable_params=params,
                prob_with=0.5 + i*0.1,
                prob_without=0.3,
                contrast=0.2 + i*0.1
            )
        
        # Set convergence info
        callback.set_convergence_info(converged=True, final_grad_norm=1e-6)
        
        repr_str = repr(callback)
        assert 'MODE: Optimization' in repr_str
        assert 'Total iterations: 3' in repr_str
        assert 'Converged: True' in repr_str
        assert '1.000000e-06' in repr_str  # gradient norm
    
    def test_callback_without_best_tracking(self):
        """Test callback with best tracking disabled."""
        callback = OptimizationCallback(save_every=1, save_best=False)
        
        # Record multiple steps
        for i in range(3):
            params = create_test_trainable_params({'theta': float(i)})
            callback(
                trainable_params=params,
                prob_with=0.5,
                prob_without=0.3,
                contrast=0.2
            )
        
        # Best parameters should not be tracked
        assert callback.best_trainable_params is None
        assert callback.best_contrast == -float('inf')
        
        # But history should still be recorded
        assert len(callback.history['epochs']) == 3
    
    def test_parameter_deep_copy(self):
        """Test that trainable parameters are deep copied."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        params = create_test_trainable_params({'theta': 1.0})
        callback(
            trainable_params=params,
            prob_with=0.7,
            prob_without=0.3,
            contrast=0.4
        )
        
        # Modify original parameters
        params.add_rotation_angles('theta', 999.0)
        
        # Callback should have a copy of the original values
        best_params = callback.get_best_trainable_params()
        best_angles = best_params.get_rotation_angles()
        assert best_angles['theta'][0] == 1.0
    
    def test_save_without_best_parameters(self):
        """Test saving when no best parameters are tracked."""
        callback = OptimizationCallback(save_every=1, save_best=False)
        
        params = create_test_trainable_params({'theta': 1.0})
        callback(
            trainable_params=params,
            prob_with=0.7,
            prob_without=0.3,
            contrast=0.4
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_no_best.npz"
            callback.save(str(filepath))
            
            loaded_data = OptimizationCallback.load(str(filepath))
            
            # Should not have best parameters
            assert 'best_parameters' not in loaded_data
            assert 'best_contrast' not in loaded_data
            
            # But should have history
            assert 'epochs' in loaded_data
            assert 'contrast' in loaded_data
    
    def test_convergence_info(self):
        """Test setting and retrieving convergence information."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Initially false
        assert callback.converged is False
        assert callback.final_grad_norm is None
        
        # Set convergence info
        callback.set_convergence_info(converged=True, final_grad_norm=1.5e-7)
        
        assert callback.converged is True
        assert callback.final_grad_norm == 1.5e-7
    
    def test_multiple_parameters_tracking(self):
        """Test tracking with multiple rotation angles."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        params = create_test_trainable_params({
            'ry1': 1.0,
            'ry2': 2.0,
            'ry3': 3.0
        })
        
        callback(
            trainable_params=params,
            prob_with=0.8,
            prob_without=0.2,
            contrast=0.6
        )
        
        best = callback.get_best_trainable_params()
        best_angles = best.get_rotation_angles()
        
        assert len(best_angles) == 3
        assert best_angles['ry1'][0] == 1.0
        assert best_angles['ry2'][0] == 2.0
        assert best_angles['ry3'][0] == 3.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

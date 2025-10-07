"""
Tests for optimization callback functionality.

Tests the OptimizationCallback class including history tracking,
best parameter tracking, saving/loading, and integration with experiments.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from qsopt import OptimizationCallback


class TestOptimizationCallback:
    """Test suite for OptimizationCallback class."""
    
    def test_callback_initialization(self):
        """Test callback initializes with correct default values."""
        callback = OptimizationCallback(save_every=5, save_best=True)
        
        assert callback.save_every == 5
        assert callback.save_best is True
        assert callback.epoch == 0
        assert callback.best_parameters is None
        assert callback.best_loss == float('inf')
        assert callback.best_metrics is None
        assert len(callback.history['epochs']) == 0
    
    def test_callback_single_call(self):
        """Test callback records single optimization step."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        params = np.array([1.0, 2.0])
        callback(
            parameters=params,
            loss=0.5,
            prob_with=0.8,
            prob_without=0.2,
            contrast=0.6
        )
        
        assert callback.epoch == 1
        assert len(callback.history['epochs']) == 1
        assert callback.history['loss'][0] == 0.5
        assert callback.history['contrast'][0] == 0.6
        assert callback.history['prob_with'][0] == 0.8
        assert callback.history['prob_without'][0] == 0.2
        assert np.array_equal(callback.history['parameters'][0], params)
    
    def test_callback_save_every(self):
        """Test callback respects save_every parameter."""
        callback = OptimizationCallback(save_every=3, save_best=False)
        
        # Call 10 times
        for i in range(10):
            params = np.array([float(i), float(i+1)])
            callback(
                parameters=params,
                loss=float(i),
                prob_with=0.5,
                prob_without=0.3,
                contrast=0.2
            )
        
        assert callback.epoch == 10
        # Should save at epochs 3, 6, 9 (3 times)
        assert len(callback.history['epochs']) == 3
        assert callback.history['epochs'] == [3, 6, 9]
    
    def test_callback_best_tracking(self):
        """Test callback tracks best parameters correctly."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # First call - should be best
        callback(
            parameters=np.array([1.0, 2.0]),
            loss=1.0,
            prob_with=0.5,
            prob_without=0.3,
            contrast=0.2
        )
        
        assert callback.best_loss == 1.0
        assert np.array_equal(callback.best_parameters, np.array([1.0, 2.0]))
        
        # Second call - worse (higher loss)
        callback(
            parameters=np.array([2.0, 3.0]),
            loss=2.0,
            prob_with=0.6,
            prob_without=0.4,
            contrast=0.2
        )
        
        # Best should not change
        assert callback.best_loss == 1.0
        assert np.array_equal(callback.best_parameters, np.array([1.0, 2.0]))
        
        # Third call - better (lower loss)
        callback(
            parameters=np.array([3.0, 4.0]),
            loss=0.5,
            prob_with=0.9,
            prob_without=0.1,
            contrast=0.8
        )
        
        # Best should update
        assert callback.best_loss == 0.5
        assert np.array_equal(callback.best_parameters, np.array([3.0, 4.0]))
        assert callback.best_metrics['contrast'] == 0.8
    
    def test_get_best_parameters(self):
        """Test retrieving best parameters."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Initially None
        assert callback.get_best_parameters() is None
        
        # After recording
        params = np.array([1.5, 2.5])
        callback(
            parameters=params,
            loss=0.3,
            prob_with=0.7,
            prob_without=0.2,
            contrast=0.5
        )
        
        best = callback.get_best_parameters()
        assert best is not None
        assert np.array_equal(best, params)
    
    def test_get_best_metrics(self):
        """Test retrieving metrics at best parameters."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Initially None
        assert callback.get_best_metrics() is None
        
        # Record some steps
        callback(
            parameters=np.array([1.0, 2.0]),
            loss=0.5,
            prob_with=0.75,
            prob_without=0.25,
            contrast=0.50
        )
        
        metrics = callback.get_best_metrics()
        assert metrics is not None
        assert metrics['epoch'] == 1
        assert metrics['loss'] == 0.5
        assert metrics['contrast'] == 0.50
        assert metrics['prob_with'] == 0.75
        assert metrics['prob_without'] == 0.25
    
    def test_get_history(self):
        """Test retrieving full optimization history."""
        callback = OptimizationCallback(save_every=1, save_best=False)
        
        # Record 3 steps
        for i in range(3):
            callback(
                parameters=np.array([float(i), float(i+1)]),
                loss=float(i),
                prob_with=0.5 + i*0.1,
                prob_without=0.3,
                contrast=0.2 + i*0.1
            )
        
        history = callback.get_history()
        assert len(history['epochs']) == 3
        assert len(history['loss']) == 3
        assert len(history['contrast']) == 3
        assert len(history['parameters']) == 3
        assert history['loss'] == [0.0, 1.0, 2.0]
    
    def test_save_and_load(self):
        """Test saving and loading callback data."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Record some data
        for i in range(5):
            callback(
                parameters=np.array([float(i), float(i+1)]),
                loss=1.0 - i*0.1,  # Decreasing loss
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
            assert 'loss' in loaded_data
            assert 'contrast' in loaded_data
            assert 'prob_with' in loaded_data
            assert 'prob_without' in loaded_data
            assert 'parameters' in loaded_data
            assert 'best_parameters' in loaded_data
            assert 'best_loss' in loaded_data
            
            # Check values
            assert len(loaded_data['epochs']) == 5
            assert np.array_equal(loaded_data['epochs'], np.array([1, 2, 3, 4, 5]))
            # Best should be at epoch 5 (lowest loss)
            assert loaded_data['best_loss'] == 0.6
            assert loaded_data['best_epoch'] == 5
    
    def test_reset(self):
        """Test resetting callback state."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        # Record some data
        callback(
            parameters=np.array([1.0, 2.0]),
            loss=0.5,
            prob_with=0.7,
            prob_without=0.3,
            contrast=0.4
        )
        
        # Verify data exists
        assert callback.epoch == 1
        assert len(callback.history['epochs']) == 1
        assert callback.best_parameters is not None
        
        # Reset
        callback.reset()
        
        # Verify reset
        assert callback.epoch == 0
        assert len(callback.history['epochs']) == 0
        assert callback.best_parameters is None
        assert callback.best_loss == float('inf')
        assert callback.best_metrics is None
    
    def test_repr(self):
        """Test string representation."""
        callback = OptimizationCallback(save_every=2, save_best=True)
        
        # Initial state
        repr_str = repr(callback)
        assert 'epoch=0' in repr_str
        assert 'best_loss=None' in repr_str
        
        # After recording
        callback(
            parameters=np.array([1.0, 2.0]),
            loss=0.5,
            prob_with=0.7,
            prob_without=0.3,
            contrast=0.4
        )
        callback(
            parameters=np.array([1.5, 2.5]),
            loss=0.3,
            prob_with=0.8,
            prob_without=0.2,
            contrast=0.6
        )
        
        repr_str = repr(callback)
        assert 'epoch=2' in repr_str
        assert 'history_size=1' in repr_str  # save_every=2, so only saved once
        assert '0.3' in repr_str  # best_loss
    
    def test_callback_without_best_tracking(self):
        """Test callback with best tracking disabled."""
        callback = OptimizationCallback(save_every=1, save_best=False)
        
        # Record multiple steps
        for i in range(3):
            callback(
                parameters=np.array([float(i), float(i+1)]),
                loss=float(i),
                prob_with=0.5,
                prob_without=0.3,
                contrast=0.2
            )
        
        # Best parameters should not be tracked
        assert callback.best_parameters is None
        assert callback.best_loss == float('inf')
        
        # But history should still be recorded
        assert len(callback.history['epochs']) == 3
    
    def test_parameter_array_copy(self):
        """Test that parameters are copied, not referenced."""
        callback = OptimizationCallback(save_every=1, save_best=True)
        
        params = np.array([1.0, 2.0])
        callback(
            parameters=params,
            loss=0.5,
            prob_with=0.7,
            prob_without=0.3,
            contrast=0.4
        )
        
        # Modify original array
        params[0] = 999.0
        
        # Callback should have a copy
        assert callback.history['parameters'][0][0] == 1.0
        assert callback.best_parameters[0] == 1.0
    
    def test_save_without_best_parameters(self):
        """Test saving when no best parameters are tracked."""
        callback = OptimizationCallback(save_every=1, save_best=False)
        
        callback(
            parameters=np.array([1.0, 2.0]),
            loss=0.5,
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
            assert 'best_loss' not in loaded_data
            
            # But should have history
            assert 'epochs' in loaded_data
            assert 'loss' in loaded_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

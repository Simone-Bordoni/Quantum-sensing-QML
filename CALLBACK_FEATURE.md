# OptimizationCallback Feature Summary

## Overview
Moved the callback functionality from `trainable_parameters.py` to a dedicated `callback.py` module with enhanced capabilities for tracking optimization metrics in quantum sensing experiments.

## New File: `src/qsopt/core/callback.py`

### OptimizationCallback Class

A comprehensive callback for tracking optimization progress with detailed metrics.

**Key Features:**
- Tracks loss function values at each epoch
- Records detection probabilities (with and without photon)
- Monitors sensing contrast evolution
- Saves parameter values throughout optimization
- Automatically tracks best parameters and metrics
- Save/load functionality using NumPy NPZ format

**Constructor:**
```python
OptimizationCallback(save_every=1, save_best=True)
```

**Parameters:**
- `save_every` (int): Save history every N epochs (default: 1)
- `save_best` (bool): Track best parameters based on loss (default: True)

**Methods:**

1. `__call__(parameters, loss, prob_with, prob_without, contrast)`
   - Records metrics from current optimization step
   - Automatically updates best tracking if enabled

2. `get_best_parameters()` → Optional[np.ndarray]
   - Returns best parameters found during optimization

3. `get_best_metrics()` → Optional[Dict[str, float]]
   - Returns metrics (epoch, loss, contrast, probabilities) at best parameters

4. `get_history()` → Dict[str, List[Any]]
   - Returns complete optimization history

5. `save(filepath: str)`
   - Saves results to NPZ file with all arrays and best parameters
   - File includes: epochs, loss, contrast, prob_with, prob_without, parameters, best_*

6. `load(filepath: str)` [static method]
   - Loads optimization results from NPZ file
   - Returns dictionary with all saved arrays

7. `reset()`
   - Resets callback to initial state

**Tracked Metrics:**
- `epochs`: List of epoch numbers where data was saved
- `loss`: Loss function values
- `contrast`: Sensing contrast (prob_with - prob_without)
- `prob_with`: Detection probability with photon interaction
- `prob_without`: Detection probability without photon interaction
- `parameters`: Parameter values as arrays
- `best_parameters`: Best parameters found (if save_best=True)
- `best_loss`: Best loss value
- `best_metrics`: Full metrics at best parameters

## Integration with SingleQubitExperiment

**Updated `experiment.py`:**
- Added optional `callback` parameter to `optimize()` method
- Callback is invoked after each optimization step with current metrics
- No changes required if callback not used (backward compatible)

**Usage Example:**
```python
from qsopt import SingleQubitExperiment, OptimizationCallback

# Create callback
callback = OptimizationCallback(save_every=1, save_best=True)

# Run optimization with callback
history = experiment.optimize(
    num_steps=100,
    learning_rate=0.05,
    verbose=True,
    callback=callback
)

# Access best results
best_params = callback.get_best_parameters()
best_metrics = callback.get_best_metrics()
print(f"Best contrast: {best_metrics['contrast']:.6f}")

# Save for later analysis
callback.save('results.npz')
```

## Example Notebook Updates

**New sections in `examples/Example.ipynb`:**

1. **Setup Callback** - Initialize callback with desired settings
2. **Run Optimization** - Pass callback to optimize() method
3. **Visualize Optimization Progress** - 4-panel plot showing:
   - Contrast evolution
   - Loss function evolution
   - Detection probabilities (with/without photon)
   - Parameter trajectories
4. **Save and Load Callback Data** - Demonstrate persistence

**Visualization Example:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot contrast
axes[0, 0].plot(callback.history['epochs'], callback.history['contrast'])
axes[0, 0].set_title('Sensing Contrast Evolution')

# Plot loss
axes[0, 1].plot(callback.history['epochs'], callback.history['loss'])
axes[0, 1].set_title('Loss Function Evolution')

# Plot probabilities
axes[1, 0].plot(callback.history['epochs'], callback.history['prob_with'], label='P(with)')
axes[1, 0].plot(callback.history['epochs'], callback.history['prob_without'], label='P(without)')
axes[1, 0].legend()

# Plot parameters
params = np.array(callback.history['parameters'])
axes[1, 1].plot(callback.history['epochs'], params[:, 0], label='θ₁')
axes[1, 1].plot(callback.history['epochs'], params[:, 1], label='θ₂')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

## Test Coverage

**New file: `src/qsopt/tests/test_callback.py`**

Comprehensive test suite with 15 test cases covering:

1. ✓ Callback initialization with default values
2. ✓ Recording single optimization step
3. ✓ Respecting save_every parameter
4. ✓ Best parameter tracking (updates when loss improves)
5. ✓ get_best_parameters() method
6. ✓ get_best_metrics() method
7. ✓ get_history() method
8. ✓ Save and load functionality with NPZ format
9. ✓ Reset functionality
10. ✓ String representation (__repr__)
11. ✓ Callback without best tracking
12. ✓ Parameter array copying (not referencing)
13. ✓ Save without best parameters
14. ✓ Load with verification of all expected keys
15. ✓ Integration scenarios

**Removed:**
- Old `ParameterCallback` class from `trainable_parameters.py`
- Related tests from `test_trainable_parameters.py`

## Documentation Updates

**README.md additions:**
- New section "Using Optimization Callbacks" after Quick Start
- Complete usage example with callback
- List of tracked metrics
- Save/load example with matplotlib plotting

## Package Exports

**Updated `src/qsopt/__init__.py`:**
```python
from .core.callback import OptimizationCallback
```

Now users can import directly:
```python
from qsopt import OptimizationCallback
```

## File Changes Summary

**New Files:**
- `src/qsopt/core/callback.py` (238 lines)
- `src/qsopt/tests/test_callback.py` (366 lines)

**Modified Files:**
- `src/qsopt/core/experiment.py` - Added callback parameter and invocation
- `src/qsopt/core/trainable_parameters.py` - Removed old ParameterCallback
- `src/qsopt/__init__.py` - Added OptimizationCallback export
- `examples/Example.ipynb` - Added 4 new cells with callback usage
- `README.md` - Added callback documentation section
- `src/qsopt/tests/test_trainable_parameters.py` - Removed old callback tests

## Benefits

1. **Better Organization**: Callback logic separated into dedicated module
2. **Enhanced Metrics**: Tracks loss, contrast, and probabilities (not just parameters)
3. **Easy Analysis**: NPZ format allows easy loading and plotting
4. **Backward Compatible**: No breaking changes to existing code
5. **Well Tested**: 15+ test cases ensure reliability
6. **Well Documented**: README, docstrings, and example notebook
7. **Visualization Ready**: Structured data format perfect for matplotlib

## Usage Patterns

### Basic Usage
```python
callback = OptimizationCallback()
history = experiment.optimize(num_steps=50, callback=callback)
best = callback.get_best_parameters()
```

### Save for Later Analysis
```python
callback.save('optimization_20250107.npz')
# Later...
data = OptimizationCallback.load('optimization_20250107.npz')
plt.plot(data['epochs'], data['contrast'])
```

### Multiple Runs Comparison
```python
callbacks = []
for lr in [0.01, 0.05, 0.1]:
    cb = OptimizationCallback()
    experiment.optimize(learning_rate=lr, callback=cb)
    callbacks.append(cb)
    
# Compare learning rates
for i, cb in enumerate(callbacks):
    plt.plot(cb.history['epochs'], cb.history['contrast'], 
             label=f'lr={[0.01, 0.05, 0.1][i]}')
```

## Commit Information

**Commit Message:**
```
Add OptimizationCallback for tracking optimization metrics

Features:
- Track loss, contrast, detection probabilities at each epoch
- Automatic best parameter tracking
- Save/load results to NPZ format for easy plotting
- Integration with SingleQubitExperiment.optimize()
- Comprehensive test suite

Updates:
- Created src/qsopt/core/callback.py with OptimizationCallback class
- Added callback parameter to experiment.optimize() method
- Updated Example.ipynb with callback usage and visualization
- Created src/qsopt/tests/test_callback.py with full test coverage
- Updated README.md with callback documentation and examples
- Exported OptimizationCallback from qsopt package
```

**Commit Hash:** `9db9474`

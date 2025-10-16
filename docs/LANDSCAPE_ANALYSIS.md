# Landscape Analysis Module Documentation

## Overview

The `landscape_analysis.py` module provides functions for systematically exploring parameter spaces to find optimal configurations for quantum sensing experiments. It enables 2D parameter landscape analysis (rotation angles) and 1D time interval landscape analysis.

## Location

- **Module**: `src/qsopt/utils/landscape_analysis.py`
- **Import**: `from qsopt.utils.landscape_analysis import compute_theta1_theta2_landscape, compute_time_interval_landscape`

## Main Functions

### 1. `compute_theta1_theta2_landscape()`

Computes a 2D landscape of sensing contrast across rotation angle parameter space (θ₁, θ₂).

**Purpose:**
- Visualize how contrast varies with rotation angles
- Identify optimal parameter regions
- Understand parameter sensitivity
- Guide initial parameter selection for optimization

**Function Signature:**
```python
def compute_theta1_theta2_landscape(
    exp_params: ExperimentalParameters,
    theta1_range: tuple = (-np.pi, np.pi),
    theta2_range: tuple = (-np.pi, np.pi),
    resolution: int = 20,
    batch_size: int = 1,
    verbose: bool = True
) -> Dict[str, Any]
```

**Arguments:**
- `exp_params` (ExperimentalParameters): System configuration
- `theta1_range` (tuple, default=(-π, π)): Range for first rotation angle [min, max]
- `theta2_range` (tuple, default=(-π, π)): Range for second rotation angle [min, max]
- `resolution` (int, default=20): Number of points per dimension (creates resolution×resolution grid)
- `batch_size` (int, default=1): Number of realizations to average (for uncertainty analysis)
  - Set > 1 when `initial_time_uncertainty` > 0 to average over multiple initial times
- `verbose` (bool, default=True): Print progress information

**Returns:**
Dictionary containing:
- `'theta1_vals'` (np.ndarray): 1D array of θ₁ values, shape (resolution,)
- `'theta2_vals'` (np.ndarray): 1D array of θ₂ values, shape (resolution,)
- `'contrast_map'` (np.ndarray): 2D contrast values, shape (resolution, resolution)
- `'detection_map'` (np.ndarray): 2D P(detect|with photon), shape (resolution, resolution)
- `'batch_size'` (int): Number of realizations averaged
- `'uncertainty'` (float): Initial time uncertainty value

**Example 1: Basic Landscape**
```python
from qsopt import *
from qsopt.utils.landscape_analysis import compute_theta1_theta2_landscape
from qsopt.utils.visualization import plot_parameter_landscape

# Setup experiment
exp_params = ExperimentalParameters(...)

# Compute landscape
data = compute_theta1_theta2_landscape(
    exp_params,
    theta1_range=(-np.pi, np.pi),
    theta2_range=(-np.pi, np.pi),
    resolution=30,
    verbose=True
)

# Visualize
fig = plot_parameter_landscape(
    data['theta1_vals'],
    data['theta2_vals'],
    data['contrast_map'],
    data['detection_map'],
    exp_params,
    save_path='parameter_landscape.png'
)
```

**Example 2: Landscape with Uncertainty Averaging**
```python
# Configure measurement with uncertainty
measurement = MeasurementProtocol(
    initial_time=-5.0,
    final_time=5.0,
    time_interval=1.0,
    initial_time_uncertainty=0.2  # ±0.2 time units
)

exp_params = ExperimentalParameters(
    physical_constants=...,
    measurement=measurement
)

# Compute landscape with batch averaging
data = compute_theta1_theta2_landscape(
    exp_params,
    resolution=25,
    batch_size=10,  # Average over 10 realizations
    verbose=True
)

# Results are averaged over uncertainty
print(f"Averaged over {data['batch_size']} realizations")
print(f"Uncertainty: ±{data['uncertainty']:.4f}")
```

**Performance Considerations:**
- Computation time scales as O(resolution² × batch_size)
- For resolution=20, batch_size=1: ~1-2 minutes
- For resolution=30, batch_size=10: ~15-30 minutes
- Consider using lower resolution for initial exploration

---

### 2. `compute_time_interval_landscape()`

Computes a 1D landscape of sensing contrast across different measurement time intervals.

**Purpose:**
- Find optimal measurement timing spacing
- Analyze temporal resolution requirements
- Understand trade-off between measurement density and contrast
- Guide measurement protocol design

**Function Signature:**
```python
def compute_time_interval_landscape(
    exp_params: ExperimentalParameters,
    theta1: float,
    theta2: float,
    resolution: int = 20,
    mode: str = 'continuous',
    batch_size: int = 1,
    verbose: bool = True
) -> Dict[str, Any]
```

**Arguments:**
- `exp_params` (ExperimentalParameters): System configuration
- `theta1` (float): First rotation angle (radians)
- `theta2` (float): Second rotation angle (radians)
- `resolution` (int, default=20): Number of time interval values to test
- `mode` (str, default='continuous'): Sampling mode
  - `'continuous'`: Linearly spaced intervals from T/100 to T
  - `'discrete'`: Integer fractions T/N for N = 1, 2, ..., resolution
- `batch_size` (int, default=1): Number of realizations to average
- `verbose` (bool, default=True): Print progress information

**Returns:**
Dictionary containing:
- `'interval_vals'` (np.ndarray): Time interval values tested, shape (resolution,)
- `'contrast_vals'` (np.ndarray): Contrast at each interval, shape (resolution,)
- `'detection_with'` (np.ndarray): P(detect|with photon), shape (resolution,)
- `'detection_without'` (np.ndarray): P(detect|without photon), shape (resolution,)
- `'n_measurements'` (np.ndarray): Number of measurements at each interval, shape (resolution,)
- `'mode'` (str): Sampling mode used
- `'batch_size'` (int): Number of realizations averaged
- `'uncertainty'` (float): Initial time uncertainty value

**Example 1: Continuous Mode Landscape**
```python
from qsopt.utils.landscape_analysis import compute_time_interval_landscape
from qsopt.utils.visualization import plot_time_interval_landscape

# Compute landscape in continuous mode
data = compute_time_interval_landscape(
    exp_params,
    theta1=np.pi/2,
    theta2=-np.pi/2,
    resolution=30,
    mode='continuous',
    verbose=True
)

# Visualize with 3-panel plot
fig = plot_time_interval_landscape(
    data['interval_vals'],
    data['contrast_vals'],
    data['detection_with'],
    data['detection_without'],
    data['n_measurements'],
    exp_params,
    theta1=np.pi/2,
    theta2=-np.pi/2,
    mode='continuous',
    save_path='time_interval_landscape.png'
)

# Find optimal interval
optimal_idx = np.argmax(data['contrast_vals'])
optimal_interval = data['interval_vals'][optimal_idx]
optimal_contrast = data['contrast_vals'][optimal_idx]
optimal_n_meas = data['n_measurements'][optimal_idx]

print(f"Optimal interval: {optimal_interval:.6f}")
print(f"Contrast achieved: {optimal_contrast:.6f}")
print(f"Number of measurements: {optimal_n_meas}")
```

**Example 2: Discrete Mode Landscape**
```python
# Compute landscape in discrete mode
# Tests intervals: T/1, T/2, T/3, ..., T/resolution
data = compute_time_interval_landscape(
    exp_params,
    theta1=1.5,
    theta2=-1.3,
    resolution=20,
    mode='discrete',  # Integer fraction mode
    verbose=True
)

# Analyze results
print(f"Mode: {data['mode']}")
print(f"Intervals tested: {len(data['interval_vals'])}")
print(f"Contrast range: [{data['contrast_vals'].min():.4f}, {data['contrast_vals'].max():.4f}]")
```

**Example 3: Landscape with Uncertainty**
```python
# Configure with initial time uncertainty
measurement = MeasurementProtocol(
    initial_time=-5.0,
    final_time=5.0,
    time_interval=1.0,
    initial_time_uncertainty=0.15
)

exp_params = ExperimentalParameters(
    measurement=measurement,
    ...
)

# Compute with batch averaging
data = compute_time_interval_landscape(
    exp_params,
    theta1=np.pi/4,
    theta2=-np.pi/4,
    resolution=25,
    mode='continuous',
    batch_size=15,  # Average over 15 realizations
    verbose=True
)

# Results account for timing uncertainty
print(f"Uncertainty-aware results (batch_size={data['batch_size']})")
```

**Mode Comparison:**

| Mode | Intervals Tested | Use Case |
|------|------------------|----------|
| `continuous` | Linearly spaced from T/100 to T | General exploration, smooth landscape |
| `discrete` | T/1, T/2, T/3, ..., T/N | Integer measurement counts, hardware constraints |

**Performance Considerations:**
- Computation time scales as O(resolution × batch_size)
- For resolution=20, batch_size=1: ~30-60 seconds
- For resolution=30, batch_size=10: ~5-10 minutes

---

## Integration Example

Complete workflow combining both landscape analyses:

```python
from qsopt import *
from qsopt.utils.landscape_analysis import (
    compute_theta1_theta2_landscape,
    compute_time_interval_landscape
)
from qsopt.utils.visualization import (
    plot_parameter_landscape,
    plot_time_interval_landscape
)
import numpy as np

# 1. Setup experiment
exp_params = ExperimentalParameters(
    physical_constants=PhysicalConstants(
        chi=0.01,
        photon_cavity_coupling=0.1,
        inverse_pulse_width=0.1
    ),
    measurement=MeasurementProtocol(
        initial_time=-5.0,
        final_time=5.0,
        time_interval=1.0
    )
)

# 2. Explore rotation angle parameter space
print("Computing rotation angle landscape...")
param_data = compute_theta1_theta2_landscape(
    exp_params,
    resolution=25,
    verbose=True
)

# Find best rotation angles from landscape
contrast_map = param_data['contrast_map']
best_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
best_theta1 = param_data['theta1_vals'][best_idx[0]]
best_theta2 = param_data['theta2_vals'][best_idx[1]]

print(f"Best angles from landscape: θ₁={best_theta1:.4f}, θ₂={best_theta2:.4f}")

# Visualize parameter landscape
fig1 = plot_parameter_landscape(
    param_data['theta1_vals'],
    param_data['theta2_vals'],
    param_data['contrast_map'],
    param_data['detection_map'],
    exp_params,
    save_path='results/parameter_landscape.png'
)

# 3. Optimize measurement timing with best angles
print("\nComputing time interval landscape...")
timing_data = compute_time_interval_landscape(
    exp_params,
    theta1=best_theta1,
    theta2=best_theta2,
    resolution=30,
    mode='continuous',
    verbose=True
)

# Find optimal time interval
optimal_idx = np.argmax(timing_data['contrast_vals'])
optimal_interval = timing_data['interval_vals'][optimal_idx]

print(f"Optimal time interval: {optimal_interval:.6f}")

# Visualize timing landscape
fig2 = plot_time_interval_landscape(
    timing_data['interval_vals'],
    timing_data['contrast_vals'],
    timing_data['detection_with'],
    timing_data['detection_without'],
    timing_data['n_measurements'],
    exp_params,
    theta1=best_theta1,
    theta2=best_theta2,
    mode='continuous',
    save_path='results/time_interval_landscape.png'
)

# 4. Run optimization with optimized initial conditions
train_params = TrainableParameters()
train_params.add_rotation_angles(['theta1', 'theta2'], [best_theta1, best_theta2])
train_params.add_measurement_interval('time_interval', optimal_interval)

exp_params.measurement.time_interval = optimal_interval
experiment = SingleQubitExperiment(exp_params, train_params)

history = experiment.optimize(
    theta_init=[best_theta1, best_theta2, optimal_interval],
    num_steps=100,
    verbose=True
)

print(f"Final optimized contrast: {history.best_contrast:.6f}")
```

## Best Practices

### 1. Resolution Selection
- **Initial exploration**: resolution=15-20
- **Fine-grained analysis**: resolution=30-50
- **Publication quality**: resolution=50-100 (time-intensive)

### 2. Batch Sizing for Uncertainty
- Use `batch_size > 1` only when `initial_time_uncertainty > 0`
- Recommended: batch_size = 10-20 for uncertainty averaging
- Higher batch sizes give smoother results but increase computation time

### 3. Mode Selection for Time Intervals
- Use `'continuous'` for general exploration and smooth landscapes
- Use `'discrete'` when hardware requires integer measurement counts
- Discrete mode ensures all intervals are integer fractions of total time

### 4. Computational Efficiency
- Start with low resolution for parameter exploration
- Increase resolution only for regions of interest
- Use batch_size=1 initially, add uncertainty analysis later
- Consider running on multiple cores for high-resolution landscapes

### 5. Interpretation
- **Parameter landscapes**: Look for global maxima and local structure
- **Time interval landscapes**: Balance between measurement density and contrast
- **Uncertainty analysis**: Broader peaks are more robust to timing jitter

## Troubleshooting

### Issue: Landscape computation is too slow
**Solution:**
- Reduce resolution (try 10-15 for initial testing)
- Set batch_size=1 initially
- Use discrete mode for time intervals (fewer computations)

### Issue: Landscape shows no clear optimum
**Solution:**
- Check parameter ranges (may need to expand)
- Verify experimental parameters are physical
- Try different rotation angle ranges
- Check if noise is too high

### Issue: Results vary significantly between runs
**Solution:**
- Increase batch_size for uncertainty averaging
- Check if initial_time_uncertainty is set appropriately
- Verify random seed is set for reproducibility

## See Also

- [Visualization Module](VISUALIZATION_MODULE.md) - For plotting landscapes
- [Experimental Parameters](experimental_parameters.md) - For configuring experiments
- [Optimization Guide](optimization.md) - For using landscape results to seed optimization

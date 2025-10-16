# Pulse Shape Visualization Documentation

## Overview

The pulse shape visualization feature provides tools to visualize the Gaussian input pulse envelope alongside the measurement protocol timing. This helps understand the temporal relationship between the pulse shape and when measurements are performed.

## Location

- **Module**: `src/qsopt/core/quantum_utils.py` (u0 function)
- **Module**: `src/qsopt/utils/visualization.py` (plotting function)
- **Import**: 
```python
from qsopt.core.quantum_utils import u0
from qsopt.utils.visualization import plot_pulse_shape_with_measurements
```

## Functions

### 1. `u0()` - Gaussian Pulse Envelope

Simple Gaussian pulse envelope function for visualization purposes.

**Function Signature:**
```python
def u0(t, **kwargs) -> jnp.ndarray
```

**Arguments:**
- `t` (float or JAX array): Time variable(s)
- `**kwargs`: Keyword arguments
  - `sigma` (float, default=0.1): Inverse pulse width (bandwidth parameter)

**Returns:**
- JAX array: Gaussian pulse amplitude at time t, normalized to 1.0 at t=0

**Formula:**
```
u₀(t) = exp(-σ²t²)
```

**Example:**
```python
from qsopt.core.quantum_utils import u0
import numpy as np

# Single time point
sigma = 0.1
amplitude_at_zero = u0(0.0, sigma=sigma)  # Returns 1.0

# Array of times
t_vals = np.linspace(-10, 10, 100)
pulse = u0(t_vals, sigma=sigma)

# Plot the pulse
import matplotlib.pyplot as plt
plt.plot(t_vals, pulse)
plt.xlabel('Time')
plt.ylabel('Pulse Amplitude')
plt.title(f'Gaussian Pulse (σ = {sigma})')
plt.show()
```

**Comparison with `gu()` function:**

| Function | Purpose | Formula | Use Case |
|----------|---------|---------|----------|
| `u0()` | Visualization | exp(-σ²t²) | Plotting pulse shape, understanding envelope |
| `gu()` | Physics simulation | Complex with erfc | Actual quantum evolution calculations |

- **`u0()`**: Simple Gaussian for visualization and intuition
- **`gu()`**: Physically accurate coupling including causality (erfc normalization)

---

### 2. `plot_pulse_shape_with_measurements()` - Visualization Function

Creates a comprehensive visualization showing the Gaussian pulse envelope with measurement time markers.

**Function Signature:**
```python
def plot_pulse_shape_with_measurements(
    exp_params: ExperimentalParameters,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure
```

**Arguments:**
- `exp_params` (ExperimentalParameters): System configuration including:
  - `inverse_pulse_width` (σ): Pulse bandwidth
  - `measurement.initial_time`: Start of measurement window
  - `measurement.final_time`: End of measurement window
  - `measurement.time_interval`: Spacing between measurements
- `save_path` (str, optional): Path to save figure (e.g., 'pulse_shape.png')
- `dpi` (int, default=300): Resolution for saved figure

**Returns:**
- matplotlib.figure.Figure: Figure object with the pulse shape plot

**Visualization Elements:**
1. **Blue curve**: Gaussian pulse envelope |u₀(t)|
   - Peak amplitude of 1.0 at t=0
   - Width determined by σ (inverse_pulse_width)
2. **Red dashed lines**: Vertical markers at each measurement time
   - Shows when photon detection measurements occur
3. **Green shaded area**: Measurement window (initial_time to final_time)
   - Indicates the active measurement period
4. **Yellow info box**: System configuration details
   - Physical parameters (σ)
   - Measurement protocol (times, intervals, count)

**Example 1: Basic Usage**
```python
from qsopt import *
from qsopt.utils.visualization import plot_pulse_shape_with_measurements

# Setup experiment
exp_params = ExperimentalParameters(
    physical_constants=PhysicalConstants(
        inverse_pulse_width=0.1
    ),
    measurement=MeasurementProtocol(
        initial_time=-5.0,
        final_time=5.0,
        time_interval=1.0
    )
)

# Visualize pulse shape with measurements
fig = plot_pulse_shape_with_measurements(
    exp_params,
    save_path='pulse_shape.png',
    dpi=300
)
plt.show()
```

**Example 2: Using Experiment Method**
```python
from qsopt.core.experiment import SingleQubitExperiment

# Create experiment
experiment = SingleQubitExperiment(exp_params, train_params)

# Convenient method that uses current experiment parameters
fig = experiment.plot_pulse_shape(save_path='results/pulse.png')
```

**Example 3: Comparing Different Measurement Protocols**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

protocols = [
    ('Dense', MeasurementProtocol(initial_time=-5, final_time=5, time_interval=0.5)),
    ('Medium', MeasurementProtocol(initial_time=-5, final_time=5, time_interval=1.0)),
    ('Sparse', MeasurementProtocol(initial_time=-5, final_time=5, time_interval=2.0)),
]

for ax, (name, protocol) in zip(axes, protocols):
    exp_params.measurement = protocol
    
    fig_temp = plot_pulse_shape_with_measurements(exp_params)
    # Extract and replot on subplot
    plt.sca(ax)
    ax.set_title(f'{name} Sampling ({protocol.time_interval})')

plt.tight_layout()
plt.savefig('measurement_protocol_comparison.png', dpi=300)
```

---

## Integration with Experiment Class

The `SingleQubitExperiment` class provides a convenience method:

### `experiment.plot_pulse_shape()`

**Method Signature:**
```python
def plot_pulse_shape(
    self,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure
```

**Example:**
```python
from qsopt import *

# Create experiment with your parameters
experiment = SingleQubitExperiment(exp_params, train_params)

# Plot pulse shape using current experiment configuration
fig = experiment.plot_pulse_shape(
    save_path='results/my_pulse.png',
    dpi=300
)
```

This method automatically uses:
- Current experimental parameters
- Current measurement protocol
- All system configuration

---

## Use Cases

### 1. Protocol Design
Visualize how many measurements fall within the pulse envelope to optimize detection efficiency.

```python
# Try different intervals
for interval in [0.5, 1.0, 2.0]:
    exp_params.measurement.time_interval = interval
    fig = plot_pulse_shape_with_measurements(
        exp_params,
        save_path=f'pulse_interval_{interval}.png'
    )
```

### 2. Pulse Width Analysis
Understand how pulse width affects temporal coverage.

```python
# Compare different pulse widths
for sigma in [0.05, 0.1, 0.2]:
    exp_params.physical_constants.inverse_pulse_width = sigma
    fig = plot_pulse_shape_with_measurements(
        exp_params,
        save_path=f'pulse_sigma_{sigma}.png'
    )
```

### 3. Pre-Optimization Planning
Before running expensive simulations, visualize the measurement setup.

```python
# Setup measurement protocol
measurement = MeasurementProtocol(
    initial_time=-10.0,
    final_time=10.0,
    time_interval=1.5
)

exp_params = ExperimentalParameters(
    physical_constants=PhysicalConstants(inverse_pulse_width=0.08),
    measurement=measurement
)

# Check if measurement times align well with pulse
fig = plot_pulse_shape_with_measurements(exp_params)
plt.show()

# Adjust if needed based on visualization
# Then proceed with optimization
```

### 4. Publication Figures
Generate publication-quality plots showing measurement protocols.

```python
fig = plot_pulse_shape_with_measurements(
    exp_params,
    save_path='figures/pulse_protocol.pdf',
    dpi=600  # High resolution for publication
)
```

---

## Interpretation Guide

### Understanding the Plot

**1. Pulse Envelope (Blue Curve)**
- Shows the normalized amplitude of the input Gaussian pulse
- Peak at t=0 indicates pulse center
- Width inversely proportional to σ (inverse_pulse_width)
- Wider σ → narrower pulse (higher bandwidth)
- Narrower σ → wider pulse (lower bandwidth)

**2. Measurement Markers (Red Dashed Lines)**
- Each line marks when a photon detection measurement occurs
- Spacing determined by `time_interval` parameter
- More measurements → denser markers
- Fewer measurements → sparser markers

**3. Measurement Window (Green Shaded Area)**
- Region between `initial_time` and `final_time`
- Measurements only occur within this window
- Pulse extends beyond window to show full decay

**4. System Info Box (Yellow)**
- **σ**: Inverse pulse width (higher = narrower pulse)
- **Initial/Final time**: Measurement window boundaries
- **Time interval**: Spacing between measurements
- **Number of measurements**: Total measurement count

### Design Recommendations

**Good Protocol:**
- Measurements concentrated where pulse amplitude is significant
- At least 3-5 measurements during peak pulse envelope
- Measurement window covers pulse FWHM (Full Width at Half Maximum)

**Poor Protocol:**
- Too sparse: Missing pulse dynamics
- Too dense: Redundant measurements, increased noise
- Window too narrow: Missing pulse tails
- Window too wide: Wasting measurements in low-signal regions

---

## Integration with Landscape Analysis

Use pulse shape visualization together with time interval landscape analysis:

```python
from qsopt.utils.landscape_analysis import compute_time_interval_landscape
from qsopt.utils.visualization import (
    plot_time_interval_landscape,
    plot_pulse_shape_with_measurements
)

# 1. Find optimal interval from landscape
data = compute_time_interval_landscape(
    exp_params,
    theta1=np.pi/2,
    theta2=-np.pi/2,
    resolution=30,
    mode='continuous'
)

optimal_idx = np.argmax(data['contrast_vals'])
optimal_interval = data['interval_vals'][optimal_idx]

# 2. Visualize landscape
fig1 = plot_time_interval_landscape(
    data['interval_vals'],
    data['contrast_vals'],
    data['detection_with'],
    data['detection_without'],
    data['n_measurements'],
    exp_params,
    theta1=np.pi/2,
    theta2=-np.pi/2,
    mode='continuous',
    save_path='time_landscape.png'
)

# 3. Update experiment with optimal interval
exp_params.measurement.time_interval = optimal_interval

# 4. Visualize what the optimized protocol looks like
fig2 = plot_pulse_shape_with_measurements(
    exp_params,
    save_path='optimized_pulse_shape.png'
)

print(f"Optimal interval: {optimal_interval:.4f}")
print(f"Number of measurements: {len(exp_params.measurement_times)}")
```

---

## Best Practices

### 1. Choose Appropriate Time Range
```python
# Ensure measurement window covers significant pulse amplitude
# Rule of thumb: ±(5-10)/σ around pulse center
sigma = exp_params.inverse_pulse_width
initial_time = -7.0 / sigma
final_time = 7.0 / sigma
```

### 2. Match Interval to Pulse Width
```python
# Nyquist-style sampling: at least 2-3 samples per pulse width
pulse_width = 1.0 / sigma  # Approximate FWHM
recommended_interval = pulse_width / 3.0
```

### 3. High-Resolution Figures
```python
# For presentations and publications
fig = plot_pulse_shape_with_measurements(
    exp_params,
    save_path='pulse_hires.pdf',  # Use PDF for vector graphics
    dpi=600
)
```

### 4. Batch Comparisons
```python
# Compare multiple configurations in a grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

configs = [
    (0.05, 1.0),  # (sigma, interval)
    (0.1, 1.0),
    (0.05, 0.5),
    (0.1, 0.5),
]

for ax, (sigma, interval) in zip(axes.flat, configs):
    exp_params.physical_constants.inverse_pulse_width = sigma
    exp_params.measurement.time_interval = interval
    # Plot each configuration
    ...
```

---

## Examples

### Complete Workflow Example

```python
from qsopt import *
from qsopt.utils.visualization import plot_pulse_shape_with_measurements
import numpy as np

# 1. Define physical system
physical_constants = PhysicalConstants(
    chi=0.01,
    photon_cavity_coupling=0.1,
    inverse_pulse_width=0.1  # σ = 0.1
)

# 2. Setup measurement protocol
measurement = MeasurementProtocol(
    initial_time=-8.0,
    final_time=8.0,
    time_interval=1.2
)

# 3. Create experimental parameters
exp_params = ExperimentalParameters(
    physical_constants=physical_constants,
    measurement=measurement
)

# 4. Visualize pulse and measurement timing
fig = plot_pulse_shape_with_measurements(
    exp_params,
    save_path='results/pulse_visualization.png',
    dpi=300
)

# 5. Analyze from plot
n_measurements = len(exp_params.measurement_times)
print(f"Number of measurements: {n_measurements}")
print(f"Measurement times: {exp_params.measurement_times}")

# 6. Iterate if needed
# Adjust interval based on visual inspection
exp_params.measurement.time_interval = 0.8
fig2 = plot_pulse_shape_with_measurements(
    exp_params,
    save_path='results/pulse_adjusted.png'
)
```

---

## Troubleshooting

### Issue: Pulse looks too wide/narrow
**Solution:** Adjust `inverse_pulse_width` (σ)
```python
exp_params.physical_constants.inverse_pulse_width = new_sigma
```

### Issue: Not enough measurements in pulse region
**Solution:** Decrease `time_interval` or expand measurement window
```python
exp_params.measurement.time_interval = smaller_value
# or
exp_params.measurement.initial_time = earlier_time
exp_params.measurement.final_time = later_time
```

### Issue: Too many measurements (cluttered plot)
**Solution:** Increase `time_interval` for sparser sampling
```python
exp_params.measurement.time_interval = larger_value
```

### Issue: Measurements outside pulse envelope
**Solution:** Adjust measurement window to center on pulse
```python
# Center window on pulse (typically at t=0)
exp_params.measurement.initial_time = -window_size/2
exp_params.measurement.final_time = window_size/2
```

---

## See Also

- [Landscape Analysis](LANDSCAPE_ANALYSIS.md) - For optimizing time intervals
- [Visualization Module](VISUALIZATION_MODULE.md) - For other plotting functions
- [Experimental Parameters](experimental_parameters.md) - For configuring measurements
- [Quantum Utils](quantum_utils.md) - For other quantum utility functions

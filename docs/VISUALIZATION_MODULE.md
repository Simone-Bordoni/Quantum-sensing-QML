# Visualization Module Documentation

## Overview

The new `visualization.py` module provides comprehensive plotting functions for quantum sensing optimization results. It creates dashboards similar to those in the `Optimization_with_noise.ipynb` notebook.

## Location

- **Module**: `src/qsopt/utils/visualization.py`
- **Import**: `from qsopt.utils.visualization import plot_optimization_dashboard`

## Main Functions

### 1. `plot_optimization_dashboard()`

Creates a comprehensive multi-panel dashboard with customizable plots.

**Features:**
- **Sensing Contrast Evolution**: Shows how the optimization objective improves over epochs
- **Gradient Magnitude Evolution**: Log-scale plot of gradient norms (useful for convergence analysis)
- **Parameter Evolution**: Tracks all rotation angles throughout optimization
- **Optimization Trajectory**: 2D visualization of the path through parameter space
- **Detection Probabilities**: Evolution of P(with photon) and P(without photon)

**Arguments:**
- `optimization_callback` (required): `OptimizationCallback` from `experiment.optimize()`
- `reference_callback` (optional): `OptimizationCallback` from `experiment.run_simulation()`
  - When provided, reference values are shown as horizontal benchmark lines
- `show_contrast` (bool, default=True): Display contrast evolution plot
- `show_gradients` (bool, default=True): Display gradient evolution plot
- `show_parameters` (bool, default=True): Display parameter evolution plot
- `show_trajectory` (bool, default=True): Display trajectory in parameter space
- `show_probabilities` (bool, default=True): Display detection probabilities plot
- `figsize` (tuple, default=(16, 18)): Figure size in inches
- `save_path` (str, optional, default=None): Path to save figure (e.g., 'opt_dashboard.pdf')
- `dpi` (int, default=300): Resolution for saved figure

**Returns:**
- `matplotlib.figure.Figure`: The created figure object

**Example:**
```python
from qsopt import *
from qsopt.utils.visualization import plot_optimization_dashboard

# Run simulation and optimization
results = experiment.run_simulation()
history = experiment.optimize(theta_init=[1.5, -1.3], num_steps=50)

# Create dashboard with all plots
fig = plot_optimization_dashboard(
    optimization_callback=history,
    reference_callback=results,
    save_path='opt_dashboard.pdf'
)
```

### 2. `plot_contrast_evolution()`

Creates a standalone plot focused on sensing contrast evolution.

**Arguments:**
- `optimization_callback` (required): Optimization history
- `reference_callback` (optional): Reference from simulation
- `figsize` (tuple, default=(10, 6)): Figure size
- `save_path` (str, optional): Path to save figure
- `dpi` (int, default=300): Resolution

**Example:**
```python
from qsopt.utils.visualization import plot_contrast_evolution

fig = plot_contrast_evolution(history, reference_callback=results)
```

### 3. `plot_parameter_trajectory()`

Creates a standalone plot of the optimization path through parameter space.

**Arguments:**
- `optimization_callback` (required): Optimization history
- `reference_callback` (optional): Reference from simulation
- `param_indices` (tuple, default=(0, 1)): Which parameters to plot
- `figsize` (tuple, default=(10, 8)): Figure size
- `save_path` (str, optional): Path to save figure
- `dpi` (int, default=300): Resolution

**Example:**
```python
from qsopt.utils.visualization import plot_parameter_trajectory

fig = plot_parameter_trajectory(history, reference_callback=results)
```

### 4. `plot_parameter_landscape()`

Creates 2D heatmap visualizations of parameter space landscapes.

**Arguments:**
- `theta1_vals` (required): 1D array of θ₁ values
- `theta2_vals` (required): 1D array of θ₂ values
- `contrast_map` (required): 2D array of contrast values
- `detection_map` (required): 2D array of detection probabilities
- `exp_params` (required): ExperimentalParameters object
- `save_path` (str, optional): Path to save figure
- `dpi` (int, default=300): Resolution

**Returns:**
- matplotlib.figure.Figure: Figure with 2-panel landscape visualization

**Example:**
```python
from qsopt.utils.landscape_analysis import compute_theta1_theta2_landscape
from qsopt.utils.visualization import plot_parameter_landscape

# Compute landscape
data = compute_theta1_theta2_landscape(
    exp_params,
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

### 5. `plot_time_interval_landscape()`

Creates 3-panel visualization of time interval landscape analysis.

**Arguments:**
- `interval_vals` (required): 1D array of time interval values
- `contrast_vals` (required): 1D array of contrast values
- `detection_with` (required): 1D array of P(detect|with photon)
- `detection_without` (required): 1D array of P(detect|without photon)
- `n_measurements` (required): 1D array of measurement counts
- `exp_params` (required): ExperimentalParameters object
- `theta1` (required): First rotation angle
- `theta2` (required): Second rotation angle
- `mode` (str, required): 'continuous' or 'discrete'
- `batch_size` (int, default=1): Number of realizations averaged
- `save_path` (str, optional): Path to save figure
- `dpi` (int, default=300): Resolution

**Returns:**
- matplotlib.figure.Figure: Figure with 3-panel landscape visualization

**Example:**
```python
from qsopt.utils.landscape_analysis import compute_time_interval_landscape
from qsopt.utils.visualization import plot_time_interval_landscape

# Compute time interval landscape
data = compute_time_interval_landscape(
    exp_params,
    theta1=np.pi/2,
    theta2=-np.pi/2,
    resolution=30,
    mode='continuous'
)

# Visualize
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
```

### 6. `plot_pulse_shape_with_measurements()`

Visualizes Gaussian pulse envelope with measurement time markers.

**Arguments:**
- `exp_params` (required): ExperimentalParameters object
- `save_path` (str, optional): Path to save figure
- `dpi` (int, default=300): Resolution

**Returns:**
- matplotlib.figure.Figure: Figure showing pulse shape and measurement markers

**Example:**
```python
from qsopt.utils.visualization import plot_pulse_shape_with_measurements

# Visualize pulse and measurement timing
fig = plot_pulse_shape_with_measurements(
    exp_params,
    save_path='pulse_shape.png'
)

# Or use experiment method
experiment = SingleQubitExperiment(exp_params, train_params)
fig = experiment.plot_pulse_shape(save_path='pulse_shape.png')
```

## Usage Patterns

### Pattern 1: Full Dashboard with Reference

Most comprehensive visualization showing optimization improvement over reference:

```python
# Run baseline simulation
results = experiment.run_simulation()

# Optimize parameters
history = experiment.optimize(theta_init=[1.5, -1.3], num_steps=50)

# Create dashboard with reference benchmarks
fig = plot_optimization_dashboard(
    optimization_callback=history,
    reference_callback=results,
    save_path='opt_dashboard.pdf'
)
plt.show()
```

### Pattern 2: Selective Plotting

Choose which plots to display:

```python
# Show only contrast, parameters, and probabilities
fig = plot_optimization_dashboard(
    optimization_callback=history,
    reference_callback=results,
    show_contrast=True,
    show_gradients=False,      # Hide gradients
    show_parameters=True,
    show_trajectory=False,     # Hide trajectory
    show_probabilities=True,
    figsize=(12, 8)
)
```

### Pattern 3: Dashboard Without Reference

When you only have optimization results (no baseline simulation):

```python
# Just optimization (no reference)
history = experiment.optimize(theta_init=[1.5, -1.3], num_steps=50)

# Dashboard without reference benchmarks
fig = plot_optimization_dashboard(
    optimization_callback=history,
    show_trajectory=True,
    save_path=None  # Display only, don't save
)
```

### Pattern 4: Individual Plots

Use standalone functions for focused analysis:

```python
# Just contrast evolution
fig1 = plot_contrast_evolution(history, reference_callback=results)

# Just parameter trajectory
fig2 = plot_parameter_trajectory(history, reference_callback=results)
```

## Dashboard Components

### 1. Sensing Contrast Evolution
- **Green line**: Optimized contrast over epochs
- **Red dashed line** (if reference provided): Reference contrast level
- Shows how optimization improves the sensing capability

### 2. Gradient Magnitude Evolution
- **Log scale** to visualize convergence
- Decreasing gradient indicates approaching local optimum
- Useful for diagnosing optimization convergence

### 3. Parameter Evolution
- Separate colored lines for each rotation angle (ry1, ry2, etc.)
- **Solid lines**: Optimized parameter values
- **Dashed lines** (if reference provided): Reference parameter values
- Shows parameter adjustment throughout optimization

### 4. Optimization Trajectory
- **2D scatter plot** with color gradient by epoch
- **Start point** (red circle): Initial parameters
- **End point** (green square): Final optimized parameters
- **Reference point** (blue triangle, if provided): Reference parameters
- Shows the path taken through parameter space

### 5. Detection Probabilities
- **Green line**: P(detection | with photon) - optimized
- **Red line**: P(detection | without photon) - optimized
- **Dashed lines** (if reference): Reference probabilities
- Shows how optimization affects both signal and background

## Integration with Existing Code

The visualization module is now automatically imported when you import qsopt:

```python
from qsopt import *  # Includes visualization functions
```

Or import specifically:

```python
from qsopt.utils.visualization import (
    plot_optimization_dashboard,
    plot_contrast_evolution,
    plot_parameter_trajectory,
    plot_parameter_landscape,
    plot_time_interval_landscape,
    plot_pulse_shape_with_measurements
)
```

## Usage Patterns by Analysis Type

### For Optimization Analysis
- `plot_optimization_dashboard()` - Comprehensive multi-panel view
- `plot_contrast_evolution()` - Focus on objective function
- `plot_parameter_trajectory()` - Understand parameter space exploration

### For Parameter Space Exploration  
- `plot_parameter_landscape()` - 2D heatmaps of rotation angles
- `plot_time_interval_landscape()` - 1D analysis of measurement timing

### For Experimental Setup
- `plot_pulse_shape_with_measurements()` - Visualize pulse and measurement protocol

## Example Notebook

See `examples/Example.ipynb` for complete examples demonstrating:
- Full dashboard with reference
- Selective plotting
- Individual plot functions
- Dashboard without reference

## Saving Figures

All functions support saving to various formats:

```python
# Save as PDF (vector graphics, recommended for publications)
fig = plot_optimization_dashboard(history, save_path='dashboard.pdf')

# Save as PNG (raster graphics)
fig = plot_optimization_dashboard(history, save_path='dashboard.png', dpi=300)

# Save as SVG (vector graphics, web-friendly)
fig = plot_optimization_dashboard(history, save_path='dashboard.svg')
```
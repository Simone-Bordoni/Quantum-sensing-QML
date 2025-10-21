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
- `optimization_callback` (required): `OptimizationCallback` from `experiment.optimize_rotations()`
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
history = experiment.optimize_rotations(theta_init=[1.5, -1.3], num_steps=50)

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

## Usage Patterns

### Pattern 1: Full Dashboard with Reference

Most comprehensive visualization showing optimization improvement over reference:

```python
# Run baseline simulation
results = experiment.run_simulation()

# Optimize parameters
history = experiment.optimize_rotations(theta_init=[1.5, -1.3], num_steps=50)

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
history = experiment.optimize_rotations(theta_init=[1.5, -1.3], num_steps=50)

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
    plot_parameter_trajectory
)
```

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

## Notes

- **Gradients**: Computed as finite differences from parameter changes (approximate)
- **Parameter Space**: Trajectory plots use the first two parameters by default
- **Auto-layout**: Dashboard automatically adjusts layout based on enabled plots
- **Reference Comparison**: When reference provided, shows improvement over baseline
- **Publication Quality**: Default DPI of 300 suitable for publications

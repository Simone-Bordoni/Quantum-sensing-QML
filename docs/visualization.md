# Visualization Documentation

Complete guide to plotting and visualization tools for quantum sensing optimization results.

## Overview

The visualization module provides:
- **Optimization dashboards** with multi-panel displays
- **Metric evolution** tracking over training
- **Parameter trajectory** visualization
- **Measurement landscape** plots
- **Pulse shape** visualization with measurement markers
- **Noise sensitivity** plots

**Module:** `src/qsopt/core/visualization.py`

---

## Quick Start

### Basic Optimization Visualization

```python
from qsopt.core import SingleQubitExperiment, plot_optimization_dashboard
from qsopt.core import ExperimentalParameters, TrainableParameters
import matplotlib.pyplot as plt

# Run optimization
experiment = SingleQubitExperiment(exp_params, params)
history = experiment.optimize_rotations(num_steps=100, learning_rate=0.05, verbose=True)

# Create dashboard
fig = plot_optimization_dashboard(
    optimization_callback=history,
    show_metric=True,
    show_gradients=True,
    show_parameters=True
)
plt.show()
```

---

## Core Visualization Functions

### plot_optimization_dashboard

Comprehensive multi-panel visualization of optimization results.

```python
plot_optimization_dashboard(
    optimization_callback: Dict,                        # Optimization history
    reference_callback: Optional[Dict] = None,          # Reference for comparison
    show_metric: bool = True,                           # Show metric panel
    show_gradients: bool = True,                        # Show gradient panel
    show_parameters: bool = True,                       # Show parameter panel
    show_loss: bool = False,                            # Show loss panel
    figsize: Tuple[int, int] = (15, 10),               # Figure size
    title: Optional[str] = None                         # Custom title
) -> plt.Figure

# Returns matplotlib Figure object
```

**Example: Full dashboard**
```python
# Optimize
history = experiment.optimize_rotations(num_steps=200, learning_rate=0.05)

# Create dashboard
fig = plot_optimization_dashboard(
    optimization_callback=history,
    show_metric=True,
    show_gradients=True,
    show_parameters=True,
    figsize=(18, 12),
    title="Quantum Sensing Optimization Results"
)

# Save figure
fig.savefig('optimization_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Example: Compare with reference**
```python
# Run two optimizations with different settings
history_noisy = experiment_noisy.optimize_rotations(num_steps=100)
history_ideal = experiment_ideal.optimize_rotations(num_steps=100)

# Compare
fig = plot_optimization_dashboard(
    optimization_callback=history_noisy,
    reference_callback=history_ideal,
    show_metric=True,
    title="Noisy vs Ideal Optimization"
)
plt.show()
```

### plot_metric_evolution

Track sensing metric over optimization steps.

```python
plot_metric_evolution(
    optimization_callback: Dict,                   # Optimization history
    reference_callback: Optional[Dict] = None,     # Optional reference
    figsize: Tuple[int, int] = (10, 6),           # Figure size
    xlabel: str = "Optimization Step",             # X-axis label
    ylabel: str = "Sensing Metric",                # Y-axis label
    title: Optional[str] = None,                   # Custom title
    show_final_value: bool = True,                 # Annotate final value
    color: str = 'blue',                           # Line color
    reference_color: str = 'red'                   # Reference line color
) -> plt.Figure
```

**Example: Single optimization**
```python
history = experiment.optimize_rotations(num_steps=150, learning_rate=0.05)

fig = plot_metric_evolution(
    optimization_callback=history,
    show_final_value=True,
    title="Metric Optimization",
    color='darkblue'
)
plt.show()
```

**Example: Multiple runs comparison**
```python
# Different learning rates
history_lr01 = experiment.optimize_rotations(num_steps=100, learning_rate=0.01)
history_lr05 = experiment.optimize_rotations(num_steps=100, learning_rate=0.05)
history_lr10 = experiment.optimize_rotations(num_steps=100, learning_rate=0.10)

# Plot all on same axes
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(history_lr01['metric'], label='LR = 0.01', linewidth=2)
ax.plot(history_lr05['metric'], label='LR = 0.05', linewidth=2)
ax.plot(history_lr10['metric'], label='LR = 0.10', linewidth=2)
ax.set_xlabel('Optimization Step', fontsize=12)
ax.set_ylabel('Sensing Metric', fontsize=12)
ax.set_title('Learning Rate Comparison', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.show()
```

### plot_parameter_trajectory

Visualize how parameters evolve during optimization.

```python
plot_parameter_trajectory(
    optimization_callback: Dict,                   # Optimization history
    parameter_names: Optional[List[str]] = None,   # Which parameters to plot
    figsize: Tuple[int, int] = (12, 6),           # Figure size
    title: Optional[str] = None,                   # Custom title
    show_final_values: bool = True,                # Annotate final values
    normalize: bool = False                        # Normalize to [0, 1]
) -> plt.Figure
```

**Example: All parameters**
```python
history = experiment.optimize_rotations(num_steps=100, learning_rate=0.05)

fig = plot_parameter_trajectory(
    optimization_callback=history,
    show_final_values=True,
    title="Parameter Evolution"
)
plt.show()
```

**Example: Specific parameters**
```python
fig = plot_parameter_trajectory(
    optimization_callback=history,
    parameter_names=['theta1', 'theta2'],  # Only these parameters
    normalize=False
)
plt.show()
```

**Example: Normalized parameters**
```python
# Normalize to [0, 1] range for comparison
fig = plot_parameter_trajectory(
    optimization_callback=history,
    normalize=True,
    title="Normalized Parameter Trajectories"
)
plt.show()
```

### plot_gradient_magnitudes

Monitor gradient magnitudes during optimization.

```python
plot_gradient_magnitudes(
    optimization_callback: Dict,                   # Optimization history
    parameter_names: Optional[List[str]] = None,   # Which gradients to plot
    figsize: Tuple[int, int] = (12, 6),           # Figure size
    title: Optional[str] = None,                   # Custom title
    log_scale: bool = False                        # Use log scale for y-axis
) -> plt.Figure
```

**Example: Gradient monitoring**
```python
history = experiment.optimize_rotations(num_steps=200, learning_rate=0.05)

fig = plot_gradient_magnitudes(
    optimization_callback=history,
    log_scale=True,  # Log scale helpful for seeing small gradients
    title="Gradient Magnitude Evolution"
)
plt.show()
```

### plot_time_interval_landscape

Visualize metric as function of measurement interval.

```python
plot_time_interval_landscape(
    time_results: Dict,                            # From optimize_measurement_times
    figsize: Tuple[int, int] = (10, 7),           # Figure size
    cmap: str = 'viridis',                         # Colormap
    show_optimal: bool = True,                     # Mark optimal point
    title: Optional[str] = None,                   # Custom title
    show_metadata: bool = True                     # Show system parameters
) -> plt.Figure
```

**Example: Measurement timing optimization**
```python
# Optimize measurement times
time_results = experiment.optimize_measurement_times(
    time_range=(-20.0, 20.0),
    resolution=50,
    verbose=True
)

# Visualize landscape
fig = plot_time_interval_landscape(
    time_results=time_results,
    show_optimal=True,
    show_metadata=True,
    title="Measurement Time Optimization Landscape"
)
plt.show()

print(f"Optimal times: {time_results['optimal_times']}")
print(f"Best metric: {time_results['best_metric']:.6f}")
```

### plot_pulse_shape_with_measurements

Show pulse shape and measurement timing.

```python
plot_pulse_shape_with_measurements(
    experimental_params: ExperimentalParameters,   # Experiment configuration
    time_range: Tuple[float, float] = (-10, 10),  # Time range to plot
    num_points: int = 200,                         # Resolution
    figsize: Tuple[int, int] = (12, 6),           # Figure size
    show_measurements: bool = True,                # Mark measurement times
    title: Optional[str] = None                    # Custom title
) -> plt.Figure
```

**Example: Pulse visualization**
```python
fig = plot_pulse_shape_with_measurements(
    experimental_params=exp_params,
    time_range=(-15.0, 15.0),
    num_points=300,
    show_measurements=True,
    title="Pulse Shape and Measurement Timing"
)
plt.show()
```

---

## Advanced Visualizations

### Parameter Space Exploration

Visualize 2D parameter landscape:

```python
import numpy as np

def plot_parameter_landscape_2d(experiment, param1_name, param2_name, 
                                 param1_range, param2_range, resolution=20):
    """Create 2D heatmap of metric over parameter space."""
    
    # Create grid
    p1_values = np.linspace(param1_range[0], param1_range[1], resolution)
    p2_values = np.linspace(param2_range[0], param2_range[1], resolution)
    metric_grid = np.zeros((resolution, resolution))
    
    # Evaluate metric at each point
    for i, p1 in enumerate(p1_values):
        for j, p2 in enumerate(p2_values):
            params = {param1_name: p1, param2_name: p2}
            experiment.trainable_params.set_parameter_dict(params)
            results = experiment.run_simulation()
            metric_grid[j, i] = results.metric
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(p1_values, p2_values, metric_grid, levels=20, cmap='RdYlGn')
    ax.set_xlabel(f'{param1_name} (rad)', fontsize=12)
    ax.set_ylabel(f'{param2_name} (rad)', fontsize=12)
    ax.set_title('Parameter Landscape', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Metric', fontsize=12)
    
    return fig

# Use it
fig = plot_parameter_landscape_2d(
    experiment=experiment,
    param1_name='theta1',
    param2_name='theta2',
    param1_range=(0, 2*np.pi),
    param2_range=(0, 2*np.pi),
    resolution=30
)
plt.show()
```

### Convergence Analysis

Analyze optimization convergence rate:

```python
def plot_convergence_analysis(history, window=10):
    """Plot convergence metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Metric improvement rate
    metric_values = np.array(history['metric'])
    improvement_rate = np.diff(metric_values)
    
    axes[0, 0].plot(improvement_rate, linewidth=2)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Metric Improvement')
    axes[0, 0].set_title('Improvement Rate per Step')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel 2: Moving average of improvement
    moving_avg = np.convolve(improvement_rate, np.ones(window)/window, mode='valid')
    
    axes[0, 1].plot(moving_avg, linewidth=2)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel(f'Moving Avg ({window} steps)')
    axes[0, 1].set_title('Smoothed Improvement Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: Parameter change magnitude
    params = np.array(history['parameters'])
    param_change = np.linalg.norm(np.diff(params, axis=0), axis=1)
    
    axes[1, 0].semilogy(param_change, linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('||Δθ||')
    axes[1, 0].set_title('Parameter Update Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: Gradient magnitude
    gradients = np.array(history['gradients'])
    grad_magnitude = np.linalg.norm(gradients, axis=1)
    
    axes[1, 1].semilogy(grad_magnitude, linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('||∇L||')
    axes[1, 1].set_title('Gradient Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Use it
history = experiment.optimize_rotations(num_steps=200, learning_rate=0.05)
fig = plot_convergence_analysis(history, window=10)
plt.show()
```

### Noise Sensitivity Visualization

Plot how noise affects optimal performance:

```python
def plot_noise_sensitivity(constants, dims, measurement, initial_state, params,
                           relaxation_range, num_points=15):
    """Visualize sensing performance vs noise strength."""
    
    relaxation_rates = np.logspace(
        np.log10(relaxation_range[0]), 
        np.log10(relaxation_range[1]), 
        num_points
    )
    
    results = []
    for rate in relaxation_rates:
        # Create noisy experiment
        noise = NoiseConfiguration(relaxation=rate, dephasing=rate/2)
        exp_params = ExperimentalParameters(
            physical_constants=constants,
            system_dims=dims,
            measurement=measurement,
            initial_state=initial_state,
            noise_config=noise
        )
        
        experiment = SingleQubitExperiment(exp_params, params)
        history = experiment.optimize_rotations(num_steps=100, verbose=False)
        
        results.append({
            'rate': rate,
            'initial_metric': history['metric'][0],
            'final_metric': history['metric'][-1],
            'improvement': history['metric'][-1] - history['metric'][0]
        })
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    rates = [r['rate'] for r in results]
    initial = [r['initial_metric'] for r in results]
    final = [r['final_metric'] for r in results]
    improvement = [r['improvement'] for r in results]
    
    # Panel 1: Metric vs noise
    axes[0].semilogx(rates, initial, 'o-', label='Initial', linewidth=2)
    axes[0].semilogx(rates, final, 's-', label='Optimized', linewidth=2)
    axes[0].set_xlabel('Relaxation Rate (rad/s)', fontsize=12)
    axes[0].set_ylabel('Metric', fontsize=12)
    axes[0].set_title('Noise Sensitivity', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Improvement vs noise
    axes[1].semilogx(rates, improvement, 'o-', color='green', linewidth=2)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Relaxation Rate (rad/s)', fontsize=12)
    axes[1].set_ylabel('Metric Improvement', fontsize=12)
    axes[1].set_title('Optimization Benefit', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Relative improvement
    relative = [r['improvement'] / r['initial_metric'] * 100 if r['initial_metric'] > 0 
                else 0 for r in results]
    axes[2].semilogx(rates, relative, 'o-', color='purple', linewidth=2)
    axes[2].set_xlabel('Relaxation Rate (rad/s)', fontsize=12)
    axes[2].set_ylabel('Relative Improvement (%)', fontsize=12)
    axes[2].set_title('Percentage Gain', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, results

# Use it
fig, results = plot_noise_sensitivity(
    constants=constants,
    dims=dims,
    measurement=measurement,
    initial_state=initial_state,
    params=params,
    relaxation_range=(1e-5 * 2*np.pi, 1e-2 * 2*np.pi),
    num_points=20
)
plt.show()
```

---

## Customization

### Color Schemes

```python
# Custom colormap for landscapes
from matplotlib.colors import LinearSegmentedColormap

# Define custom colormap
colors = ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Use in landscape plot
fig = plot_time_interval_landscape(
    time_results=time_results,
    cmap=cmap
)
```

### Figure Styles

```python
# Set publication-quality style
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Create plots with this style
fig = plot_optimization_dashboard(history)
fig.savefig('results.pdf', bbox_inches='tight')
```

### Annotations

```python
# Add custom annotations to plots
fig = plot_metric_evolution(history)
ax = fig.axes[0]

# Mark specific event
ax.axvline(x=50, color='red', linestyle='--', alpha=0.7)
ax.text(52, 0.5, 'Learning rate decreased', fontsize=10, color='red')

# Add custom title with metadata
chi = exp_params.physical_constants.chi
ax.set_title(f'Optimization Results (χ = {chi:.4f} rad/s)', fontsize=14)

plt.show()
```

---

## Export and Saving

### High-Resolution Exports

```python
# Save as publication-quality figure
fig = plot_optimization_dashboard(history)

# PDF (vector format)
fig.savefig('optimization.pdf', format='pdf', bbox_inches='tight', dpi=300)

# PNG (raster format)
fig.savefig('optimization.png', format='png', bbox_inches='tight', dpi=300)

# SVG (vector format, editable)
fig.savefig('optimization.svg', format='svg', bbox_inches='tight')
```

### Batch Exports

```python
# Save multiple figures in loop
for idx, history in enumerate(optimization_histories):
    fig = plot_metric_evolution(history, title=f'Run {idx+1}')
    fig.savefig(f'run_{idx+1}_metric.png', dpi=200, bbox_inches='tight')
    plt.close(fig)  # Close to free memory
```

### Data Export

```python
# Export data for external plotting
import pandas as pd

# Convert history to DataFrame
df = pd.DataFrame({
    'step': range(len(history['metric'])),
    'metric': history['metric'],
    'theta1': [p['theta1'] for p in history['parameters']],
    'theta2': [p['theta2'] for p in history['parameters']]
})

# Save to CSV
df.to_csv('optimization_data.csv', index=False)

# Save to Excel
df.to_excel('optimization_data.xlsx', index=False)
```

---

## Best Practices

### Figure Size Selection

```python
# Presentation (16:9 aspect ratio)
fig = plot_optimization_dashboard(history, figsize=(16, 9))

# Publication (square or 4:3)
fig = plot_metric_evolution(history, figsize=(8, 6))

# Poster (large, high DPI)
fig = plot_optimization_dashboard(history, figsize=(24, 18))
fig.savefig('poster_figure.png', dpi=300)
```

### Color Accessibility

```python
# Use colorblind-friendly palettes
colorblind_colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161']

plt.plot(history['metric'], color=colorblind_colors[0], linewidth=2)
```

### Grid and Styling

```python
# Professional grid styling
ax.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
ax.grid(True, which='minor', linestyle=':', alpha=0.2, linewidth=0.5)
ax.minorticks_on()
```

---

## Interactive Visualizations

### Jupyter Notebook Integration

```python
# In Jupyter notebook with interactive backend
%matplotlib widget

import ipywidgets as widgets
from IPython.display import display

def interactive_parameter_plot(experiment, param_name, param_range):
    """Create interactive parameter slider."""
    
    @widgets.interact(value=widgets.FloatSlider(
        min=param_range[0], max=param_range[1], 
        step=(param_range[1]-param_range[0])/50, value=np.pi/4
    ))
    def update_plot(value):
        experiment.trainable_params.set_parameter_dict({param_name: value})
        results = experiment.run_simulation()
        
        print(f"{param_name} = {value:.4f} rad")
        print(f"Metric = {results.metric:.6f}")

# Use it
interactive_parameter_plot(experiment, 'theta1', (0, 2*np.pi))
```

### Animation

```python
from matplotlib.animation import FuncAnimation

def animate_optimization(history, interval=100):
    """Animate optimization process."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', linewidth=2)
    ax.set_xlim(0, len(history['metric']))
    ax.set_ylim(0, max(history['metric']) * 1.1)
    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel('Sensing Metric', fontsize=12)
    ax.set_title('Optimization Progress', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    def init():
        line.set_data([], [])
        return line,
    
    def update(frame):
        x = list(range(frame + 1))
        y = history['metric'][:frame + 1]
        line.set_data(x, y)
        return line,
    
    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=len(history['metric']),
        interval=interval, blit=True
    )
    
    return anim

# Create animation
anim = animate_optimization(history, interval=50)
anim.save('optimization.gif', writer='pillow', fps=20)
plt.show()
```

---

## Troubleshooting

### Figure Not Displaying

```python
# Ensure matplotlib backend is set
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system

import matplotlib.pyplot as plt

# Call plt.show() explicitly
fig = plot_optimization_dashboard(history)
plt.show()
```

### Memory Issues with Large Datasets

```python
# Downsample data for plotting
def downsample(data, target_points=1000):
    """Reduce number of points for plotting."""
    if len(data) <= target_points:
        return data
    step = len(data) // target_points
    return data[::step]

# Use downsampled data
metric_downsampled = downsample(history['metric'])
plt.plot(metric_downsampled)
```

### Font Rendering Issues

```python
# Use system fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# Or use LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
```

---

## See Also

- [Experiments Documentation](./experiments.md) - Running optimizations to visualize
- [Gates & Circuits Documentation](./gates_and_circuits.md) - Visualizing circuit parameters
- [Matplotlib Documentation](https://matplotlib.org/) - Advanced plotting techniques
- [Seaborn Documentation](https://seaborn.pydata.org/) - Statistical visualization

# Quantum Sensing Optimization Library Documentation

This document provides comprehensive documentation for the Quantum Sensing Optimization Library (qsopt).

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Core Modules](#core-modules)
3. [API Reference](#api-reference)
4. [Examples](#examples)
5. [Advanced Topics](#advanced-topics)

## Quick Start Guide

### Installation

```bash
git clone https://github.com/Simone-Bordoni/Quantum-sensing-QML.git
cd Quantum-sensing-QML
pip install -e .
```

### Your First Experiment

```python
from qsopt import *
import numpy as np

# Define system parameters
gm = 0.03 * 2 * np.pi
constants = PhysicalConstants(
    chi=0.5 * gm,
    photon_cavity_coupling=gm,
    inverse_pulse_width=0.1 * gm
)

# Configure experiment
exp_params = ExperimentalParameters(
    physical_constants=constants,
    system_dims=SystemDimensions(cavity_levels=2, qubit_levels=2),
    measurement=MeasurementProtocol(measurement_times=[-5.0, 0.0, 5.0]),
    initial_state=InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
)

# Define trainable parameters
params = TrainableParameters()
params.add_rotation_angles(['ry1', 'ry2'], [np.pi/2, -np.pi/2])

# Create experiment
experiment = SingleQubitExperiment(exp_params, params)

# Run simulation
results = experiment.run_simulation()
print(results)

# Optimize parameters
history = experiment.optimize(theta_init=[1.5, -1.3], num_steps=50)
print(history)
```

## Core Modules

### 1. Experimental Parameters

Configure your quantum system with precise control over physical constants, dimensions, noise, and measurements.

**Key Classes:**
- `PhysicalConstants`: Define chi, coupling, pulse width
- `SystemDimensions`: Set Hilbert space dimensions  
- `NoiseConfiguration`: Configure decoherence parameters
- `MeasurementProtocol`: Define measurement times
- `InitialStateConfig`: Set initial quantum state

**See:** [Experimental Parameters Guide](./experimental_parameters.md)

### 2. Trainable Parameters

Define and manage parameters for optimization.

**Key Classes:**
- `TrainableParameters`: Container for optimizable parameters
- `ParameterType`: Enum for parameter types
- `ParameterConstraints`: Define bounds and periodicity

**See:** [Trainable Parameters Guide](./trainable_parameters.md)

### 3. Experiment Class

The `SingleQubitExperiment` class is the main interface for running simulations and optimizations.

**Key Methods:**
- `run_simulation()`: Execute quantum evolution with current parameters
- `optimize()`: Gradient-based parameter optimization

**See:** [Experiment Class Reference](./experiment.md)

### 4. Optimization Callbacks

Track and analyze optimization progress.

**Key Classes:**
- `OptimizationCallback`: Record metrics during optimization

**See:** [Callbacks Guide](./callbacks.md)

### 5. Visualization

Create comprehensive plots and dashboards.

**Key Functions:**
- `plot_optimization_dashboard()`: Multi-panel optimization visualization
- `plot_contrast_evolution()`: Tracking sensing contrast
- `plot_parameter_trajectory()`: Parameter space exploration
- `plot_parameter_landscape()`: 2D parameter landscape heatmaps
- `plot_time_interval_landscape()`: Time interval optimization analysis
- `plot_pulse_shape_with_measurements()`: Pulse envelope with measurement markers

**See:** [Visualization Module](./VISUALIZATION_MODULE.md)

### 6. Landscape Analysis

Systematically explore parameter spaces for optimization.

**Key Functions:**
- `compute_theta1_theta2_landscape()`: 2D rotation angle parameter analysis
- `compute_time_interval_landscape()`: Measurement timing optimization

**See:** [Landscape Analysis Module](./LANDSCAPE_ANALYSIS.md)

### 7. Experiment Loader

Load and reconstruct experiments from saved reports for reproducibility.

**Key Functions:**
- `load_experiment_from_report()`: Load experiment configuration from JSON

**See:** [Experiment Loader](./EXPERIMENT_LOADER.md)

## API Reference

### Core Classes

#### ExperimentalParameters

```python
ExperimentalParameters(
    physical_constants: PhysicalConstants,
    system_dims: SystemDimensions,
    measurement: MeasurementProtocol,
    initial_state: InitialStateConfig,
    noise_config: NoiseConfiguration = NoiseConfiguration()
)
```

Comprehensive configuration for quantum sensing experiments.

**Attributes:**
- `chi`: Dispersive coupling strength
- `cavity_levels`: Resonator Hilbert space dimension
- `measurement_times`: List of measurement time points
- `state_type`: Initial quantum state type
- `relaxation`: T1 relaxation rate
- `dephasing`: T2 dephasing rate

#### SingleQubitExperiment

```python
SingleQubitExperiment(
    experimental_params: ExperimentalParameters,
    trainable_params: TrainableParameters
)
```

Main experiment class for quantum sensing simulations.

**Methods:**
- `run_simulation() -> OptimizationCallback`: Execute single simulation
- `optimize(theta_init, num_steps, verbose) -> OptimizationCallback`: Run optimization
- `get_sensing_contrast() -> float`: Calculate current contrast

#### TrainableParameters

```python
TrainableParameters()
```

Container for optimization parameters.

**Methods:**
- `add_rotation_angles(names, values, optimizer)`: Add rotation gate parameters
- `add_custom_parameters(names, values, optimizer)`: Add custom parameters
- `get_rotation_angles() -> Dict`: Get all rotation parameters
- `set_parameter_vector(values)`: Update parameter values

### Visualization Functions

#### plot_optimization_dashboard

```python
plot_optimization_dashboard(
    optimization_callback: OptimizationCallback,
    reference_callback: Optional[OptimizationCallback] = None,
    show_contrast: bool = True,
    show_gradients: bool = True,
    show_parameters: bool = True,
    show_trajectory: bool = True,
    show_probabilities: bool = True,
    save_path: Optional[str] = None
) -> Figure
```

Create comprehensive optimization visualization.

**Returns:** matplotlib Figure object

## Examples

### Example 1: Basic Simulation

```python
from qsopt import *

# Create experiment (using default parameters)
experiment = SingleQubitExperiment(exp_params, trainable_params)

# Run single simulation
results = experiment.run_simulation()
print(f"Detection probability (with photon): {results.best_metrics['prob_with']:.4f}")
print(f"Detection probability (no photon): {results.best_metrics['prob_without']:.4f}")
print(f"Sensing contrast: {results.best_metrics['contrast']:.4f}")
```

### Example 2: Parameter Optimization

```python
# Run optimization
history = experiment.optimize(
    theta_init=[np.pi/2, -np.pi/2],  # Initial guess
    num_steps=100,                    # Optimization steps
    verbose=True,                     # Show progress
    tolerance=1e-6                    # Convergence threshold
)

# Access results
best_params = history.get_best_trainable_params()
best_contrast = history.best_contrast
print(f"Best contrast achieved: {best_contrast:.6f}")
```

### Example 3: Visualization

```python
from qsopt.utils.visualization import plot_optimization_dashboard

# Create dashboard
fig = plot_optimization_dashboard(
    optimization_callback=history,
    reference_callback=results,
    save_path='my_optimization.pdf'
)
```

### Example 4: Noise Sensitivity

```python
import matplotlib.pyplot as plt

contrasts = []
noise_levels = [0.0001, 0.001, 0.01, 0.1]

for noise in noise_levels:
    # Update noise configuration
    noise_config = NoiseConfiguration(
        relaxation=noise * 2 * np.pi,
        dephasing=noise * 2 * np.pi
    )
    
    # Create new experiment with updated noise
    exp_params_noisy = ExperimentalParameters(
        physical_constants=constants,
        system_dims=dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise_config
    )
    
    exp = SingleQubitExperiment(exp_params_noisy, params)
    history = exp.optimize(theta_init=[1.5, -1.3], num_steps=50)
    contrasts.append(history.best_contrast)

# Plot results
plt.plot(noise_levels, contrasts, 'o-')
plt.xlabel('Noise Level')
plt.ylabel('Best Contrast')
plt.xscale('log')
plt.show()
```

## Advanced Topics

### Custom Optimizers

```python
import optax

# Define custom optimizer
custom_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=0.01)
)

# Add parameters with custom optimizer
params = TrainableParameters()
params.add_rotation_angles(
    names=['theta1', 'theta2'],
    initial_values=[1.0, 2.0],
    optimizer=custom_optimizer
)
```

### Parameter Constraints

```python
from qsopt import ParameterConstraints

# Define periodic constraints (for rotation angles)
constraints = ParameterConstraints(
    lower_bound=0.0,
    upper_bound=2 * np.pi,
    periodic=True
)

# Apply during optimization
# (constraints are automatically applied after each step)
```

### Custom Initial States

```python
# Define custom superposition state
custom_amplitudes = {
    (0, 0, 0): 0.7 + 0.0j,  # |000⟩
    (1, 0, 0): 0.3 + 0.0j,  # |100⟩
}

initial_state = InitialStateConfig(
    state_type=InitialStateType.CUSTOM,
    custom_amplitudes=custom_amplitudes
)
```

## Best Practices

### 1. Choose Appropriate System Dimensions
- Start with `cavity_levels=2` for faster computation
- Increase to 3-5 for coherent/thermal states if needed

### 2. Optimize Measurement Times
- Normalize times by relevant system timescale
- Use 3-5 measurement points for balance of accuracy and speed

### 3. Set Reasonable Learning Rates
- Start with `learning_rate=0.05` for SGD
- Use `learning_rate=0.01` for Adam/RMSprop
- Monitor gradient norms for convergence

### 4. Use Callbacks for Monitoring
- Always use callbacks to track optimization progress
- Save results periodically with `callback.save()`

### 5. Benchmark Against Reference
- Run `run_simulation()` before optimization
- Use reference values in visualization for comparison


## Citation

```bibtex
@software{qsopt2025,
  title={Quantum Sensing Optimization Library},
  author={Bordoni, Simone and Gargioni, Nathan},
  year={2025},
  url={https://github.com/Simone-Bordoni/Quantum-sensing-QML}
}
```

### Example 5: Saving and Loading Experiment Reports

Save your experimental configuration for reproducibility:

```python
# After running an optimization
experiment = SingleQubitExperiment(exp_params, params)
history = experiment.optimize(theta_init=[1.5, -1.3], num_steps=100)

# Save comprehensive report with all parameters and optimization data
experiment.save_experiment_report('results/my_experiment.json')
# This creates:
#   - results/my_experiment.json (all experimental parameters, metadata)
#   - results/my_experiment_callback.npz (detailed optimization data)

# Later, load the configuration
from qsopt.utils.experiment_loader import load_experiment_from_report
exp_params, train_params, metadata = load_experiment_from_report('results/my_experiment.json')

# Recreate experiment
experiment = SingleQubitExperiment(exp_params, train_params)

# Access optimization data
if 'callback_data' in metadata:
    optimization_data = metadata['callback_data']
    print(f"Epochs: {len(optimization_data['epochs'])}")
    print(f"Final contrast: {optimization_data['contrast'][-1]:.6f}")
```

### Example 6: Parameter Landscape Analysis

Systematically explore parameter space before optimization:

```python
from qsopt.utils.landscape_analysis import compute_theta1_theta2_landscape
from qsopt.utils.visualization import plot_parameter_landscape

# Compute 2D landscape of rotation angles
data = compute_theta1_theta2_landscape(
    exp_params,
    theta1_range=(-np.pi, np.pi),
    theta2_range=(-np.pi, np.pi),
    resolution=30,  # 30×30 grid
    verbose=True
)

# Visualize landscape
fig = plot_parameter_landscape(
    data['theta1_vals'],
    data['theta2_vals'],
    data['contrast_map'],
    data['detection_map'],
    exp_params,
    save_path='parameter_landscape.png'
)

# Find best parameters from landscape
best_idx = np.unravel_index(np.argmax(data['contrast_map']), data['contrast_map'].shape)
best_theta1 = data['theta1_vals'][best_idx[0]]
best_theta2 = data['theta2_vals'][best_idx[1]]

print(f"Best parameters from landscape: θ₁={best_theta1:.4f}, θ₂={best_theta2:.4f}")

# Use as starting point for optimization
history = experiment.optimize(theta_init=[best_theta1, best_theta2], num_steps=50)
```

### Example 7: Time Interval Optimization

Find optimal measurement timing:

```python
from qsopt.utils.landscape_analysis import compute_time_interval_landscape
from qsopt.utils.visualization import plot_time_interval_landscape

# Analyze how contrast varies with measurement interval
data = compute_time_interval_landscape(
    exp_params,
    theta1=np.pi/2,
    theta2=-np.pi/2,
    resolution=30,
    mode='continuous',  # Linearly spaced intervals
    verbose=True
)

# Visualize 3-panel landscape
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

# Find and apply optimal interval
optimal_idx = np.argmax(data['contrast_vals'])
optimal_interval = data['interval_vals'][optimal_idx]

print(f"Optimal time interval: {optimal_interval:.6f}")
exp_params.measurement.time_interval = optimal_interval
```

### Example 8: Pulse Shape Visualization

Visualize Gaussian pulse envelope with measurement markers:

```python
from qsopt.utils.visualization import plot_pulse_shape_with_measurements

# Visualize pulse and measurement timing
fig = plot_pulse_shape_with_measurements(
    exp_params,
    save_path='pulse_shape.png',
    dpi=300
)

# Or use experiment method
experiment = SingleQubitExperiment(exp_params, train_params)
fig = experiment.plot_pulse_shape(save_path='pulse_shape.png')
```

The JSON report includes:
- Physical constants (chi, coupling strengths, pulse width)
- System dimensions (cavity, qubit, field levels)
- Measurement protocol (times, strategies)
- Initial state configuration (type, parameters)
- Noise configuration (depolarizing, dephasing, relaxation)
- Trainable parameters (rotation angles, field strengths)
- Optimization summary (if available)

## Further Reading

- [QuTiP Documentation](https://qutip.org/docs/latest/)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)

---

**Last Updated:** October 2025  
**Version:** 0.1.0

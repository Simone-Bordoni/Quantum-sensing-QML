# Quantum Sensing Optimization Library Documentation

Welcome to the documentation for the Quantum Sensing Optimization Library (qsopt)!

## 📖 Table of Contents

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
- `get_sensing_contrast()`: Calculate detection metric

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

**See:** [Visualization Module](./VISUALIZATION_MODULE.md)

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

## Troubleshooting

### Common Issues

**Issue: Optimization doesn't converge**
- Try different initial values
- Reduce learning rate
- Check if gradients are vanishing (use dashboard)

**Issue: Numerical instabilities**
- Reduce system dimensions
- Check measurement time normalization
- Ensure physical parameters are reasonable

**Issue: Slow performance**
- Reduce `cavity_levels` and `qubit_levels`
- Use fewer measurement points
- Enable JAX JIT compilation (automatic)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{qsopt2025,
  title={Quantum Sensing Optimization Library},
  author={Bordoni, Simone and Gargioni, Nathan},
  year={2025},
  url={https://github.com/Simone-Bordoni/Quantum-sensing-QML}
}
```

## Further Reading

- [QuTiP Documentation](https://qutip.org/docs/latest/)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)

---

**Last Updated:** January 2025  
**Version:** 0.1.0

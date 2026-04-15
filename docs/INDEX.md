# Quantum Sensing Optimization Library Documentation

This document provides comprehensive documentation for the Quantum Sensing Optimization Library (qsopt), a specialized toolkit for gradient-based optimization of quantum sensing protocols with JAX-compatible quantum circuits.

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Core Modules](#core-modules)
3. [API Reference](#api-reference)
4. [Examples](#examples)
5. [Advanced Topics](#advanced-topics)
6. [Detailed Documentation](#detailed-documentation)

## Quick Start Guide

### Installation

```bash
git clone https://github.com/Simone-Bordoni/Quantum-sensing-QML.git
cd Quantum-sensing-QML
pip install -e .
```

### Your First Quantum Sensing Experiment

```python
from qsopt.core import (
    ExperimentalParameters, PhysicalConstants, SystemDimensions,
    NoiseConfiguration, MeasurementProtocol, InitialStateConfig,
    InitialStateType, TrainableParameters, SingleQubitExperiment
)
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
    system_dims=SystemDimensions(cavity_levels=2, qubit_levels=2, field_levels=2),
    measurement=MeasurementProtocol(measurement_times=[-5.0, 0.0, 5.0]),
    initial_state=InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON),
    noise_config=NoiseConfiguration(relaxation=0.0001 * 2 * np.pi, dephasing=0.0001 * 2 * np.pi)
)

# Define trainable parameters
params = TrainableParameters()
params.add_rotation_angles(
    names=['theta1', 'theta2'],
    initial_values=[np.pi/2, -np.pi/2]
)

# Create and run experiment
experiment = SingleQubitExperiment(exp_params, params)
results = experiment.run_simulation()

# Optimize rotation parameters
history = experiment.optimize_rotations(num_steps=100, learning_rate=0.05, verbose=True)
print(f"Final metric: {history['metric'][-1]:.6f}")
```

### Building Quantum Circuits for State Preparation

```python
from qsopt.core.circuit import QuantumCircuit, create_layer
from qsopt.core.gates import RXGate, RYGate, CNOTGate, HadamardGate
import jax.numpy as jnp

# Create 2-qubit parameterized circuit
circuit = QuantumCircuit(num_qubits=2)
circuit.add_gate(HadamardGate(), target=0)
circuit.add_gate(RXGate(theta=jnp.pi/4, trainable=True), target=1)
circuit.add_gate(CNOTGate(), target=(0, 1))

# Get circuit unitary for state preparation
U = circuit.get_unitary_jax()

# Access trainable parameters
params = circuit.get_trainable_parameters()
print(f"Trainable parameters: {params}")
```

## Core Modules

### 1. Quantum Sensing Experiments

Run and optimize quantum sensing protocols with realistic noise models.

**Key Classes:**
- `SingleQubitExperiment`: Main experiment interface for 1-qubit sensing
- `TwoQubitExperiment`: Two-qubit sensing protocols
- `ExperimentalParameters`: System configuration
- `PhysicalConstants`: Chi coupling, photon coupling, pulse parameters
- `SystemDimensions`: Hilbert space dimensions
- `NoiseConfiguration`: Relaxation and dephasing rates

**See:** [Experiments Documentation](./experiments.md)

### 2. Quantum Gates & Circuits

JAX-compatible quantum gates and circuit construction for state preparation and gradient-based optimization.

**Gate Classes:**
- `RXGate`, `RYGate`, `RZGate`: Rotation gates with trainable parameters
- `HadamardGate`, `CNOTGate`, `CZGate`: Fixed gates
- `Gate`: Base class with parameter management

**Circuit Classes:**
- `QuantumCircuit`: Multi-qubit circuit builder
- `GateApplication`: Gate with target qubit specification
- Utility functions: `create_layer()`, `create_entangling_layer()`

**See:** [Gates & Circuits Documentation](./gates_and_circuits.md)

### 3. Visualization Tools

Comprehensive plotting and analysis tools for optimization results.

**Key Functions:**
- `plot_optimization_dashboard()`: Multi-panel optimization visualization
- `plot_contrast_evolution()`: Track sensing contrast over epochs
- `plot_parameter_trajectory()`: Parameter space exploration
- `plot_time_interval_landscape()`: Measurement interval landscapes
- `plot_pulse_shape_with_measurements()`: Pulse visualization

**See:** [Visualization Documentation](./visualization.md)

### 4. Trainable Parameters

Manage optimization parameters with gradient control.

**Key Classes:**
- `TrainableParameters`: Container for optimizable parameters
- `ParameterType`: Parameter type enumeration
- `ParameterConstraints`: Bounds and periodicity

**See:** [Trainable Parameters Guide](./trainable_parameters.md)

### 5. Optimization Callbacks

Track and analyze optimization progress with detailed metrics.

**Key Classes:**
- `OptimizationCallback`: Record metrics during optimization

**See:** [Callbacks Guide](./callbacks.md)

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
- `plot_time_interval_landscape()`: Measurement interval landscape with uncertainty
- `plot_pulse_shape_with_measurements()`: Pulse envelope annotated with measurement times

**See:** [Visualization Module](./VISUALIZATION_MODULE.md)

### 6. Measurement Time Optimization

Optimize, visualize, and interpret measurement schedules alongside rotation controls.

**Key Functions:**
- `compute_time_interval_landscape()`: Evaluate sensing contrast across interval grids
- `optimize_measurement_times()`: Execute the adaptive measurement interval search
- `plot_time_interval_landscape()`: Present interval landscapes with system metadata

**See:** [Measurement Time Optimization Guide](./measurement_time_optimization.md)

## API Reference

### Quantum Gates API

#### Gate Base Class

```python
class Gate:
    """Base class for quantum gates with parameter management."""
    
    def __init__(self, num_qubits: int, parameters: Dict[str, GateParameter] = None)
    def matrix(self) -> Qobj
    def enable_gradients(self, param_name: Optional[str] = None) -> None
    def disable_gradients(self, param_name: Optional[str] = None) -> None
    def get_trainable_parameters(self) -> Dict[str, float]
    def set_trainable_parameters(self, values: Dict[str, float]) -> None
```

#### Rotation Gates

```python
# Single-qubit rotation gates with trainable angles
RXGate(theta: float = 0.0, trainable: bool = False)  # Rotation around X
RYGate(theta: float = 0.0, trainable: bool = False)  # Rotation around Y
RZGate(theta: float = 0.0, trainable: bool = False)  # Rotation around Z
```

#### Fixed Gates

```python
HadamardGate()           # Hadamard gate (basis change)
CNOTGate()               # Controlled-NOT (2-qubit entangling)
CZGate()                 # Controlled-Z (2-qubit phase)
```

### Quantum Circuit API

```python
class QuantumCircuit:
    """Build multi-qubit quantum circuits with parameter tracking."""
    
    def __init__(self, num_qubits: int)
    
    def add_gate(self, gate: Gate, target: Union[int, Tuple[int, int]]) -> None
        """Add gate to circuit. target is qubit index or (control, target) tuple."""
    
    def get_unitary(self) -> Qobj
        """Get circuit unitary as QuTiP Qobj."""
    
    def get_unitary_jax(self) -> jnp.ndarray
        """Get circuit unitary as JAX array for autodiff."""
    
    def get_trainable_parameters(self) -> Dict[str, float]
        """Get all trainable parameters from circuit gates."""
    
    def set_trainable_parameters(self, values: Dict[str, float]) -> None
        """Update trainable parameter values."""

# Utility Functions
def create_layer(gate_class: Type[Gate], num_qubits: int, **kwargs) -> List[GateApplication]
    """Create parallel layer of single-qubit gates."""

def create_entangling_layer(gate_class: Type[Gate], num_qubits: int, 
                           pattern: str = "linear") -> List[GateApplication]
    """Create entangling layer with specified connectivity pattern."""
```

### Experiment API

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
- `optimize_rotations(theta_init, num_steps, verbose) -> OptimizationCallback`: Run optimization
- `optimize_measurement_times(resolution, mode, batch_size, ...) -> Dict`: Search over measurement intervals
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
    show_metric: bool = True,
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
print(f"Sensing metric: {results.best_metrics['metric']:.4f}")
```

### Example 2: Parameter Optimization

```python
# Run optimization
history = experiment.optimize_rotations(
    theta_init=[np.pi/2, -np.pi/2],  # Initial guess
    num_steps=100,                    # Optimization steps
    verbose=True,                     # Show progress
    tolerance=1e-6                    # Convergence threshold
)

# Access results
best_params = history.get_best_trainable_params()
best_metric = history.best_metric
print(f"Best metric achieved: {best_metric:.6f}")
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

metric_values = []
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
    history = exp.optimize_rotations(theta_init=[1.5, -1.3], num_steps=50)
    metric_values.append(history.best_metric)

# Plot results
plt.plot(noise_levels, metric_values, 'o-')
plt.xlabel('Noise Level')
plt.ylabel('Best Metric')
plt.xscale('log')
plt.show()
```

### Example 5: Measurement Time Optimization

```python
# Required imports
import numpy as np
from qsopt.utils.landscape_analysis import compute_time_interval_landscape
from qsopt.utils.visualization import plot_time_interval_landscape

# Compute contrast landscape over time intervals
landscape = compute_time_interval_landscape(
    exp_params,
    theta1=np.pi / 2,
    theta2=-np.pi / 2,
    resolution=40,
    mode='continuous',
    batch_size=10,
    verbose=False
)

# Plot with measurement-count subplot enabled
fig = plot_time_interval_landscape(
    landscape,
    exp_params,
    show_measurement_count=True
)

# Run adaptive measurement-time optimization
measurement_results = experiment.optimize_measurement_times(
    resolution=60,
    mode='continuous',
    batch_size=15,
    min_interval=0.05,
    max_interval=1.5,
    verbose=True
)

print(f"Optimal interval: {measurement_results['best_interval']:.4f}")
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

## Detailed Documentation

For comprehensive guides on specific modules:

- **[Quantum Gates & Circuits](./gates_and_circuits.md)** - Complete guide to JAX-compatible gates, circuit construction, parameter management, and integration with sensing experiments. Includes all gate types (RX, RY, RZ, H, CNOT, CZ), circuit builder API, and 86 verified tests.

- **[Quantum Sensing Experiments](./experiments.md)** - Full documentation of experiment configuration, optimization methods, parameter management, noise modeling, and integration with circuit-based state preparation.

- **[Visualization Tools](./visualization.md)** - Comprehensive plotting functions including optimization dashboards, contrast evolution, parameter trajectories, measurement landscapes, and advanced customization options.

---

## Examples

### Example 1: Basic Quantum Sensing Simulation

```python
from qsopt.core import (
    ExperimentalParameters, PhysicalConstants, SystemDimensions,
    MeasurementProtocol, InitialStateConfig, InitialStateType,
    NoiseConfiguration, TrainableParameters, SingleQubitExperiment
)
import numpy as np

# System parameters
gm = 0.03 * 2 * np.pi
constants = PhysicalConstants(
    chi=0.5 * gm,
    photon_cavity_coupling=gm,
    inverse_pulse_width=0.1 * gm
)

# Experiment configuration
exp_params = ExperimentalParameters(
    physical_constants=constants,
    system_dims=SystemDimensions(cavity_levels=2, qubit_levels=2, field_levels=2),
    measurement=MeasurementProtocol(measurement_times=[-5.0, 0.0, 5.0]),
    initial_state=InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON),
    noise_config=NoiseConfiguration(relaxation=0.0001 * 2 * np.pi, dephasing=0.0001 * 2 * np.pi)
)

# Trainable parameters
params = TrainableParameters()
params.add_rotation_angles(['theta1', 'theta2'], [np.pi/2, -np.pi/2])

# Run simulation
experiment = SingleQubitExperiment(exp_params, params)
results = experiment.run_simulation()
print(f"Sensing contrast: {results.contrast:.6f}")
```

### Example 2: Gradient-Based Optimization

```python
# Optimize rotation parameters
history = experiment.optimize_rotations(
    num_steps=100,
    learning_rate=0.05,
    verbose=True
)

# Visualize results
from qsopt.core.visualization import plot_optimization_dashboard
fig = plot_optimization_dashboard(history, show_metric=True, show_gradients=True)
plt.show()

print(f"Initial metric: {history['metric'][0]:.6f}")
print(f"Final metric: {history['metric'][-1]:.6f}")
print(f"Improvement: {history['metric'][-1] - history['metric'][0]:.6f}")
```

### Example 3: Circuit-Based State Preparation

```python
from qsopt.core.circuit import QuantumCircuit
from qsopt.core.gates import RYGate, CNOTGate, HadamardGate
from qutip import basis
import jax.numpy as jnp

# Build parameterized 2-qubit circuit
circuit = QuantumCircuit(num_qubits=2)
circuit.add_gate(HadamardGate(), target=0)
circuit.add_gate(RYGate(theta=jnp.pi/4, trainable=True), target=1)
circuit.add_gate(CNOTGate(), target=(0, 1))

# Get unitary for optimization
U = circuit.get_unitary_jax()  # JAX array for autodiff

# Access and update parameters
params = circuit.get_trainable_parameters()
# {'gate_1_theta': 0.785...}

circuit.set_trainable_parameters({'gate_1_theta': jnp.pi/2})
```

### Example 4: Measurement Time Optimization

```python
# Search for optimal measurement intervals
time_results = experiment.optimize_measurement_times(
    time_range=(-15.0, 15.0),
    resolution=40,
    mode='exhaustive',
    verbose=True
)

# Visualize landscape
from qsopt.core.visualization import plot_time_interval_landscape
fig = plot_time_interval_landscape(time_results, show_optimal=True)
plt.show()

print(f"Optimal times: {time_results['optimal_times']}")
print(f"Best metric: {time_results['best_metric']:.6f}")
```

### Example 5: Noise Sensitivity Analysis

```python
# Study performance vs noise strength
relaxation_rates = np.logspace(-5, -2, 10) * 2 * np.pi
results_noise = []

for rate in relaxation_rates:
    noise = NoiseConfiguration(relaxation=rate, dephasing=rate/2)
    exp_params_noisy = ExperimentalParameters(
        physical_constants=constants,
        system_dims=dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise
    )
    
    experiment_noisy = SingleQubitExperiment(exp_params_noisy, params)
    history = experiment_noisy.optimize_rotations(num_steps=100, verbose=False)
    
    results_noise.append({
        'rate': rate,
        'metric': history['metric'][-1]
    })

# Plot results
import matplotlib.pyplot as plt
rates = [r['rate'] for r in results_noise]
metric_values = [r['metric'] for r in results_noise]
plt.semilogx(rates, metric_values, 'o-', linewidth=2)
plt.xlabel('Relaxation Rate (rad/s)')
plt.ylabel('Optimized Metric')
plt.title('Noise Sensitivity')
plt.grid(True)
plt.show()
```

### Example 6: Multi-Qubit Circuit with Layers

```python
from qsopt.core.circuit import create_layer, create_entangling_layer

# Create 3-qubit variational circuit
circuit = QuantumCircuit(num_qubits=3)

# Layer 1: Rotation gates
rot_layer_1 = create_layer(RYGate, num_qubits=3, theta=0.0, trainable=True)
for gate_app in rot_layer_1:
    circuit.add_gate(gate_app.gate, target=gate_app.target)

# Layer 2: Entangling gates
ent_layer = create_entangling_layer(CNOTGate, num_qubits=3, pattern="linear")
for gate_app in ent_layer:
    circuit.add_gate(gate_app.gate, target=gate_app.target)

# Layer 3: More rotations
rot_layer_2 = create_layer(RYGate, num_qubits=3, theta=0.0, trainable=True)
for gate_app in rot_layer_2:
    circuit.add_gate(gate_app.gate, target=gate_app.target)

# 6 trainable parameters total
params = circuit.get_trainable_parameters()
print(f"Number of trainable parameters: {len(params)}")
```

---

## Advanced Topics

### Custom Optimizers

Use different JAX/Optax optimizers:

```python
import optax

# Adam optimizer
history = experiment.optimize_rotations(
    num_steps=100,
    learning_rate=0.01,
    optimizer='adam'  # default
)

# SGD with momentum
history = experiment.optimize_rotations(
    num_steps=100,
    learning_rate=0.05,
    optimizer='sgd'
)

# RMSprop
history = experiment.optimize_rotations(
    num_steps=100,
    learning_rate=0.01,
    optimizer='rmsprop'
)
```

### Parameter Constraints

Define bounds and periodicity:

```python
from qsopt.core import ParameterConstraints

# Periodic constraints for rotation angles
constraints = ParameterConstraints(
    lower_bound=0.0,
    upper_bound=2 * np.pi,
    periodic=True  # θ and θ + 2π are equivalent
)

params.add_rotation_angles(
    names=['theta1'],
    initial_values=[np.pi/4],
    constraints=constraints
)
```

### Custom Initial States from Circuits

Integrate circuit-prepared states with sensing experiments:

```python
# Create circuit for state preparation
def prepare_custom_state(theta: float):
    circuit = QuantumCircuit(num_qubits=1)
    circuit.add_gate(RYGate(theta=theta), target=0)
    U = circuit.get_unitary()
    psi0 = basis(2, 0)
    return U * psi0

# Use in experiment
custom_psi = prepare_custom_state(np.pi/3)
initial_state = InitialStateConfig(
    state_type=InitialStateType.CUSTOM,
    custom_state=custom_psi
)

exp_params = ExperimentalParameters(
    physical_constants=constants,
    system_dims=dims,
    measurement=measurement,
    initial_state=initial_state,
    noise_config=noise
)
```

### JAX-Compatible Loss Functions

Define custom loss functions for optimization:

```python
import jax
import jax.numpy as jnp

def custom_loss_fn(params_dict, circuit, target_metric):
    """Custom loss function using JAX autodiff."""
    circuit.set_trainable_parameters(params_dict)
    U = circuit.get_unitary_jax()
    
    # Define loss (example: minimize distance from target unitary)
    loss = jnp.sum(jnp.abs(U - target_metric)**2)
    return loss

# Use with JAX optimizer
params = circuit.get_trainable_parameters()
loss, grads = jax.value_and_grad(custom_loss_fn)(params, circuit, target)
```

---

## Best Practices

### 1. System Dimensions
- Start with `cavity_levels=2` for faster computation
- Increase to 3-5 only if modeling coherent/thermal states
- `qubit_levels=2` is standard for qubit systems
- `field_levels=2` for binary external field detection

### 2. Measurement Timing
- Normalize times by system timescale (e.g., 1/χ or 1/g)
- Use 3-5 measurement points for balance of accuracy and speed
- Span the pulse duration (typically -5 to +5 in normalized units)

### 3. Learning Rates
- Start with `learning_rate=0.05` for SGD
- Use `learning_rate=0.01-0.03` for Adam optimizer
- Monitor gradient magnitudes via visualization
- Decrease learning rate if optimization oscillates

### 4. Parameter Initialization
- Use physically motivated initial values (e.g., π/2 for Hadamard-like)
- Try multiple random initializations for global optimization
- Check if parameters are stuck in local minima

### 5. Noise Modeling
- Start with ideal (noiseless) case to establish upper bound
- Add realistic noise: T1 relaxation > T2 pure dephasing
- Typical rates: ~10^-4 × 2π rad/s for high-Q systems

### 6. Circuit Depth
- Keep circuits shallow (3-5 gates per qubit) for better optimization
- Use layer structure: rotation → entangling → rotation
- Verify circuit unitarity with tests

### 7. Visualization
- Always use `plot_optimization_dashboard` to monitor convergence
- Check gradient magnitudes to diagnose optimization issues
- Save figures for reproducibility

---

## Troubleshooting

### Optimization Not Converging

**Symptoms:** Contrast oscillates or doesn't improve

**Solutions:**
1. Reduce learning rate (try 0.01 instead of 0.05)
2. Increase number of steps (try 200 instead of 100)
3. Check gradient magnitudes via dashboard
4. Try different initial parameter values
5. Verify physical parameters are reasonable

```python
# Debug: Monitor gradients
history = experiment.optimize_rotations(num_steps=100, learning_rate=0.05, verbose=True)
avg_grad = np.mean([np.abs(g) for g in history['gradients']])
print(f"Average gradient magnitude: {avg_grad:.6e}")

# If too small (< 1e-6): increase learning rate
# If too large (> 1.0): decrease learning rate
```

### Low Sensing Contrast

**Symptoms:** Optimized contrast remains < 0.1

**Solutions:**
1. Check measurement times span the pulse timescale
2. Verify physical constants are in correct units (rad/s)
3. Test ideal (noiseless) limit to establish upper bound
4. Try different initial state types
5. Increase system dimensions if truncation too aggressive

```python
# Debug: Test ideal limit
ideal_noise = NoiseConfiguration(relaxation=0.0, dephasing=0.0)
exp_params_ideal = ExperimentalParameters(..., noise_config=ideal_noise)
experiment_ideal = SingleQubitExperiment(exp_params_ideal, params)
ideal_contrast = experiment_ideal.run_simulation().contrast
print(f"Ideal contrast (no noise): {ideal_contrast:.6f}")
```

### Memory Issues

**Symptoms:** Out of memory errors for large Hilbert spaces

**Solutions:**
1. Reduce `cavity_levels` (use 2 instead of 3+)
2. Reduce number of measurement times
3. Use fewer optimization steps or smaller batches

```python
# Memory-efficient configuration
dims_small = SystemDimensions(
    cavity_levels=2,  # Minimal for qubit sensing
    qubit_levels=2,
    field_levels=2
)

measurement_sparse = MeasurementProtocol(
    measurement_times=[-5.0, 0.0, 5.0]  # Only 3 points
)
```

### JAX Tracer Errors

**Symptoms:** "Cannot multiply JAX tracer with QuTiP Qobj"

**Solutions:**
1. Use `get_unitary_jax()` instead of `get_unitary()` for JAX operations
2. Ensure gate parameters are JAX arrays (jnp.array)
3. Don't mix QuTiP Qobj with JAX autodiff directly

```python
# Correct: Use JAX-compatible method
U_jax = circuit.get_unitary_jax()  # Returns jnp.ndarray

# Incorrect: Using QuTiP Qobj in JAX function
U_qobj = circuit.get_unitary()  # Returns Qobj - don't use in jax.grad
```

---

## Project Structure

```
Quantum-sensing-QML/
├── src/qsopt/
│   └── core/
│       ├── gates.py                 # JAX-compatible quantum gates (472 lines)
│       ├── circuit.py               # Quantum circuit builder
│       ├── experiment.py            # Sensing experiment classes
│       ├── trainable_parameters.py  # Parameter management
│       ├── visualization.py         # Plotting functions
│       └── callbacks.py             # Optimization callbacks
│   └── tests/
│       └── test_circuit.py          # Circuit tests (33 passing)
├── tests/
│   └── test_gates.py                # Gate tests (53 passing)
├── docs/
│   ├── INDEX.md                     # This file
│   ├── gates_and_circuits.md        # Gates & circuits guide
│   ├── experiments.md               # Experiments guide
│   └── visualization.md             # Visualization guide
├── examples/
│   ├── basic_sensing.py             # Basic sensing example
│   ├── circuit_state_prep.py        # Circuit examples
│   └── optimization_tutorial.py     # Optimization walkthrough
└── README.md                        # Project overview
```

**Test Coverage:** 86 total tests (all passing)
- Gate tests: 53 (verified against QuTiP)
- Circuit tests: 33 (verified against QuTiP circuit builder)

---

## References

### Core Technologies
- **QuTiP:** Quantum Toolbox in Python - https://qutip.org/
- **JAX:** Autograd and XLA - https://jax.readthedocs.io/
- **Optax:** Gradient optimization library - https://optax.readthedocs.io/

### Theory Background
- **Dispersive Readout:** Blais et al., Phys. Rev. A 69, 062320 (2004)
- **Circuit QED:** See Bibliography/Circuit QED/ for references
- **Quantum Sensing:** See Bibliography/QML for quantum sensing/

### Related Documentation
- [Gates & Circuits](./gates_and_circuits.md) - Quantum gate implementation details
- [Experiments](./experiments.md) - Sensing protocol documentation  
- [Visualization](./visualization.md) - Plotting and analysis tools

---

## Contributing

Contributions welcome! Areas of interest:
1. Additional gate types (Toffoli, SWAP, etc.)
2. Multi-qubit sensing protocols
3. Advanced optimization algorithms
4. Hardware-specific noise models
5. Extended visualization tools

---

## License

[Specify license information]

---

## Contact

[Repository owner contact information]

---

*Documentation last updated: January 2025*
*Library version: 1.0*
*Test suite: 86/86 passing*

- Use fewer measurement points
- Enable JAX JIT compilation (automatic)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{qsopt2025,
  title={Quantum Sensing Optimization Library},
  author={Bordoni, Simone and Campioni, Nathan},
  year={2025},
  url={https://github.com/Simone-Bordoni/Quantum-sensing-QML}
}
```

### Example 6: Saving and Loading Experiment Reports

Save your experimental configuration for reproducibility:

```python
# After running an optimization
experiment = SingleQubitExperiment(exp_params, params)
history = experiment.optimize_rotations(theta_init=[1.5, -1.3], num_steps=100)

# Save comprehensive report with all parameters and optimization data
experiment.save_experiment_report('results/my_experiment.json')
# This creates:
#   - results/my_experiment.json (all experimental parameters, metadata)
#   - results/my_experiment_callback.npz (detailed optimization data)

# Later, load the configuration
loaded = SingleQubitExperiment.load_experiment_report('results/my_experiment.json')

# Access all experiment details
exp_config = loaded['experimental_params_dict']
trainable_config = loaded['trainable_params_dict']
optimization_data = loaded['callback_data']  # epochs, metric, loss, etc.

print(f"Experiment type: {loaded['experiment_type']}")
print(f"Final metric: {optimization_data['metric'][-1]}")
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

**Last Updated:** January 2025  
**Version:** 0.1.0

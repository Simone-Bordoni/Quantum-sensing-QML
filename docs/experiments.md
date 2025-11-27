# Quantum Sensing Experiments Documentation

Complete guide to quantum sensing simulations and optimization in the qsopt library.

## Overview

The experiments module provides tools for:
- **Quantum sensing protocol simulation** with realistic noise models
- **Gradient-based optimization** using JAX autodiff
- **Custom state preparation** including circuit-based states
- **Measurement protocol design** with adaptive timing
- **Parameter management** for sensing optimization

**Module:** `src/qsopt/core/experiment.py`

---

## Quick Start

### Basic Sensing Experiment

```python
from qsopt.core import (
    SingleQubitExperiment, ExperimentalParameters, PhysicalConstants,
    SystemDimensions, MeasurementProtocol, InitialStateConfig,
    InitialStateType, NoiseConfiguration, TrainableParameters
)
import numpy as np

# Define physical system
gm = 0.03 * 2 * np.pi
constants = PhysicalConstants(
    chi=0.5 * gm,                    # Dispersive coupling
    photon_cavity_coupling=gm,       # Cavity-field coupling
    inverse_pulse_width=0.1 * gm     # Pulse shaping
)

# Configure experiment
exp_params = ExperimentalParameters(
    physical_constants=constants,
    system_dims=SystemDimensions(
        cavity_levels=2,   # Fock states in cavity
        qubit_levels=2,    # Qubit states
        field_levels=2     # External field states
    ),
    measurement=MeasurementProtocol(
        measurement_times=[-5.0, 0.0, 5.0]  # Three measurement points
    ),
    initial_state=InitialStateConfig(
        state_type=InitialStateType.SINGLE_PHOTON  # |1⟩ in cavity
    ),
    noise_config=NoiseConfiguration(
        relaxation=0.0001 * 2 * np.pi,   # T1 decay
        dephasing=0.0001 * 2 * np.pi     # T2 dephasing
    )
)

# Define trainable rotation angles
params = TrainableParameters()
params.add_rotation_angles(
    names=['theta1', 'theta2'],
    initial_values=[np.pi/2, -np.pi/2]
)

# Create experiment
experiment = SingleQubitExperiment(exp_params, params)

# Run single simulation
results = experiment.run_simulation()
print(f"Initial contrast: {results.contrast:.6f}")

# Optimize rotations
history = experiment.optimize_rotations(
    num_steps=100,
    learning_rate=0.05,
    verbose=True
)
print(f"Optimized contrast: {history['contrast'][-1]:.6f}")
```

---

## ExperimentalParameters

Complete configuration for quantum sensing experiments.

### PhysicalConstants

Physical parameters of the quantum system:

```python
@dataclass
class PhysicalConstants:
    """Physical constants for the sensing protocol."""
    chi: float                      # Dispersive coupling strength (rad/s)
    photon_cavity_coupling: float   # Cavity-field coupling g (rad/s)
    inverse_pulse_width: float      # 1/τ for pulse shaping (rad/s)
    
# Example: Circuit QED parameters
gm = 0.03 * 2 * np.pi  # 30 kHz typical cavity coupling
constants = PhysicalConstants(
    chi=0.5 * gm,          # Chi ~ g/2
    photon_cavity_coupling=gm,
    inverse_pulse_width=0.1 * gm  # Pulse width ~ 10/g
)
```

### SystemDimensions

Hilbert space truncation:

```python
@dataclass
class SystemDimensions:
    """Dimensions for truncated Hilbert space."""
    cavity_levels: int    # Number of Fock states in cavity
    qubit_levels: int     # Qubit levels (typically 2)
    field_levels: int     # External field states (2 for binary)
    
# Typical configuration
dims = SystemDimensions(
    cavity_levels=3,   # |0⟩, |1⟩, |2⟩ photon states
    qubit_levels=2,    # |g⟩, |e⟩ qubit states
    field_levels=2     # Field present or absent
)
```

### MeasurementProtocol

Define when measurements occur:

```python
@dataclass
class MeasurementProtocol:
    """Measurement timing configuration."""
    measurement_times: List[float]  # Time points for measurements (arbitrary units)
    
# Example: Three-point measurement
measurement = MeasurementProtocol(
    measurement_times=[-5.0, 0.0, 5.0]
)

# Example: Dense sampling
measurement = MeasurementProtocol(
    measurement_times=np.linspace(-10, 10, 21).tolist()
)
```

### InitialStateConfig

Configure initial quantum state:

```python
@dataclass
class InitialStateConfig:
    """Initial state configuration."""
    state_type: InitialStateType    # Predefined state type
    custom_state: Optional[Qobj]    # Custom state (if state_type=CUSTOM)

class InitialStateType(Enum):
    """Available initial state types."""
    GROUND = "ground"               # |0⟩ cavity, |g⟩ qubit
    SINGLE_PHOTON = "single_photon" # |1⟩ cavity, |g⟩ qubit
    SUPERPOSITION = "superposition" # (|0⟩ + |1⟩)/√2 cavity
    CUSTOM = "custom"               # User-defined state

# Predefined states
config1 = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)

# Custom state from circuit
from qsopt.core.circuit import QuantumCircuit
from qsopt.core.gates import RYGate
circuit = QuantumCircuit(num_qubits=1)
circuit.add_gate(RYGate(theta=np.pi/4), target=0)
custom_psi = circuit.get_unitary() * basis(2, 0)

config2 = InitialStateConfig(
    state_type=InitialStateType.CUSTOM,
    custom_state=custom_psi
)
```

### NoiseConfiguration

Decoherence and noise parameters:

```python
@dataclass
class NoiseConfiguration:
    """Noise and decoherence parameters."""
    relaxation: float = 0.0     # T1 relaxation rate (rad/s)
    dephasing: float = 0.0      # T2 dephasing rate (rad/s)
    thermal_photons: float = 0.0  # Thermal occupation number

# Realistic noise
noise = NoiseConfiguration(
    relaxation=0.0001 * 2 * np.pi,    # Weak T1 decay
    dephasing=0.0001 * 2 * np.pi,     # Weak pure dephasing
    thermal_photons=0.01              # Small thermal background
)

# Ideal (noiseless) limit
ideal_noise = NoiseConfiguration(
    relaxation=0.0,
    dephasing=0.0,
    thermal_photons=0.0
)
```

### Complete Example

```python
# Full experimental configuration
exp_params = ExperimentalParameters(
    physical_constants=PhysicalConstants(
        chi=0.5 * gm,
        photon_cavity_coupling=gm,
        inverse_pulse_width=0.1 * gm
    ),
    system_dims=SystemDimensions(
        cavity_levels=3,
        qubit_levels=2,
        field_levels=2
    ),
    measurement=MeasurementProtocol(
        measurement_times=[-5.0, -2.5, 0.0, 2.5, 5.0]
    ),
    initial_state=InitialStateConfig(
        state_type=InitialStateType.SINGLE_PHOTON
    ),
    noise_config=NoiseConfiguration(
        relaxation=0.0001 * 2 * np.pi,
        dephasing=0.0001 * 2 * np.pi,
        thermal_photons=0.0
    )
)
```

---

## TrainableParameters

Manage parameters for gradient-based optimization.

### Adding Parameters

```python
from qsopt.core import TrainableParameters

params = TrainableParameters()

# Add rotation angles (most common)
params.add_rotation_angles(
    names=['theta1', 'theta2', 'theta3'],
    initial_values=[0.5, 1.0, -0.3]
)

# Add custom parameters
params.add_custom_parameters(
    names=['coupling_strength', 'pulse_width'],
    initial_values=[0.05, 2.0]
)
```

### Parameter Constraints

```python
from qsopt.core import ParameterConstraints

# Constrained parameter
params.add_rotation_angles(
    names=['theta'],
    initial_values=[np.pi/4],
    constraints=ParameterConstraints(
        lower_bound=0.0,
        upper_bound=2*np.pi,
        periodic=True  # θ and θ+2π are equivalent
    )
)
```

### Accessing Parameters

```python
# Get all rotation angles
rotations = params.get_rotation_angles()
# {'theta1': 0.5, 'theta2': 1.0, 'theta3': -0.3}

# Get parameter vector for optimization
param_vector = params.get_parameter_vector()
# [0.5, 1.0, -0.3]

# Update parameters
params.set_parameter_vector([0.6, 1.1, -0.2])
```

---

## SingleQubitExperiment

Main experiment class for single-qubit sensing protocols.

### Initialization

```python
experiment = SingleQubitExperiment(
    experimental_params: ExperimentalParameters,
    trainable_params: TrainableParameters
)
```

### Running Simulations

```python
# Single simulation run
results = experiment.run_simulation()

# Results contain:
results.contrast              # Sensing contrast metric
results.measurement_outcomes  # Measurement results at each time
results.evolved_states        # Quantum states at measurement times
results.expectation_values    # Observable expectations
```

### Optimization Methods

#### optimize_rotations

Optimize rotation angles for maximum sensing contrast:

```python
history = experiment.optimize_rotations(
    num_steps: int = 100,           # Number of optimization steps
    learning_rate: float = 0.05,    # Adam learning rate
    verbose: bool = True,           # Print progress
    callback_freq: int = 10         # Callback frequency
)

# Returns dictionary with:
history['contrast']       # Contrast at each step
history['parameters']     # Parameter trajectory
history['gradients']      # Gradient magnitudes
history['loss']          # Loss values (1 - contrast)
```

**Example: Standard optimization**
```python
history = experiment.optimize_rotations(
    num_steps=200,
    learning_rate=0.05,
    verbose=True
)

import matplotlib.pyplot as plt
plt.plot(history['contrast'])
plt.xlabel('Optimization Step')
plt.ylabel('Sensing Contrast')
plt.title('Contrast Evolution')
plt.show()
```

#### optimize_measurement_times

Search for optimal measurement intervals:

```python
results = experiment.optimize_measurement_times(
    time_range: Tuple[float, float] = (-10.0, 10.0),  # Search range
    resolution: int = 50,                              # Grid resolution
    mode: str = 'exhaustive',                          # Search mode
    batch_size: int = 10,                              # Parallel batch size
    verbose: bool = True
)

# Returns dictionary:
results['optimal_times']      # Best measurement times found
results['best_contrast']      # Contrast at optimal times
results['contrast_landscape'] # Full landscape grid
results['computation_time']   # Search duration
```

**Example: Find optimal 3-point measurement**
```python
results = experiment.optimize_measurement_times(
    time_range=(-15.0, 15.0),
    resolution=30,
    mode='exhaustive'
)

print(f"Optimal times: {results['optimal_times']}")
print(f"Best contrast: {results['best_contrast']:.6f}")
```

### Computing Sensing Contrast

```python
# Get current contrast
contrast = experiment.get_sensing_contrast()

# Contrast definition: C = |⟨O⟩_field - ⟨O⟩_no_field| / (⟨O⟩_field + ⟨O⟩_no_field)
# where O is the measured observable
```

---

## Optimization Workflows

### Basic Workflow

```python
# 1. Setup
exp_params = ExperimentalParameters(...)
params = TrainableParameters()
params.add_rotation_angles(['theta1', 'theta2'], [0.5, -0.5])
experiment = SingleQubitExperiment(exp_params, params)

# 2. Initial evaluation
initial_results = experiment.run_simulation()
print(f"Initial contrast: {initial_results.contrast:.6f}")

# 3. Optimize
history = experiment.optimize_rotations(num_steps=100, learning_rate=0.05)

# 4. Evaluate optimized
final_results = experiment.run_simulation()
print(f"Final contrast: {final_results.contrast:.6f}")
print(f"Improvement: {final_results.contrast - initial_results.contrast:.6f}")
```

### Advanced Workflow with Multiple Optimizations

```python
# Multi-stage optimization
experiment = SingleQubitExperiment(exp_params, params)

# Stage 1: Coarse optimization
history1 = experiment.optimize_rotations(
    num_steps=50,
    learning_rate=0.1,  # Higher LR for coarse search
    verbose=True
)

# Stage 2: Fine-tuning
history2 = experiment.optimize_rotations(
    num_steps=100,
    learning_rate=0.01,  # Lower LR for fine-tuning
    verbose=True
)

# Stage 3: Optimize measurement times with optimized rotations
time_results = experiment.optimize_measurement_times(
    time_range=(-20.0, 20.0),
    resolution=40
)

print(f"Final contrast: {time_results['best_contrast']:.6f}")
```

### Parameter Sweep

```python
# Sweep over different initial conditions
initial_angles = [
    [0.0, 0.0],
    [np.pi/4, np.pi/4],
    [np.pi/2, -np.pi/2],
    [np.pi, 0.0]
]

results_sweep = []
for init_angles in initial_angles:
    params = TrainableParameters()
    params.add_rotation_angles(['theta1', 'theta2'], init_angles)
    
    experiment = SingleQubitExperiment(exp_params, params)
    history = experiment.optimize_rotations(num_steps=100, verbose=False)
    
    results_sweep.append({
        'initial': init_angles,
        'final_contrast': history['contrast'][-1],
        'final_params': history['parameters'][-1]
    })

# Find best initialization
best_result = max(results_sweep, key=lambda x: x['final_contrast'])
print(f"Best initialization: {best_result['initial']}")
print(f"Best contrast: {best_result['final_contrast']:.6f}")
```

---

## Noise Sensitivity Analysis

Study how noise affects sensing performance:

```python
# Define noise levels to test
relaxation_rates = np.logspace(-5, -2, 10) * 2 * np.pi
results_noise = []

for rate in relaxation_rates:
    # Create experiment with specific noise
    noise = NoiseConfiguration(relaxation=rate, dephasing=rate/2)
    exp_params_noisy = ExperimentalParameters(
        physical_constants=constants,
        system_dims=dims,
        measurement=measurement,
        initial_state=initial_state,
        noise_config=noise
    )
    
    experiment = SingleQubitExperiment(exp_params_noisy, params)
    history = experiment.optimize_rotations(num_steps=100, verbose=False)
    
    results_noise.append({
        'relaxation_rate': rate,
        'final_contrast': history['contrast'][-1]
    })

# Plot noise sensitivity
rates = [r['relaxation_rate'] for r in results_noise]
contrasts = [r['final_contrast'] for r in results_noise]

plt.semilogx(rates, contrasts, 'o-')
plt.xlabel('Relaxation Rate (rad/s)')
plt.ylabel('Optimized Contrast')
plt.title('Noise Sensitivity')
plt.grid(True)
plt.show()
```

---

## Integration with Circuits

Use circuit-prepared states in experiments:

```python
from qsopt.core.circuit import QuantumCircuit
from qsopt.core.gates import RYGate, HadamardGate
from qutip import basis, tensor

# Create parameterized initial state
def prepare_initial_state(theta: float):
    """Prepare RY(θ)|0⟩ state."""
    circuit = QuantumCircuit(num_qubits=1)
    circuit.add_gate(RYGate(theta=theta), target=0)
    U = circuit.get_unitary()
    return U * basis(2, 0)

# Use in experiment
initial_state = InitialStateConfig(
    state_type=InitialStateType.CUSTOM,
    custom_state=prepare_initial_state(np.pi/3)
)

exp_params = ExperimentalParameters(
    physical_constants=constants,
    system_dims=dims,
    measurement=measurement,
    initial_state=initial_state,
    noise_config=noise
)

experiment = SingleQubitExperiment(exp_params, trainable_params)
```

### Optimizing Both Circuit and Sensing Parameters

```python
# Joint optimization of state preparation and sensing rotations
circuit = QuantumCircuit(num_qubits=1)
circuit.add_gate(RYGate(theta=0.0, trainable=True), target=0)

# Sensing parameters
sensing_params = TrainableParameters()
sensing_params.add_rotation_angles(['theta_sense'], [0.5])

# Optimization loop
for epoch in range(10):
    # Update initial state from circuit
    U = circuit.get_unitary()
    psi0 = U * basis(2, 0)
    
    exp_params.initial_state.custom_state = psi0
    experiment = SingleQubitExperiment(exp_params, sensing_params)
    
    # Optimize sensing parameters
    history = experiment.optimize_rotations(num_steps=20, verbose=False)
    
    # Update circuit parameters (gradient step)
    circuit_params = circuit.get_trainable_parameters()
    # ... gradient update logic ...
    
    print(f"Epoch {epoch}, Contrast: {history['contrast'][-1]:.6f}")
```

---

## Two-Qubit Experiments

The `TwoQubitExperiment` class extends sensing to two-qubit systems:

```python
from qsopt.core import TwoQubitExperiment

# Configure 2-qubit system
dims_2q = SystemDimensions(
    cavity_levels=2,
    qubit_levels=2,  # Per qubit
    field_levels=2
)

# Two-qubit parameters
params_2q = TrainableParameters()
params_2q.add_rotation_angles(
    names=['theta1_q1', 'theta2_q1', 'theta1_q2', 'theta2_q2'],
    initial_values=[0.5, -0.5, 0.3, -0.3]
)

# Create experiment
experiment_2q = TwoQubitExperiment(exp_params_2q, params_2q)

# Optimize
history = experiment_2q.optimize_rotations(num_steps=150, learning_rate=0.03)
```

---

## Best Practices

### Initialization

```python
# Good: Physically motivated initial angles
params.add_rotation_angles(
    names=['theta1', 'theta2'],
    initial_values=[np.pi/2, -np.pi/2]  # Hadamard-like operations
)

# Less ideal: Random initialization
initial_values=[np.random.uniform(0, 2*np.pi) for _ in range(2)]
```

### Learning Rates

```python
# Start with moderate learning rate
history = experiment.optimize_rotations(
    num_steps=50,
    learning_rate=0.05
)

# If converging slowly, increase
history = experiment.optimize_rotations(
    num_steps=50,
    learning_rate=0.1
)

# If oscillating, decrease
history = experiment.optimize_rotations(
    num_steps=100,
    learning_rate=0.01
)
```

### Measurement Timing

```python
# Good: Measurements around pulse timescale
pulse_width = 1.0 / constants.inverse_pulse_width
measurement_times = [-2*pulse_width, 0, 2*pulse_width]

# Less ideal: Measurements too far from pulse
measurement_times = [-100, 0, 100]  # May miss signal
```

### Noise Modeling

```python
# Realistic noise: T1 > T2_pure
noise = NoiseConfiguration(
    relaxation=0.0001 * 2 * np.pi,      # T1 = 10000 time units
    dephasing=0.00015 * 2 * np.pi,      # T2_pure shorter than T1
    thermal_photons=0.01
)
```

---

## Troubleshooting

### Low Contrast

**Problem:** Optimized contrast remains low

**Solutions:**
1. Check measurement times - ensure they span the pulse timescale
2. Try different initial parameter values
3. Verify physical constants are reasonable
4. Increase system dimensions if truncation is too aggressive
5. Reduce noise levels to establish ideal limit

```python
# Debug: Check ideal limit
ideal_noise = NoiseConfiguration(relaxation=0.0, dephasing=0.0)
exp_params_ideal = ExperimentalParameters(..., noise_config=ideal_noise)
experiment_ideal = SingleQubitExperiment(exp_params_ideal, params)
ideal_contrast = experiment_ideal.run_simulation().contrast
print(f"Ideal contrast: {ideal_contrast:.6f}")
```

### Optimization Not Converging

**Problem:** Contrast oscillates or doesn't improve

**Solutions:**
1. Reduce learning rate
2. Increase number of steps
3. Try multi-stage optimization
4. Check gradient magnitudes

```python
# Monitor gradients
history = experiment.optimize_rotations(num_steps=100, learning_rate=0.05, verbose=True)
avg_grad = np.mean([np.abs(g) for g in history['gradients']])
print(f"Average gradient magnitude: {avg_grad:.6f}")

# If too small: increase learning rate
# If too large: decrease learning rate
```

### Memory Issues

**Problem:** Out of memory for large Hilbert spaces

**Solutions:**
1. Reduce system dimensions
2. Use fewer measurement times
3. Process in batches

```python
# Reduce dimensions
dims = SystemDimensions(
    cavity_levels=2,  # Instead of 3 or more
    qubit_levels=2,
    field_levels=2
)

# Fewer measurements
measurement = MeasurementProtocol(
    measurement_times=[-5.0, 0.0, 5.0]  # Instead of 21 points
)
```

---

## See Also

- [Gates & Circuits Documentation](./gates_and_circuits.md) - Circuit-based state preparation
- [Visualization Documentation](./visualization.md) - Plotting optimization results
- [Trainable Parameters Guide](./trainable_parameters.md) - Advanced parameter management
- [Examples](../examples/) - Complete sensing experiment examples

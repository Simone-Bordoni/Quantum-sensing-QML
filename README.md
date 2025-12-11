# Quantum Sensing Optimization Library (qsopt)

A specialized Python library for **gradient-based optimization of quantum sensing protocols** using JAX-compatible quantum circuits and automatic differentiation. Designed for optimizing dispersive qubit-cavity interactions in quantum sensing experiments, with applications to dark matter detection and quantum state preparation.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![QuTiP](https://img.shields.io/badge/QuTiP-JAX%20compatible-green.svg)](https://qutip.org/)
[![JAX](https://img.shields.io/badge/JAX-autodiff-orange.svg)](https://jax.readthedocs.io/)
[![Test Windows](https://github.com/Simone-Bordoni/Quantum-sensing-QML/workflows/Test%20Installation%20and%20Run%20Tests/badge.svg)](https://github.com/Simone-Bordoni/Quantum-sensing-QML/actions/workflows/test-windows.yml)
[![CI Tests](https://github.com/Simone-Bordoni/Quantum-sensing-QML/workflows/Comprehensive%20CI%20Tests/badge.svg)](https://github.com/Simone-Bordoni/Quantum-sensing-QML/actions/workflows/ci.yml)

## Overview

This library focuses on **quantum sensing optimization** through gradient-based parameter tuning of rotation gates and initial state preparation. The core functionality enables:

- **Protocol optimization**: Maximize photon detection sensitivity in dispersive readout schemes
- **Noise-aware optimization**: Realistic decoherence models with relaxation and dephasing
- **Parameter landscape analysis**: Systematic exploration of sensing contrast vs rotation angles
- **Gradient-based state preparation**: Optimize initial quantum states for enhanced sensing
- **Benchmarking**: Compare optimized protocols against standard approaches

### Supporting Infrastructure: JAX-Compatible Quantum Circuits

To enable gradient-based optimization of initial states and rotation protocols, the library includes a **JAX-compatible quantum circuit framework**:

- **Parameterized gates**: RX, RY, RZ with trainable rotation angles
- **Fixed gates**: Hadamard, CNOT, CZ for state preparation
- **Parameter management**: Automatic tracking of trainable parameters with gradient control
- **JAX integration**: Circuit unitaries as JAX arrays for automatic differentiation
- **QuTiP verified**: All gates match QuTiP implementations exactly (86 tests ✓)

This circuit infrastructure supports the main sensing optimization by providing:
- Gradient-optimizable initial state preparation circuits
- Differentiable rotation gate operations
- Efficient parameter management for optimization loops

### System Architecture

**Primary: Quantum Sensing Framework** (three-subsystem composite Hilbert space):
1. **Input Cavity Mode**: Controls photon injection with temporal pulse shaping
2. **Resonator Cavity Mode**: Main sensing element coupled to the qubit detector  
3. **Two-Level Qubit**: Quantum sensor subject to relaxation and dephasing noise

**Sensing Optimization Workflow:**
```
Initial State |ψ₀⟩ → Ry(θ₁) → H(t) Evolution → Ry(θ₂) → Measurement → Contrast
                ↓                                                          ↓
         Gradient ← ← ← ← ← ← JAX Autodiff ← ← ← ← ← ← ← ← ← ← ← ← ← ←┘
                ↓
         Optimizer → Updated (|ψ₀⟩, θ₁, θ₂)
```

**Supporting: Quantum Circuit Framework** (for state preparation):
```
Gate → Matrix(θ) → JAX Array → Circuit Unitary → Initial State |ψ₀⟩
  ↑                                                               ↓
  └← ← ← ← Gradient ← ← JAX Autodiff ← ← Sensing Contrast ← ← ←┘
```

## Quick Start

## Installation

**Quick Start:**
```bash
pip install -e .
```

**With development tools:**
```bash
pip install -e ".[dev]"
```

**Using Poetry:**
```bash
poetry install
```

For detailed installation instructions, troubleshooting, and requirements verification, see **[INSTALLATION.md](./INSTALLATION.md)**.

### Verification

Test your installation:
```bash
pytest src/qsopt/tests/test_installation.py -v
```

## Usage Examples

### Example 1: Quantum Sensing Optimization (Primary Use Case)

```python
import numpy as np
from qsopt.core import (
    ExperimentalParameters, PhysicalConstants, SystemDimensions,
    NoiseConfiguration, MeasurementProtocol, InitialStateConfig,
    InitialStateType, TrainableParameters, SingleQubitExperiment
)

# Define physical system
physical_constants = PhysicalConstants(
    chi=0.5 * 0.03 * 2 * np.pi,  # Dispersive coupling
    photon_cavity_coupling=0.03 * 2 * np.pi,
    inverse_pulse_width=0.1 * 0.03 * 2 * np.pi
)

system_dims = SystemDimensions(
    cavity_levels=2, qubit_levels=2, field_levels=2
)

# Configure measurement and noise
measurement = MeasurementProtocol(measurement_times=[-5.0, 0.0, 5.0])
noise_config = NoiseConfiguration(
    relaxation=0.0001 * 2 * np.pi,
    dephasing=0.0001 * 2 * np.pi
)
initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)

# Create experimental parameters
exp_params = ExperimentalParameters(
    physical_constants=physical_constants,
    system_dims=system_dims,
    measurement=measurement,
    initial_state=initial_state,
    noise_config=noise_config
)

# Define trainable parameters
trainable_params = TrainableParameters()
trainable_params.add_rotation_angles(
    names=['theta1', 'theta2'],
    initial_values=[np.pi/2, -np.pi/2]
)

# Create and optimize experiment
experiment = SingleQubitExperiment(exp_params, trainable_params)
history = experiment.optimize_rotations(num_steps=100, learning_rate=0.05, verbose=True)

print(f"Final contrast: {history['contrast'][-1]:.6f}")
print(f"Optimized θ₁: {trainable_params.parameters[0].value:.3f}")
print(f"Optimized θ₂: {trainable_params.parameters[1].value:.3f}")
```

### Example 2: Building Quantum Circuits for State Preparation

```python
import numpy as np
import jax.numpy as jnp
from qsopt.core.circuit import QuantumCircuit
from qsopt.core.gates import RXGate, RYGate, CNOTGate, HadamardGate

# Create a 2-qubit parameterized quantum circuit
circuit = QuantumCircuit(num_qubits=2)

# Add gates with target qubits
circuit.add_gate(HadamardGate(), target=0)
circuit.add_gate(RXGate(theta=jnp.pi/4, trainable=True), target=1)
circuit.add_gate(CNOTGate(), target=(0, 1))
circuit.add_gate(RYGate(theta=jnp.pi/2, trainable=True), target=0)

# Get circuit unitary as JAX array (ready for autodiff)
U = circuit.get_unitary_jax()
print(f"Unitary shape: {U.shape}, dtype: {U.dtype}")

# Access and update trainable parameters
params = circuit.get_trainable_parameters()
print(f"Trainable parameters: {params}")

# Update parameters (e.g., from optimizer)
circuit.set_trainable_parameters({
    "gate_1_theta": jnp.array(jnp.pi/2),
    "gate_3_theta": jnp.array(jnp.pi)
})

# Visualize circuit
print(circuit.draw())
```

**Output:**
```
q0: --[H]------------------------*--------[RY(theta=3.1416)]--
q1: ---------[RX(theta=1.5708)]--[CNOT]---------------------
```

### Example 3: Gradient-Based Initial State Optimization

```python
import jax
import jax.numpy as jnp
from qsopt.core.circuit import QuantumCircuit
from qsopt.core.gates import RXGate, RYGate
import qutip as qt

# Create parameterized initial state preparation circuit
def prepare_initial_state(params):
    circuit = QuantumCircuit(num_qubits=1)
    circuit.add_gate(RYGate(theta=params[0], trainable=True), target=0)
    circuit.add_gate(RXGate(theta=params[1], trainable=True), target=0)
    
    U = circuit.get_unitary()
    psi0 = qt.basis(2, 0)
    return U * psi0

# Optimize initial state for sensing (connect to sensing experiment)
@jax.jit
def sensing_objective(params):
    psi_init = prepare_initial_state(params)
    # ... connect to sensing experiment, compute contrast
    # contrast = experiment.compute_contrast(psi_init)
    return -contrast  # Minimize negative contrast

# Gradient-based optimization of initial state
grad_fn = jax.grad(sensing_objective)
params = jnp.array([0.0, 0.0])

import optax
optimizer = optax.adam(learning_rate=0.05)
opt_state = optimizer.init(params)

for step in range(100):
    gradient = grad_fn(params)
    updates, opt_state = optimizer.update(gradient, opt_state)
    params = optax.apply_updates(params, updates)
    
print(f"Optimized state preparation: RY({params[0]:.4f}) RX({params[1]:.4f})")
```

For complete examples, see the [examples/](./examples/) directory.

## Key Features

### Quantum Sensing Optimization (Primary Focus)

**Core Capabilities:**
- **Time-Dependent Hamiltonians**: Full support for time-varying coupling with Gaussian pulse functions
- **JAX Automatic Differentiation**: End-to-end differentiable quantum simulations
- **Lindblad Master Equation**: Realistic open quantum system dynamics with configurable noise
- **Composite Hilbert Space**: Three-subsystem architecture (input ⊗ resonator ⊗ qubit)
- **Flexible Optimization**: Built-in optax integration with multiple optimizer choices

**Optimization Callbacks:**

Track detailed optimization metrics including loss, contrast, and detection probabilities:

```python
from qsopt import OptimizationCallback

# Create callback to track optimization progress
callback = OptimizationCallback(save_every=1, save_best=True)

# Run optimization with callback
history = experiment.optimize_rotations(
    num_steps=100,
    learning_rate=0.05,
    verbose=True,
    callback=callback
)

# Access tracked metrics
print(f"Best contrast: {callback.best_metrics['contrast']:.6f}")
print(f"Best parameters: {callback.get_best_parameters()}")

# Save results for later analysis
callback.save('optimization_results.npz')
```

**Tracked Metrics:**
- Loss function values and gradients
- Sensing contrast (P_with - P_without)
- Detection probabilities
- Parameter evolution
- Best parameters and corresponding metrics

### JAX-Compatible Quantum Circuits (Supporting Infrastructure)

**Gate Library:**
- **Rotation Gates**: `RXGate(θ)`, `RYGate(θ)`, `RZGate(θ)` with differentiable parameters
- **Fixed Gates**: `HadamardGate()`, `CNOTGate()`, `CZGate()`
- **Parameter Control**: Enable/disable gradients per gate or parameter
- **QuTiP Verified**: All gates match QuTiP implementations exactly (tested)

**Circuit Features (for state preparation and gradient optimization):**
- Multi-qubit circuits with flexible gate targeting
- Automatic parameter tracking and management
- JAX array unitaries for autodiff
- Circuit visualization with `draw()` method
- Utility functions for layered circuit construction

**Example - Preparing Optimizable Initial States:**
```python
from qsopt.core.circuit import QuantumCircuit, create_layer
from qsopt.core.gates import RYGate, RXGate

# Create parameterized initial state preparation
prep_circuit = QuantumCircuit(num_qubits=1)
create_layer(prep_circuit, RYGate, [0.1], trainable=True)
create_layer(prep_circuit, RXGate, [0.2], trainable=True)

# Get initial state for sensing experiment
U_prep = prep_circuit.get_unitary_jax()
# ... use U_prep in sensing optimization

# Get trainable parameters for joint optimization
params = prep_circuit.get_trainable_parameters()
```

### Previous: Quantum Sensing Optimization

**Optimization Callbacks:**

Track detailed optimization metrics including loss, contrast, and detection probabilities:

```python
from qsopt import OptimizationCallback

# Create callback to track optimization progress
callback = OptimizationCallback(save_every=1, save_best=True)

# Run optimization with callback
history = experiment.optimize_rotations(
    num_steps=100,
    learning_rate=0.05,
    verbose=True,
    callback=callback
)

# Access tracked metrics
print(f"Best contrast: {callback.best_metrics['contrast']:.6f}")
print(f"Best parameters: {callback.get_best_parameters()}")

# Save results for later analysis
callback.save('optimization_results.npz')
```

**Tracked Metrics:**
- Loss function values and gradients
- Sensing contrast (P_with - P_without)
- Detection probabilities
- Parameter evolution
- Best parameters and corresponding metrics

**Core Capabilities:**
- **Time-Dependent Hamiltonians**: Full support for time-varying coupling with Gaussian pulse functions
- **JAX Automatic Differentiation**: End-to-end differentiable quantum simulations
- **Lindblad Master Equation**: Realistic open quantum system dynamics with configurable noise
- **Composite Hilbert Space**: Three-subsystem architecture (input ⊗ resonator ⊗ qubit)
- **Flexible Optimization**: Built-in optax integration with multiple optimizer choices

## Project Structure

```
Quantum-sensing-QML/
├── src/qsopt/                      # Main library package
│   ├── core/                       # Core components
│   │   ├── gates.py               # JAX-compatible quantum gates
│   │   ├── circuit.py             # Quantum circuit builder
│   │   ├── experiment/            # Quantum sensing experiments
│   │   ├── experimental_parameters.py  # System configuration
│   │   ├── trainable_parameters.py     # Parameter management
│   │   ├── loss_functions.py      # Optimization objectives
│   │   └── callback.py            # Optimization tracking
│   ├── tests/                     # Unit tests
│   │   ├── test_gates.py          # Gate implementation tests (53 tests ✓)
│   │   └── test_circuit.py        # Circuit tests (33 tests ✓)
│   ├── optimization/              # Optimization algorithms
│   ├── protocols/                 # Sensing protocols
│   ├── analysis/                  # Visualization & benchmarking
│   └── utils/                     # Utilities
├── examples/                      # Jupyter notebooks & demos
│   ├── jax_quantum_gates_optimization.ipynb    # Gate optimization demo
│   ├── quantum_circuit.ipynb                   # Circuit construction
│   ├── single_qubit_tutorial.ipynb            # 1-qubit sensing
│   ├── two_qubit_tutorial.ipynb               # 2-qubit sensing
│   └── sweep_experiments_tutorial.ipynb       # Parameter sweeps
├── experiments/                   # Research experiments & data
├── tests/                        # Integration tests
└── docs/                         # Documentation
```

## Examples and Tutorials

### Quantum Sensing Examples (Primary Focus)

**1. Single Qubit Tutorial** (`single_qubit_tutorial.ipynb`)
- Basic quantum sensing setup and protocol optimization
- Gradient-based rotation angle optimization
- Noise analysis and robustness studies

**2. Two Qubit Tutorial** (`two_qubit_tutorial.ipynb`)
- Two-qubit sensing protocols
- Multi-qubit parameter optimization
- Entanglement effects in sensing

**3. Sweep Experiments** (`sweep_experiments_tutorial.ipynb`)
- Parameter space exploration
- Systematic sweeps over physical parameters
- Performance analysis across parameter ranges

### Quantum Circuit Examples (Supporting Tools)

**4. JAX Quantum Gates Optimization** (`jax_quantum_gates_optimization.ipynb`)
- Demonstrates gradient-based optimization of quantum gates
- Custom VJP (vector-Jacobian product) for JAX compatibility
- Single and multi-gate optimization examples
- Foundation for state preparation optimization

**5. Quantum Circuit Construction** (`quantum_circuit.ipynb`)
- Building circuits for initial state preparation
- Parameter management for optimization
- Circuit unitary computation
- Integration with sensing experiments

## Testing

The library includes comprehensive test suites:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_gates.py          # Gate implementation tests (53 tests)
pytest src/qsopt/tests/test_circuit.py  # Circuit tests (33 tests)

# Run with coverage
pytest --cov=qsopt --cov-report=html
```

**Test Coverage:**
- ✓ All rotation gates (RX, RY, RZ) match QuTiP implementations exactly
- ✓ All fixed gates (H, CNOT, CZ) verified against QuTiP
- ✓ Gate unitarity (U†U = I) confirmed for all gates
- ✓ Circuit unitaries match QuTiP circuit builder results
- ✓ Parameter management (get/set/update) functionality
- ✓ Gradient control (enable/disable) per gate
- ✓ Multi-qubit gate expansion to full Hilbert space
- ✓ Bell state preparation circuits
- ✓ JAX array format for autodiff integration

## API Reference

### Gates Module (`qsopt.core.gates`)

**Rotation Gates** (parametrized):
```python
RXGate(theta, trainable=True)   # Rotation around X-axis
RYGate(theta, trainable=True)   # Rotation around Y-axis  
RZGate(theta, trainable=True)   # Rotation around Z-axis
```

**Fixed Gates** (non-parametrized):
```python
HadamardGate()    # Hadamard gate
CNOTGate()        # Controlled-NOT gate
CZGate()          # Controlled-Z gate
```

**Gate Methods:**
- `matrix()` - Get gate matrix as QuTiP Qobj
- `get_parameter(name)` - Get parameter value
- `set_parameter(value, name)` - Update parameter
- `enable_gradients()` / `disable_gradients()` - Control gradient tracing
- `is_trainable(name)` - Check if parameter is trainable

### Circuit Module (`qsopt.core.circuit`)

**QuantumCircuit Class:**
```python
circuit = QuantumCircuit(num_qubits=2)
circuit.add_gate(gate, target)           # Add gate to circuit
circuit.get_unitary()                    # Get unitary as QuTiP Qobj
circuit.get_unitary_jax()                # Get unitary as JAX array
circuit.get_trainable_parameters()       # Get all trainable params
circuit.set_trainable_parameters(dict)   # Update parameters
circuit.enable_gradients()               # Enable gradient tracing
circuit.disable_gradients()              # Disable gradient tracing
circuit.draw()                           # Visualize circuit
```

**Utility Functions:**
```python
create_layer(circuit, gate_type, params, qubits, trainable)
create_entangling_layer(circuit, gate_type, pattern)
```

### Experimental Parameters (`qsopt.core`)

**Configuration Classes:**
- `PhysicalConstants` - System coupling strengths and frequencies
- `SystemDimensions` - Hilbert space dimensions
- `NoiseConfiguration` - Relaxation and dephasing rates
- `MeasurementProtocol` - Measurement times and settings
- `InitialStateConfig` - Initial state specification
- `TrainableParameters` - Parameter management for optimization

## Research Applications

This library has been developed for and applied to:

**Primary Applications:**
- **Dark Matter Detection**: Optimizing quantum sensors for axion searches via dispersive readout
- **Protocol Optimization**: Maximizing sensing contrast through gradient-based parameter tuning
- **Noise Resilience Analysis**: Understanding optimization benefits under realistic decoherence
- **Parameter Space Studies**: Exploring rotation angle landscapes for optimal sensing
- **Benchmarking**: Quantifying improvements over standard π/2 rotation protocols

**Supporting Applications:**
- **Initial State Optimization**: Using quantum circuits to prepare optimal sensing states
- **Gradient-Based State Preparation**: Differentiable state engineering for enhanced sensitivity
- **Multi-Parameter Optimization**: Joint optimization of initial states and rotation angles

## Dependencies

**Core Dependencies:**
- **QuTiP** (≥4.7): Quantum Toolbox in Python
- **qutip-qip** (≥0.3): Quantum information processing add-on
- **JAX** (≥0.4): Automatic differentiation and JIT compilation  
- **Optax** (≥0.1): JAX-based optimization algorithms
- **NumPy** (≥1.21): Numerical computing
- **SciPy** (≥1.8): Scientific computing utilities

**Optional:**
- **Matplotlib** (≥3.5): Visualization and plotting
- **Jupyter**: Interactive notebooks

**JAX Backend Configuration:**

For optimal performance with QuTiP, configure the JAX backend:
```python
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Use CPU backend
# For GPU: os.environ["JAX_PLATFORM_NAME"] = "gpu"
```

## Performance

**Gradient Computation:**
- Automatic differentiation through quantum circuits via JAX
- Custom VJP for gates where needed for QuTiP compatibility
- JIT compilation for accelerated execution

**Benchmarks:**
- Single-qubit gate optimization: ~50 iterations in <1 second
- Multi-gate circuit gradients: Computed in milliseconds
- Circuit unitaries: Cached and reused during optimization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

**Areas for Contribution:**
- Additional quantum gates (SWAP, Toffoli, etc.)
- More circuit construction patterns
- Advanced optimization strategies
- Extended test coverage
- Documentation improvements

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/Simone-Bordoni/Quantum-sensing-QML.git
cd Quantum-sensing-QML
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=qsopt

# Format code
black src/
isort src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **QuTiP Development Team** for the quantum toolbox and JAX backend
- **Google JAX Team** for automatic differentiation framework
- **DeepMind Optax Team** for optimization algorithms

## Contact

**Simone Bordoni**  
Email: simone.bordoni@uniroma1.it  
GitHub: [@Simone-Bordoni](https://github.com/Simone-Bordoni)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{quantum_sensing_qml_2025,
  title={Quantum Sensing \& QML Library (qsopt)},
  author={Bordoni, Simone},
  year={2025},
  url={https://github.com/Simone-Bordoni/Quantum-sensing-QML},
  note={JAX-compatible quantum circuits and sensing optimization}
}
```

## References

**Quantum Sensing:**
- Circuit QED and dispersive readout techniques
- Quantum sensors for dark matter detection
- Parameter optimization in quantum protocols

**Quantum Machine Learning:**
- Parameterized quantum circuits (PQC)
- Variational quantum algorithms
- Gradient-based quantum optimization

---

*For detailed examples and tutorials, see the [examples/](./examples/) directory.*

*For API documentation, see the docstrings in the source code.*

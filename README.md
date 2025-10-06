# Quantum Sensing Optimization Library (qsopt)

A specialized Python library for **parameter optimization in quantum sensing experiments** using QuTiP-JAX backend for automatic differentiation. This library focuses on optimizing rotation gate parameters to maximize photon detection sensitivity through dispersive qubit-cavity interactions.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![QuTiP](https://img.shields.io/badge/QuTiP-JAX%20compatible-green.svg)](https://qutip.org/)
[![JAX](https://img.shields.io/badge/JAX-autodiff-orange.svg)](https://jax.readthedocs.io/)

## 🔬 Overview

This library emerged from quantum sensing research for dark matter detection experiments, providing tools to:

- **Optimize quantum sensing protocols** under realistic noise conditions
- **Compare parameterization strategies** (θ₁,θ₂ vs θ,Δθ approaches)  
- **Analyze parameter space landscapes** with visualization tools
- **Benchmark against standard protocols** with comprehensive metrics
- **Perform noise sensitivity analysis** across different decoherence regimes

### System Architecture

The simulation employs a **three-subsystem composite Hilbert space**:

1. **Input Cavity Mode**: Controls photon injection with temporal pulse shaping
2. **Resonator Cavity Mode**: Main sensing element coupled to the qubit detector  
3. **Two-Level Qubit**: Quantum sensor subject to relaxation and dephasing noise

**Quantum Sensing Workflow:**
```
|ψ₀⟩ → Ry(θ₁) → H(t) Evolution → Ry(θ₂) → Measurement → Detection Probability
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Simone-Bordoni/Quantum-sensing-QML.git
cd Quantum-sensing-QML

# Install dependencies
pip install -r requirements.txt

# Install the library in development mode
pip install -e .
```

### Basic Usage

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
history = experiment.optimize(num_steps=100, learning_rate=0.05, verbose=True)

print(f"Final contrast: {history['contrast'][-1]:.6f}")
print(f"Optimized θ₁: {trainable_params.parameters[0].value:.3f}")
print(f"Optimized θ₂: {trainable_params.parameters[1].value:.3f}")
```

For a complete walkthrough, see the [Example notebook](./examples/Example.ipynb).

## 📊 Key Features

### ⚡ Completed Implementation

The `SingleQubitExperiment` class provides a complete quantum sensing framework with:

- **Time-Dependent Hamiltonians**: Full support for time-varying coupling using Gaussian pulse functions
- **JAX Automatic Differentiation**: End-to-end differentiable quantum simulations for gradient-based optimization
- **Lindblad Master Equation**: Realistic open quantum system dynamics with configurable noise models
- **Composite Hilbert Space**: Three-subsystem architecture (input cavity ⊗ resonator ⊗ qubit)
- **Flexible Optimization**: Built-in optax integration with customizable optimizers and learning rates

### 🎯 Parameter Optimization
- **Multiple optimizers**: Adam, SGD, RMSprop, AdamW with automatic differentiation via JAX
- **Learning rate scheduling**: Exponential decay and adaptive strategies
- **Convergence monitoring**: Real-time gradient tracking and early stopping
- **Time-dependent gradients**: Fully differentiable through complex time-evolution operators

### 📈 Analysis & Visualization
- **Training dashboards**: 6-panel optimization monitoring with parameter evolution
- **Parameter landscapes**: 2D visualization of sensing contrast vs rotation angles
- **Benchmarking tools**: Performance comparison against standard protocols
- **Noise sensitivity**: Systematic analysis across decoherence strength levels

### 🔧 Protocol Comparison
- **Standard protocol**: θ₁=π/2, θ₂=-π/2 baseline performance
- **Optimized protocols**: Data-driven parameter selection
- **Multiple parameterizations**: Direct θ₁,θ₂ vs centered θ,Δθ strategies

## 📁 Project Structure

```
quantum-sensing-opt/
├── src/qsopt/                    # Main library package
│   ├── core/                     # Simulation engine & measurements  
│   ├── optimization/             # Optimizers & objective functions
│   ├── protocols/                # Standard & optimized sensing protocols
│   ├── analysis/                 # Visualization & benchmarking
│   └── utils/                    # JAX compatibility & parameterizations
├── notebooks/                    # Jupyter examples & tutorials
│   ├── Multiple_measurement_opt.ipynb    # Multi-measurement optimization
│   ├── Parameter_space_landscape.ipynb  # Parameter space analysis
│   └── Optimization_with_noise.ipynb    # Noise robustness studies
├── examples/                     # Standalone Python examples
└── tests/                        # Unit tests
```

## 📚 Examples & Notebooks

### 1. Multiple Measurement Optimization
Demonstrates gradient-based optimization across sequential measurements with comprehensive performance analysis.

### 2. Parameter Space Landscape Analysis  
Comparative study of different parameterization strategies with 2D landscape visualization.

### 3. Noise Robustness Studies
Systematic analysis of optimization benefits under varying decoherence conditions.

## 🛠️ Core Components

### Optimization Engine
```python
from qsopt.optimization import QuantumSensingOptimizer

optimizer = QuantumSensingOptimizer(
    optimizer_name='adam',
    learning_rate=0.05,
    use_lr_schedule=True
)
```

### Visualization Tools
```python
from qsopt.analysis import plot_optimization_dashboard, plot_parameter_landscape

# Training progress dashboard
plot_optimization_dashboard(optimization_history)

# Parameter space analysis
plot_parameter_landscape(theta_range, contrast_data)
```

### Protocol Benchmarking
```python
from qsopt.analysis import compare_protocols

comparison = compare_protocols(
    standard_results, 
    optimized_results,
    metrics=['contrast', 'detection_prob', 'improvement_ratio']
)
```

## 🔬 Research Applications

This library has been used for:

- **Dark matter detection**: Optimizing quantum sensors for axion searches
- **Parameter space studies**: Comparing θ₁,θ₂ vs θ,Δθ parameterization strategies  
- **Noise resilience analysis**: Understanding optimization benefits under decoherence
- **Protocol benchmarking**: Quantifying improvements over standard sensing approaches

## 📖 Dependencies

- **QuTiP** (≥4.7): Quantum Toolbox in Python with JAX backend
- **JAX** (≥0.4): Automatic differentiation and JIT compilation  
- **Optax** (≥0.1): JAX-based optimization algorithms
- **NumPy** (≥1.21): Numerical computing foundation
- **Matplotlib** (≥3.5): Visualization and plotting
- **SciPy** (≥1.8): Scientific computing utilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

**Simone Bordoni**  
Email: [your.email@domain.com]  
GitHub: [@Simone-Bordoni](https://github.com/Simone-Bordoni)

## � Code Quality

This project maintains high code quality through automated tools:

- **Pylint**: Static code analysis with scientific Python configuration
- **Black**: Consistent code formatting (100 characters line length)
- **isort**: Import statement organization
- **GitHub Actions**: Automated testing and linting on all PRs

### Running Quality Checks

```bash
# Run complete analysis with automatic fixes
python run_pylint.py --fix

# Windows users
run_pylint.bat

# Unix/Linux users  
./run_pylint.sh
```

See [LINTING.md](./LINTING.md) for detailed information about code quality tools and configuration.

## �📚 Citation

If you use this library in your research, please consider citing:

```bibtex
@misc{quantum-sensing-opt,
  title={Quantum Sensing Optimization Library},
  author={Bordoni, Simone},
  year={2025},
  url={https://github.com/Simone-Bordoni/Quantum-sensing-QML}
}
```

---

*For detailed examples and advanced usage, see the [notebooks](./notebooks/) directory and [documentation](./docs/).*

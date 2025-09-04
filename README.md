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
from qsopt import QuantumSensingOptimizer, standard_protocol

# Initialize optimizer
optimizer = QuantumSensingOptimizer('adam', learning_rate=0.05)

# Set up initial parameters
initial_params = [np.pi/2 + 0.01, -np.pi/2 + 0.01]  # θ₁, θ₂

# Run optimization
result = optimizer.optimize_sensing_contrast(
    initial_params, 
    max_iterations=200,
    tolerance=1e-6
)

print(f"Optimized angles: θ₁={result.theta1:.3f}, θ₂={result.theta2:.3f}")
print(f"Sensing contrast: {result.contrast:.6f}")
```

## 📊 Key Features

### 🎯 Parameter Optimization
- **Multiple optimizers**: Adam, SGD, RMSprop, AdamW with automatic differentiation
- **Learning rate scheduling**: Exponential decay and adaptive strategies
- **Convergence monitoring**: Real-time gradient tracking and early stopping

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

## 📚 Citation

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

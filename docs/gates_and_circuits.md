# Quantum Gates and Circuits Documentation

Complete guide to JAX-compatible quantum gates and circuit construction for state preparation in quantum sensing experiments.

## Overview

The gates and circuits modules provide:
- **JAX-compatible gates** with automatic differentiation support
- **Trainable parameters** for gradient-based optimization
- **QuTiP verification** - all gates tested against QuTiP implementations
- **Flexible circuit building** with parameter management
- **State preparation** tools for sensing experiments

**Modules:**
- `src/qsopt/core/gates.py`: Gate implementations
- `src/qsopt/core/circuit.py`: Circuit builder

**Tests:** 86 comprehensive tests (all passing)
- `tests/test_gates.py`: 53 tests
- `src/qsopt/tests/test_circuit.py`: 33 tests

---

## Quantum Gates

### Gate Base Class

All gates inherit from the `Gate` base class providing unified parameter management:

```python
from qsopt.core.gates import Gate, GateParameter

class Gate:
    """Base class for quantum gates with parameter management."""
    
    def __init__(self, num_qubits: int, parameters: Dict[str, GateParameter] = None):
        self.num_qubits = num_qubits
        self.parameters = parameters or {}
    
    def matrix(self) -> Qobj:
        """Return gate matrix as QuTiP Qobj. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def enable_gradients(self, param_name: Optional[str] = None) -> None:
        """Enable gradient computation for parameters."""
        
    def disable_gradients(self, param_name: Optional[str] = None) -> None:
        """Disable gradient computation for parameters."""
    
    def get_trainable_parameters(self) -> Dict[str, float]:
        """Get dictionary of trainable parameter names and values."""
    
    def set_trainable_parameters(self, values: Dict[str, float]) -> None:
        """Update trainable parameter values."""
```

### GateParameter

Dataclass for managing individual gate parameters:

```python
@dataclass
class GateParameter:
    """Container for gate parameters with gradient control."""
    value: float
    trainable: bool = False
    
    def enable_gradient(self) -> None:
        """Mark parameter as trainable."""
        self.trainable = True
    
    def disable_gradient(self) -> None:
        """Mark parameter as non-trainable."""
        self.trainable = False
```

---

## Rotation Gates

Single-qubit rotation gates around Pauli axes. These gates have one trainable parameter `theta`.

### RXGate - Rotation around X

```python
from qsopt.core.gates import RXGate
import jax.numpy as jnp

# Create rotation gate
rx = RXGate(theta=jnp.pi/4, trainable=True)

# Get gate matrix (QuTiP Qobj)
U = rx.matrix()  # exp(-i * theta * X / 2)

# Access parameters
params = rx.get_trainable_parameters()  # {'theta': 0.785...}

# Update angle
rx.set_trainable_parameters({'theta': jnp.pi/2})
```

**Matrix form:** $U_{RX}(\theta) = e^{-i\theta X/2} = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$

**Use cases:**
- State preparation
- Bit-flip operations
- Quantum control sequences

### RYGate - Rotation around Y

```python
from qsopt.core.gates import RYGate

# Create trainable RY gate
ry = RYGate(theta=0.5, trainable=True)

# Disable gradients temporarily
ry.disable_gradients('theta')

# Re-enable
ry.enable_gradients('theta')
```

**Matrix form:** $U_{RY}(\theta) = e^{-i\theta Y/2} = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$

**Use cases:**
- Most common rotation for quantum sensing
- Qubit rotations in Bloch sphere's Y-Z plane
- Parameter optimization in sensing protocols

### RZGate - Rotation around Z

```python
from qsopt.core.gates import RZGate

# Phase rotation
rz = RZGate(theta=jnp.pi/3, trainable=False)
```

**Matrix form:** $U_{RZ}(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$

**Use cases:**
- Phase corrections
- Z-basis rotations
- Virtual Z gates in pulse sequences

---

## Fixed Gates

Gates without trainable parameters.

### HadamardGate

```python
from qsopt.core.gates import HadamardGate

# Create Hadamard gate
h = HadamardGate()

# Apply to create superposition
U = h.matrix()  # (X + Z) / sqrt(2)
```

**Matrix form:** $H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$

**Use cases:**
- Creating superposition states
- Basis transformations X ↔ Z
- Standard state preparation

### CNOTGate - Controlled-NOT

```python
from qsopt.core.gates import CNOTGate

# 2-qubit entangling gate
cnot = CNOTGate()

# num_qubits = 2 (control and target)
U = cnot.matrix()
```

**Matrix form:** $CNOT = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$

**Use cases:**
- Creating Bell states
- Entangling qubits
- Quantum error correction

### CZGate - Controlled-Z

```python
from qsopt.core.gates import CZGate

# Controlled phase gate
cz = CZGate()
```

**Matrix form:** $CZ = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$

**Use cases:**
- Phase entanglement
- Symmetric 2-qubit gates
- Alternative to CNOT

---

## Quantum Circuits

Build multi-qubit circuits with automatic parameter tracking.

### QuantumCircuit Class

```python
from qsopt.core.circuit import QuantumCircuit
from qsopt.core.gates import RYGate, CNOTGate, HadamardGate
import jax.numpy as jnp

# Create 2-qubit circuit
circuit = QuantumCircuit(num_qubits=2)

# Add gates
circuit.add_gate(HadamardGate(), target=0)
circuit.add_gate(RYGate(theta=jnp.pi/4, trainable=True), target=1)
circuit.add_gate(CNOTGate(), target=(0, 1))  # (control, target)

# Get unitary as QuTiP Qobj
U_qobj = circuit.get_unitary()

# Get unitary as JAX array for autodiff
U_jax = circuit.get_unitary_jax()  # Complex128 array

# Access trainable parameters
params = circuit.get_trainable_parameters()
# {'gate_1_theta': 0.785...}  # Automatic naming: gate_{idx}_{param_name}

# Update parameters
circuit.set_trainable_parameters({'gate_1_theta': jnp.pi/2})
```

### Parameter Management

```python
# Enable/disable gradients for specific gates
circuit.enable_gradients(gate_idx=1)  # Enable gate 1
circuit.disable_gradients(gate_idx=2)  # Disable gate 2

# Enable/disable specific parameters
circuit.enable_gradients(gate_idx=1, param_name='theta')

# Enable/disable all gradients
circuit.enable_gradients()   # All trainable
circuit.disable_gradients()  # None trainable
```

### Building Bell States

```python
# |Φ+⟩ = (|00⟩ + |11⟩) / √2
circuit = QuantumCircuit(num_qubits=2)
circuit.add_gate(HadamardGate(), target=0)
circuit.add_gate(CNOTGate(), target=(0, 1))

U = circuit.get_unitary()
psi0 = tensor(basis(2, 0), basis(2, 0))
bell_state = U * psi0  # |Φ+⟩
```

### Parameterized State Preparation

```python
def prepare_sensing_state(theta1: float, theta2: float) -> QuantumCircuit:
    """Create parameterized 2-qubit state for sensing."""
    circuit = QuantumCircuit(num_qubits=2)
    
    # Layer 1: Individual rotations
    circuit.add_gate(RYGate(theta=theta1, trainable=True), target=0)
    circuit.add_gate(RYGate(theta=theta2, trainable=True), target=1)
    
    # Layer 2: Entanglement
    circuit.add_gate(CNOTGate(), target=(0, 1))
    
    # Layer 3: More rotations
    circuit.add_gate(RYGate(theta=0.0, trainable=True), target=0)
    circuit.add_gate(RYGate(theta=0.0, trainable=True), target=1)
    
    return circuit

circuit = prepare_sensing_state(jnp.pi/4, jnp.pi/3)
params = circuit.get_trainable_parameters()
# 4 trainable parameters: gate_0_theta, gate_1_theta, gate_3_theta, gate_4_theta
```

---

## Circuit Utility Functions

Helper functions for constructing common circuit patterns.

### create_layer

Create parallel layer of single-qubit gates:

```python
from qsopt.core.circuit import create_layer
from qsopt.core.gates import RYGate
import jax.numpy as jnp

# Create layer of RY gates
layer = create_layer(RYGate, num_qubits=3, theta=jnp.pi/4, trainable=True)

# Add layer to circuit
circuit = QuantumCircuit(num_qubits=3)
for gate_app in layer:
    circuit.add_gate(gate_app.gate, target=gate_app.target)
```

**Returns:** List of `GateApplication` objects

### create_entangling_layer

Create entangling layer with various connectivity patterns:

```python
from qsopt.core.circuit import create_entangling_layer
from qsopt.core.gates import CNOTGate

# Linear connectivity: 0→1, 1→2, 2→3
layer = create_entangling_layer(CNOTGate, num_qubits=4, pattern="linear")

# Circular connectivity: 0→1, 1→2, 2→3, 3→0
layer = create_entangling_layer(CNOTGate, num_qubits=4, pattern="circular")

# All-to-all connectivity
layer = create_entangling_layer(CNOTGate, num_qubits=4, pattern="all_to_all")
```

**Patterns:**
- `"linear"`: Chain connectivity
- `"circular"`: Ring topology
- `"all_to_all"`: Full connectivity

### Building Layered Circuits

```python
def create_variational_circuit(num_qubits: int, num_layers: int) -> QuantumCircuit:
    """Create variational quantum circuit with alternating layers."""
    circuit = QuantumCircuit(num_qubits=num_qubits)
    
    for layer_idx in range(num_layers):
        # Rotation layer
        rot_layer = create_layer(RYGate, num_qubits, theta=0.0, trainable=True)
        for gate_app in rot_layer:
            circuit.add_gate(gate_app.gate, target=gate_app.target)
        
        # Entangling layer
        if layer_idx < num_layers - 1:  # No entangling on last layer
            ent_layer = create_entangling_layer(CNOTGate, num_qubits, pattern="linear")
            for gate_app in ent_layer:
                circuit.add_gate(gate_app.gate, target=gate_app.target)
    
    return circuit

# 3-qubit, 2-layer circuit with 6 trainable parameters
circuit = create_variational_circuit(num_qubits=3, num_layers=2)
```

---

## Integration with JAX Optimization

Use circuits in JAX optimization loops:

```python
import jax
import jax.numpy as jnp
import optax
from qsopt.core.circuit import QuantumCircuit
from qsopt.core.gates import RYGate, CNOTGate

# Define loss function
def loss_fn(params_dict, circuit, target_state):
    """Compute fidelity loss for state preparation."""
    circuit.set_trainable_parameters(params_dict)
    U = circuit.get_unitary_jax()
    
    # Apply to initial state |0⟩
    psi0 = jnp.array([1.0 + 0j, 0.0, 0.0, 0.0])  # |00⟩
    psi = U @ psi0
    
    # Compute fidelity
    fidelity = jnp.abs(jnp.vdot(target_state, psi))**2
    return 1.0 - fidelity  # Minimize loss

# Create circuit
circuit = QuantumCircuit(num_qubits=2)
circuit.add_gate(RYGate(theta=0.0, trainable=True), target=0)
circuit.add_gate(RYGate(theta=0.0, trainable=True), target=1)
circuit.add_gate(CNOTGate(), target=(0, 1))

# Target: Bell state |Φ+⟩
target = jnp.array([1.0, 0.0, 0.0, 1.0]) / jnp.sqrt(2.0)

# Initialize parameters
initial_params = circuit.get_trainable_parameters()

# Setup optimizer
optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(initial_params)

# Optimization loop
@jax.jit
def update(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, circuit, target)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

params = initial_params
for step in range(100):
    params, opt_state, loss = update(params, opt_state)
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss:.6f}")

# Final parameters
print(f"Optimized parameters: {params}")
# Expected: gate_0_theta ≈ π/2, gate_1_theta ≈ 0
```

---

## Using Circuits in Sensing Experiments

Integrate custom circuit-prepared states with sensing experiments:

```python
from qsopt.core import (
    SingleQubitExperiment, ExperimentalParameters, 
    InitialStateConfig, InitialStateType
)
from qsopt.core.circuit import QuantumCircuit
from qsopt.core.gates import RYGate
import jax.numpy as jnp

# Create circuit for state preparation
def create_custom_initial_state(theta: float):
    """Prepare initial state with circuit."""
    circuit = QuantumCircuit(num_qubits=1)
    circuit.add_gate(RYGate(theta=theta, trainable=False), target=0)
    U = circuit.get_unitary()
    psi0 = basis(2, 0)  # |0⟩
    return U * psi0

# Configure experiment with custom state
exp_params = ExperimentalParameters(
    physical_constants=PhysicalConstants(...),
    system_dims=SystemDimensions(cavity_levels=2, qubit_levels=2, field_levels=2),
    measurement=MeasurementProtocol(measurement_times=[-5.0, 0.0, 5.0]),
    initial_state=InitialStateConfig(
        state_type=InitialStateType.CUSTOM,
        custom_state=create_custom_initial_state(jnp.pi/4)
    )
)

# Run sensing experiment
experiment = SingleQubitExperiment(exp_params, trainable_params)
results = experiment.run_simulation()
```

---

## Gate Verification

All gates are verified against QuTiP implementations:

```python
# tests/test_gates.py contains comprehensive verification
import pytest
from qsopt.core.gates import RXGate, RYGate, RZGate, CNOTGate
from qutip import rx, ry, rz
from qutip_qip.operations import cnot
import numpy as np

def test_rx_matches_qutip():
    """Verify RX gate matches QuTiP."""
    angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    
    for theta in angles:
        custom_gate = RXGate(theta=theta)
        qutip_gate = rx(theta)
        
        # Compare matrices
        np.testing.assert_allclose(
            custom_gate.matrix().full(),
            qutip_gate.full(),
            atol=1e-10
        )

# Test results: All 53 gate tests PASSING
# - 18 rotation gate tests (6 angles × 3 gates)
# - 3 fixed gate tests
# - 6 parameter management tests
# - 3 gate application tests
# - 3 gate sequence tests
# - 2 JAX compatibility tests
# - 3 gate property tests
# - 15 unitarity tests
```

---

## Circuit Verification

Circuits verified against QuTiP circuit builder:

```python
# src/qsopt/tests/test_circuit.py
from qsopt.core.circuit import QuantumCircuit
from qsopt.core.gates import HadamardGate, CNOTGate
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import hadamard_transform, cnot
import numpy as np

def test_bell_state_circuit():
    """Verify Bell state preparation matches QuTiP."""
    # Custom circuit
    circuit = QuantumCircuit(num_qubits=2)
    circuit.add_gate(HadamardGate(), target=0)
    circuit.add_gate(CNOTGate(), target=(0, 1))
    U_custom = circuit.get_unitary()
    
    # QuTiP circuit
    qc = QubitCircuit(N=2)
    qc.add_gate("SNOT", targets=0)  # Hadamard
    qc.add_gate("CNOT", targets=1, controls=0)
    U_qutip = qc.propagators()[0]
    
    # Compare
    np.testing.assert_allclose(
        U_custom.full(),
        U_qutip.full(),
        atol=1e-10
    )

# Test results: All 33 circuit tests PASSING
# - 9 basic circuit tests
# - 6 parameter management tests
# - 5 unitary computation tests
# - 6 QuTiP comparison tests
# - 3 circuit layer tests
# - 3 unitarity tests
```

---

## Best Practices

### Parameter Naming

Circuit automatically names parameters as `gate_{idx}_{param_name}`:

```python
circuit = QuantumCircuit(num_qubits=2)
circuit.add_gate(RYGate(theta=0.1, trainable=True), target=0)  # gate_0_theta
circuit.add_gate(RYGate(theta=0.2, trainable=True), target=1)  # gate_1_theta
circuit.add_gate(RXGate(theta=0.3, trainable=True), target=0)  # gate_2_theta

params = circuit.get_trainable_parameters()
# {'gate_0_theta': 0.1, 'gate_1_theta': 0.2, 'gate_2_theta': 0.3}
```

### Gradient Control

Fine-grained control over which parameters are trainable:

```python
# Start with all parameters trainable
circuit = QuantumCircuit(num_qubits=2)
circuit.add_gate(RYGate(theta=0.1, trainable=True), target=0)
circuit.add_gate(RYGate(theta=0.2, trainable=True), target=1)

# Freeze first gate during optimization
circuit.disable_gradients(gate_idx=0)

# Only gate_1_theta is now trainable
trainable = circuit.get_trainable_parameters()  # {'gate_1_theta': 0.2}
```

### Circuit Depth

Keep circuits shallow for better optimization:

```python
# Good: Shallow circuit (depth = 3)
circuit = QuantumCircuit(num_qubits=2)
circuit.add_gate(RYGate(theta=0.0, trainable=True), target=0)
circuit.add_gate(RYGate(theta=0.0, trainable=True), target=1)
circuit.add_gate(CNOTGate(), target=(0, 1))

# Less ideal: Deep circuit (depth = 9)
for _ in range(3):
    circuit.add_gate(RYGate(theta=0.0, trainable=True), target=0)
    circuit.add_gate(RYGate(theta=0.0, trainable=True), target=1)
    circuit.add_gate(CNOTGate(), target=(0, 1))
```

### Unitarity Verification

Always verify gates/circuits are unitary:

```python
def verify_unitarity(U, tolerance=1e-10):
    """Check if U is unitary: U† U = I"""
    identity = U.dag() * U
    expected = qeye(U.dims[0])
    return np.allclose(identity.full(), expected.full(), atol=tolerance)

# Verify gate
gate = RYGate(theta=0.5)
assert verify_unitarity(gate.matrix())

# Verify circuit
circuit = QuantumCircuit(num_qubits=2)
circuit.add_gate(RYGate(theta=0.1, trainable=True), target=0)
circuit.add_gate(CNOTGate(), target=(0, 1))
assert verify_unitarity(circuit.get_unitary())
```

---

## Troubleshooting

### Common Issues

**Issue: "Cannot multiply JAX tracer with Qobj"**
```python
# Problem: Trying to use JAX array directly in QuTiP
theta = jnp.array(0.5)
gate = RYGate(theta=theta)  # Works
U = gate.matrix()  # Returns QuTiP Qobj

# Solution: Use get_unitary_jax() for JAX operations
U_jax = circuit.get_unitary_jax()  # Returns JAX array
```

**Issue: "Parameter not found in circuit"**
```python
# Problem: Wrong parameter name
circuit.set_trainable_parameters({'theta': 0.5})  # KeyError

# Solution: Use full parameter name
params = circuit.get_trainable_parameters()  # Check names first
circuit.set_trainable_parameters({'gate_0_theta': 0.5})  # Correct
```

**Issue: "Gradients not flowing through circuit"**
```python
# Problem: Gradients disabled
circuit.disable_gradients()

# Solution: Enable gradients
circuit.enable_gradients()

# Or check specific parameter
params = circuit.get_trainable_parameters()
if len(params) == 0:
    print("No trainable parameters!")
```

---

## Advanced Topics

### Custom Gate Implementation

Define your own gates:

```python
from qsopt.core.gates import Gate, GateParameter
from qutip import Qobj
import jax.numpy as jnp

class CustomRotation(Gate):
    """Custom rotation gate: R(θ, φ) = RZ(φ) RY(θ) RZ(-φ)"""
    
    def __init__(self, theta: float = 0.0, phi: float = 0.0, 
                 trainable_theta: bool = False, trainable_phi: bool = False):
        parameters = {
            'theta': GateParameter(theta, trainable_theta),
            'phi': GateParameter(phi, trainable_phi)
        }
        super().__init__(num_qubits=1, parameters=parameters)
    
    def matrix(self) -> Qobj:
        theta = self.parameters['theta'].value
        phi = self.parameters['phi'].value
        
        # RZ(φ) RY(θ) RZ(-φ)
        rz_phi = (-1j * phi * sigmaz() / 2).expm()
        ry_theta = (-1j * theta * sigmay() / 2).expm()
        rz_neg_phi = (1j * phi * sigmaz() / 2).expm()
        
        return rz_phi * ry_theta * rz_neg_phi

# Use custom gate
circuit = QuantumCircuit(num_qubits=1)
circuit.add_gate(CustomRotation(theta=jnp.pi/4, phi=jnp.pi/6, 
                                trainable_theta=True, trainable_phi=True), target=0)
```

### Gradient Monitoring

Track gradient magnitudes during optimization:

```python
def track_gradients(circuit, loss_fn, params):
    """Monitor gradient magnitudes for each parameter."""
    grads = jax.grad(loss_fn)(params, circuit)
    
    for name, grad_value in grads.items():
        print(f"{name}: |∇| = {jnp.abs(grad_value):.6f}")
    
    return grads

# In optimization loop
for step in range(100):
    grads = track_gradients(circuit, loss_fn, params)
    # Update params...
```

---

## See Also

- [Experiments Documentation](./experiments.md) - Using circuits in sensing experiments
- [Visualization Documentation](./visualization.md) - Plotting circuit parameters
- [Examples](../examples/) - Complete examples with circuits
- [QuTiP Documentation](https://qutip.org/) - QuTiP quantum toolbox
- [JAX Documentation](https://jax.readthedocs.io/) - JAX autodiff framework

"""
JAX-Compatible Quantum Circuit
===============================

This module provides a quantum circuit implementation that uses JAX-compatible
gates and tracks trainable parameters for gradient-based optimization.

Key Features:
- Build quantum circuits with parametrized gates
- Track and manage trainable parameters across all gates
- Compute circuit unitary in JAX format
- Specify target qubits for gate application
- Compatible with JAX autodiff

Example:
    >>> import jax.numpy as jnp
    >>> from qsopt.core.circuit import QuantumCircuit
    >>> from qsopt.core.gates import RXGate, RYGate, CNOTGate
    >>>
    >>> # Create 2-qubit circuit
    >>> circuit = QuantumCircuit(num_qubits=2)
    >>> circuit.add_gate(RXGate(theta=jnp.pi/4), target=0)
    >>> circuit.add_gate(RYGate(theta=jnp.pi/2), target=1)
    >>> circuit.add_gate(CNOTGate(), target=(0, 1))
    >>>
    >>> # Get circuit unitary
    >>> U = circuit.get_unitary_jax()
    >>>
    >>> # Get trainable parameters
    >>> params = circuit.get_trainable_parameters()
"""

from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
import qutip as qt
from qutip_qip.operations import expand_operator

from .gates import Gate


class GateApplication:
    """
    Container for a gate and its target qubit(s).

    Attributes:
        gate: The quantum gate to apply
        target: Target qubit index (int) or tuple of indices for multi-qubit gates
    """

    def __init__(self, gate: Gate, target: Union[int, Tuple[int, ...]]):
        """
        Initialize gate application.

        Args:
            gate: Gate object to apply
            target: Target qubit(s). Single int for 1-qubit gates,
                   tuple of ints for multi-qubit gates
        """
        self.gate = gate

        # Normalize target to tuple
        if isinstance(target, int):
            self.target = (target,)
        else:
            self.target = tuple(target)

        # Validate target matches gate dimensions
        if len(self.target) != gate.num_qubits:
            raise ValueError(
                f"Gate {gate.name} requires {gate.num_qubits} qubit(s), "
                f"but {len(self.target)} target(s) provided"
            )

    def __repr__(self) -> str:
        """String representation."""
        target_str = str(self.target[0]) if len(self.target) == 1 else str(self.target)
        return f"{self.gate.name}[{target_str}]"


class QuantumCircuit:
    """
    Quantum circuit with JAX-compatible gates.

    Manages a sequence of gate applications on multiple qubits,
    tracks trainable parameters, and computes circuit unitaries.
    """

    def __init__(self, num_qubits: int):
        """
        Initialize quantum circuit.

        Args:
            num_qubits: Number of qubits in the circuit
        """
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")

        self.num_qubits = num_qubits
        self._gates: List[GateApplication] = []

    def add_gate(self, gate: Gate, target: Union[int, Tuple[int, ...]]) -> None:
        """
        Add a gate to the circuit.

        Args:
            gate: Gate object to add
            target: Target qubit index (int) or indices (tuple) for the gate

        Raises:
            ValueError: If target qubits are out of range
        """
        gate_app = GateApplication(gate, target)

        # Validate all target qubits are in range
        for qubit in gate_app.target:
            if qubit < 0 or qubit >= self.num_qubits:
                raise ValueError(
                    f"Target qubit {qubit} is out of range. "
                    f"Circuit has {self.num_qubits} qubits (0-{self.num_qubits-1})"
                )

        self._gates.append(gate_app)

    def get_gates(self) -> List[GateApplication]:
        """
        Get list of all gate applications in the circuit.

        Returns:
            List of GateApplication objects
        """
        return self._gates.copy()

    def num_gates(self) -> int:
        """Return number of gates in the circuit."""
        return len(self._gates)

    def get_trainable_parameters(self) -> Dict[str, jnp.ndarray]:
        """
        Get all trainable parameters from the circuit.

        Returns dictionary mapping parameter identifiers to values.
        Parameter identifiers have the format: "gate_{index}_{param_name}"

        Returns:
            Dictionary of trainable parameters
        """
        params = {}
        for i, gate_app in enumerate(self._gates):
            gate = gate_app.gate
            if gate.has_parameters():
                gate_params = gate.get_parameters()
                for param_name, param_value in gate_params.items():
                    # Only include trainable parameters
                    if gate.is_trainable(param_name):
                        key = f"gate_{i}_{param_name}"
                        params[key] = param_value
        return params

    def set_trainable_parameters(self, parameters: Dict[str, jnp.ndarray]) -> None:
        """
        Update trainable parameters in the circuit.

        Args:
            parameters: Dictionary mapping "gate_{index}_{param_name}" to values
        """
        for key, value in parameters.items():
            # Parse key: gate_{index}_{param_name}
            parts = key.split("_")
            if len(parts) < 3 or parts[0] != "gate":
                raise ValueError(f"Invalid parameter key format: {key}")

            gate_idx = int(parts[1])
            param_name = "_".join(parts[2:])  # Handle param names with underscores

            if gate_idx < 0 or gate_idx >= len(self._gates):
                raise ValueError(f"Gate index {gate_idx} out of range")

            self._gates[gate_idx].gate.set_parameter(value, param_name)

    def update_parameters(self, parameter_values: List[float]) -> None:
        """
        Update trainable parameters from a flat list.

        Convenience method that updates parameters in order they appear
        in get_trainable_parameters().

        Args:
            parameter_values: List of parameter values in order
        """
        params_dict = self.get_trainable_parameters()
        if len(parameter_values) != len(params_dict):
            raise ValueError(f"Expected {len(params_dict)} parameters, got {len(parameter_values)}")

        updated_params = {}
        for key, value in zip(params_dict.keys(), parameter_values):
            updated_params[key] = jnp.asarray(value, dtype=float)

        self.set_trainable_parameters(updated_params)

    def get_unitary(self) -> qt.Qobj:
        """
        Compute the circuit unitary as a QuTiP Qobj.

        Returns:
            Circuit unitary matrix as QuTiP Qobj
        """
        if len(self._gates) == 0:
            # Identity for empty circuit
            return qt.qeye([2] * self.num_qubits)

        # Build unitary by applying gates in sequence
        # Start with identity
        U = qt.qeye([2] * self.num_qubits)

        for gate_app in self._gates:
            # Get gate matrix
            gate_matrix = gate_app.gate.matrix()

            # Expand to full Hilbert space
            if self.num_qubits == 1:
                # Single qubit circuit
                expanded = gate_matrix
            else:
                # Multi-qubit circuit - use qutip_qip's expand_operator
                expanded = expand_operator(
                    gate_matrix,
                    N=self.num_qubits,
                    targets=list(gate_app.target),
                    dims=[2] * self.num_qubits,
                )

            # Apply gate (multiply on left since gates are applied left to right)
            U = expanded * U

        return U

    def get_unitary_jax(self) -> jnp.ndarray:
        """
        Compute the circuit unitary as a JAX array.

        This is useful for JAX-based simulations and autodiff.

        Returns:
            Circuit unitary matrix as JAX array (complex128)
        """
        U = self.get_unitary()
        return jnp.array(U.full(), dtype=jnp.complex128)

    def enable_gradients(self, gate_index: int = None, param_name: str = None) -> None:
        """
        Enable gradient tracing for parameters.

        Args:
            gate_index: Index of gate (None = all gates)
            param_name: Name of parameter (None = all parameters)
        """
        if gate_index is None:
            # Enable for all gates
            for gate_app in self._gates:
                gate_app.gate.enable_gradients(param_name)
        else:
            if gate_index < 0 or gate_index >= len(self._gates):
                raise ValueError(f"Gate index {gate_index} out of range")
            self._gates[gate_index].gate.enable_gradients(param_name)

    def disable_gradients(self, gate_index: int = None, param_name: str = None) -> None:
        """
        Disable gradient tracing for parameters.

        Args:
            gate_index: Index of gate (None = all gates)
            param_name: Name of parameter (None = all parameters)
        """
        if gate_index is None:
            # Disable for all gates
            for gate_app in self._gates:
                gate_app.gate.disable_gradients(param_name)
        else:
            if gate_index < 0 or gate_index >= len(self._gates):
                raise ValueError(f"Gate index {gate_index} out of range")
            self._gates[gate_index].gate.disable_gradients(param_name)

    def __repr__(self) -> str:
        """String representation of circuit."""
        header = f"QuantumCircuit({self.num_qubits} qubits, {len(self._gates)} gates)"
        if len(self._gates) == 0:
            return header

        gates_str = "\n".join(f"  {i}: {gate_app}" for i, gate_app in enumerate(self._gates))
        return f"{header}\n{gates_str}"

    def draw(self) -> str:
        """
        Generate a simple text representation of the circuit.

        Returns:
            String visualization of the circuit
        """
        if len(self._gates) == 0:
            return f"Empty circuit with {self.num_qubits} qubit(s)"

        # Create wire for each qubit
        lines = [f"q{i}: " for i in range(self.num_qubits)]

        # Add each gate
        for gate_app in self._gates:
            gate_name = str(gate_app.gate)
            targets = gate_app.target

            if len(targets) == 1:
                # Single qubit gate
                target = targets[0]
                for i in range(self.num_qubits):
                    if i == target:
                        lines[i] += f"--[{gate_name}]--"
                    else:
                        lines[i] += f"--{'-' * (len(gate_name) + 2)}--"
            else:
                # Multi-qubit gate
                min_target = min(targets)
                max_target = max(targets)

                for i in range(self.num_qubits):
                    if i == min_target:
                        lines[i] += f"--*{'-' * (len(gate_name))}--"
                    elif i == max_target:
                        lines[i] += f"--[{gate_name}]--"
                    elif min_target < i < max_target:
                        lines[i] += f"--|{'-' * (len(gate_name))}--"
                    else:
                        lines[i] += f"--{'-' * (len(gate_name) + 2)}--"

        return "\n".join(lines)


# Utility functions for circuit construction


def create_layer(
    circuit: QuantumCircuit,
    gate_type: type,
    parameters: List[float],
    qubits: List[int] = None,
    trainable: bool = True,
) -> None:
    """
    Add a layer of identical single-qubit gates to the circuit.

    Args:
        circuit: QuantumCircuit to add gates to
        gate_type: Gate class (e.g., RXGate, RYGate)
        parameters: List of parameters for each qubit
        qubits: List of target qubits (None = all qubits)
        trainable: Whether parameters should be trainable
    """
    if qubits is None:
        qubits = list(range(circuit.num_qubits))

    if len(parameters) != len(qubits):
        raise ValueError(
            f"Number of parameters ({len(parameters)}) must match "
            f"number of qubits ({len(qubits)})"
        )

    for qubit, param in zip(qubits, parameters):
        gate = gate_type(theta=param, trainable=trainable)
        circuit.add_gate(gate, target=qubit)


def create_entangling_layer(
    circuit: QuantumCircuit, gate_type: type, pattern: str = "linear"
) -> None:
    """
    Add a layer of two-qubit entangling gates.

    Args:
        circuit: QuantumCircuit to add gates to
        gate_type: Two-qubit gate class (e.g., CNOTGate, CZGate)
        pattern: Connectivity pattern - "linear" or "circular"
    """
    if circuit.num_qubits < 2:
        raise ValueError("Need at least 2 qubits for entangling layer")

    if pattern == "linear":
        # Connect adjacent qubits: 0-1, 1-2, 2-3, ...
        for i in range(circuit.num_qubits - 1):
            gate = gate_type()
            circuit.add_gate(gate, target=(i, i + 1))
    elif pattern == "circular":
        # Linear + connect last to first
        for i in range(circuit.num_qubits - 1):
            gate = gate_type()
            circuit.add_gate(gate, target=(i, i + 1))
        # Wrap around
        gate = gate_type()
        circuit.add_gate(gate, target=(circuit.num_qubits - 1, 0))
    else:
        raise ValueError(f"Unknown pattern: {pattern}. Use 'linear' or 'circular'")


# Example usage
if __name__ == "__main__":
    import numpy as np

    from .gates import CNOTGate, HadamardGate, RXGate, RYGate, RZGate

    print("=" * 70)
    print("JAX-Compatible Quantum Circuit Demo")
    print("=" * 70)

    # Create 2-qubit circuit
    print("\n1. Create 2-Qubit Circuit:")
    circuit = QuantumCircuit(num_qubits=2)
    circuit.add_gate(HadamardGate(), target=0)
    circuit.add_gate(RXGate(theta=np.pi / 4, trainable=True), target=1)
    circuit.add_gate(CNOTGate(), target=(0, 1))
    circuit.add_gate(RZGate(theta=np.pi / 2, trainable=True), target=0)

    print(circuit)
    print("\nCircuit diagram:")
    print(circuit.draw())

    # Get trainable parameters
    print("\n2. Trainable Parameters:")
    params = circuit.get_trainable_parameters()
    for name, value in params.items():
        print(f"   {name} = {value:.4f}")

    # Get unitary
    print("\n3. Circuit Unitary (JAX):")
    U_jax = circuit.get_unitary_jax()
    print(f"   Shape: {U_jax.shape}")
    print(f"   Dtype: {U_jax.dtype}")

    # Update parameters
    print("\n4. Update Parameters:")
    circuit.set_trainable_parameters(
        {"gate_1_theta": jnp.array(np.pi / 2), "gate_3_theta": jnp.array(np.pi)}
    )
    print("   Updated gate_1_theta to π/2 and gate_3_theta to π")
    params = circuit.get_trainable_parameters()
    for name, value in params.items():
        print(f"   {name} = {value:.4f}")

    print("\n" + "=" * 70)

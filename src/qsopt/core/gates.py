"""
JAX-Compatible Quantum Gates
=============================

This module provides JAX-compatible quantum gate implementations with
trainable parameters and gradient control.

Key Features:
- Rotation gates (RX, RY, RZ) with differentiable parameters
- Fixed gates (CNOT, CZ, Hadamard)
- Parameter management with gradient tracing control
- Compatible with JAX autodiff and optimization

Example:
    >>> import jax.numpy as jnp
    >>> from qsopt.core.gates import RXGate, CNOTGate
    >>>
    >>> # Create rotation gate with trainable parameter
    >>> rx = RXGate(theta=jnp.pi/4, trainable=True)
    >>> theta = rx.get_parameter()
    >>> rx.set_parameter(jnp.pi/2)
    >>>
    >>> # Get gate matrix
    >>> matrix = rx.matrix()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import qutip as qt
from qutip_qip.operations import cnot, hadamard_transform


@dataclass
class GateParameter:
    """
    Container for gate parameters with gradient control.

    Attributes:
        value: Parameter value (JAX array)
        trainable: Whether parameter should be traced by JAX
        name: Parameter name for identification
    """

    value: jnp.ndarray
    trainable: bool = True
    name: str = "param"

    def get(self) -> jnp.ndarray:
        """Get parameter value with appropriate gradient handling."""
        if self.trainable:
            return self.value
        else:
            return jax.lax.stop_gradient(self.value)

    def set(self, value: Union[float, jnp.ndarray]) -> None:
        """Update parameter value."""
        self.value = jnp.asarray(value, dtype=float)

    def enable_gradients(self) -> None:
        """Enable gradient tracing for this parameter."""
        self.trainable = True

    def disable_gradients(self) -> None:
        """Disable gradient tracing for this parameter."""
        self.trainable = False


class Gate(ABC):
    """
    Abstract base class for quantum gates.

    All gates must implement the matrix() method to return their
    matrix representation. Parametrized gates should also implement
    parameter management methods.
    """

    def __init__(self, name: str, num_qubits: int):
        """
        Initialize gate.

        Args:
            name: Gate name (e.g., "RX", "CNOT")
            num_qubits: Number of qubits gate acts on
        """
        self.name = name
        self.num_qubits = num_qubits
        self._parameters: Dict[str, GateParameter] = {}

    @abstractmethod
    def matrix(self) -> qt.Qobj:
        """
        Return the gate's matrix representation.

        Returns:
            QuTiP Qobj representing the gate matrix
        """
        pass

    def has_parameters(self) -> bool:
        """Check if gate has trainable parameters."""
        return len(self._parameters) > 0

    def get_parameters(self) -> Dict[str, jnp.ndarray]:
        """
        Get all parameters with gradient handling.

        Returns:
            Dictionary mapping parameter names to values
        """
        return {name: param.get() for name, param in self._parameters.items()}

    def get_parameter(self, name: str = None) -> jnp.ndarray:
        """
        Get specific parameter value.

        Args:
            name: Parameter name. If None and only one parameter exists, returns it.

        Returns:
            Parameter value
        """
        if name is None:
            if len(self._parameters) == 1:
                return list(self._parameters.values())[0].get()
            else:
                raise ValueError("Must specify parameter name when multiple parameters exist")
        return self._parameters[name].get()

    def set_parameter(self, value: Union[float, jnp.ndarray], name: str = None) -> None:
        """
        Update parameter value.

        Args:
            value: New parameter value
            name: Parameter name. If None and only one parameter exists, updates it.
        """
        if name is None:
            if len(self._parameters) == 1:
                list(self._parameters.values())[0].set(value)
            else:
                raise ValueError("Must specify parameter name when multiple parameters exist")
        else:
            self._parameters[name].set(value)

    def enable_gradients(self, name: str = None) -> None:
        """
        Enable gradient tracing for parameters.

        Args:
            name: Parameter name. If None, enables all parameters.
        """
        if name is None:
            for param in self._parameters.values():
                param.enable_gradients()
        else:
            self._parameters[name].enable_gradients()

    def disable_gradients(self, name: str = None) -> None:
        """
        Disable gradient tracing for parameters.

        Args:
            name: Parameter name. If None, disables all parameters.
        """
        if name is None:
            for param in self._parameters.values():
                param.disable_gradients()
        else:
            self._parameters[name].disable_gradients()

    def is_trainable(self, name: str = None) -> bool:
        """
        Check if parameter is trainable.

        Args:
            name: Parameter name. If None and only one parameter exists, checks it.

        Returns:
            True if parameter is trainable
        """
        if name is None:
            if len(self._parameters) == 1:
                return list(self._parameters.values())[0].trainable
            else:
                raise ValueError("Must specify parameter name when multiple parameters exist")
        return self._parameters[name].trainable

    def __repr__(self) -> str:
        """String representation of gate."""
        if self.has_parameters():
            params_str = ", ".join(
                f"{name}={param.value:.4f}" for name, param in self._parameters.items()
            )
            return f"{self.name}({params_str})"
        return f"{self.name}"


# ============================================================================
# Rotation Gates (Parametrized)
# ============================================================================


class RXGate(Gate):
    """
    Rotation gate about X-axis.

    Matrix: RX(θ) = exp(-i θ σₓ/2)

    Example:
        >>> rx = RXGate(theta=jnp.pi/4, trainable=True)
        >>> matrix = rx.matrix()
        >>> rx.set_parameter(jnp.pi/2)
    """

    def __init__(self, theta: Union[float, jnp.ndarray] = 0.0, trainable: bool = True):
        """
        Initialize RX gate.

        Args:
            theta: Rotation angle in radians
            trainable: Whether theta should be traced by JAX
        """
        super().__init__("RX", num_qubits=1)
        self._parameters["theta"] = GateParameter(
            value=jnp.asarray(theta, dtype=float), trainable=trainable, name="theta"
        )

    def matrix(self) -> qt.Qobj:
        """Return RX gate matrix."""
        theta = self.get_parameter("theta")
        sx = qt.sigmax()
        return (-1j * theta * sx / 2).expm()


class RYGate(Gate):
    """
    Rotation gate about Y-axis.

    Matrix: RY(θ) = exp(-i θ σᵧ/2)

    Example:
        >>> ry = RYGate(theta=jnp.pi/4, trainable=True)
        >>> matrix = ry.matrix()
    """

    def __init__(self, theta: Union[float, jnp.ndarray] = 0.0, trainable: bool = True):
        """
        Initialize RY gate.

        Args:
            theta: Rotation angle in radians
            trainable: Whether theta should be traced by JAX
        """
        super().__init__("RY", num_qubits=1)
        self._parameters["theta"] = GateParameter(
            value=jnp.asarray(theta, dtype=float), trainable=trainable, name="theta"
        )

    def matrix(self) -> qt.Qobj:
        """Return RY gate matrix."""
        theta = self.get_parameter("theta")
        sy = qt.sigmay()
        return (-1j * theta * sy / 2).expm()


class RZGate(Gate):
    """
    Rotation gate about Z-axis.

    Matrix: RZ(θ) = exp(-i θ σᵧ/2)

    Example:
        >>> rz = RZGate(theta=jnp.pi/4, trainable=True)
        >>> matrix = rz.matrix()
    """

    def __init__(self, theta: Union[float, jnp.ndarray] = 0.0, trainable: bool = True):
        """
        Initialize RZ gate.

        Args:
            theta: Rotation angle in radians
            trainable: Whether theta should be traced by JAX
        """
        super().__init__("RZ", num_qubits=1)
        self._parameters["theta"] = GateParameter(
            value=jnp.asarray(theta, dtype=float), trainable=trainable, name="theta"
        )

    def matrix(self) -> qt.Qobj:
        """Return RZ gate matrix."""
        theta = self.get_parameter("theta")
        sz = qt.sigmaz()
        return (-1j * theta * sz / 2).expm()


# ============================================================================
# Fixed Gates (Non-parametrized)
# ============================================================================


class HadamardGate(Gate):
    """
    Hadamard gate.

    Matrix: H = (1/√2) [[1, 1], [1, -1]]

    Example:
        >>> h = HadamardGate()
        >>> matrix = h.matrix()
    """

    def __init__(self):
        """Initialize Hadamard gate."""
        super().__init__("H", num_qubits=1)

    def matrix(self) -> qt.Qobj:
        """Return Hadamard gate matrix."""
        return hadamard_transform()


class CNOTGate(Gate):
    """
    Controlled-NOT (CNOT) gate.

    Matrix: CNOT = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]]

    Control qubit: first qubit (index 0)
    Target qubit: second qubit (index 1)

    Example:
        >>> cnot = CNOTGate()
        >>> matrix = cnot.matrix()
    """

    def __init__(self):
        """Initialize CNOT gate."""
        super().__init__("CNOT", num_qubits=2)

    def matrix(self) -> qt.Qobj:
        """Return CNOT gate matrix."""
        return cnot()


class CZGate(Gate):
    """
    Controlled-Z (CZ) gate.

    Matrix: CZ = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, -1]]

    Applies phase flip to |11⟩ state.

    Example:
        >>> cz = CZGate()
        >>> matrix = cz.matrix()
    """

    def __init__(self):
        """Initialize CZ gate."""
        super().__init__("CZ", num_qubits=2)

    def matrix(self) -> qt.Qobj:
        """Return CZ gate matrix."""
        # CZ = diag(1, 1, 1, -1)
        return qt.Qobj(
            jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex),
            dims=[[2, 2], [2, 2]],
        )


# ============================================================================
# Utility Functions
# ============================================================================


def apply_gate(gate: Gate, state: qt.Qobj) -> qt.Qobj:
    """
    Apply gate to quantum state.

    Args:
        gate: Gate to apply
        state: Quantum state (ket or density matrix)

    Returns:
        Transformed state
    """
    gate_matrix = gate.matrix()
    if state.type == "ket":
        return gate_matrix * state
    elif state.type == "oper":
        return gate_matrix * state * gate_matrix.dag()
    else:
        raise ValueError(f"Unsupported state type: {state.type}")


def gate_sequence_product(gates: list) -> qt.Qobj:
    """
    Compute product of gate matrices (gates applied in list order).

    Gates are applied in the order they appear in the list. The matrix
    product is computed so that gates[0] is applied first, gates[1] second, etc.
    This means: U = gates[-1] * ... * gates[1] * gates[0]

    Example:
        gates = [RX, RY, RZ]  # RX applied first, then RY, then RZ
        U = RZ * RY * RX
        |ψ'⟩ = U|ψ⟩ = RZ(RY(RX|ψ⟩))

    Args:
        gates: List of Gate objects

    Returns:
        Combined gate matrix
    """
    if not gates:
        raise ValueError("Gate list is empty")

    # Build product: gates[n-1] * ... * gates[1] * gates[0]
    # So gates[0] is applied first to a state
    result = gates[0].matrix()
    for gate in gates[1:]:
        result = gate.matrix() * result
    return result


def get_all_trainable_parameters(gates: list) -> Dict[str, jnp.ndarray]:
    """
    Extract all trainable parameters from gate sequence.

    Args:
        gates: List of Gate objects

    Returns:
        Dictionary mapping "gate_index.param_name" to parameter values
    """
    all_params = {}
    for i, gate in enumerate(gates):
        if gate.has_parameters():
            params = gate.get_parameters()
            for name, value in params.items():
                if gate.is_trainable(name):
                    all_params[f"gate_{i}.{name}"] = value
    return all_params


def set_trainable_parameters(gates: list, parameters: Dict[str, jnp.ndarray]) -> None:
    """
    Update trainable parameters in gate sequence.

    Args:
        gates: List of Gate objects
        parameters: Dictionary mapping "gate_index.param_name" to new values
    """
    for key, value in parameters.items():
        gate_idx, param_name = key.split(".")
        idx = int(gate_idx.replace("gate_", ""))
        gates[idx].set_parameter(value, param_name)

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
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import qutip as qt


@dataclass
class GateParameter:
    """
    Container for gate parameters with gradient control.

    Attributes:
        value: Parameter value (JAX array)
        trainable: Whether parameter should be traced by JAX
    """

    value: jnp.ndarray
    trainable: bool = True
    name: Optional[str] = '[nome]'

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
    matrix representation. Parametrized gates can have at most one parameter.
    """

    def __init__(self, name: str, target: Union[int, Tuple[int, ...]]):
        """
        Initialize gate.

        Args:
            name: Gate name (RX, RY, RZ, H, CNOT, CZ)
            target: Target qubit(s) - int for single-qubit, tuple for multi-qubit gates
        """
        self.name = name
        self.target: Union[int, Tuple[int, ...]] = target
        self._parameter: Optional[GateParameter] = None



    @abstractmethod
    def matrix(self, qutip: bool = True) -> Union[qt.Qobj, jnp.ndarray]:
        """
        Return the gate's matrix representation.

        Args:
            qutip: If True, return QuTiP Qobj; if False, return JAX array

        Returns:
            QuTiP Qobj or JAX array depending on qutip flag
        """
        pass

    def has_parameter(self) -> bool:
        """Check if gate has a parameter."""
        return self._parameter is not None

    def get_parameter(self) -> jnp.ndarray:
        """
        Get parameter value.

        Returns:
            Parameter value with appropriate gradient handling
        """
        if self._parameter is None:
            raise ValueError(f"Gate {self.name} has no parameters")
        return self._parameter.get()


    def set_parameter(self, value: Union[float, jnp.ndarray]) -> None:
        """
        Update parameter value.

        Args:
            value: New parameter value
        """
        if self._parameter is None:
            raise ValueError(f"Gate {self.name} has no parameters")
        self._parameter.set(value)

    def __repr__(self, params = True) -> str:
        """String representation of gate."""
        if params & self.has_parameter():
            return f"{self.name}[{self.target}](param={self._parameter.value:.4f})"
        return f"{self.name}[{self.target}]"


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

    def __init__(self, theta: Union[float, jnp.ndarray], target: int, trainable: bool = True):
        """
        Initialize RX gate.

        Args:
            theta: Rotation angle in radians
            target: Target qubit index
            trainable: Whether theta should be traced by JAX
        """
        super().__init__("RX", target=target)
        self._parameter = GateParameter(
            value=jnp.asarray(theta, dtype=float), trainable=trainable, name=f'theta_x_qb{target}'
        )

    def matrix(self, qutip: bool = True) -> Union[qt.Qobj, jnp.ndarray]:
        """Return RX gate matrix.

        Args:
            qutip: If True, return QuTiP Qobj; if False, return JAX array
        """
        theta = self.get_parameter()
        half_theta = theta / 2.0
        c = jnp.cos(half_theta)
        s = jnp.sin(half_theta)
        matrix_data = jnp.array(
            [[c, -1j * s], [-1j * s, c]],
            dtype=jnp.complex128,
        )
        # Return JAX array or wrap in Qobj
        return qt.Qobj(matrix_data, dims=[[2],[2]]) if qutip else matrix_data


class RYGate(Gate):
    """
    Rotation gate about Y-axis.

    Matrix: RY(θ) = exp(-i θ σᵧ/2)

    Example:
        >>> ry = RYGate(theta=jnp.pi/4, trainable=True)
        >>> matrix = ry.matrix()
    """

    def __init__(self, theta: Union[float, jnp.ndarray], target: int, trainable: bool = True):
        """
        Initialize RY gate.

        Args:
            theta: Rotation angle in radians
            target: Target qubit index
            trainable: Whether theta should be traced by JAX
        """
        super().__init__("RY", target=target)
        self._parameter = GateParameter(
            value=jnp.asarray(theta, dtype=float), trainable=trainable, name=f'theta_y_qb{target}'
        )

    def matrix(self, qutip: bool = True) -> Union[qt.Qobj, jnp.ndarray]:
        """Return RY gate matrix.

        Args:
            qutip: If True, return QuTiP Qobj; if False, return JAX array
        """
        theta = self.get_parameter()
        half_theta = theta / 2.0
        c = jnp.cos(half_theta)
        s = jnp.sin(half_theta)
        matrix_data = jnp.array(
            [[c, -s], [s, c]],
            dtype=jnp.complex128,
        )
        # Return JAX array or wrap in Qobj
        return qt.Qobj(matrix_data, dims=[[2],[2]]) if qutip else matrix_data


class RZGate(Gate):
    """
    Rotation gate about Z-axis.

    Matrix: RZ(θ) = exp(-i θ σᵧ/2)

    Example:
        >>> rz = RZGate(theta=jnp.pi/4, trainable=True)
        >>> matrix = rz.matrix()
    """

    def __init__(self, theta: Union[float, jnp.ndarray], target: int, trainable: bool = True):
        """
        Initialize RZ gate.

        Args:
            theta: Rotation angle in radians
            target: Target qubit index
            trainable: Whether theta should be traced by JAX
        """
        super().__init__("RZ", target=target)
        self._parameter = GateParameter(
            value=jnp.asarray(theta, dtype=float), trainable=trainable, name=f'theta_z_qb{target}'
        )

    def matrix(self, qutip: bool = True) -> Union[qt.Qobj, jnp.ndarray]:
        """Return RZ gate matrix.

        Args:
            qutip: If True, return QuTiP Qobj; if False, return JAX array
        """
        theta = self.get_parameter()
        phase_minus = jnp.exp(-0.5j * theta)
        phase_plus = jnp.exp(0.5j * theta)
        matrix_data = jnp.array(
            [[phase_minus, 0.0], [0.0, phase_plus]],
            dtype=jnp.complex128,
        )
        # Return JAX array or wrap in Qobj
        return qt.Qobj(matrix_data, dims=[[2],[2]]) if qutip else matrix_data


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

    def __init__(self, target: int):
        """Initialize Hadamard gate.

        Args:
            target: Target qubit index
        """
        super().__init__("H", target=target)

    def matrix(self, qutip: bool = True) -> Union[qt.Qobj, jnp.ndarray]:
        """Return Hadamard gate matrix.

        Args:
            qutip: If True, return QuTiP Qobj; if False, return JAX array
        """
        # H = (1/√2) * [[1, 1], [1, -1]]
        h_data = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
        return qt.Qobj(h_data, dims=[[2],[2]]) if qutip else h_data


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

    def __init__(self, target: Tuple[int, int]):
        """Initialize CNOT gate.

        Args:
            target: Tuple of (control, target) qubit indices
        """
        super().__init__("CNOT", target=target)

    def matrix(self, qutip: bool = True) -> Union[qt.Qobj, jnp.ndarray]:
        """Return CNOT gate matrix.

        Args:
            qutip: If True, return QuTiP Qobj; if False, return JAX array
        """
        # CNOT = [[1, 0, 0, 0],
        #         [0, 1, 0, 0],
        #         [0, 0, 0, 1],
        #         [0, 0, 1, 0]]
        cnot_data = jnp.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex128
        )
        return qt.Qobj(cnot_data, dims=[[2,2],[2,2]]) if qutip else cnot_data


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

    def __init__(self, target: Tuple[int, int]):
        """Initialize CZ gate.

        Args:
            target: Tuple of (control, target) qubit indices
        """
        super().__init__("CZ", target=target)

    def matrix(self, qutip: bool = True) -> Union[qt.Qobj, jnp.ndarray]:
        """Return CZ gate matrix.

        Args:
            qutip: If True, return QuTiP Qobj; if False, return JAX array
        """
        # CZ = diag(1, 1, 1, -1)
        cz_data = jnp.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=jnp.complex128
        )
        return qt.Qobj(cz_data, dims=[[2,2],[2,2]]) if qutip else cz_data

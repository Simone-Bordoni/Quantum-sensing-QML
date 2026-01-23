"""
Tests for JAX-compatible quantum gates.

Verifies that custom gate implementations match QuTiP's standard gates
and that parameter management functions work correctly.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import qutip as qt
from qutip_qip.operations import cnot, hadamard_transform, rx, ry, rz

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsopt.core.gates import (
    CNOTGate,
    CZGate,
    HadamardGate,
    RXGate,
    RYGate,
    RZGate,
)


# --------------------------------------------------------------------------
# Tests for gate matrices against QuTiP reference
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Tests for gate matrices against QuTiP reference
# --------------------------------------------------------------------------

class TestGateMatrices:
    """Test gate matrices match QuTiP reference."""

    @pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3])
    def test_rx_gate(self, theta):
        """Test RX gate matrix matches QuTiP."""
        custom_rx = RXGate(theta=theta, target=0)
        qutip_rx = rx(theta)
        np.testing.assert_allclose(custom_rx.matrix(qutip=True).full(), qutip_rx.full(), rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_ry_gate(self, theta):
        """Test RY gate matrix matches QuTiP."""
        custom_ry = RYGate(theta=theta, target=0)
        qutip_ry = ry(theta)
        np.testing.assert_allclose(custom_ry.matrix(qutip=True).full(), qutip_ry.full(), rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi])
    def test_rz_gate(self, theta):
        """Test RZ gate matrix matches QuTiP."""
        custom_rz = RZGate(theta=theta, target=0)
        qutip_rz = rz(theta)
        np.testing.assert_allclose(custom_rz.matrix(qutip=True).full(), qutip_rz.full(), rtol=1e-10)

    def test_hadamard_gate(self):
        """Test Hadamard gate matrix matches QuTiP."""
        custom_h = HadamardGate(target=0)
        qutip_h = hadamard_transform()
        np.testing.assert_allclose(custom_h.matrix(qutip=True).full(), qutip_h.full(), rtol=1e-10)

    def test_cnot_gate(self):
        """Test CNOT gate matrix matches QuTiP."""
        custom_cnot = CNOTGate(target=(0, 1))
        qutip_cnot = cnot()
        np.testing.assert_allclose(custom_cnot.matrix(qutip=True).full(), qutip_cnot.full(), rtol=1e-10)

    def test_cz_gate(self):
        """Test CZ gate matrix."""
        custom_cz = CZGate(target=(0, 1))
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)
        np.testing.assert_allclose(custom_cz.matrix(qutip=True).full(), expected, rtol=1e-10)


# --------------------------------------------------------------------------
# Tests for JAX array return type
# --------------------------------------------------------------------------

class TestJAXArrays:
    """Test gates return JAX arrays when qutip=False."""

    def test_rotation_gates_return_jax_arrays(self):
        """Test rotation gates return JAX arrays."""
        rx = RXGate(theta=np.pi / 4, target=0)
        matrix_jax = rx.matrix(qutip=False)
        assert isinstance(matrix_jax, jnp.ndarray)
        assert matrix_jax.dtype == jnp.complex128
        np.testing.assert_allclose(matrix_jax, rx.matrix(qutip=True).full(), rtol=1e-10)

    def test_fixed_gates_return_jax_arrays(self):
        """Test fixed gates return JAX arrays."""
        h = HadamardGate(target=0)
        matrix_jax = h.matrix(qutip=False)
        assert isinstance(matrix_jax, jnp.ndarray)
        assert matrix_jax.dtype == jnp.complex128


# --------------------------------------------------------------------------
# Tests for parameter management
# --------------------------------------------------------------------------

class TestParameterManagement:
    """Test parameter management functionality."""

    def test_has_parameter(self):
        """Test has_parameter method."""
        rx = RXGate(theta=0.5, target=0)
        h = HadamardGate(target=0)
        assert rx.has_parameter() is True
        assert h.has_parameter() is False

    def test_get_set_parameter(self):
        """Test getting and setting gate parameters."""
        rx = RXGate(theta=0.5, target=0)
        np.testing.assert_allclose(rx.get_parameter(), 0.5, rtol=1e-10)
        
        rx.set_parameter(1.5)
        np.testing.assert_allclose(rx.get_parameter(), 1.5, rtol=1e-10)

    def test_trainable_flag(self):
        """Test trainable parameter flag."""
        rx_trainable = RXGate(theta=0.5, target=0, trainable=True)
        rx_fixed = RXGate(theta=0.5, target=0, trainable=False)
        
        assert rx_trainable._parameter.trainable is True
        assert rx_fixed._parameter.trainable is False

    def test_enable_disable_gradients(self):
        """Test enabling and disabling gradients."""
        rx = RXGate(theta=0.5, target=0, trainable=True)
        assert rx._parameter.trainable is True
        
        rx._parameter.disable_gradients()
        assert rx._parameter.trainable is False
        
        rx._parameter.enable_gradients()
        assert rx._parameter.trainable is True


# --------------------------------------------------------------------------
# Tests for gate properties
# --------------------------------------------------------------------------

class TestGateProperties:
    """Test gate properties."""

    def test_gate_names(self):
        """Test gate name property."""
        assert RXGate(theta=0.5, target=0).name == "RX"
        assert RYGate(theta=0.5, target=0).name == "RY"
        assert RZGate(theta=0.5, target=0).name == "RZ"
        assert HadamardGate(target=0).name == "H"
        assert CNOTGate(target=(0, 1)).name == "CNOT"
        assert CZGate(target=(0, 1)).name == "CZ"

    def test_gate_target(self):
        """Test gate target property."""
        rx = RXGate(theta=0.5, target=2)
        cnot = CNOTGate(target=(1, 3))
        assert rx.target == 2
        assert cnot.target == (1, 3)

    def test_gate_repr(self):
        """Test gate string representation."""
        rx = RXGate(theta=0.5, target=0)
        h = HadamardGate(target=1)
        assert "RX" in repr(rx)
        assert "0.5" in repr(rx)
        assert "H" in repr(h)


# --------------------------------------------------------------------------
# Tests for gate unitarity
# --------------------------------------------------------------------------

class TestGateUnitarity:
    """Test that gates are unitary."""

    @pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_rotation_gates_unitary(self, theta):
        """Test rotation gates are unitary."""
        for gate_class in [RXGate, RYGate, RZGate]:
            gate = gate_class(theta=theta, target=0)
            U = gate.matrix(qutip=True)
            identity = U.dag() * U
            np.testing.assert_allclose(identity.full(), qt.qeye(2).full(), rtol=1e-10, atol=1e-12)

    def test_fixed_gates_unitary(self):
        """Test fixed gates are unitary."""
        h = HadamardGate(target=0)
        U = h.matrix(qutip=True)
        identity = U.dag() * U
        np.testing.assert_allclose(identity.full(), qt.qeye(2).full(), rtol=1e-10, atol=1e-12)
        
        cnot = CNOTGate(target=(0, 1))
        U = cnot.matrix(qutip=True)
        identity = U.dag() * U
        np.testing.assert_allclose(identity.full(), qt.qeye([2, 2]).full(), rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


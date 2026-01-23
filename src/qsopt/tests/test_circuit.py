"""
Tests for JAX-compatible quantum circuits.

Verifies that custom circuit implementations produce correct unitaries
that match QuTiP's circuit builder results.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import qutip as qt
from qutip_qip.circuit import QubitCircuit

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qsopt.core.circuit import (
    QuantumCircuit,
    create_entangling_layer,
    create_layer,
)
from qsopt.core.gates import CNOTGate, HadamardGate, RXGate, RYGate, RZGate


# --------------------------------------------------------------------------
# Basic circuit functionality tests
# --------------------------------------------------------------------------

class TestCircuitBasics:
    """Test basic circuit functionality."""

    def test_circuit_creation_and_gate_addition(self):
        """Test creating circuit and adding gates."""
        circuit = QuantumCircuit(num_qubits=3)
        assert circuit.num_qubits == 3
        assert len(circuit._gates) == 0

        # Add single and two-qubit gates
        circuit.add_gate(RXGate(theta=0.5, target=0))
        circuit.add_gate(CNOTGate(target=(0, 1)))
        assert len(circuit._gates) == 2

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="at least 1"):
            QuantumCircuit(num_qubits=0)

        circuit = QuantumCircuit(num_qubits=2)
        with pytest.raises(ValueError, match="out of range"):
            circuit.add_gate(RXGate(theta=0.5, target=5))

    def test_circuit_repr(self):
        """Test string representation."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.5, target=0))
        assert "2 qubits" in repr(circuit) and "1 gates" in repr(circuit)


# --------------------------------------------------------------------------
# Parameter management tests
# --------------------------------------------------------------------------

class TestParameterManagement:
    """Test parameter management in circuits."""

    def test_get_and_count_trainable_parameters(self):
        """Test getting and counting trainable parameters."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.1, target=0, trainable=True))
        circuit.add_gate(RYGate(theta=0.2, target=1, trainable=True))
        circuit.add_gate(HadamardGate(target=0))
        circuit.add_gate(RZGate(theta=0.3, target=1, trainable=False))

        params = circuit.get_trainable_parameters()
        assert len(params) == 2
        assert circuit.count_trainable_parameters() == 2
        np.testing.assert_allclose(params[0], 0.1, rtol=1e-10)
        np.testing.assert_allclose(params[1], 0.2, rtol=1e-10)

    def test_set_and_update_parameters(self):
        """Test setting and updating trainable parameters."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.1, target=0, trainable=True))
        circuit.add_gate(RYGate(theta=0.2, target=1, trainable=True))

        # Test set_trainable_parameters
        circuit.set_trainable_parameters([jnp.array(1.5), jnp.array(2.5)])
        params = circuit.get_trainable_parameters()
        np.testing.assert_allclose(params[0], 1.5, rtol=1e-10)
        np.testing.assert_allclose(params[1], 2.5, rtol=1e-10)

        # Test update_parameters
        circuit.update_parameters([0.5, 0.7])
        params = circuit.get_trainable_parameters()
        np.testing.assert_allclose(params[0], 0.5, rtol=1e-10)
        np.testing.assert_allclose(params[1], 0.7, rtol=1e-10)


# --------------------------------------------------------------------------
# Unitary computation tests
# --------------------------------------------------------------------------

class TestUnitaryComputation:
    """Test circuit unitary computation."""

    def test_empty_circuit_and_jax_format(self):
        """Test empty circuit unitary and JAX array format."""
        circuit = QuantumCircuit(num_qubits=2)
        
        # Empty circuit should be identity
        U_qutip = circuit.get_unitary(qutip=True)
        np.testing.assert_allclose(U_qutip.full(), qt.qeye([2, 2]).full(), rtol=1e-10)

        # Test JAX array format
        U_jax = circuit.get_unitary(qutip=False)
        assert isinstance(U_jax, jnp.ndarray)
        assert U_jax.dtype == jnp.complex128
        np.testing.assert_allclose(U_jax, U_qutip.full(), rtol=1e-10)

    @pytest.mark.parametrize("num_qubits", [1, 2, 3])
    def test_unitarity_property(self, num_qubits):
        """Test that circuit unitaries satisfy U†U = I."""
        circuit = QuantumCircuit(num_qubits=num_qubits)
        circuit.add_gate(RXGate(theta=0.5, target=0))
        circuit.add_gate(RYGate(theta=0.7, target=0))
        if num_qubits >= 2:
            circuit.add_gate(CNOTGate(target=(0, 1)))

        U = circuit.get_unitary(qutip=True)
        identity = U.dag() * U
        np.testing.assert_allclose(identity.full(), qt.qeye([2] * num_qubits).full(), rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------
# Circuit vs QuTiP comparison tests
# --------------------------------------------------------------------------

class TestCircuitVsQuTiP:
    """Compare circuit unitaries with QuTiP reference."""

    @pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_single_qubit_gates(self, theta):
        """Test single qubit gates match QuTiP."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(RXGate(theta=theta, target=0))
        U_custom = circuit.get_unitary(qutip=True)

        qc = QubitCircuit(1)
        qc.add_gate("RX", targets=0, arg_value=theta)
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10, atol=1e-12)

    def test_bell_state_circuit(self):
        """Test Bell state preparation (H + CNOT) matches QuTiP."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(HadamardGate(target=0))
        circuit.add_gate(CNOTGate(target=(0, 1)))
        U_custom = circuit.get_unitary(qutip=True)

        qc = QubitCircuit(2)
        qc.add_gate("SNOT", targets=0)
        qc.add_gate("CNOT", controls=0, targets=1)
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10)

    def test_multi_gate_circuit(self):
        """Test multi-gate circuit matches QuTiP."""
        theta1, theta2, theta3 = np.pi / 4, np.pi / 3, np.pi / 6

        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=theta1, target=0))
        circuit.add_gate(RYGate(theta=theta2, target=1))
        circuit.add_gate(CNOTGate(target=(0, 1)))
        circuit.add_gate(RZGate(theta=theta3, target=0))
        U_custom = circuit.get_unitary(qutip=True)

        qc = QubitCircuit(2)
        qc.add_gate("RX", targets=0, arg_value=theta1)
        qc.add_gate("RY", targets=1, arg_value=theta2)
        qc.add_gate("CNOT", controls=0, targets=1)
        qc.add_gate("RZ", targets=0, arg_value=theta3)
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10)

    @pytest.mark.parametrize("target", [0, 1])
    def test_gate_on_different_qubits(self, target):
        """Test gates on different qubits match QuTiP."""
        theta = np.pi / 3
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=theta, target=target))
        U_custom = circuit.get_unitary(qutip=True)

        qc = QubitCircuit(2)
        qc.add_gate("RX", targets=target, arg_value=theta)
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10)

    @pytest.mark.parametrize("control,target", [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)])
    def test_cnot_configurations(self, control, target):
        """Test CNOT with different control-target pairs matches QuTiP."""
        num_qubits = max(control, target) + 1
        circuit = QuantumCircuit(num_qubits=num_qubits)
        circuit.add_gate(CNOTGate(target=(control, target)))
        U_custom = circuit.get_unitary(qutip=True)

        qc = QubitCircuit(num_qubits)
        qc.add_gate("CNOT", controls=control, targets=target)
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10,
                                   err_msg=f"Failed for CNOT({control}, {target})")


# --------------------------------------------------------------------------
# Circuit application to quantum states
# --------------------------------------------------------------------------

class TestCircuitApplication:
    """Test applying circuits to quantum states."""

    def test_apply_to_pure_state_qutip(self):
        """Test applying circuit to pure state (QuTiP Qobj)."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(HadamardGate(target=0))

        # Start with |0⟩
        psi0 = qt.basis(2, 0)
        rho_final = circuit(psi0, qutip=True)

        # H|0⟩|0⟩⟨0|⟨0|H† = |+⟩⟨+|
        # Should have equal probabilities for |0⟩ and |1⟩
        assert isinstance(rho_final, qt.Qobj)
        np.testing.assert_allclose(rho_final.diag(), [0.5, 0.5], rtol=1e-10)

    def test_apply_to_pure_state_jax_vector(self):
        """Test applying circuit to pure state (JAX vector)."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(HadamardGate(target=0))

        # Start with |0⟩ as JAX array
        psi0 = jnp.array([1.0, 0.0], dtype=jnp.complex128)
        rho_final = circuit(psi0, qutip=False)

        # Check it's a density matrix
        assert isinstance(rho_final, jnp.ndarray)
        assert rho_final.shape == (2, 2)
        np.testing.assert_allclose(jnp.diag(rho_final), [0.5, 0.5], rtol=1e-10)

    def test_apply_to_density_matrix_qutip(self):
        """Test applying circuit to density matrix (QuTiP)."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(RXGate(theta=np.pi / 2, target=0))

        # Start with |0⟩⟨0| density matrix
        rho0 = qt.ket2dm(qt.basis(2, 0))
        rho_final = circuit(rho0, qutip=True)

        # RX(π/2) rotates |0⟩ to (|0⟩ - i|1⟩)/√2
        # Density matrix should have equal diagonal elements
        assert isinstance(rho_final, qt.Qobj)
        np.testing.assert_allclose(rho_final.diag(), [0.5, 0.5], rtol=1e-10)

    def test_apply_to_density_matrix_jax(self):
        """Test applying circuit to density matrix (JAX array)."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(RXGate(theta=np.pi / 2, target=0))

        # Start with |0⟩⟨0| density matrix as JAX array
        rho0 = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
        rho_final = circuit(rho0, qutip=False)

        assert isinstance(rho_final, jnp.ndarray)
        np.testing.assert_allclose(jnp.diag(rho_final), [0.5, 0.5], rtol=1e-10)

    def test_bell_state_preparation(self):
        """Test Bell state preparation using circuit application."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(HadamardGate(target=0))
        circuit.add_gate(CNOTGate(target=(0, 1)))

        # Start with |00⟩
        psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
        rho_final = circuit(psi0, qutip=True)

        # Should get |Φ+⟩ = (|00⟩ + |11⟩)/√2
        # Density matrix should have ρ[0,0] = ρ[3,3] = 0.5, ρ[0,3] = ρ[3,0] = 0.5
        expected_diag = [0.5, 0.0, 0.0, 0.5]
        np.testing.assert_allclose(rho_final.diag(), expected_diag, rtol=1e-10)
        np.testing.assert_allclose(abs(rho_final.full()[0, 3]), 0.5, rtol=1e-10)

    def test_multi_qubit_state_evolution(self):
        """Test evolving multi-qubit state through circuit."""
        theta1, theta2 = np.pi / 4, np.pi / 3
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=theta1, target=0))
        circuit.add_gate(RYGate(theta=theta2, target=1))

        # Start with |00⟩
        psi0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex128)
        rho_final_jax = circuit(psi0, qutip=False)

        # Compare with QuTiP
        U = circuit.get_unitary(qutip=True)
        psi0_qutip = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
        rho0_qutip = qt.ket2dm(psi0_qutip)
        rho_expected = U * rho0_qutip * U.dag()

        np.testing.assert_allclose(rho_final_jax, rho_expected.full(), rtol=1e-10)

    def test_apply_with_default_ground_state(self):
        """Test applying circuit with default ground state."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(HadamardGate(target=0))

        # Use default ground state |00⟩
        rho_final = circuit(qutip=True)

        # Should be same as explicitly providing |00⟩
        psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
        rho_expected = circuit(psi0, qutip=True)

        np.testing.assert_allclose(rho_final.full(), rho_expected.full(), rtol=1e-10)

    def test_default_ground_state_single_qubit(self):
        """Test default ground state for single qubit."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(RXGate(theta=np.pi / 2, target=0))

        # Apply with default ground state
        rho_final = circuit(qutip=False)

        # Should start from |0⟩⟨0|
        assert rho_final.shape == (2, 2)
        # RX(π/2) on |0⟩ gives equal superposition
        np.testing.assert_allclose(jnp.diag(rho_final), [0.5, 0.5], rtol=1e-10)


# --------------------------------------------------------------------------
# Utility function tests
# --------------------------------------------------------------------------

class TestCircuitUtilities:
    """Test circuit utility functions."""

    def test_create_layer(self):
        """Test creating rotation layer."""
        circuit = QuantumCircuit(num_qubits=3)
        create_layer(circuit, RXGate, [0.1, 0.2, 0.3], trainable=True)

        assert len(circuit._gates) == 3
        assert circuit.count_trainable_parameters() == 3

    def test_create_entangling_layers(self):
        """Test creating entangling layers."""
        # Linear pattern
        circuit = QuantumCircuit(num_qubits=4)
        create_entangling_layer(circuit, CNOTGate, pattern="linear")
        assert len(circuit._gates) == 3  # (0,1), (1,2), (2,3)

        # Circular pattern
        circuit = QuantumCircuit(num_qubits=4)
        create_entangling_layer(circuit, CNOTGate, pattern="circular")
        assert len(circuit._gates) == 4  # (0,1), (1,2), (2,3), (3,0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



if __name__ == "__main__":
    pytest.main([__file__, "-v"])

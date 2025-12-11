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
from qutip_qip.operations import cnot, hadamard_transform, rx, ry, rz

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qsopt.core.circuit import (
    GateApplication,
    QuantumCircuit,
    create_entangling_layer,
    create_layer,
)
from qsopt.core.gates import CNOTGate, CZGate, HadamardGate, RXGate, RYGate, RZGate


class TestGateApplication:
    """Test GateApplication class."""

    def test_single_qubit_gate(self):
        """Test single qubit gate application."""
        gate = RXGate(theta=0.5)
        gate_app = GateApplication(gate, target=0)

        assert gate_app.target == (0,)
        assert gate_app.gate.name == "RX"

    def test_two_qubit_gate(self):
        """Test two qubit gate application."""
        gate = CNOTGate()
        gate_app = GateApplication(gate, target=(0, 1))

        assert gate_app.target == (0, 1)
        assert gate_app.gate.name == "CNOT"

    def test_target_mismatch_error(self):
        """Test error when target qubits don't match gate dimensions."""
        gate = CNOTGate()  # 2-qubit gate

        with pytest.raises(ValueError, match="requires 2 qubit"):
            GateApplication(gate, target=0)  # Only 1 target


class TestQuantumCircuitBasics:
    """Test basic QuantumCircuit functionality."""

    def test_circuit_creation(self):
        """Test creating empty circuit."""
        circuit = QuantumCircuit(num_qubits=3)
        assert circuit.num_qubits == 3
        assert circuit.num_gates() == 0

    def test_invalid_num_qubits(self):
        """Test error for invalid number of qubits."""
        with pytest.raises(ValueError, match="at least 1"):
            QuantumCircuit(num_qubits=0)

    def test_add_single_qubit_gate(self):
        """Test adding single qubit gate."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.5), target=0)

        assert circuit.num_gates() == 1

    def test_add_two_qubit_gate(self):
        """Test adding two qubit gate."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(CNOTGate(), target=(0, 1))

        assert circuit.num_gates() == 1

    def test_invalid_target_qubit(self):
        """Test error for out of range target qubit."""
        circuit = QuantumCircuit(num_qubits=2)

        with pytest.raises(ValueError, match="out of range"):
            circuit.add_gate(RXGate(theta=0.5), target=5)

    def test_get_gates(self):
        """Test retrieving gate list."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.5), target=0)
        circuit.add_gate(RYGate(theta=0.3), target=1)

        gates = circuit.get_gates()
        assert len(gates) == 2
        assert gates[0].gate.name == "RX"
        assert gates[1].gate.name == "RY"


class TestParameterManagement:
    """Test parameter management in circuits."""

    def test_get_trainable_parameters(self):
        """Test getting trainable parameters."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.1, trainable=True), target=0)
        circuit.add_gate(RYGate(theta=0.2, trainable=True), target=1)
        circuit.add_gate(HadamardGate(), target=0)  # No parameters

        params = circuit.get_trainable_parameters()

        assert len(params) == 2
        assert "gate_0_theta" in params
        assert "gate_1_theta" in params
        np.testing.assert_allclose(params["gate_0_theta"], 0.1, rtol=1e-10)
        np.testing.assert_allclose(params["gate_1_theta"], 0.2, rtol=1e-10)

    def test_exclude_non_trainable_parameters(self):
        """Test that non-trainable parameters are excluded."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.1, trainable=True), target=0)
        circuit.add_gate(RYGate(theta=0.2, trainable=False), target=1)

        params = circuit.get_trainable_parameters()

        assert len(params) == 1
        assert "gate_0_theta" in params
        assert "gate_1_theta" not in params

    def test_set_trainable_parameters(self):
        """Test updating trainable parameters."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.1, trainable=True), target=0)
        circuit.add_gate(RYGate(theta=0.2, trainable=True), target=1)

        # Update parameters
        new_params = {"gate_0_theta": jnp.array(1.5), "gate_1_theta": jnp.array(2.5)}
        circuit.set_trainable_parameters(new_params)

        # Verify update
        params = circuit.get_trainable_parameters()
        np.testing.assert_allclose(params["gate_0_theta"], 1.5, rtol=1e-10)
        np.testing.assert_allclose(params["gate_1_theta"], 2.5, rtol=1e-10)

    def test_update_parameters_from_list(self):
        """Test updating parameters from flat list."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.1, trainable=True), target=0)
        circuit.add_gate(RYGate(theta=0.2, trainable=True), target=1)

        circuit.update_parameters([1.5, 2.5])

        params = circuit.get_trainable_parameters()
        np.testing.assert_allclose(params["gate_0_theta"], 1.5, rtol=1e-10)
        np.testing.assert_allclose(params["gate_1_theta"], 2.5, rtol=1e-10)

    def test_enable_disable_gradients(self):
        """Test enabling and disabling gradients."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.1, trainable=True), target=0)
        circuit.add_gate(RYGate(theta=0.2, trainable=True), target=1)

        # Disable all gradients
        circuit.disable_gradients()
        params = circuit.get_trainable_parameters()
        assert len(params) == 0  # No trainable parameters

        # Enable all gradients
        circuit.enable_gradients()
        params = circuit.get_trainable_parameters()
        assert len(params) == 2

        # Disable specific gate
        circuit.disable_gradients(gate_index=0)
        params = circuit.get_trainable_parameters()
        assert len(params) == 1
        assert "gate_1_theta" in params


class TestUnitaryComputation:
    """Test circuit unitary computation."""

    def test_empty_circuit_unitary(self):
        """Test unitary of empty circuit is identity."""
        circuit = QuantumCircuit(num_qubits=2)
        U = circuit.get_unitary()

        expected = qt.qeye([2, 2])
        np.testing.assert_allclose(U.full(), expected.full(), rtol=1e-10)

    def test_single_gate_circuit(self):
        """Test unitary of single gate circuit."""
        theta = np.pi / 4
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(RXGate(theta=theta), target=0)

        U = circuit.get_unitary()
        expected = rx(theta)

        np.testing.assert_allclose(U.full(), expected.full(), rtol=1e-10)

    def test_two_gate_sequence(self):
        """Test unitary of two-gate sequence."""
        theta1 = np.pi / 4
        theta2 = np.pi / 3

        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(RXGate(theta=theta1), target=0)
        circuit.add_gate(RYGate(theta=theta2), target=0)

        U_custom = circuit.get_unitary()

        # Build equivalent QuTiP circuit
        qc = QubitCircuit(1)
        qc.add_gate("RX", targets=0, arg_value=theta1)
        qc.add_gate("RY", targets=0, arg_value=theta2)
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10)

    def test_unitary_jax_format(self):
        """Test getting unitary as JAX array."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(HadamardGate(), target=0)
        circuit.add_gate(CNOTGate(), target=(0, 1))

        U_jax = circuit.get_unitary_jax()

        assert isinstance(U_jax, jnp.ndarray)
        assert U_jax.dtype == jnp.complex128
        assert U_jax.shape == (4, 4)

    def test_multi_qubit_gate_expansion(self):
        """Test proper expansion of gates in multi-qubit circuits."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=np.pi / 2), target=0)

        U_custom = circuit.get_unitary()

        # Build equivalent QuTiP circuit
        qc = QubitCircuit(2)
        qc.add_gate("RX", targets=0, arg_value=np.pi / 2)
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10)


class TestCircuitVsQuTiP:
    """Compare circuit unitaries with QuTiP's circuit builder."""

    def test_single_qubit_rotations(self):
        """Test single qubit rotation circuits match QuTiP."""
        angles = [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi]

        for theta in angles:
            # Custom circuit
            circuit = QuantumCircuit(num_qubits=1)
            circuit.add_gate(RXGate(theta=theta), target=0)
            U_custom = circuit.get_unitary()

            # QuTiP circuit
            qc = QubitCircuit(1)
            qc.add_gate("RX", targets=0, arg_value=theta)
            U_qutip = qc.compute_unitary()

            np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10, atol=1e-12)

    def test_hadamard_circuit(self):
        """Test Hadamard circuit matches QuTiP."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(HadamardGate(), target=0)
        U_custom = circuit.get_unitary()

        qc = QubitCircuit(1)
        qc.add_gate("SNOT", targets=0)  # SNOT is Hadamard in QuTiP
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10)

    def test_bell_state_circuit(self):
        """Test Bell state preparation circuit matches QuTiP."""
        # H on qubit 0, then CNOT(0,1)
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(HadamardGate(), target=0)
        circuit.add_gate(CNOTGate(), target=(0, 1))
        U_custom = circuit.get_unitary()

        qc = QubitCircuit(2)
        qc.add_gate("SNOT", targets=0)
        qc.add_gate("CNOT", controls=0, targets=1)
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10)

    def test_complex_multi_gate_circuit(self):
        """Test complex circuit with multiple gates matches QuTiP."""
        theta1, theta2, theta3 = np.pi / 4, np.pi / 3, np.pi / 6

        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=theta1), target=0)
        circuit.add_gate(RYGate(theta=theta2), target=1)
        circuit.add_gate(CNOTGate(), target=(0, 1))
        circuit.add_gate(RZGate(theta=theta3), target=0)
        U_custom = circuit.get_unitary()

        qc = QubitCircuit(2)
        qc.add_gate("RX", targets=0, arg_value=theta1)
        qc.add_gate("RY", targets=1, arg_value=theta2)
        qc.add_gate("CNOT", controls=0, targets=1)
        qc.add_gate("RZ", targets=0, arg_value=theta3)
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10)

    def test_three_qubit_circuit(self):
        """Test three-qubit circuit matches QuTiP."""
        circuit = QuantumCircuit(num_qubits=3)
        circuit.add_gate(HadamardGate(), target=0)
        circuit.add_gate(HadamardGate(), target=1)
        circuit.add_gate(CNOTGate(), target=(0, 2))
        circuit.add_gate(CNOTGate(), target=(1, 2))
        U_custom = circuit.get_unitary()

        qc = QubitCircuit(3)
        qc.add_gate("SNOT", targets=0)
        qc.add_gate("SNOT", targets=1)
        qc.add_gate("CNOT", controls=0, targets=2)
        qc.add_gate("CNOT", controls=1, targets=2)
        U_qutip = qc.compute_unitary()

        np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10)

    def test_gate_on_different_qubits(self):
        """Test gates applied to different qubits in multi-qubit circuit."""
        theta = np.pi / 3

        # Test RX on qubit 0 vs qubit 1
        for target_qubit in [0, 1]:
            circuit = QuantumCircuit(num_qubits=2)
            circuit.add_gate(RXGate(theta=theta), target=target_qubit)
            U_custom = circuit.get_unitary()

            qc = QubitCircuit(2)
            qc.add_gate("RX", targets=target_qubit, arg_value=theta)
            U_qutip = qc.compute_unitary()

            np.testing.assert_allclose(U_custom.full(), U_qutip.full(), rtol=1e-10)


class TestCircuitLayers:
    """Test circuit layer creation utilities."""

    def test_create_rotation_layer(self):
        """Test creating layer of rotation gates."""
        circuit = QuantumCircuit(num_qubits=3)
        params = [0.1, 0.2, 0.3]
        create_layer(circuit, RXGate, params, trainable=True)

        assert circuit.num_gates() == 3
        params_dict = circuit.get_trainable_parameters()
        assert len(params_dict) == 3

    def test_create_entangling_layer_linear(self):
        """Test creating linear entangling layer."""
        circuit = QuantumCircuit(num_qubits=4)
        create_entangling_layer(circuit, CNOTGate, pattern="linear")

        # Should have 3 CNOTs: (0,1), (1,2), (2,3)
        assert circuit.num_gates() == 3

    def test_create_entangling_layer_circular(self):
        """Test creating circular entangling layer."""
        circuit = QuantumCircuit(num_qubits=4)
        create_entangling_layer(circuit, CNOTGate, pattern="circular")

        # Should have 4 CNOTs: (0,1), (1,2), (2,3), (3,0)
        assert circuit.num_gates() == 4


class TestCircuitUnitarity:
    """Test that circuit unitaries are actually unitary."""

    @pytest.mark.parametrize("num_qubits", [1, 2, 3])
    def test_unitary_property(self, num_qubits):
        """Test that U†U = I for various circuits."""
        circuit = QuantumCircuit(num_qubits=num_qubits)

        # Add random gates
        circuit.add_gate(RXGate(theta=0.5), target=0)
        circuit.add_gate(RYGate(theta=0.7), target=0)
        if num_qubits >= 2:
            circuit.add_gate(CNOTGate(), target=(0, 1))

        U = circuit.get_unitary()
        identity = U.dag() * U
        expected_identity = qt.qeye([2] * num_qubits)

        np.testing.assert_allclose(
            identity.full(), expected_identity.full(), rtol=1e-10, atol=1e-12
        )


class TestCircuitVisualization:
    """Test circuit visualization methods."""

    def test_repr(self):
        """Test string representation."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(RXGate(theta=0.5), target=0)

        repr_str = repr(circuit)
        assert "2 qubits" in repr_str
        assert "1 gates" in repr_str

    def test_draw(self):
        """Test circuit drawing."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(HadamardGate(), target=0)
        circuit.add_gate(CNOTGate(), target=(0, 1))

        drawing = circuit.draw()
        assert "q0:" in drawing
        assert "q1:" in drawing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

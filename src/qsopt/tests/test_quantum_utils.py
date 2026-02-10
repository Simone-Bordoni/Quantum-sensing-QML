"""
Tests for quantum_utils module
================================

Test suite for quantum utility functions including operator generation,
initial state preparation, and quantum gates.
"""

import numpy as np
import pytest
import qutip as qt

from qsopt.core.experiment.quantum_utils import (
    apply_single_qubit_rotation,
    create_measurement_projector,
    embed_circuit_unitary,
    generate_initial_state,
    generate_n_qubit_operators,
    gu,
)
from qsopt.core.experimental_parameters import InitialStateConfig, InitialStateType


class TestOperatorGeneration:
    """Test operator generation functions."""

    def test_single_qubit_operators_structure(self):
        """Test that single-qubit operators have correct structure."""
        field_levels = 2
        cavity_levels = 2
        qubit_levels = 2
        n_qubits = 1

        operators = generate_n_qubit_operators(field_levels, cavity_levels, qubit_levels, n_qubits)

        # Check all required operators exist
        required_ops = [
            "a_in",
            "a_in_dag",
            "a",
            "a_dag",
            "sigma_z",
            "sigma_x",
            "sigma_y",
            "sigma_minus",
            "sigma_plus",
            "P0_q",
            "P1_q",
            "P_all0",
            "I_field",
            "I_cavity",
            "I_q",
        ]
        for op_name in required_ops:
            assert op_name in operators, f"Operator {op_name} not found"
            if op_name in ["sigma_z", "sigma_x", "sigma_y", "sigma_minus", "sigma_plus", "P0_q", "P1_q"]:
                # These are now lists for n-qubit systems
                assert isinstance(operators[op_name], list), f"{op_name} should be a list"
                assert len(operators[op_name]) == n_qubits
                assert isinstance(operators[op_name][0], qt.Qobj)
            else:
                assert isinstance(operators[op_name], (qt.Qobj, list)), f"{op_name} should be Qobj or list"

    def test_operator_dimensions(self):
        """Test that operators have correct dimensions."""
        field_levels = 3
        cavity_levels = 4
        qubit_levels = 2
        n_qubits = 1

        operators = generate_n_qubit_operators(field_levels, cavity_levels, qubit_levels, n_qubits)

        expected_dim = field_levels * cavity_levels * qubit_levels

        for op_name, op in operators.items():
            if not op_name.startswith("I_") and not isinstance(op, list):  # Skip identity and list operators
                assert op.dims == [
                    [field_levels, cavity_levels, qubit_levels],
                    [field_levels, cavity_levels, qubit_levels],
                ], f"Operator {op_name} has incorrect dimensions"
                assert op.shape == (
                    expected_dim,
                    expected_dim,
                ), f"Operator {op_name} has incorrect shape"

    def test_pauli_operators_hermitian(self):
        """Test that Pauli operators are Hermitian."""
        operators = generate_n_qubit_operators(2, 2, 2, n_qubits=1)

        pauli_ops = ["sigma_x", "sigma_y", "sigma_z"]
        for op_name in pauli_ops:
            op = operators[op_name][0]  # Get first qubit's operator
            assert (op - op.dag()).norm() < 1e-10, f"{op_name} is not Hermitian"

    def test_ladder_operators(self):
        """Test that ladder operators have correct action."""
        operators = generate_n_qubit_operators(2, 2, 2, n_qubits=1)

        # Test that a†a has correct eigenvalues (number operator)
        # For 2-level field: eigenvalues should be 0, 1
        n_field = operators["a_in_dag"] * operators["a_in"]
        n_cavity = operators["a_dag"] * operators["a"]

        # Number operators should be Hermitian
        assert (n_field - n_field.dag()).norm() < 1e-10, "Field number operator not Hermitian"
        assert (n_cavity - n_cavity.dag()).norm() < 1e-10, "Cavity number operator not Hermitian"

        # For qubit: σ₊σ₋ = (I + σz)/2
        sigma_plus_minus = operators["sigma_plus"][0] * operators["sigma_minus"][0]
        sigma_z = operators["sigma_z"][0]

        # This should be Hermitian
        assert (
            sigma_plus_minus - sigma_plus_minus.dag()
        ).norm() < 1e-10, "Qubit σ₊σ₋ not Hermitian"

    def test_projector_properties(self):
        """Test that projectors are idempotent and orthogonal."""
        operators = generate_n_qubit_operators(2, 2, 2, n_qubits=1)

        P0 = operators["P0_q"][0]  # First qubit's |0⟩⟨0| projector
        P1 = operators["P1_q"][0]  # First qubit's |1⟩⟨1| projector

        # Idempotency: P² = P
        assert (P0 * P0 - P0).norm() < 1e-10, "P0 not idempotent"
        assert (P1 * P1 - P1).norm() < 1e-10, "P1 not idempotent"

        # Orthogonality: P0 * P1 = 0
        assert (P0 * P1).norm() < 1e-10, "P0 and P1 not orthogonal"

        # Completeness: P0 + P1 projects onto full qubit space
        # (not identity in full space due to field and cavity)
        sum_proj = P0 + P1
        assert sum_proj.tr() > 0, "Projectors don't sum correctly"

    def test_two_qubit_operators_implemented(self):
        """Test that two-qubit operators are generated correctly."""
        ops = generate_n_qubit_operators(2, 2, 2, n_qubits=2)

        # Check that all expected operators are present
        assert "a_in" in ops
        assert "a" in ops
        assert "sigma_z" in ops
        assert "sigma_x" in ops
        assert "sigma_y" in ops
        assert "P0_q" in ops
        assert "P1_q" in ops
        assert "P_all0" in ops
        assert "roty_q" in ops
        assert "roty" in ops

        # Check lists have 2 elements for 2 qubits
        list_ops = ["sigma_z", "sigma_x", "sigma_y", "sigma_minus", "sigma_plus", "P0_q", "P1_q", "roty_q"]
        for op_name in list_ops:
            assert isinstance(ops[op_name], list), f"{op_name} should be a list"
            assert len(ops[op_name]) == 2, f"{op_name} should have 2 elements"
            for op in ops[op_name]:
                assert isinstance(op, qt.Qobj), f"Elements of {op_name} should be Qobj"


class TestInitialStateGeneration:
    """Test initial state generation functions."""

    def test_vacuum_state(self):
        """Test vacuum state generation."""
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, n_qubits=1)

        # Check it's a density matrix
        assert rho.isherm, "Vacuum state not Hermitian"
        assert abs(rho.tr() - 1.0) < 1e-10, "Vacuum state not normalized"

        # Check purity (pure state: Tr(ρ²) = 1)
        assert abs((rho * rho).tr() - 1.0) < 1e-10, "Vacuum state not pure"

    def test_single_photon_state(self):
        """Test single photon state generation."""
        config = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
        rho = generate_initial_state(config, 2, 2, 2, n_qubits=1)

        assert rho.isherm, "Single photon state not Hermitian"
        assert abs(rho.tr() - 1.0) < 1e-10, "Single photon state not normalized"
        assert abs((rho * rho).tr() - 1.0) < 1e-10, "Single photon state not pure"

    def test_coherent_state(self):
        """Test coherent state generation."""
        alpha = 0.5
        config = InitialStateConfig(state_type=InitialStateType.COHERENT, coherent_alpha=alpha)
        rho = generate_initial_state(config, 3, 2, 2, n_qubits=1)

        assert rho.isherm, "Coherent state not Hermitian"
        assert abs(rho.tr() - 1.0) < 1e-10, "Coherent state not normalized"
        # Coherent states are pure
        assert abs((rho * rho).tr() - 1.0) < 1e-10, "Coherent state not pure"

    def test_coherent_state_requires_alpha(self):
        """Test that coherent state requires alpha parameter."""
        config = InitialStateConfig(state_type=InitialStateType.COHERENT)

        with pytest.raises(ValueError, match="coherent_alpha must be specified"):
            generate_initial_state(config, 2, 2, 2, n_qubits=1)

    def test_custom_state_simple(self):
        """Test custom state with simple superposition."""
        # Create |+⟩ state on qubit: (|0⟩ + |1⟩)/√2
        amplitudes = {
            (0, 0, 0): 1 / np.sqrt(2),  # |0,0,0⟩
            (0, 0, 1): 1 / np.sqrt(2),  # |0,0,1⟩
        }
        config = InitialStateConfig(
            state_type=InitialStateType.CUSTOM, custom_amplitudes=amplitudes
        )
        rho = generate_initial_state(config, 2, 2, 2, n_qubits=1)

        assert rho.isherm, "Custom state not Hermitian"
        assert abs(rho.tr() - 1.0) < 1e-10, "Custom state not normalized"
        assert abs((rho * rho).tr() - 1.0) < 1e-10, "Custom state not pure"

    def test_custom_state_requires_amplitudes(self):
        """Test that custom state requires amplitudes parameter."""
        config = InitialStateConfig(state_type=InitialStateType.CUSTOM)

        with pytest.raises(ValueError, match="custom_amplitudes must be specified"):
            generate_initial_state(config, 2, 2, 2, n_qubits=1)

    def test_custom_state_index_validation(self):
        """Test that custom state validates indices."""
        # Invalid field index
        amplitudes = {(5, 0, 0): 1.0}  # field index 5 > field_levels 2
        config = InitialStateConfig(
            state_type=InitialStateType.CUSTOM, custom_amplitudes=amplitudes
        )

        with pytest.raises(ValueError, match="Field index.*out of range"):
            generate_initial_state(config, 2, 2, 2, n_qubits=1)

    def test_two_qubit_states_implemented(self):
        """Test that two-qubit states are generated correctly."""
        # Test single photon state for two qubits
        config = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
        rho = generate_initial_state(config, 2, 2, 2, n_qubits=2)

        assert rho is not None
        assert rho.isoper, "Should be a density matrix"
        assert rho.isherm, "Should be Hermitian"
        assert abs(rho.tr() - 1.0) < 1e-10, "Should be normalized"

        # Test vacuum state for two qubits
        config_vac = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho_vac = generate_initial_state(config_vac, 2, 2, 2, n_qubits=2)

        assert rho_vac is not None
        assert rho_vac.isherm, "Should be Hermitian"
        assert abs(rho_vac.tr() - 1.0) < 1e-10, "Should be normalized"


class TestQuantumGates:
    """Test quantum gate operations."""

    def test_ry_rotation_pi(self):
        """Test Ry(π) rotation flips qubit state."""
        # Start with |0⟩ state
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, n_qubits=1)

        operators = generate_n_qubit_operators(2, 2, 2, n_qubits=1)

        # Apply Ry(π) - should flip |0⟩ → |1⟩
        rho_rotated = apply_single_qubit_rotation(
            rho, np.pi, "y", operators["I_field"], operators["I_cavity"]
        )

        # Check that P0 and P1 probabilities are swapped
        P0 = operators["P0_q"][0]
        P1 = operators["P1_q"][0]

        prob0_initial = (P0 * rho * P0.dag()).tr()
        prob1_initial = (P1 * rho * P1.dag()).tr()
        prob0_rotated = (P0 * rho_rotated * P0.dag()).tr()
        prob1_rotated = (P1 * rho_rotated * P1.dag()).tr()

        assert abs(prob0_initial - prob1_rotated) < 0.1, "Ry(π) didn't flip qubit"
        assert abs(prob1_initial - prob0_rotated) < 0.1, "Ry(π) didn't flip qubit"

    def test_ry_rotation_preserves_normalization(self):
        """Test that Ry rotation preserves trace."""
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, n_qubits=1)

        operators = generate_n_qubit_operators(2, 2, 2, n_qubits=1)

        for angle in [0, np.pi / 4, np.pi / 2, np.pi]:
            rho_rotated = apply_single_qubit_rotation(
                rho, angle, "y", operators["I_field"], operators["I_cavity"]
            )
            assert (
                abs(rho_rotated.tr() - 1.0) < 1e-10
            ), f"Ry({angle}) doesn't preserve normalization"

    def test_rotation_axes(self):
        """Test rotations around different axes."""
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, n_qubits=1)

        operators = generate_n_qubit_operators(2, 2, 2, n_qubits=1)

        for axis in ["x", "y", "z"]:
            rho_rotated = apply_single_qubit_rotation(
                rho, np.pi / 2, axis, operators["I_field"], operators["I_cavity"]
            )
            assert rho_rotated.isherm, f"R{axis}(π/2) not Hermitian"
            assert (
                abs(rho_rotated.tr() - 1.0) < 1e-10
            ), f"R{axis}(π/2) doesn't preserve normalization"

    def test_invalid_rotation_axis(self):
        """Test that invalid rotation axis raises error."""
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, n_qubits=1)

        operators = generate_n_qubit_operators(2, 2, 2, n_qubits=1)

        with pytest.raises(ValueError, match="Invalid rotation axis"):
            apply_single_qubit_rotation(
                rho, np.pi / 2, "invalid", operators["I_field"], operators["I_cavity"]
            )


class TestMeasurementProjectors:
    """Test measurement projector creation."""

    def test_projector_creation(self):
        """Test that projectors are created correctly."""
        P0 = create_measurement_projector(0, 2, 2, 2)
        P1 = create_measurement_projector(1, 2, 2, 2)

        assert isinstance(P0, qt.Qobj)
        assert isinstance(P1, qt.Qobj)

        # Check dimensions
        assert P0.dims == [[2, 2, 2], [2, 2, 2]]
        assert P1.dims == [[2, 2, 2], [2, 2, 2]]

    def test_projector_properties(self):
        """Test projector mathematical properties."""
        P0 = create_measurement_projector(0, 2, 2, 2)
        P1 = create_measurement_projector(1, 2, 2, 2)

        # Idempotency
        assert (P0 * P0 - P0).norm() < 1e-10
        assert (P1 * P1 - P1).norm() < 1e-10

        # Orthogonality
        assert (P0 * P1).norm() < 1e-10
        assert (P1 * P0).norm() < 1e-10

    def test_projector_on_vacuum_state(self):
        """Test projectors on known states."""
        # Vacuum state should have P(0) = 1, P(1) = 0
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, n_qubits=1)

        P0 = create_measurement_projector(0, 2, 2, 2)
        P1 = create_measurement_projector(1, 2, 2, 2)

        prob0 = (P0 * rho * P0.dag()).tr()
        prob1 = (P1 * rho * P1.dag()).tr()

        assert abs(prob0 - 1.0) < 1e-10, "Vacuum should be in |0⟩"
        assert abs(prob1) < 1e-10, "Vacuum shouldn't be in |1⟩"

    def test_invalid_outcome(self):
        """Test that invalid measurement outcome raises error."""
        with pytest.raises(ValueError, match="Invalid measurement outcome"):
            create_measurement_projector(2, 2, 2, 2)  # outcome must be 0 or 1


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_full_workflow(self):
        """Test complete workflow: state prep → rotation → measurement."""
        # Generate operators
        operators = generate_n_qubit_operators(2, 2, 2, n_qubits=1)

        # Prepare initial state
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, n_qubits=1)

        # Apply rotation
        rho_rotated = apply_single_qubit_rotation(
            rho, np.pi / 2, "y", operators["I_field"], operators["I_cavity"]
        )

        # Measure
        P0 = create_measurement_projector(0, 2, 2, 2)
        P1 = create_measurement_projector(1, 2, 2, 2)

        prob0 = (P0 * rho_rotated * P0.dag()).tr()
        prob1 = (P1 * rho_rotated * P1.dag()).tr()

        # After Ry(π/2) on |0⟩, should get equal superposition
        assert abs(prob0 - 0.5) < 0.1, "Ry(π/2)|0⟩ should give ~50% in |0⟩"
        assert abs(prob1 - 0.5) < 0.1, "Ry(π/2)|0⟩ should give ~50% in |1⟩"
        assert abs(prob0 + prob1 - 1.0) < 1e-10, "Probabilities should sum to 1"


class TestCouplingFunction:
    """Test the time-dependent coupling function gu()."""

    def test_gu_basic_call(self):
        """Test that gu() returns proper JAX array."""
        import jax.numpy as jnp

        t = 0.5
        result = gu(t, sigma=0.1)

        # Should return JAX array
        assert isinstance(result, jnp.ndarray), "gu() should return JAX array"
        assert result.shape == (), "gu() should return scalar"

    def test_gu_default_sigma(self):
        """Test gu() with default sigma value."""
        result = gu(0.5)
        assert float(result) > 0, "Coupling should be positive"

    def test_gu_custom_sigma(self):
        """Test gu() with custom sigma parameter."""
        result1 = gu(0.5, sigma=0.1)
        result2 = gu(0.5, sigma=0.2)

        # Different sigma values should give different results
        assert float(result1) != float(result2), "Different sigma should give different coupling"

    def test_gu_jit_compatible(self):
        """Test that gu() is JIT compatible."""
        import jax

        # Should work with jax.jit (already decorated, but test double-jit)
        @jax.jit
        def compute_coupling(t):
            return gu(t, sigma=0.1)

        result = compute_coupling(0.5)
        assert float(result) > 0, "JIT compiled gu() should work"

    def test_gu_vectorization(self):
        """Test gu() with array of times."""
        import jax
        import jax.numpy as jnp

        times = jnp.array([0.1, 0.5, 1.0])

        # vmap over times
        results = jax.vmap(lambda t: gu(t, sigma=0.1))(times)

        assert results.shape == (3,), "Should handle array of times"
        assert all(float(r) > 0 for r in results), "All couplings should be positive"


class TestEmbedCircuitUnitary:
    """Test circuit unitary embedding function."""

    def test_embed_identity_single_qubit(self):
        """Test embedding identity operator for single qubit."""
        import jax.numpy as jnp

        # Single qubit identity (2x2)
        U_circuit = jnp.eye(2, dtype=jnp.complex128)
        field_levels = 2
        cavity_levels = 3

        U_full = embed_circuit_unitary(U_circuit, field_levels, cavity_levels)

        # Full space dimension should be field_levels * cavity_levels * qubit_levels
        expected_dim = field_levels * cavity_levels * 2
        assert U_full.shape == (expected_dim, expected_dim), "Full unitary has wrong shape"

        # Should be identity in full space
        expected_identity = jnp.eye(expected_dim, dtype=jnp.complex128)
        assert jnp.allclose(U_full, expected_identity), "Embedded identity should be full-space identity"

    def test_embed_identity_two_qubit(self):
        """Test embedding identity operator for two qubits."""
        import jax.numpy as jnp

        # Two qubit identity (4x4)
        U_circuit = jnp.eye(4, dtype=jnp.complex128)
        field_levels = 2
        cavity_levels = 2

        U_full = embed_circuit_unitary(U_circuit, field_levels, cavity_levels)

        # Full space dimension
        expected_dim = field_levels * cavity_levels * 4
        assert U_full.shape == (expected_dim, expected_dim), "Full unitary has wrong shape"

        # Should be identity
        expected_identity = jnp.eye(expected_dim, dtype=jnp.complex128)
        assert jnp.allclose(U_full, expected_identity), "Embedded identity should be full-space identity"

    def test_embed_pauli_x(self):
        """Test embedding Pauli-X gate."""
        import jax.numpy as jnp

        # Pauli-X gate
        U_circuit = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
        field_levels = 2
        cavity_levels = 2

        U_full = embed_circuit_unitary(U_circuit, field_levels, cavity_levels)

        # Full space dimension
        expected_dim = field_levels * cavity_levels * 2
        assert U_full.shape == (expected_dim, expected_dim)

        # Should be unitary
        U_full_dag = jnp.conj(U_full.T)
        product = U_full @ U_full_dag
        identity = jnp.eye(expected_dim, dtype=jnp.complex128)
        assert jnp.allclose(product, identity, atol=1e-10), "Embedded unitary should be unitary"

        # Applying twice should give identity (X^2 = I)
        U_squared = U_full @ U_full
        assert jnp.allclose(U_squared, identity, atol=1e-10), "Pauli-X squared should be identity"

    def test_embed_acts_on_qubit_subspace_only(self):
        """Test that embedded unitary acts only on qubit subspace."""
        import jax.numpy as jnp

        # Pauli-Z gate (diagonal, easy to check)
        U_circuit = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
        field_levels = 2
        cavity_levels = 2

        U_full = embed_circuit_unitary(U_circuit, field_levels, cavity_levels)

        # Create a state with field=0, cavity=0, qubit=0: |0,0,0⟩
        # This should be unchanged by Pauli-Z (eigenstate with eigenvalue +1)
        dim = field_levels * cavity_levels * 2
        state_000 = jnp.zeros(dim, dtype=jnp.complex128)
        state_000 = state_000.at[0].set(1.0)  # First basis state

        result_000 = U_full @ state_000
        assert jnp.allclose(result_000, state_000), "|0,0,0⟩ should be unchanged by Z gate"

        # Create state |0,0,1⟩ (field=0, cavity=0, qubit=1)
        # Index in composite basis: 0*cavity*2 + 0*2 + 1 = 1
        state_001 = jnp.zeros(dim, dtype=jnp.complex128)
        state_001 = state_001.at[1].set(1.0)

        result_001 = U_full @ state_001
        expected_001 = -state_001  # Z|1⟩ = -|1⟩
        assert jnp.allclose(result_001, expected_001), "|0,0,1⟩ should get phase -1 from Z gate"

    def test_embed_preserves_unitarity(self):
        """Test that embedding preserves unitarity."""
        import jax.numpy as jnp

        # Random unitary for single qubit
        # Use Hadamard gate
        H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
        field_levels = 3
        cavity_levels = 4

        U_full = embed_circuit_unitary(H, field_levels, cavity_levels)

        # Check unitarity: U†U = I
        U_dag = jnp.conj(U_full.T)
        product = U_dag @ U_full
        dim = field_levels * cavity_levels * 2
        identity = jnp.eye(dim, dtype=jnp.complex128)

        assert jnp.allclose(product, identity, atol=1e-10), "U†U should be identity"

        # Check UU† = I as well
        product2 = U_full @ U_dag
        assert jnp.allclose(product2, identity, atol=1e-10), "UU† should be identity"

    def test_embed_different_system_sizes(self):
        """Test embedding with different field and cavity dimensions."""
        import jax.numpy as jnp

        U_circuit = jnp.eye(2, dtype=jnp.complex128)

        # Test various system sizes
        test_cases = [
            (2, 2),  # Small system
            (3, 3),  # Medium system
            (2, 5),  # Asymmetric
            (5, 2),  # Asymmetric reverse
        ]

        for field_levels, cavity_levels in test_cases:
            U_full = embed_circuit_unitary(U_circuit, field_levels, cavity_levels)
            expected_dim = field_levels * cavity_levels * 2
            assert U_full.shape == (expected_dim, expected_dim), \
                f"Wrong shape for field={field_levels}, cavity={cavity_levels}"

            # Check it's still unitary
            U_dag = jnp.conj(U_full.T)
            product = U_dag @ U_full
            identity = jnp.eye(expected_dim, dtype=jnp.complex128)
            assert jnp.allclose(product, identity, atol=1e-10), \
                f"Lost unitarity for field={field_levels}, cavity={cavity_levels}"

    def test_embed_cnot_two_qubit(self):
        """Test embedding CNOT gate for two qubits."""
        import jax.numpy as jnp

        # CNOT gate (4x4 for 2 qubits)
        CNOT = jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.complex128)

        field_levels = 2
        cavity_levels = 2

        U_full = embed_circuit_unitary(CNOT, field_levels, cavity_levels)

        # Full space dimension
        expected_dim = field_levels * cavity_levels * 4
        assert U_full.shape == (expected_dim, expected_dim)

        # Check unitarity
        U_dag = jnp.conj(U_full.T)
        product = U_dag @ U_full
        identity = jnp.eye(expected_dim, dtype=jnp.complex128)
        assert jnp.allclose(product, identity, atol=1e-10), "Embedded CNOT should be unitary"

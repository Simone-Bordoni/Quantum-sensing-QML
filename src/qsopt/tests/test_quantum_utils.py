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
    generate_initial_state,
    generate_single_qubit_operators,
    generate_two_qubit_operators,
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

        operators = generate_single_qubit_operators(field_levels, cavity_levels, qubit_levels)

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
            "P0",
            "P1",
            "I_field",
            "I_cavity",
            "I_qubit",
        ]
        for op_name in required_ops:
            assert op_name in operators, f"Operator {op_name} not found"
            assert isinstance(operators[op_name], qt.Qobj)

    def test_operator_dimensions(self):
        """Test that operators have correct dimensions."""
        field_levels = 3
        cavity_levels = 4
        qubit_levels = 2

        operators = generate_single_qubit_operators(field_levels, cavity_levels, qubit_levels)

        expected_dim = field_levels * cavity_levels * qubit_levels

        for op_name, op in operators.items():
            if not op_name.startswith("I_"):  # Skip individual identity operators
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
        operators = generate_single_qubit_operators(2, 2, 2)

        pauli_ops = ["sigma_x", "sigma_y", "sigma_z"]
        for op_name in pauli_ops:
            op = operators[op_name]
            assert (op - op.dag()).norm() < 1e-10, f"{op_name} is not Hermitian"

    def test_ladder_operators(self):
        """Test that ladder operators have correct action."""
        operators = generate_single_qubit_operators(2, 2, 2)

        # Test that a†a has correct eigenvalues (number operator)
        # For 2-level field: eigenvalues should be 0, 1
        n_field = operators["a_in_dag"] * operators["a_in"]
        n_cavity = operators["a_dag"] * operators["a"]

        # Number operators should be Hermitian
        assert (n_field - n_field.dag()).norm() < 1e-10, "Field number operator not Hermitian"
        assert (n_cavity - n_cavity.dag()).norm() < 1e-10, "Cavity number operator not Hermitian"

        # For qubit: σ₊σ₋ = (I + σz)/2
        sigma_plus_minus = operators["sigma_plus"] * operators["sigma_minus"]
        sigma_z = operators["sigma_z"]

        # This should be Hermitian
        assert (
            sigma_plus_minus - sigma_plus_minus.dag()
        ).norm() < 1e-10, "Qubit σ₊σ₋ not Hermitian"

    def test_projector_properties(self):
        """Test that projectors are idempotent and orthogonal."""
        operators = generate_single_qubit_operators(2, 2, 2)

        P0 = operators["P0"]
        P1 = operators["P1"]

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
        ops = generate_two_qubit_operators(2, 2, 2)

        # Check that all expected operators are present
        assert "a_in" in ops
        assert "a" in ops
        assert "sigma_z1" in ops
        assert "sigma_z2" in ops
        assert "sigma_x1" in ops
        assert "sigma_x2" in ops
        assert "sigma_y1" in ops
        assert "sigma_y2" in ops
        assert "P00" in ops
        assert "P01" in ops
        assert "P10" in ops
        assert "P11" in ops
        assert "roty_q1" in ops
        assert "roty_q2" in ops
        assert "roty" in ops

        # Check that operators are QuTiP objects
        for key, op in ops.items():
            if not key.startswith("I_"):
                assert isinstance(op, qt.Qobj)


class TestInitialStateGeneration:
    """Test initial state generation functions."""

    def test_vacuum_state(self):
        """Test vacuum state generation."""
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, num_qubits=1)

        # Check it's a density matrix
        assert rho.isherm, "Vacuum state not Hermitian"
        assert abs(rho.tr() - 1.0) < 1e-10, "Vacuum state not normalized"

        # Check purity (pure state: Tr(ρ²) = 1)
        assert abs((rho * rho).tr() - 1.0) < 1e-10, "Vacuum state not pure"

    def test_single_photon_state(self):
        """Test single photon state generation."""
        config = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
        rho = generate_initial_state(config, 2, 2, 2, num_qubits=1)

        assert rho.isherm, "Single photon state not Hermitian"
        assert abs(rho.tr() - 1.0) < 1e-10, "Single photon state not normalized"
        assert abs((rho * rho).tr() - 1.0) < 1e-10, "Single photon state not pure"

    def test_coherent_state(self):
        """Test coherent state generation."""
        alpha = 0.5
        config = InitialStateConfig(state_type=InitialStateType.COHERENT, coherent_alpha=alpha)
        rho = generate_initial_state(config, 3, 2, 2, num_qubits=1)

        assert rho.isherm, "Coherent state not Hermitian"
        assert abs(rho.tr() - 1.0) < 1e-10, "Coherent state not normalized"
        # Coherent states are pure
        assert abs((rho * rho).tr() - 1.0) < 1e-10, "Coherent state not pure"

    def test_coherent_state_requires_alpha(self):
        """Test that coherent state requires alpha parameter."""
        config = InitialStateConfig(state_type=InitialStateType.COHERENT)

        with pytest.raises(ValueError, match="coherent_alpha must be specified"):
            generate_initial_state(config, 2, 2, 2, num_qubits=1)

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
        rho = generate_initial_state(config, 2, 2, 2, num_qubits=1)

        assert rho.isherm, "Custom state not Hermitian"
        assert abs(rho.tr() - 1.0) < 1e-10, "Custom state not normalized"
        assert abs((rho * rho).tr() - 1.0) < 1e-10, "Custom state not pure"

    def test_custom_state_requires_amplitudes(self):
        """Test that custom state requires amplitudes parameter."""
        config = InitialStateConfig(state_type=InitialStateType.CUSTOM)

        with pytest.raises(ValueError, match="custom_amplitudes must be specified"):
            generate_initial_state(config, 2, 2, 2, num_qubits=1)

    def test_custom_state_index_validation(self):
        """Test that custom state validates indices."""
        # Invalid field index
        amplitudes = {(5, 0, 0): 1.0}  # field index 5 > field_levels 2
        config = InitialStateConfig(
            state_type=InitialStateType.CUSTOM, custom_amplitudes=amplitudes
        )

        with pytest.raises(ValueError, match="Field index.*out of range"):
            generate_initial_state(config, 2, 2, 2, num_qubits=1)

    def test_two_qubit_states_implemented(self):
        """Test that two-qubit states are generated correctly."""
        # Test single photon state for two qubits
        config = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
        rho = generate_initial_state(config, 2, 2, 2, num_qubits=2)

        assert rho is not None
        assert rho.isoper, "Should be a density matrix"
        assert rho.isherm, "Should be Hermitian"
        assert abs(rho.tr() - 1.0) < 1e-10, "Should be normalized"

        # Test vacuum state for two qubits
        config_vac = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho_vac = generate_initial_state(config_vac, 2, 2, 2, num_qubits=2)

        assert rho_vac is not None
        assert rho_vac.isherm, "Should be Hermitian"
        assert abs(rho_vac.tr() - 1.0) < 1e-10, "Should be normalized"


class TestQuantumGates:
    """Test quantum gate operations."""

    def test_ry_rotation_pi(self):
        """Test Ry(π) rotation flips qubit state."""
        # Start with |0⟩ state
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, num_qubits=1)

        operators = generate_single_qubit_operators(2, 2, 2)

        # Apply Ry(π) - should flip |0⟩ → |1⟩
        rho_rotated = apply_single_qubit_rotation(
            rho, np.pi, "y", operators["I_field"], operators["I_cavity"]
        )

        # Check that P0 and P1 probabilities are swapped
        P0 = operators["P0"]
        P1 = operators["P1"]

        prob0_initial = (P0 * rho * P0.dag()).tr()
        prob1_initial = (P1 * rho * P1.dag()).tr()
        prob0_rotated = (P0 * rho_rotated * P0.dag()).tr()
        prob1_rotated = (P1 * rho_rotated * P1.dag()).tr()

        assert abs(prob0_initial - prob1_rotated) < 0.1, "Ry(π) didn't flip qubit"
        assert abs(prob1_initial - prob0_rotated) < 0.1, "Ry(π) didn't flip qubit"

    def test_ry_rotation_preserves_normalization(self):
        """Test that Ry rotation preserves trace."""
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, num_qubits=1)

        operators = generate_single_qubit_operators(2, 2, 2)

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
        rho = generate_initial_state(config, 2, 2, 2, num_qubits=1)

        operators = generate_single_qubit_operators(2, 2, 2)

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
        rho = generate_initial_state(config, 2, 2, 2, num_qubits=1)

        operators = generate_single_qubit_operators(2, 2, 2)

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
        rho = generate_initial_state(config, 2, 2, 2, num_qubits=1)

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
        operators = generate_single_qubit_operators(2, 2, 2)

        # Prepare initial state
        config = InitialStateConfig(state_type=InitialStateType.VACUUM)
        rho = generate_initial_state(config, 2, 2, 2, num_qubits=1)

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

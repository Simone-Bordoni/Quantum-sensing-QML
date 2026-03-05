"""
JAX-Compatible Quantum Circuit
===============================

This module provides a quantum circuit implementation that uses JAX-compatible
gates and tracks trainable parameters for gradient-based optimization.

Example:
    >>> import jax.numpy as jnp
    >>> from qsopt.core.circuit import QuantumCircuit
    >>> from qsopt.core.gates import RXGate, RYGate, CNOTGate
    >>>
    >>> # Create 2-qubit circuit
    >>> circuit = QuantumCircuit(n_qubits=2)
    >>> rx = RXGate(theta=jnp.pi/4)
    >>> rx.target = 0
    >>> circuit.add_gate(rx)
    >>> ry = RYGate(theta=jnp.pi/2)
    >>> ry.target = 1
    >>> circuit.add_gate(ry)
    >>> cnot = CNOTGate()
    >>> cnot.target = (0, 1)
    >>> circuit.add_gate(cnot)
    >>>
    >>> # Get circuit unitary
    >>> U = circuit.get_unitary(qutip=False)
    >>>
    >>> # Get trainable parameters
    >>> params = circuit.get_trainable_parameters()
"""

from typing import List, Union, Optional, Callable
import jax
import jax.numpy as jnp
import qutip as qt
import qutip_jax
from .gates import Gate
import math

@jax.tree_util.register_pytree_node_class
class QuantumCircuit:
    """
    Quantum circuit with JAX-compatible gates.

    Manages a sequence of gate applications on multiple qubits,
    tracks trainable parameters, and computes circuit unitaries.
    """

    def __init__(self, n_qubits: int = 2):
        """
        Initialize quantum circuit.

        Args:
            n_qubits: Number of qubits in the circuit
        """
        if n_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")

        self.n_qubits = n_qubits
        self._gates: List[Gate] = []
        self._cached_unitary_jax: Optional[jnp.ndarray] = None
        self._cached_unitary_qutip: Optional[qt.Qobj] = None
        self._cached_params: Optional[List] = None
        self._dynamic_fields = ("_gates", "_cached_unitary_jax", "_cached_unitary_qutip", "_cached_params")

    # Pytree construction from class for jax
    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in self._dynamic_fields)
        aux_data = {k: v for k, v in self.__dict__.items()
                    if k not in self._dynamic_fields}
        return children, aux_data

    # Class reconstruction from pytree for jax
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        # restore static data first
        obj.__dict__.update(aux_data)

        # restore dynamic fields
        for name, value in zip(obj._dynamic_fields, children):
            setattr(obj, name, value)

        return obj

    def add_gate(self, gate: Gate) -> None:
        """
        Add a gate to the circuit.

        Args:
            gate: Gate object to add

        Raises:
            ValueError: If target qubits are out of range or not set
        """

        # Normalize target to tuple for validation
        target_tuple = (gate.target,) if isinstance(gate.target, int) else tuple(gate.target)

        # Validate all target qubits are in range
        for qubit in target_tuple:
            if qubit < 0 or qubit >= self.n_qubits:
                raise ValueError(
                    f"Target qubit {qubit} is out of range. "
                    f"Circuit has {self.n_qubits} qubits (0-{self.n_qubits-1})"
                )

        self._gates.append(gate)

    def get_trainable_parameters(self) -> List[jnp.ndarray]:
        """
        Get all trainable parameters from the circuit.

        Returns:
            List of trainable parameter values
        """
        params = []
        for gate in self._gates:
            if gate.has_parameter() and gate._parameter.trainable:
                params.append(gate.get_parameter())
        return params

        #params = {}
        #for gate in self._gates:
        #    if gate.has_parameter() and gate._parameter.trainable:
        #        params[gate._parameter.name] = gate.get_parameter())
        #return params

    def count_trainable_parameters(self) -> int:
        """
        Count the number of trainable parameters in the circuit.

        Returns:
            Number of trainable parameters
        """
        count = 0
        for gate in self._gates:
            if gate.has_parameter() and gate._parameter.trainable:
                count += 1
        return count

    def set_trainable_parameters(self, parameters: List[jnp.ndarray]) -> None:
        """
        Update trainable parameters in the circuit.

        Args:
            parameters: List of parameter values (must match number of trainable parameters)
        """
        expected_count = self.count_trainable_parameters()
        if len(parameters) != expected_count:
            raise ValueError(f"Expected {expected_count} parameters, got {len(parameters)}")

        param_idx = 0
        for gate in self._gates:
            if gate.has_parameter() and gate._parameter.trainable:
                gate.set_parameter(parameters[param_idx])
                param_idx += 1

        # Invalidate cached unitary since parameters changed
        self._cached_unitary_jax = None
        self._cached_unitary_qutip = None
        self._cached_params = None

    def get_unitary(self, qutip: bool = True) -> Union[jnp.ndarray, qt.Qobj]:
        """
        Compute the circuit unitary.

        Args:
            qutip: If True, return QuTiP Qobj; if False, return JAX array

        Returns:
            Circuit unitary matrix as QuTiP Qobj or JAX array
        """
        # Check if we have a cached unitary for current parameters
        current_params = self.get_trainable_parameters()
        params_match = False
        if self._cached_params is not None and len(current_params) == len(self._cached_params):
            # Check if all parameters match (using allclose for numerical arrays)
            params_match = all(
                jnp.allclose(p1, p2) for p1, p2 in zip(current_params, self._cached_params)
            )

        # Return cached unitary if parameters match
        if params_match:
            if qutip and self._cached_unitary_qutip is not None:
                return self._cached_unitary_qutip
            elif not qutip and self._cached_unitary_jax is not None:
                return self._cached_unitary_jax

        # Otherwise, compute unitary
        if len(self._gates) == 0:
            # Identity for empty circuit
            dim = 2 ** self.n_qubits
            identity = jnp.eye(dim, dtype=jnp.complex128)
            if qutip:
                return qt.Qobj(identity, dims=[[2]*self.n_qubits, [2]*self.n_qubits])
            return identity

        # Build unitary by applying gates in sequence
        dim = 2 ** self.n_qubits
        U_jax = jnp.eye(dim, dtype=jnp.complex128)

        for gate in self._gates:
            # Get gate matrix as JAX array directly
            gate_jax = gate.matrix(qutip=False)

            # Expand to full Hilbert space using JAX operations
            if self.n_qubits == 1:
                # Single qubit circuit
                expanded_jax = gate_jax
            else:
                # Multi-qubit circuit - manually expand using Kronecker products
                target_tuple = (gate.target,) if isinstance(gate.target, int) else gate.target
                expanded_jax = self._expand_gate_jax(gate_jax, target_tuple)

            # Apply gate (multiply on left since gates are applied left to right)
            U_jax = expanded_jax @ U_jax

        # Cache both JAX and QuTiP versions
        self._cached_unitary_jax = U_jax
        self._cached_unitary_qutip = qt.Qobj(U_jax, dims=[[2]*self.n_qubits, [2]*self.n_qubits])
        self._cached_params = [jnp.array(p) for p in current_params]  # Store copy of params

        # Return appropriate version
        if qutip:
            return self._cached_unitary_qutip
        return U_jax

    @jax.jit
    def get_parametric_circuit(self) -> Callable[List[jnp.ndarray, jnp.ndarray]]:
        """
        Compute the circuit unitary from a set of parameters

        Returns:
            Circuit unitary matrix as QuTiP Qobj or JAX array
        """

        # Compute unitary
        if len(self._gates) == 0:
            # Identity for empty circuit
            dim = 2 ** self.n_qubits
            identity = jnp.eye(dim, dtype=jnp.complex128)
            return identity

        # Build unitary by applying gates in sequence
        dim = 2 ** self.n_qubits
        U_jax = jnp.eye(dim, dtype=jnp.complex128)

        for gate in self._gates:
            # Get gate matrix as JAX array directly
            gate_jax = gate.matrix(qutip=False)

            # Expand to full Hilbert space using JAX operations
            if self.n_qubits == 1:
                # Single qubit circuit
                expanded_jax = gate_jax
            else:
                # Multi-qubit circuit - manually expand using Kronecker products
                target_tuple = (gate.target,) if isinstance(gate.target, int) else gate.target
                expanded_jax = self._expand_gate_jax(gate_jax, target_tuple)

            # Apply gate (multiply on left since gates are applied left to right)
            U_jax = expanded_jax @ U_jax

        # Cache both JAX and QuTiP versions
        self._cached_unitary_jax = U_jax
        self._cached_unitary_qutip = qt.Qobj(U_jax, dims=[[2]*self.n_qubits, [2]*self.n_qubits])
        self._cached_params = [jnp.array(p) for p in parameters]  # Store copy of params

        # Return appropriate version
        return U_jax


    def __call__(self, state: Optional[Union[jnp.ndarray, qt.Qobj]] = None, qutip: bool = True) -> Union[jnp.ndarray, qt.Qobj]:
        """
        Apply the circuit to a quantum state.

        The circuit is applied by evolving the density matrix through the unitary:
        ρ_final = U ρ_initial U†

        Args:
            state: Initial quantum state (can be JAX array or QuTiP Qobj, pure state or density matrix).
                   If None, uses ground state |0...0⟩ as initial state.
            qutip: If True, return QuTiP Qobj; if False, return JAX array

        Returns:
            Final quantum state after circuit application

        Example:
            >>> circuit = QuantumCircuit(n_qubits=2)
            >>> circuit.add_gate(HadamardGate(target=0))
            >>> # Use ground state by default
            >>> rho_final = circuit()
            >>> # Or provide initial state
            >>> psi0 = qt.basis(2, 0)
            >>> psi_final = circuit(psi0)
        """
        # Get circuit unitary as JAX array
        U = self.get_unitary(qutip=False)

        # If no state provided, use ground state |0...0⟩
        if state is None:
            # Create ground state as density matrix |0...0⟩⟨0...0|
            dim = 2 ** self.n_qubits
            rho = jnp.zeros((dim, dim), dtype=jnp.complex128)
            rho = rho.at[0, 0].set(1.0)
        else:
            # Convert state to JAX array if it's a QuTiP object
            if isinstance(state, qt.Qobj):
                state_jax = jnp.array(state.full(), dtype=jnp.complex128)
            else:
                state_jax = jnp.asarray(state, dtype=jnp.complex128)

            # Check if state is a pure state (vector) or density matrix
            if state_jax.ndim == 1:
                # Pure state vector - convert to column vector then to density matrix
                state_jax = state_jax.reshape(-1, 1)
                rho = state_jax @ state_jax.conj().T
            elif state_jax.shape[1] == 1:
                # Column vector - convert to density matrix
                rho = state_jax @ state_jax.conj().T
            else:
                # Already a density matrix
                rho = state_jax

        # Apply circuit: ρ_final = U ρ U†
        rho_final = U @ rho @ U.conj().T

        # Return in requested format
        if qutip:
            return qt.Qobj(rho_final, dims=[[2]*self.n_qubits, [2]*self.n_qubits])
        return rho_final

    def _expand_gate_jax(self, gate_matrix: jnp.ndarray, targets: tuple) -> jnp.ndarray:
        """
        Expand a gate matrix to act on specific qubits in the full Hilbert space.

        Uses JAX operations (Kronecker products) to build the expanded operator,
        maintaining JAX traceability for autodiff.

        Args:
            gate_matrix: Gate matrix as JAX array
            targets: Tuple of target qubit indices

        Returns:
            Expanded gate matrix as JAX array
        """
        if len(targets) == 1:
            # Single-qubit gate
            target = targets[0]
            matrices = []
            for i in range(self.n_qubits):
                if i == target:
                    matrices.append(gate_matrix)
                else:
                    matrices.append(jnp.eye(2, dtype=jnp.complex128))

            # Build full operator using Kronecker products
            result = matrices[0]
            for mat in matrices[1:]:
                result = jnp.kron(result, mat)
            return result

        elif len(targets) == 2:
            # Two-qubit gate (e.g., CNOT)
            # Build the full operator by expanding the 4x4 gate matrix to the full Hilbert space
            # while preserving the control-target relationship

            q0, q1 = targets  # Qubits as specified in gate (e.g., control, target for CNOT)

            # Create full operator matrix
            dim = 2 ** self.n_qubits
            full_matrix = jnp.zeros((dim, dim), dtype=jnp.complex128)

            # Iterate over all basis states
            for i in range(dim):
                for j in range(dim):
                    # Convert i and j to binary representations (basis states)
                    # QuTiP uses big-endian ordering: |q_{n-1}...q_1 q_0⟩
                    # So qubit k's bit is at position (n_qubits - 1 - k)
                    i_bits = [(i >> (self.n_qubits - 1 - k)) & 1 for k in range(self.n_qubits)]
                    j_bits = [(j >> (self.n_qubits - 1 - k)) & 1 for k in range(self.n_qubits)]

                    # Check if all other qubits (not q0, q1) are the same
                    other_qubits_match = all(
                        i_bits[k] == j_bits[k] for k in range(self.n_qubits) if k != q0 and k != q1
                    )

                    if other_qubits_match:
                        # Get the 2-qubit state indices for gate application
                        # The gate matrix expects: |q0, q1⟩ ordering
                        # So index 0 = |00⟩, 1 = |01⟩, 2 = |10⟩, 3 = |11⟩ for qubits (q0, q1)
                        i_gate = i_bits[q0] * 2 + i_bits[q1]  # Row index in 4x4 gate matrix
                        j_gate = j_bits[q0] * 2 + j_bits[q1]  # Col index in 4x4 gate matrix

                        # Apply the gate matrix element
                        full_matrix = full_matrix.at[i, j].set(gate_matrix[i_gate, j_gate])

            return full_matrix
    
    def add_layer(
        self, gate_type: type,
        parameters: Optional[Union[float,List[float]]] = None,
        targets: Optional[List[int]] = None,
        trainable: bool = True
    ) -> None:
        """
        Add a layer of identical single-qubit gates to the circuit.

        Args:
            gate_type: Gate class (e.g., RXGate, RYGate)
            parameters: List of parameters for each qubit (None = pi/2 for each qubit)
            targets: List of target qubits (None = all qubits)
            trainable: Whether parameters should be trainable
        """
        from .gates import CNOTGate, CZGate, HadamardGate

        if any(gate == gate_type for gate in [CNOTGate, CZGate]):
            raise ValueError(f"For entangling gates ({gate_type}) use the method: circuit.add_entangling_layer(gate_type, pattern)")
        
        if targets is None:
            targets = list(range(self.n_qubits))
        
        if gate_type == HadamardGate:
            for qubit in targets:
                gate = gate_type(target=qubit)
                self.add_gate(gate) 

        else:
            
            if parameters is None:
                parameters = [np.pi / 2] * len(targets)
            elif isinstance(parameters,float):
                parameters = [parameters] * len(targets)

            if len(parameters) != len(targets):
                raise ValueError(
                    f"Number of parameters ({len(parameters)}) must match "
                    f"number of targets ({len(targets)})"
                )

            for qubit, param in zip(targets, parameters):
                gate = gate_type(theta=param, target=qubit, trainable=trainable)
                self.add_gate(gate)

    def add_entangling_layer(
        self , gate_type: type,
        pattern: str = "linear",
        targets: Optional[List[int]] = None
    ) -> None:
        """
        Add a layer of two-qubit entangling gates to the circuit.

        Args:
            gate_type: Two-qubit gate class (e.g., CNOTGate, CZGate)
            pattern: Connectivity pattern - "linear" or "circular"
            targets: Ordered list of target qubits 
                        if None -> all qubits ordered by index
        """

        if targets is None:
            targets = list(range(self.n_qubits))

        if len(targets) < 2:
            raise ValueError("Need at least 2 qubits for entangling layer")

        if pattern == "linear":
            # Connect adjacent qubits: 0-1, 1-2, 2-3, ...
            for i in targets:
                gate = gate_type(target=(i, i + 1))
                self.add_gate(gate)
        elif pattern == "circular":
            # Linear + connect last to first
            for i in targets:
                gate = gate_type(target=(i, i + 1))
                self.add_gate(gate)
            # Wrap around
            gate = gate_type(target=(targets[-1], targets[0]))
            self.add_gate(gate)
        else:
            raise ValueError(f"Unknown pattern: {pattern}. Use 'linear' or 'circular'")

    def __repr__(self) -> str:
        """String representation of circuit."""
        header = f"QuantumCircuit({self.n_qubits} qubits, {len(self._gates)} gates)"
        if len(self._gates) == 0:
            return header

        gates_str = "\n".join(
            f"  {i}: {gate}[{gate.target}]" for i, gate in enumerate(self._gates)
        )
        return f"{header}\n{gates_str}"



# Utility functions for circuit construction

def create_ry_circuit_layer(
    n_qubits: int,
    theta_values: Optional[Union[List[float],float]] = None,
    trainable: bool = True,
) -> QuantumCircuit:
    """
    Create a quantum circuit with RY gates on all qubits.

    This is a utility function for creating default rotation layers,
    commonly used in quantum sensing and variational quantum algorithms.

    Args:
        n_qubits: Number of qubits in the circuit
        theta_values: Rotation angle value applied to all qubits or a list (one per qubit).
                      If None initializes all to π/2
        trainable: Whether the rotation parameters should be trainable

    Returns:
        QuantumCircuit with RY gate on each qubit

    Example:
        >>> # Create 2-qubit circuit with trainable RY gates
        >>> circuit = create_ry_circuit_layer(n_qubits=2, theta_values=[np.pi/4, -np.pi/4])
        >>>
        >>> # Default initialization (all π/2)
        >>> circuit = create_ry_circuit_layer(n_qubits=3)
    """
    from .gates import RYGate
    import numpy as np

    if theta_values is None:
        theta_values = [np.pi / 2] * n_qubits
    elif isinstance(theta_values,float):
        theta_values = [theta_values] * n_qubits

    if len(theta_values) != n_qubits:
        raise ValueError(
            f"Lenght of the list theta values ({len(theta_values)}) must match "
            f"number of qubits ({n_qubits}), or it must be a float"
        )

    circuit = QuantumCircuit(n_qubits=n_qubits)

    circuit.add_layer(RYGate, parameters=theta_values, trainable=trainable)

    return circuit

"""Two-qubit RY detector: 5 optimizations at increasing noise, accuracy vs noise.

Uses the values from the two_qubit_ry_angle_sweep tutorial (see noise_sweep_common); shares
everything with the 1-qubit run except the circuit.
Run:  python noise_sweep_2qb.py
"""

from noise_sweep_common import run_noise_sweep


def main():
    run_noise_sweep(n_qubits=2, name="2qb")


if __name__ == "__main__":
    main()

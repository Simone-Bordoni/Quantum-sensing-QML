"""Single-qubit RY detector: 5 optimizations at increasing noise, accuracy vs noise.

Shares all values with the 2-qubit run (see noise_sweep_common); only the circuit changes.
Run:  python noise_sweep_1qb.py
"""

from noise_sweep_common import run_noise_sweep


def main():
    run_noise_sweep(n_qubits=1, name="1qb")


if __name__ == "__main__":
    main()

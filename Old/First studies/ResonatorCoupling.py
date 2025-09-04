import matplotlib.pyplot as plt
import numpy as np
from qutip import *

N = 5
# frequencies
wc = 5.0
w1 = 3.0
w2 = 2.0
# coupling strengths (use symmetric)
g1 = 0.03
g2 = 0.03
g_eff = g1*g2/(np.sqrt(g1**2+g2**2))

# Start interaction time
T_0 =5
# Interaction time
T_gate= np.pi / (2*g_eff)  

tlist = np.linspace(0, 100, 500)

# cavity operators
a = tensor(destroy(N), qeye(2), qeye(2))
n = a.dag() * a

# operators for qubit 1
sm1 = tensor(qeye(N), destroy(2), qeye(2))
sz1 = tensor(qeye(N), sigmaz(), qeye(2))
n1 = sm1.dag() * sm1

# oeprators for qubit 2
sm2 = tensor(qeye(N), qeye(2), destroy(2))
sz2 = tensor(qeye(N), qeye(2), sigmaz())
n2 = sm2.dag() * sm2

# Hamiltonian
Hc = a.dag() * a
H1 = -0.5 * sz1
H2 = -0.5 * sz2
Hc1 = g1 * (a.dag() * sm1 + a * sm1.dag())
Hc2 = g2 * (a.dag() * sm2 + a * sm2.dag())
Hint = Hc1 + Hc2

def step_t(w1, w2, t0, tgate, t):
    """
    Step function that goes from w1 to w2 at time t0 and returns to w1 at t0+tgate as a function of t.
    Used to change the qubit frequency during the simulation.
    """
    return w1 + (w2 - w1) * (t > t0) * (t < t0 + tgate)

def wc_t(t, args=None):
    return wc

def w1_t(t, args=None):
    return step_t(w1, wc, T_0, T_gate, t)

def w2_t(t, args=None):
    return step_t(w2, wc, T_0, T_gate, t)

# Hamiltonian with time-dependent frequencies
H_t = [[Hc, wc_t], [H1, w1_t], [H2, w2_t], Hint]
# Initial state
psi0 = tensor(basis(N, 0), basis(2, 1), basis(2, 0))
# Evolution
res = mesolve(H_t, psi0, tlist, [], [])

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

axes[0].plot(
    tlist,
    np.array(list(map(wc_t, tlist))),
    "r",
    linewidth=2,
    label="cavity",
)
axes[0].plot(
    tlist,
    np.array(list(map(w1_t, tlist))),
    "b",
    linewidth=2,
    label="qubit 1",
)
axes[0].plot(
    tlist,
    np.array(list(map(w2_t, tlist))),
    "g",
    linewidth=2,
    label="qubit 2",
)
axes[0].set_ylim(1, 6)
axes[0].set_ylabel("w", fontsize=16)
axes[0].legend()

axes[1].plot(tlist, np.real(expect(n, res.states)), "r",
             linewidth=2, label="cavity")
axes[1].plot(tlist, np.real(expect(n1, res.states)), "b",
             linewidth=2, label="qubit 1")
axes[1].plot(tlist, np.real(expect(n2, res.states)), "g",
             linewidth=2, label="qubit 2")
axes[1].set_ylim(0, 1)

axes[1].set_xlabel("Time (ns)", fontsize=16)
axes[1].set_ylabel("Occupation probability", fontsize=16)
axes[1].legend()

fig.tight_layout()
plt.savefig("ResonatorCoupling.png")
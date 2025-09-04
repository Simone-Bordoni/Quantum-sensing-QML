from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters
w1 = 1.0
w2 = 2.0
g12 = 0.01
gamma1 = 0.001
gamma2 = 0.001

t_int = np.pi / (2 * g12)       # Interaction time
t_0 = 20                         # Start of the interaction
tlist = np.linspace(0, 250, 500)

# Initial state
psi0 = tensor(basis(2, 1), basis(2, 0))

# operators for qubit 1
sm1 = tensor(destroy(2), qeye(2))
sz1 = tensor(sigmaz(), qeye(2))
n1 = sm1.dag() * sm1

# oeprators for qubit 2
sm2 = tensor(qeye(2), destroy(2))
sz2 = tensor(qeye(2), sigmaz())
n2 = sm2.dag() * sm2

# Hamiltonian for the qubits
H1 = -0.5 * sz1
H2 = -0.5 * sz2
# Interaction Hamiltonian for capacitive coupling
Hc = g12/2 * (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()))

# Losses
c_ops = [np.sqrt(gamma1) * sm1, np.sqrt(gamma2) * sm2]

def step_t(w1, w2, t0, width, t):
    """
    Step function that goes from w1 to w2 at time t0 and returns to w1 at t_0+width as a function of t.
    Used to change the qubit frequency during the simulation.
    """
    return w1 + (w2 - w1) * (t > t0) * (t < t0 + width)

def H1_coeff(t, args=None):
    return step_t(w1, w2, t_0, t_int, t)

H_t = [[H1, H1_coeff],  w2*H2, Hc]

if gamma1 > 0 or gamma2 > 0:
    res = mesolve(H_t, psi0, tlist, c_ops, [])
else:
    res = mesolve(H_t, psi0, tlist, [], [])
print('Initial state')
print(psi0)
print('Final state')
print(res.states[-1])

# Plot the results
figure_title = "iSWAP: " + "w1 = " + str(w1) + ", w2 = " + str(w2) + ", g12 = " + str(g12)
fig, axes = plt.subplots(1, 1)
axes.set_title(figure_title)
axes.plot(tlist, np.real(expect(n1, res.states)),
             linewidth=2, label="qubit 1")
axes.plot(tlist, np.real(expect(n2, res.states)),
             linewidth=2, label="qubit 2")
axes.set_ylim(0, 1)

axes.set_xlabel("Time (ns)", fontsize=16)
axes.set_ylabel("Occupation probability", fontsize=16)
axes.legend()

plt.savefig('iSWAP_losses.png')


from qutip import *
import numpy as np
import matplotlib.pyplot as plt

wc = 1.5  # cavity frequency
wq = 1.0  # qubit frenqency
g = 0.1  # coupling strength
kappa = 0  # cavity dissipation rate

# cavity mode operator
a = tensor(destroy(5), qeye(2))     # destroy operator for the cavity
pn = a.dag() * a                    # photon number operator

# qubit operators
sz = tensor(qeye(5), sigmaz())      # sigma-z operator
sm = tensor(qeye(5), destroy(2))    # sigma-minus operator
sp = sm.dag()                       # sigma-plus operator

# the Jaynes-Cumming Hamiltonian
H = wc * pn - 0.5 * wq * sz + g * (a * sp + a.dag() * sm)

# excited qubit and no photons in the cavity
psi0 = tensor(basis(5, 0), basis(2, 1))
# times at which the solution is computed
tlist=np.linspace(0, 500, 200)

# evolve the system and calculate the expectation values of qubit state and num photons
qubit_state = sp * sm
if kappa == 0:
    result = mesolve(H, psi0, tlist, [], [qubit_state, pn])
else:
    c_ops = np.sqrt(kappa) * a  # collapse operators
    result = mesolve(H, psi0, tlist, [c_ops], [qubit_state, pn])

# plot the cavity occupation and qubit excitation probabilities
plot_title = "g=" + str(g) + ", wc=" + str(wc) + ", wq=" + str(wq) + ", kappa=" + str(kappa)
fig, axes = plt.subplots(1, 1)
axes.plot(tlist, result.expect[1], label="n_photons")
axes.plot(tlist, result.expect[0], label="q_state")
axes.set_xlabel("t", fontsize=20)
axes.legend(loc=2)
axes.set_title(plot_title)
plt.savefig(plot_title + ".png")


from qutip import *
import numpy as np
import matplotlib.pyplot as plt

N = 5  # number of cavity levels
wc = 5.0  # cavity frequency
wq = 3.0  # qubit frenqency
g_r = 0.05  # coupling strength readout
g_drive = 0.05  # coupling strength drive
kappa = 0  # cavity dissipation rate

# Parameters for the drive pulse (external field)
gate_type = 'X'  # type of gate: 'X', 'Y', 'I' or set manually the parameters
v_0 = 0.5  # amplitude of the drive pulse
w_d = wq  # frequency of the drive pulse
t_0 = 10  # start of the drive pulse or center of the gaussian envelope
gaussian = False  # gaussian envelope or not
sigma = 30  # width of the gaussian envelope
if gate_type == 'X':
    d_phase = 0.0  # phase of the drive pulse
    width = np.pi / (2 * g_drive * v_0)
elif gate_type == 'Y':
    d_phase = np.pi / 2.0  # phase of the drive pulse
    width = np.pi / (2 * g_drive * v_0)
elif gate_type == 'I':
    width = 0.0
    d_phase = 0.0  # phase of the drive pulse
else:
    width = 20.0  # width of the drive pulse
    d_phase = 0.0  # phase of the drive pulse

print("Drive pulse time:", width)
print("Drive pulse phase:", d_phase)

tlist = np.linspace(0, 50, 300)
# initial state of the qubit
psi0_q = basis(2, 0)  # ground state
# psi0_q = 1/np.sqrt(2)*(basis(2, 0) + basis(2,1))  # superposition state
psi0 = tensor(basis(N, 0), psi0_q)

# cavity mode operator
a = tensor(destroy(N), qeye(2))     # destroy operator for the cavity
pn = a.dag() * a                    # photon number operator

# qubit operators
sz = tensor(qeye(N), sigmaz())      # sigma-z operator
sx = tensor(qeye(N), sigmax())      # sigma-x operator
sy = tensor(qeye(N), sigmay())      # sigma-y operator
sm = tensor(qeye(N), destroy(2))    # sigma-minus operator
sp = sm.dag()                       # sigma-plus operator
nq = sm.dag() * sm              # qubit number operator

# drive pulse
def gaussian_envelope(t, t_0, sigma):
    '''Gaussian envelope function, normalized to one.'''
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-((t - t_0) ** 2) / (2 * sigma ** 2))

def pulse(t, d_phase, t_0, width, w_d, sigma=20):
    if gaussian:
        return width * gaussian_envelope(t, t_0, sigma) * np.sin(w_d * t + d_phase)
    return np.sin(w_d * t + d_phase ) * (t > t_0) * (t < t_0 + width)

def V_t(t, args=None):
    return pulse(t, d_phase, t_0, width, w_d)

# Hamiltonians
H_q = - 0.5 * wq * sz
H_c = wc * pn 
H_cq = g_r * (a * sp + a.dag() * sm)
H_drive = g_drive * sy

H_t = [[H_drive, V_t],  H_q, H_c, H_cq]

# Evolution
res = mesolve(H_t, psi0, tlist, [], [])

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
axes[0].plot(
    tlist,
    np.array(list(map(V_t, tlist))),
    "r",
    linewidth=2,
    label="drive",)

axes[0].set_ylabel("V(t)", fontsize=16)
axes[0].set_xlabel("t", fontsize=16)
axes[0].legend()
axes[1].plot(tlist, np.real(expect(nq, res.states)), "b",
             linewidth=2, label="qubit")
axes[1].plot(tlist, np.real(expect(pn, res.states)), "r",
             linewidth=2, label="cavity")
axes[1].set_xlabel("t", fontsize=16)
axes[1].set_ylabel("Occupation probability", fontsize=16)
axes[1].legend()
fig.tight_layout()
plt.savefig("Occupation.png")
plt.close(fig)

# Extract expectation values for pauli matrices
exp_sx_circ = expect(sx, res.states)
exp_sy_circ = expect(sy, res.states)
exp_sz_circ = expect(sz, res.states)
exp_sx_circ, exp_sy_circ, exp_sz_circ = (
    np.array(exp_sx_circ),
    np.array(exp_sy_circ),
    np.array(exp_sz_circ),
)

# Create Bloch sphere plot
sphere = Bloch()
sphere.add_points([exp_sx_circ, exp_sy_circ, exp_sz_circ], meth="l")
sphere.add_states(psi0_q)
bloch_fig = sphere.show()
plt.savefig("Bloch.png")




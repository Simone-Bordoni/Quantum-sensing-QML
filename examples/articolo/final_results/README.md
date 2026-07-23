## Articolo — consolidated best results

This folder distills every experiment run under `examples/articolo/` down to its  
**architecture**, its **best optimized circuit values**, and its **key results / plots**.  
Once you have verified this is complete, everything else under `examples/articolo/`  
(the `results/`, `results_amp7/`, per-level / per-run / per-spacing `.npz` files, and the  
diagnostic `.txt` files) can be deleted — the source scripts themselves live in the repo and  
are not touched here.

There are **four kinds of experiment**:

| # | Experiment | Detector physics | Best result (headline) |
| --- | --- | --- | --- |
| 1 | Coherent-drive 2-qubit corner sweep | persistent dispersive cavity, drive vs no-drive | validation **0.783** @ chi=1, kappa=12, amp=20, spacing=3.2 |
| 2 | Ensemble build-up chains (2 qubits) | persistent dispersive detector, drive vs no-drive | accuracy **0.843** (amp=7 regime, all chains converge here) |
| 3 | Multi-photon number detector (0/1/2 photons) | transient itinerant photon, 2 dispersive qubits | accuracy **0.823** @ measurement spacing 5.0 (low noise) |
| 4 | Noise sweeps (1qb / 2qb / 2qb-entangled) | transient single-photon RY detector | accuracy **0.999** clean → 0.50 at noise 0.2 |

All accuracies are the **mean of the confusion-matrix diagonal** (per-configuration  
true-positive rate, excluding the prediction-only `mixed` column). "validation" is the  
optimizer's detection metric. Circuit parameter order is always  
`initial_circuit` (setup) params first, then `final_circuit` (decode) params, in gate-layer order.

Reload any `.npz` with:

```python
from qsopt import OptimizationCallback           # for callback files (circuits + confusion)
cb = OptimizationCallback.load_callback("....npz")
cb.get_best_trainable_params()   # (setup_params, decode_params)
# or plain numpy for summaries:
import numpy as np; d = np.load("...summary.npz", allow_pickle=True)
```

## 1\. Coherent-drive 2-qubit corner sweep

Folder: `1_coherent_drive_corner_sweep/` — source: `coherent_drive_2qb_corner_sweep.py`

**Architecture.** 1 cavity (4 Fock levels) dispersively coupled to qubit 0, cavity dissipation  
`kappa`; 2 qubits. Two configurations: `no drive` (ground) vs `driven` (coherent cavity drive of  
amplitude = 2·chi). Noise 0.01 on all channels. Persistent (time-independent) Hamiltonian.  
Trainable circuits: `RY⊗2 + CNOT(circular)` for both setup and decode.  
Metric: **max trace distance**. This is a **parameter sweep** (not an optimization): a 4-D grid  
over `(chi, kappa, amplitude, time_interval)`, 4 points per axis (256 points), batch 32, scored  
over a fixed window `[-2, 8]`.

**Best result.** validation **0.7829** at **chi = 1.0, kappa = 12.0, amplitude = 20.0,**  
**time\_interval (spacing) = 3.2**. Key trend: chi = 1 dominates; higher chi caps near 0.44. Full grid  
in `coherent_drive_2qb_corner_sweep.npz` (`result_validation` is a 4×4×4×4 array; axis values in  
`axis_vals_0..3`, order chi/kappa/amplitude/time\_interval). Corner plot: `coherent_drive_2qb_corner.pdf`.

## 2\. Ensemble build-up chains (2-qubit coherent-drive detector)

Folder: `2_ensemble_buildup_chains/` — sources: `ensemble/ensemble_common.py`,  
`ensemble_ry_entangle.py`, `ensemble_rxrz_entangle.py`, `ensemble_ry_depth.py`

**Architecture.** Same persistent dispersive detector (drive vs no-drive), but scored with the  
**max computational distance** criterion and optimized (SGD, lr 0.1, 1500 steps, 3 random restarts,  
best kept). Two physics regimes were run:

*   **amp7 regime** (`amp7_best_regime/`, source uses `chi=1, amplitude=7, kappa=20, cavity_levels=8, spacing=3.2`, batch 1 no-shift) — **this is the better regime.**
*   **amp20 regime** (`amp20_regime/`) — the original weaker-metric regime.

Each chain is a build-up: every stage only _adds_ gate layers to the previous stage's circuits and  
is warm-started from the parent's best params. Three chains:

*   `ry_entangle`: RY → +CNOT(circular)+RY into setup, then decode, then +RZ layers (5 stages).
*   `rxrz_entangle`: RX → +CNOT(linear)+RZ into setup, then decode, then a 2nd CNOT+RX block (4 stages).
*   `ry_depth`: RY depth only, **no entanglement** (control chain), +RZ/+RX layers (5 stages).

**Best result.** In the **amp7 regime all three chains converge to accuracy ≈ 0.843 / validation**  
**0.6856** — added depth and entanglement give essentially no gain over the clean RY pair (0.8425),  
i.e. the plain rotation detector is already near-optimal here. (amp20 regime tops out at accuracy  
≈ 0.615.) Per-chain accuracy-vs-stage curves in `*_accuracy.pdf`; full optimization + confusion  
dashboards in `*_dashboard.pdf`; per-stage metrics in `*_summary.npz`.

**Best-stage optimized circuit values** (`*_BESTSTAGE_*.npz`, order = setup then decode params):

_amp7 regime (validation 0.6856, accuracy 0.843 — the best):_  
| chain | best stage | setup params | decode params |  
|-------|-----------|--------------|---------------|  
| ry\_entangle | setup\_rz | `[1.5708, -1.0619, 2.6327, -2.8815, 3.095, 3.0212]` | `[-1.5708, -2.9913, -3.1416, -3.1416]` |  
| rxrz\_entangle| setup\_deep| `[1.5708, -3.1416, 2.002, -1.093, -0.0, 2.9651]` | `[-1.5708, -3.1141, -2.8841, -3.0377]` |  
| ry\_depth | setup\_rx | `[1.5716, -1.4465, 3.1882, -2.9306, 3.124, 3.0212]` | `[-1.5708, -3.0377, -2.9637, -2.3607]` |

_amp20 regime (validation ≈ 0.23, accuracy ≈ 0.615):_  
| chain | best stage | setup params | decode params |  
|-------|-----------|--------------|---------------|  
| ry\_entangle | setup\_rz | `[1.593, 2.5093, 2.197, 0.2503, 3.2527, 3.0181]` | `[1.571, 1.4282, -3.1518, -3.1409]` |  
| rxrz\_entangle| setup\_deep| `[1.5825, 2.463, 2.9288, -3.2358, 3.1235, 3.0239]` | `[1.5708, 1.0114, -1.2584, -0.4858]` |  
| ry\_depth | decode\_rz | `[1.5768, 2.5934, 3.255, 0.2605]` | `[1.5708, 1.442, -2.9637, -2.3607]` |

## 3\. Multi-photon number detector (0 / 1 / 2 photons)

Folder: `3_multiphoton_detector/` — sources: `make_protocol_3meas.py`,  
`make_protocol_spacing_sweep.py`, `make_protocol_spacing_sweep_lownoise.py`

**Architecture.** One trained model, `multi-photon_data_TRAINED_MODEL.npz` (= the original  
`multi-photon_data.npz`): **transient** itinerant single-photon released into a cavity via a Gaussian  
input-output pulse, read by **2 dispersive qubits** (chi = \[7.5, 3.75\], kappa 15, sigma 1, weak  
ZZ = 0.001). Three configs: **0 / 1 / 2 photons**. Trainable circuits:  
setup = `RY⊗2 + CNOT(circular) + RY⊗2 + RZ⊗2`, decode = `RY⊗2 + CNOT(circular) + RY⊗2`.  
Trained model: **validation 0.8730** (best epoch 696).

Optimized circuit values (same across every downstream analysis):

*   setup: `[3.1339, 1.5444, -1.6467, 3.2026, 2.8853, 2.6145]`
*   decode: `[-0.0501, -1.5702, 1.5739, 3.1379]`
*   states→label map: `0-photons→['01']`, `1-photons→['10','11']`, `2-photons→['00']`.

The two sub-experiments **re-derive the deployable confusion matrix** of this same model under  
different measurement protocols (they do not retrain):

**3a. Measurement-spacing sweep** (`spacing_sweep/`, the definitive fine-grid low-noise run).  
Sweeps measurement spacing over a fixed scoring window `[-4, 4]` at low hardware noise (1e-4,  
≈ T1/T2 ~ 10⁴ σ). Accuracy rises monotonically with spacing; **best spacing = 5.0 → accuracy**  
**0.8231**. Confusion diagonal there ≈ `[0.841, 0.922, 0.706]` for 0/1/2 photons.  
Files: `..._BEST.npz` (callback + confusion at spacing 5), `..._BEST_confusion.pdf`,  
`..._accuracy.pdf` (accuracy vs spacing), `..._summary.npz` (full 21-point curve, 3.0→5.0).

**3b. Three-measurement protocol** (`3meas_protocol/`). Same model re-scored under a 3-measurement  
protocol (t = -2, 4, 10; spacing 6, collective timing offset), batch 16. Accuracy diagonal  
`[0.720, 0.843, 0.752]`. Files: `multi-photon_data_3meas_b16.npz` + `_confusion.pdf`.

## 4\. Noise sweeps (single-photon RY detector)

Folder: `4_noise_sweeps/` — sources: `noise sweep/noise_sweep_common.py`, `noise_sweep_1qb.py`,  
`noise_sweep_2qb.py`, `noise_sweep_2qb_entangled.py`, `plot_noise_sweeps_combined.py`

**Architecture.** **Transient** single-photon detector (tutorial values): 1 cavity (2 levels) + 1  
field + N qubits dispersively coupled (chi = \[7.5, 4.5\] = \[0.5, 0.3\]·K, K=15). Two configs:  
`no interaction` (0 photons) vs `with interaction` (1 photon via Gaussian input-output pulse).  
Optimized (SGD lr 0.1, 1500 steps, batch 64) at 8 noise levels `[0, 0.005…0.2]` (equal per-channel  
depolarizing = dephasing = relaxation). Three detector variants:

*   **1qb**: single RY setup + RY decode.
*   **2qb**: RY⊗2 setup + RY⊗2 decode.
*   **2qb\_entangled**: RY → CNOT(circular) → RY (setup and decode), warm-started from the plain 2qb.

**Best result.** Accuracy vs noise (mean confusion diagonal):

| noise | 1qb | 2qb | 2qb\_entangled |
| --- | --- | --- | --- |
| 0.000 | **0.998** | 0.988 | **0.999** |
| 0.005 | 0.912 | 0.896 | 0.896 |
| 0.009 | 0.852 | 0.838 | 0.838 |
| 0.017 | 0.765 | 0.755 | 0.710 |
| 0.032 | 0.662 | 0.579 | 0.648 |
| 0.059 | 0.571 | 0.554 | 0.537 |
| 0.108 | 0.500 | 0.524 | 0.514 |
| 0.200 | 0.500 | 0.506 | 0.503 |

All three degrade to chance (~0.5) by noise ≈ 0.1; entanglement gives no robustness advantage.  
Curves in `noise_sweep_*_accuracy.pdf` and the overlay `noise_sweep_combined_accuracy.pdf`; full  
per-level accuracies/validations in `noise_sweep_*.npz`; the clean (noise 0) optimal circuits in  
`noise_sweep_*_CLEAN_level0.npz`.

Clean-detector optimized circuit values (setup / decode):

*   1qb: `[1.5708]` / `[-1.5708]`
*   2qb: `[1.5745, 1.3422]` / `[-1.5717, -1.3421]`
*   2qb\_entangled: `[0.344, 0.2758, 1.4612, 1.3075]` / `[-0.2174, -0.8197, -0.8754, -1.219]`

## Not carried over (safe to delete, no unique scientific result)

*   `diagnose_sweep_timeout.py` + `results/diagnose_sweep_timeout_interval*.txt` — a debugging probe  
    for a diffrax solver timeout; each `.txt` just records "time\_interval=X: OK in Ns".
*   `*_FAILED.txt` — SLURM failure tracebacks.
*   The coarse spacing sweeps `multi-photon_data_spacing_sweep_b32*` (5-point) — superseded by the  
    21-point fine-grid low-noise sweep kept in `3_multiphoton_detector/spacing_sweep/`.
*   All per-noise-level (`noise_sweep_*_level1..7.npz`), per-run (`ensemble_*_run1/2/3.npz` for  
    non-best stages), and per-spacing (`..._spacing3p1.npz` …) intermediates — the kept summaries,  
    best-stage callbacks and plots capture their outcome.
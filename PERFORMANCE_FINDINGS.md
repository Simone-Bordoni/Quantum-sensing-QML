# Optimization Performance Findings (single-qubit `optimize_rotations`)

_Date: 2026-06-18. Investigator: Claude (Opus 4.8). All numbers are wall-clock on this
machine (Tesla V100-SXM2-32GB, 80 CPU cores, 503 GB RAM)._

## TL;DR

1. **The GPU "doesn't help" only because the single-qubit problem is tiny.** At 1 qubit
   (Hilbert dim 8) CPU≈GPU. The GPU advantage explodes with system size:
   **~1.4× (n=1) → ~10× (n=2) → ~53× (n=3)**. Nothing is wrong with the GPU path; the
   single-qubit tutorial is simply the worst case for it.
2. **`testing` is faster than `main`/`modular` mostly because of a BUG**, not a real
   optimization. An exhausted-`zip` in `testing`'s `simulation` zeroes the state after the
   first measurement interval, so it silently **skips ~half the ODE solves** (and drops
   measurement intervals 3 & 4 from the physics). Fixing it makes `testing` ≈ `main`.
3. After removing that bug, **`modular` is slower than `main`/fixed-`testing`** by a factor
   that **grows with system size** (≈1.2× at n=1, ≈1.5× at n=2). See the UPDATE section for
   the decomposition: at n=1 it was the expensive time-dependent *reference* config (fixed by
   the faster tutorial setup); the residual at n≥2 is the **time-dependent pulse-driven
   config's ODE solve itself**. Matching "buggy `testing`" is not possible without dropping
   measurement intervals.
4. **Dicts are NOT the slowdown** (tested directly): modular's dict-based detection metric is
   186 µs/step (~0.1% of a step), and dict-vs-list containers compile to the same runtime —
   JAX bakes dicts into a static graph at trace time. See UPDATE section.
5. **The faster tutorial setup helps CPU, not GPU** (GPU is latency-bound at small n).

Branch lineage: `main` → `testing` → `modular` (modular is newest, will supersede).

## Environment

- `.venv` = CPU JAX (jax 0.9.2, cpu). `.venv_gpu` = GPU JAX (jax-cuda12, CudaDevice).
- Both venvs editable-install `qsopt` from `src/`. Main/testing benchmarked via
  `git worktree` + `PYTHONPATH` override (working dir stayed on `modular-hamiltonian`).
- Stack: JAX + QuTiP (`qt.MESolver`, `method="diffrax"`), optax SGD. Open-system Lindblad
  master equation, so the solver works with the **Liouvillian superoperator** (dim²×dim²).

## Methodology

- Same physics across branches (1 cavity + 1 field + N qubits, all 2 levels; dispersive
  cavity-qubit; input-output cavity-field; single-photon vs vacuum configs; depol/deph/relax
  noise = 0.01). `optimize_rotations(batch_size=1, verbose=False)`.
- Per-step time = steady-state median from a timing callback (excludes one-time JIT compile
  and the final `run_simulation`). Single-qubit table also reports compile+final overhead
  from a linear fit of total time vs num_steps.
- Scripts in `/tmp`: `bench_main.py`, `bench_modular.py`, `qubit_scaling.py` (modular API),
  `qubit_scaling_main.py` (main/testing API), `diag_testing.py` / `diag_modular.py` (trace
  diagnostics).

## Single-qubit `optimize_rotations` (the tutorial case)

| Branch  | CPU /step | GPU /step | CPU compile+final | GPU compile+final | GPU/CPU per-step |
|---------|-----------|-----------|-------------------|-------------------|------------------|
| modular | 220 ms    | 152 ms    | 14.3 s            | 15.9 s            | 0.69×            |
| main    | 188 ms    | 177 ms    | 8.9 s             | 13.5 s            | 0.94×            |
| testing | 116 ms    | 154 ms    | 16.5 s            | 25.1 s            | 1.33× (GPU slower)|

100-step run ≈ compile + 100×per-step + final-sim. Reproduces the user's report (modular
~36 s CPU / ~31 s GPU for 100 steps). Note GPU barely helps — or hurts — at n=1.

## Qubit-scaling, per-step (batch_size=1) — the representative axis

Hilbert dim = 4·2^n. Cost is dominated by the dim²×dim² Liouvillian, so it grows ~dim⁴.

| n_qubits | dim | modular CPU | modular GPU | main CPU | main GPU | testing(buggy) CPU | testing(buggy) GPU |
|----------|-----|-------------|-------------|----------|----------|--------------------|--------------------|
| 1        | 8   | 216 ms      | 153 ms      | 190 ms   | 178 ms   | 111 ms             | 128 ms             |
| 2        | 16  | 3031 ms     | 290 ms      | 1878 ms  | 300 ms   | 993 ms             | 178 ms             |
| 3        | 32  | 85123 ms    | 1598 ms     | (pending)| 1569 ms  | (pending)          | 756 ms             |
| 4        | 64  | —           | OOM         | —        | —        | —                  | —                  |

**GPU speedup (modular):** n=1 → 1.4×, n=2 → 10.5×, n=3 → **53×**. The GPU absolutely pays
off — just not at 1 qubit. (n=4 OOMs: the Liouvillian superoperator at dim 128 is a
16384×16384 dense matrix ≈ 4 GiB, and reverse-mode AD needs several copies.)

Scaling *cavity/field levels* instead is a poor axis: it inflates the Liouvillian (OOM at
dim≈50) without adding the trainable-parameter/circuit work we actually optimize over.

## Root cause of `testing`'s speed: an exhausted-`zip` bug

In `testing` `src/qsopt/core/experiment/experiment.py` `simulation()`:

```python
zipped_reset = zip(self.operators['measure_reset'], self.operators['measure_reset_dag'])  # created ONCE, before the loop
for t0, t1 in zip(measurements[:-1], measurements[1:]):
    ...
    rho_reset = [op * rho_final * op_dag for op, op_dag in zipped_reset]  # consumes the zip on interval 0
    rho_current = sum(rho_reset)                                          # interval >=1: sum([]) == 0  -> state zeroed
```

A `zip` is a **one-shot iterator**: it is fully consumed on the first interval, so from the
2nd interval on `rho_reset == []`, `rho_current == 0`, and every later `solver.run` integrates
the **zero state** (trivial/instant). Verified directly via `diag_testing.py`:

```
testing  trace(rho_final) per interval: [1.0, 1.0, 0.0, 0.0]   <- intervals 2,3 zeroed
modular  trace(rho_final) per interval: [1.0, 1.0, 1.0, 1.0]   <- all correct
```

So with the tutorial's 5 measurement times (4 intervals), `testing` only really simulates ~2
of them → roughly half the ODE work → ~2× faster, but **physically wrong** (drops the last
measurement intervals from the detection).

- **Not present in `main`:** `main` builds the `zip` *inside* a per-interval jitted function
  `f_measure_reset(rho)` (`experiment.py:261`), so each interval gets a fresh, non-exhausted
  zip. `main` is correct.
- **Fixed in `modular`:** uses a pre-built list `measure_reset_pairs`, reused every interval.

### Controlled test — fix `testing`'s zip (`zip(...)` → `list(zip(...))`)

| | buggy testing | **fixed** testing | main | modular |
|---|---|---|---|---|
| CPU n=1 | 111 ms | 173 ms | 190 ms | 216 ms |
| CPU n=2 | 993 ms | **1803 ms** | 1878 ms | 3031 ms |

Fixed `testing` ≈ `main`: **the entire `testing`-vs-`main` advantage was the bug.** Modular
remains ~1.7× above fixed-`testing`/`main` (3031 vs ~1800 at n=2) — the legitimate overhead.

## Why GPU is underutilized at n=1 (mechanism)

Single qubit ⇒ 8×8 density matrices, vectorized state of 64 complex numbers. The ODE
integration is inherently **sequential** (each diffrax step depends on the previous), and at
`batch_size=1` there is **no batch axis** to parallelize. So the V100 is latency/kernel-launch
bound and mostly idle; CPU matches or beats it. Evidence: CPU per-step scales ~14× from
n=1→n=2 (compute-bound), while GPU scales only ~1.9× (153→290 ms) — it had idle cores to
absorb the extra work. GPU pays off once per-step matrix work is large enough (≥2 qubits, or
large batch via `vmap` over measurement-uncertainty realizations).

## Remaining work / target

The legitimate `modular` overhead (~1.7× vs `main`/fixed-`testing`) is the safe optimization
target. Suspects under investigation (per-step / per-ODE-RHS drivers):
- `modular` `simulation` returns **full density-matrix lists**; the new **`max computational
  distance`** detection metric processes them at the end (vs `main`'s incremental scalar
  `prob` aggregation with the cheap `any excited` metric).
- Hamiltonian / collapse-operator structure built from modular `Interaction` objects
  (`QobjEvo([H_static] + H_time_dependent])` + per-config rebuild) — number/merging of terms
  is evaluated on **every** ODE RHS step, so a less-merged structure slows every solver step.
- Per-config `rho_dict = {config: simulation(...)}` dict handling + threaded `epoch_fraction`.

---

# UPDATE — faster tutorial setup, dict research, and where the residual gap lives

## CPU n=3 (one point obtained before the slow CPU runs were stopped)

| n_qubits | dim | modular CPU | modular GPU | GPU speedup |
|----------|-----|-------------|-------------|-------------|
| 3 | 32 | **85,123 ms** | 1,598 ms | **53×** |

Confirms the GPU advantage explodes with size; CPU at n=3 is impractical (~85 s/step).

## Faster tutorial setup (`faster_single_qubit_tutorial.ipynb`)

The faster tutorial moves the time-dependent INPUT_OUTPUT pulse out of the global
`PhysicalModel.interactions` and into **only** the `with interaction`
`SystemConfiguration` (per-config `interactions=[...]`). The global model keeps just the
constant DISPERSIVE term, so the **`no interaction` reference config becomes
constant-coefficient** (no per-step pulse evaluation, far fewer diffrax steps) — the same
cheap reference that `main`/`testing` already had.

Single-qubit `optimize_rotations`, modular, OLD vs FASTER setup:

| | per-step CPU | compile CPU | per-step GPU |
|---|---|---|---|
| modular OLD (pulse on both configs) | 220 ms | 14.3 s | 152 ms |
| modular FASTER (pulse on with-int only) | **179 ms** | **10.1 s** | 158 ms |
| (reference) main / fixed-testing CPU | 188 / 173 ms | 8.9 / — s | — |

- **CPU n=1: 220 → 179 ms, now on par with main/fixed-testing.** Compile 14.3 → 10.1 s.
- **GPU: unchanged (~155 ms).** At 1 qubit the V100 is latency-bound, so making the
  reference constant saves nothing there.

Qubit-scaling with the faster setup (per-step):

| n | modular OLD CPU | modular FASTER CPU | fixed-testing CPU | modular FASTER GPU |
|---|---|---|---|---|
| 1 | 216 ms | 190 ms | 173 ms | 153 ms |
| 2 | 3031 ms | 2702 ms | 1803 ms | 285 ms |

The faster setup helps at n=1 but only ~11% at n=2 — **a residual gap remains and grows
with n.**

## Where the residual gap lives (per-config forward ODE timing)

Forward (no-gradient) solve time per config, faster modular:

| config | n=1 | n=2 |
|---|---|---|
| `no interaction` (constant) | 7.4 ms | 13 ms |
| `with interaction` (time-dependent pulse) | 49 ms | **1126 ms** |

The whole step cost is the **single time-dependent pulse-driven solve** (reference is
already cheap). At n=2 that one solve (1126 ms forward → ~2700 ms with gradient) is ~1.5×
`testing`'s equivalent. So the residual modular overhead is **in the time-dependent ODE
integration itself**, not in dicts, the detection metric, or the reference config.

Likely sub-causes to profile next (not yet isolated): number of diffrax adaptive steps for
modular's pulse formulation; the per-RHS cost of modular's jitted `_wrap_time_modulation`
coefficient (it remaps args through the large global `global_args` dict every evaluation);
operator-term count in the time-dependent `QobjEvo`.

## Dicts are NOT the slowdown (directly measured)

- modular's real jitted detection metric (dict-based, `max computational distance`), timed
  in isolation: **186 µs/call** — ~0.1% of a 150–220 ms gradient step.
- Identical detection-style math, dict vs list containers: **runtime 87 µs vs 82 µs**
  (within noise); dict adds only ~50 ms one-time compile. JAX flattens the dict pytree into
  a static graph at trace time, so there is **no per-step dict overhead**.

Conclusion: the lists→dicts change is not a meaningful cost. Optimisation effort should go
into the time-dependent ODE solve (e.g. diffrax step control / solver tolerances / a leaner
time-modulation coefficient), and into using the GPU for the multi-qubit regime where it is
10–53× faster.

## How to reproduce

Scripts in `/tmp`: `bench_modular.py` (old), `bench_modular_faster.py` (faster setup),
`bench_main.py` (main/testing via `PYTHONPATH` to the worktree), `qubit_scaling*.py`,
`diag_testing.py`/`diag_modular.py` (zip-bug trace check), `dict_probe.py` (dict research),
`percfg_modular*.py` (per-config ODE timing). Worktrees: `/tmp/qsopt-main`,
`/tmp/qsopt-testing` (the latter has the `zip`→`list(zip)` fix applied for the controlled test).

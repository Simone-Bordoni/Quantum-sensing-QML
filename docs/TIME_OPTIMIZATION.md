# Measurement Time Optimization Guide

This document outlines the workflow for optimizing measurement intervals in the `qsopt` package. The utilities span landscape analysis, adaptive optimization, and visualization support.

## Overview

Quantum sensing performance depends critically on when measurements are performed during the system evolution. The measurement time optimization toolkit provides:

- Deterministic landscape generation across discretized time-interval grids.
- Adaptive refinement of the interval via simulated experiments.
- Visualization utilities to interpret contrast trends, detection probabilities, and measurement counts.

## Workflow Summary

1. **Prepare experimental parameters** with `MeasurementProtocol` configured in interval mode (initial time, final time, interval guess).
2. **Evaluate the landscape** using `compute_time_interval_landscape()` to understand contrast sensitivity versus interval choices.
3. **Visualize the results** with `plot_time_interval_landscape()` and optionally overlay measurement counts.
4. **Run adaptive optimization** using `optimize_measurement_times()` from `SingleQubitExperiment` to refine the interval automatically.
5. **Inspect pulse alignment** with `plot_pulse_shape_with_measurements()` to confirm measurement timing relative to the drive envelope.

## Key Functions

### compute_time_interval_landscape

```python
from qsopt.utils.landscape_analysis import compute_time_interval_landscape
```

Generate sensing contrast, detection probabilities, and measurement-count arrays for a grid of time intervals. Supports continuous and discrete modes, batch averaging for timing uncertainty, and custom interval bounds.

**Important parameters:**
- `theta1`, `theta2`: Fixed rotation angles used during the sweep.
- `resolution`: Number of interval samples (minimum 2).
- `mode`: `'continuous'` (default) or `'discrete'`.
- `batch_size`: Number of uncertainty realizations; set >1 to incorporate `initial_time_uncertainty`.

### optimize_measurement_times

```python
results = experiment.optimize_measurement_times(
    resolution=60,
    mode="continuous",
    batch_size=15,
    min_interval=0.05,
    max_interval=1.5,
    verbose=True,
)
```

Performs a sweep over candidate measurement intervals while re-running the sensing experiment. The returned dictionary includes all arrays from `compute_time_interval_landscape` and adds:
- `results['best_interval']`: Interval that achieved the highest contrast.
- `results['best_contrast']`: Maximum contrast value located in the sweep.
- `results['best_index']`: Index of the best-performing interval.

### plot_time_interval_landscape

```python
from qsopt.utils.visualization import plot_time_interval_landscape
fig = plot_time_interval_landscape(
    landscape,
    exp_params,
    show_measurement_count=True,
    save_path="results/time_interval_landscape.png"
)
```

Creates a publication-ready figure with:
- Contrast versus interval (with optimal marker).
- Detection probabilities with and without photon.
- Optional measurement-count subplot.
- Detailed system summary box, including batch-uncertainty metadata and interval statistics.

### plot_pulse_shape_with_measurements

```python
from qsopt.utils.visualization import plot_pulse_shape_with_measurements
fig = plot_pulse_shape_with_measurements(exp_params)
```

Displays the Gaussian pulse envelope together with measurement markers, derived from the current `MeasurementProtocol`. Useful for confirming measurement placements relative to control pulses and uncertainty spreads.

## Example Script

```python
import numpy as np
from qsopt import SingleQubitExperiment
from qsopt.utils.landscape_analysis import compute_time_interval_landscape
from qsopt.utils.visualization import (
    plot_time_interval_landscape,
    plot_pulse_shape_with_measurements,
)

# Assume `experiment` and `exp_params` are preconfigured
landscape = compute_time_interval_landscape(
    exp_params,
    theta1=np.pi / 2,
    theta2=-np.pi / 2,
    resolution=50,
    mode="continuous",
    batch_size=10,
    verbose=False,
)

# Visual diagnostics
plot_time_interval_landscape(landscape, exp_params, show_measurement_count=True)
plot_pulse_shape_with_measurements(exp_params)

# Adaptive optimization
mt_results = experiment.optimize_measurement_times(
    resolution=80,
    mode="continuous",
    batch_size=20,
    min_interval=0.05,
    max_interval=1.2,
    verbose=True,
)

print(
    "Optimal interval: "
    f"{mt_results['best_interval']:.6f}"
    f" (contrast={mt_results['best_contrast']:.6f})"
)
```

## Best Practices

- Ensure `MeasurementProtocol.time_interval` is a sensible starting point; the optimizer refines rather than fully re-initializes the schedule.
- When using batch averaging, increase `batch_size` until contrast statistics stabilize.
- Compare landscape-derived optima with the adaptive optimizer output for validation.
- Persist figures and optimization callbacks to track progress across experiments.

## Related Documentation

- [Visualization Module](./VISUALIZATION_MODULE.md)
- [Experimental Parameters Guide](./experimental_parameters.md)
- [Index](./INDEX.md)

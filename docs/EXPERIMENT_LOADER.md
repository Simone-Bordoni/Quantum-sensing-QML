# Experiment Loader Documentation

## Overview

The `experiment_loader.py` module provides functionality to load and reconstruct quantum sensing experiments from saved JSON report files. This enables reproducibility, experiment sharing, and resuming previous work.

## Location

- **Module**: `src/qsopt/utils/experiment_loader.py`
- **Import**: `from qsopt.utils.experiment_loader import load_experiment_from_report`

## Main Function

### `load_experiment_from_report()`

Loads and reconstructs experiment configuration from a JSON report file.

**Function Signature:**
```python
def load_experiment_from_report(
    json_path: str
) -> Tuple[ExperimentalParameters, TrainableParameters, Dict[str, Any]]
```

**Arguments:**
- `json_path` (str): Path to the JSON report file (created by `experiment.save_experiment_report()`)

**Returns:**
Tuple containing three elements:
1. `ExperimentalParameters`: Reconstructed experimental parameters
2. `TrainableParameters`: Reconstructed trainable parameters
3. `Dict[str, Any]`: Metadata dictionary containing:
   - `'experiment_type'`: Type of experiment (e.g., 'SingleQubitExperiment')
   - `'version'`: Report format version
   - `'callback_info'`: Summary of optimization run (if available)
   - `'callback_data'`: Full optimization history from NPZ file (if available)

**Example 1: Basic Loading**
```python
from qsopt.utils.experiment_loader import load_experiment_from_report
from qsopt.core.experiment import SingleQubitExperiment

# Load saved experiment configuration
exp_params, train_params, metadata = load_experiment_from_report('results/my_experiment.json')

# Recreate experiment
experiment = SingleQubitExperiment(exp_params, train_params)

# Access metadata
print(f"Experiment type: {metadata['experiment_type']}")
print(f"Report version: {metadata['version']}")

# Run with loaded configuration
results = experiment.run_simulation()
print(f"Contrast with loaded parameters: {results.best_contrast:.6f}")
```

**Example 2: Loading Optimization Data**
```python
# Load experiment that was previously optimized
exp_params, train_params, metadata = load_experiment_from_report('results/optimized.json')

# Check if optimization data is available
if 'callback_info' in metadata and metadata['callback_info'] is not None:
    callback_info = metadata['callback_info']
    
    print(f"Optimization mode: {callback_info['mode']}")
    print(f"Total epochs: {callback_info['total_epochs']}")
    print(f"Converged: {callback_info['converged']}")
    
    if 'best_metrics' in callback_info:
        best = callback_info['best_metrics']
        print(f"\nBest Results:")
        print(f"  Epoch: {best['epoch']}")
        print(f"  Contrast: {best['contrast']:.6f}")
        print(f"  P(detect|with): {best['prob_with']:.6f}")
        print(f"  P(detect|without): {best['prob_without']:.6f}")
    
    if 'best_parameters' in callback_info:
        best_params = callback_info['best_parameters']
        for param_name, param_info in best_params.items():
            print(f"  {param_name}: {param_info['value_rad']:.6f} rad ({param_info['value_deg']:.2f}°)")

# Access full optimization history if available
if 'callback_data' in metadata:
    callback_data = metadata['callback_data']
    
    # Plot optimization history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(callback_data['epochs'], callback_data['contrast'])
    plt.xlabel('Epoch')
    plt.ylabel('Contrast')
    plt.title('Optimization Progress')
    
    plt.subplot(132)
    plt.plot(callback_data['epochs'], callback_data['prob_with'], label='With photon')
    plt.plot(callback_data['epochs'], callback_data['prob_without'], label='Without photon')
    plt.xlabel('Epoch')
    plt.ylabel('Detection Probability')
    plt.legend()
    
    plt.subplot(133)
    plt.semilogy(callback_data['epochs'], callback_data['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    
    plt.tight_layout()
    plt.savefig('loaded_optimization_history.png')
```

**Example 3: Resuming Optimization**
```python
# Load previous experiment
exp_params, train_params, metadata = load_experiment_from_report('results/partial_opt.json')

# Get best parameters from previous run
if 'callback_info' in metadata and 'best_parameters' in metadata['callback_info']:
    best_params = metadata['callback_info']['best_parameters']
    
    # Extract angles for new optimization
    theta1 = best_params['ry1']['value_rad']
    theta2 = best_params['ry2']['value_rad']
    
    # Continue optimization from best point
    experiment = SingleQubitExperiment(exp_params, train_params)
    history = experiment.optimize(
        theta_init=[theta1, theta2],
        num_steps=100,  # Continue for more steps
        verbose=True
    )
    
    print(f"Previous best contrast: {metadata['callback_info']['best_metrics']['contrast']:.6f}")
    print(f"New best contrast: {history.best_contrast:.6f}")
    print(f"Improvement: {history.best_contrast - metadata['callback_info']['best_metrics']['contrast']:.6f}")
```

---

## Integration with Save Functionality

### Saving Experiments

Use `experiment.save_experiment_report()` to create files that can be loaded:

```python
from qsopt import *

# Create and run experiment
experiment = SingleQubitExperiment(exp_params, train_params)
history = experiment.optimize(theta_init=[1.5, -1.3], num_steps=100)

# Save complete configuration and results
experiment.save_experiment_report('results/my_experiment.json')
```

This creates two files:
1. **JSON file** (`my_experiment.json`): All experimental parameters, configuration, metadata
2. **NPZ file** (`my_experiment_callback.npz`): Detailed optimization history arrays

### Complete Save/Load Workflow

```python
from qsopt import *
from qsopt.utils.experiment_loader import load_experiment_from_report

# === SESSION 1: Initial Experiment ===
print("Running initial experiment...")

# Setup
exp_params = ExperimentalParameters(...)
train_params = TrainableParameters()
train_params.add_rotation_angles(['theta1', 'theta2'], [1.5, -1.3])

# Run optimization
experiment = SingleQubitExperiment(exp_params, train_params)
history = experiment.optimize(theta_init=[1.5, -1.3], num_steps=50)

# Save everything
experiment.save_experiment_report('results/experiment_v1.json')
print(f"Saved with contrast: {history.best_contrast:.6f}")

# === SESSION 2: Later Analysis (Different Python Session) ===
print("\nLoading previous experiment...")

# Load saved configuration
exp_params_loaded, train_params_loaded, metadata = load_experiment_from_report('results/experiment_v1.json')

# Verify parameters match
print("Loaded configuration:")
print(f"  chi: {exp_params_loaded.chi}")
print(f"  cavity_levels: {exp_params_loaded.cavity_levels}")
print(f"  measurement times: {len(exp_params_loaded.measurement_times)}")

# Access optimization results
if 'callback_data' in metadata:
    print(f"\nOptimization completed:")
    print(f"  Epochs: {len(metadata['callback_data']['epochs'])}")
    print(f"  Final contrast: {metadata['callback_data']['contrast'][-1]:.6f}")

# Re-run or continue optimization
experiment_new = SingleQubitExperiment(exp_params_loaded, train_params_loaded)
results = experiment_new.run_simulation()
print(f"Re-simulation contrast: {results.best_contrast:.6f}")
```

---

## Report File Structure

### JSON Report Contents

The JSON file contains a structured representation of all experiment details:

```json
{
  "experiment_type": "SingleQubitExperiment",
  "version": "0.1.0",
  
  "experimental_parameters": {
    "physical_constants": {
      "chi": 0.01,
      "photon_cavity_coupling": 0.1,
      "inverse_pulse_width": 0.1
    },
    "system_dimensions": {
      "cavity_levels": 2,
      "qubit_levels": 2,
      "field_levels": 2
    },
    "measurement_protocol": {
      "mode": "interval",
      "initial_time": -5.0,
      "final_time": 5.0,
      "time_interval": 1.0,
      "initial_time_uncertainty": 0.0,
      "computed_times": [-5.0, -4.0, -3.0, ...],
      "num_measurements": 11
    },
    "initial_state": {
      "state_type": "single_photon",
      "coherent_alpha": null,
      "thermal_n_bar": null
    },
    "noise_configuration": {
      "depolarizing": 0.0001,
      "dephasing": 0.0001,
      "relaxation": 0.0001
    }
  },
  
  "trainable_parameters": {
    "parameters": [
      {
        "name": "theta1",
        "type": "rotation_angle",
        "value": 1.489788,
        "trainable": true
      },
      {
        "name": "theta2",
        "type": "rotation_angle",
        "value": -1.339683,
        "trainable": true
      }
    ],
    "num_parameters": 2,
    "num_trainable": 2
  },
  
  "callback_info": {
    "mode": "optimization",
    "total_epochs": 100,
    "converged": true,
    "final_gradient_norm": 0.000012,
    "best_metrics": {
      "epoch": 98,
      "contrast": 0.244651,
      "prob_with": 0.750262,
      "prob_without": 0.505612
    },
    "best_parameters": {
      "ry1": {
        "value_rad": 1.489788,
        "value_deg": 85.358571
      },
      "ry2": {
        "value_rad": -1.339683,
        "value_deg": -76.758155
      }
    },
    "callback_data_path": "results/experiment_v1_callback.npz",
    "optimization_summary": {
      "initial_contrast": 0.234500,
      "final_contrast": 0.244651,
      "best_contrast": 0.244651,
      "improvement": 0.010151
    }
  }
}
```

### NPZ Callback Data

The NPZ file contains arrays with full optimization history:

```python
import numpy as np

# Load NPZ file directly
data = np.load('results/experiment_v1_callback.npz')

# Available arrays:
# - epochs: Epoch numbers
# - contrast: Contrast at each epoch
# - prob_with: P(detect|with photon) at each epoch
# - prob_without: P(detect|without photon) at each epoch
# - loss: Loss function value at each epoch
# - parameters: Parameter values at each epoch (shape: epochs × n_params)
# - gradient_norms: Gradient magnitude at each epoch
# - best_epoch: Epoch with best contrast
# - best_contrast: Best contrast achieved
# - best_parameters: Parameter values at best epoch
# - best_prob_with: P(detect|with) at best epoch
# - best_prob_without: P(detect|without) at best epoch
```

---

## Use Cases

### 1. Reproducible Research
```python
# Save experiment for paper
experiment.save_experiment_report('paper_data/figure3_experiment.json')

# Later, reviewers can load and verify
exp_params, train_params, metadata = load_experiment_from_report('paper_data/figure3_experiment.json')
experiment = SingleQubitExperiment(exp_params, train_params)
results = experiment.run_simulation()
# Results should match published values
```

### 2. Parameter Transfer
```python
# Load parameters optimized for one condition
exp_params_base, train_params_base, _ = load_experiment_from_report('results/low_noise.json')

# Modify for new condition (higher noise)
exp_params_noisy = exp_params_base
exp_params_noisy.noise_config.relaxation = 0.01  # Higher noise

# Use optimized parameters as starting point
experiment_noisy = SingleQubitExperiment(exp_params_noisy, train_params_base)
history = experiment_noisy.optimize(
    theta_init=train_params_base.get_parameter_vector(),
    num_steps=50
)
```

### 3. Batch Analysis
```python
import os
from pathlib import Path

# Load and analyze multiple experiments
experiment_dir = Path('results/parameter_sweep')
results_summary = []

for json_file in experiment_dir.glob('*.json'):
    exp_params, train_params, metadata = load_experiment_from_report(str(json_file))
    
    if 'callback_info' in metadata and metadata['callback_info'] is not None:
        best_contrast = metadata['callback_info']['best_metrics']['contrast']
        chi_value = exp_params.chi
        
        results_summary.append({
            'file': json_file.name,
            'chi': chi_value,
            'best_contrast': best_contrast
        })

# Sort and display
results_summary.sort(key=lambda x: x['best_contrast'], reverse=True)
print("Best Experiments:")
for result in results_summary[:5]:
    print(f"  {result['file']}: chi={result['chi']:.4f}, contrast={result['best_contrast']:.6f}")
```

### 4. Configuration Templates
```python
# Create template configuration
template_params = ExperimentalParameters(
    physical_constants=PhysicalConstants(chi=0.01, ...),
    system_dims=SystemDimensions(cavity_levels=2, ...),
    ...
)
template_train = TrainableParameters()
template_train.add_rotation_angles(['theta1', 'theta2'], [np.pi/2, -np.pi/2])

# Save template
template_exp = SingleQubitExperiment(template_params, template_train)
template_exp.save_experiment_report('templates/default_config.json')

# Later, users can load template and modify
exp_params, train_params, _ = load_experiment_from_report('templates/default_config.json')
# Customize as needed
exp_params.chi = 0.02  # Different coupling
# Then run experiment
```

---

## Best Practices

### 1. Descriptive Filenames
```python
# Include key information in filename
filename = f'results/chi_{chi:.3f}_noise_{noise:.4f}_epoch{num_steps}.json'
experiment.save_experiment_report(filename)
```

### 2. Version Control
```python
# Include version or date in report path
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment.save_experiment_report(f'results/exp_{timestamp}.json')
```

### 3. Validate Loaded Parameters
```python
# Always verify loaded parameters are reasonable
exp_params, train_params, metadata = load_experiment_from_report(json_path)

assert exp_params.chi > 0, "Invalid chi value"
assert exp_params.cavity_levels >= 2, "Invalid cavity levels"
assert len(exp_params.measurement_times) >= 2, "Insufficient measurements"
```

### 4. Check File Existence
```python
from pathlib import Path

json_path = 'results/my_experiment.json'
if not Path(json_path).exists():
    print(f"Error: File not found: {json_path}")
else:
    exp_params, train_params, metadata = load_experiment_from_report(json_path)
```

---

## Troubleshooting

### Issue: FileNotFoundError when loading
**Solution:** Check file path and ensure JSON file exists
```python
from pathlib import Path
if Path(json_path).exists():
    data = load_experiment_from_report(json_path)
else:
    print(f"File not found: {json_path}")
```

### Issue: Callback NPZ file not found
**Solution:** NPZ file and JSON file should be in same directory
```python
# Ensure both files are present
json_path = Path('results/experiment.json')
npz_path = json_path.with_stem(json_path.stem + '_callback').with_suffix('.npz')

if not npz_path.exists():
    print(f"Warning: Callback data not found at {npz_path}")
    print("Optimization history will not be available")
```

### Issue: Invalid JSON format
**Solution:** Validate JSON file is not corrupted
```python
import json
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
    print("JSON file is valid")
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
```

### Issue: Parameter mismatch after loading
**Solution:** Verify the loaded parameters match expected values
```python
exp_params, train_params, metadata = load_experiment_from_report(json_path)

# Check critical parameters
print(f"Chi: {exp_params.chi}")
print(f"Cavity levels: {exp_params.cavity_levels}")
print(f"Number of parameters: {len(train_params.parameters)}")

# Compare with expected
assert abs(exp_params.chi - expected_chi) < 1e-6, "Chi mismatch"
```

---

## See Also

- [Experiment Class](experiment.md) - For `save_experiment_report()` method
- [Callbacks](callbacks.md) - For optimization history data
- [Experimental Parameters](experimental_parameters.md) - For parameter configuration
- [Trainable Parameters](trainable_parameters.md) - For trainable parameter setup

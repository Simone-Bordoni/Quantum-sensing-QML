"""
Demonstration of experiment report save/load functionality.
"""
import numpy as np
from pathlib import Path
from qsopt import (
    SingleQubitExperiment,
    ExperimentalParameters,
    PhysicalConstants,
    SystemDimensions,
    MeasurementProtocol,
    InitialStateConfig,
    InitialStateType,
    TrainableParameters
)


def main():
    """Demonstrate saving and loading experiment reports."""
    
    print("=" * 70)
    print("Quantum Sensing QML - Experiment Report Demo")
    print("=" * 70)
    
    # Create experiment
    print("\n1. Creating experiment...")
    gm = 0.03 * 2 * np.pi
    
    constants = PhysicalConstants(
        chi=0.5 * gm,
        photon_cavity_coupling=gm,
        inverse_pulse_width=0.1 * gm
    )
    
    dims = SystemDimensions(cavity_levels=2, qubit_levels=2, field_levels=2)
    
    measurement = MeasurementProtocol(
        measurement_times=list(np.array([-5.0, 0.0, 5.0]) / (0.1 * gm))
    )
    
    initial_state = InitialStateConfig(state_type=InitialStateType.SINGLE_PHOTON)
    
    exp_params = ExperimentalParameters(
        physical_constants=constants,
        system_dims=dims,
        measurement=measurement,
        initial_state=initial_state
    )
    
    params = TrainableParameters()
    params.add_rotation_angles(['ry1', 'ry2'], [np.pi/2, -np.pi/2])
    
    experiment = SingleQubitExperiment(exp_params, params)
    print("   ✓ Experiment created")
    
    # Run short optimization
    print("\n2. Running optimization (10 steps)...")
    history = experiment.optimize(
        theta_init=[1.5, -1.3],
        num_steps=10,
        verbose=False
    )
    print(f"   ✓ Optimization complete")
    print(f"   Initial contrast: {history.history['contrast'][0]:.6f}")
    print(f"   Final contrast: {history.history['contrast'][-1]:.6f}")
    
    # Save report
    print("\n3. Saving experiment report...")
    report_path = "demo_experiment_report.json"
    experiment.save_experiment_report(report_path)
    print(f"   ✓ Report saved to: {report_path}")
    
    # Check file size
    json_size = Path(report_path).stat().st_size / 1024  # KB
    npz_path = report_path.replace('.json', '_callback.npz')
    npz_size = Path(npz_path).stat().st_size / 1024  # KB
    print(f"   JSON size: {json_size:.2f} KB")
    print(f"   NPZ size: {npz_size:.2f} KB")
    
    # Load report
    print("\n4. Loading experiment report...")
    loaded = SingleQubitExperiment.load_experiment_report(report_path)
    print(f"   ✓ Report loaded successfully")
    print(f"   Experiment type: {loaded['experiment_type']}")
    print(f"   Contains {len(loaded['callback_data']['epochs'])} optimization steps")
    
    # Display some key parameters
    print("\n5. Key parameters from loaded report:")
    exp_config = loaded['experimental_params_dict']
    print(f"   • System dimensions: {exp_config['system_dimensions']}")
    print(f"   • Initial state: {exp_config['initial_state']['state_type']}")
    print(f"   • Measurement times: {len(exp_config['measurement_protocol']['measurement_times'])} points")
    
    trainable = loaded['trainable_params_dict']
    print(f"   • Rotation angles: {trainable['rotation_angles']}")
    
    # Cleanup demo files
    print("\n6. Cleaning up demo files...")
    Path(report_path).unlink()
    Path(npz_path).unlink()
    print("   ✓ Demo files removed")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

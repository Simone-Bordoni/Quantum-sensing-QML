from qsopt import *
import optax

# Create experiment
exp_params = ExperimentalParameters()
trainable_params = TrainableParameters()
trainable_params.add_rotation_angles(['ry1', 'ry2'], [1.0, 1.0], 
                                     optimizer=optax.adam(0.01))

experiment = SingleQubitExperiment(exp_params, trainable_params)

# Track optimization with callback
callback = OptimizationCallback(save_every=1, save_best=True)

# Run optimization (JAX autodiff works!)
history = experiment.optimize(num_steps=20, learning_rate=0.05, 
                             callback=callback, verbose=True)

# Save results
callback.save('results.npz')
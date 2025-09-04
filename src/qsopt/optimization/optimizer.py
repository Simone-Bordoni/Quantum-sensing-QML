"""
Quantum Sensing Parameter Optimization
=====================================

Advanced optimization algorithms for quantum sensing protocol parameter tuning
using JAX automatic differentiation and professional optimization libraries.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class OptimizerType(Enum):
    """Available optimization algorithms."""
    ADAM = "adam"
    ADAMW = "adamw" 
    RMSPROP = "rmsprop"
    SGD = "sgd"
    ADAMAX = "adamax"
    ADAGRAD = "adagrad"


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    optimizer_type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.1
    max_iterations: int = 100
    tolerance: float = 1e-6
    use_lr_schedule: bool = False
    decay_rate: float = 0.95
    decay_steps: Optional[int] = None
    beta1: float = 0.9  # Adam momentum parameter
    beta2: float = 0.999  # Adam second moment parameter
    epsilon: float = 1e-8  # Numerical stability
    
    def __post_init__(self):
        """Set default decay steps if not provided."""
        if self.decay_steps is None:
            self.decay_steps = self.max_iterations // 4


@dataclass 
class OptimizationResult:
    """Results from optimization run."""
    optimal_params: jnp.ndarray
    final_loss: float
    final_contrast: float
    iterations: int
    converged: bool
    history: Dict[str, List[float]]
    metadata: Dict[str, Any]


class Optimizer:
    """
    Professional optimization engine for quantum sensing parameter tuning.
    
    Supports state-of-the-art optimization algorithms with automatic differentiation,
    learning rate scheduling, and comprehensive progress tracking.
    
    Features:
    - Multiple optimization algorithms (Adam, RMSprop, SGD, etc.)
    - Automatic learning rate scheduling
    - Gradient norm convergence checking
    - Comprehensive optimization history tracking
    - JAX-compatible automatic differentiation
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """
        Initialize optimizer with configuration.
        
        Args:
            config: OptimizationConfig object with algorithm settings
        """
        self.config = config or OptimizationConfig()
        self._optimizer = None
        self._lr_schedule = None
        self._setup_optimizer()
        
    def _setup_optimizer(self) -> None:
        """Configure the optimization algorithm and learning rate schedule."""
        # Setup learning rate schedule
        base_lr = self.config.learning_rate
        
        if self.config.use_lr_schedule:
            self._lr_schedule = optax.exponential_decay(
                base_lr, 
                self.config.decay_steps, 
                self.config.decay_rate
            )
        else:
            self._lr_schedule = base_lr
        
        # Configure optimizer algorithms
        optimizer_dict = {
            OptimizerType.ADAM: optax.adam(
                self._lr_schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=self.config.epsilon
            ),
            OptimizerType.ADAMW: optax.adamw(
                self._lr_schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=self.config.epsilon
            ),
            OptimizerType.RMSPROP: optax.rmsprop(
                self._lr_schedule,
                decay=0.9,
                eps=self.config.epsilon
            ),
            OptimizerType.SGD: optax.sgd(self._lr_schedule),
            OptimizerType.ADAMAX: optax.adamax(
                self._lr_schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=self.config.epsilon
            ),
            OptimizerType.ADAGRAD: optax.adagrad(self._lr_schedule)
        }
        
        self._optimizer = optimizer_dict[self.config.optimizer_type]
    
    def optimize_sensing_contrast(self, 
                                experiment,
                                initial_params: List[float],
                                measurement_times: Dict[float, int],
                                verbose: bool = True) -> OptimizationResult:
        """
        Optimize quantum sensing protocol to maximize detection contrast.
        
        Optimizes rotation angles to maximize the difference between detection
        probability with and without input photons.
        
        Args:
            experiment: Initialized Experiment object
            initial_params: Initial rotation angles [theta1, theta2]
            measurement_times: Dict mapping time points to expected outcomes
            verbose: Whether to print optimization progress
            
        Returns:
            OptimizationResult: Complete optimization results and history
        """
        # Initialize parameters as JAX array
        params = jnp.array(initial_params, dtype=float)
        
        def objective_function(theta_params: jnp.ndarray) -> float:
            """
            Objective function for contrast maximization.
            
            Returns negative sensing contrast for minimization.
            """
            theta1, theta2 = theta_params
            
            # Calculate detection probabilities with and without photon
            prob_with = self._simulate_detection(
                experiment, [theta1, theta2], measurement_times, with_photon=True
            )
            prob_without = self._simulate_detection(
                experiment, [theta1, theta2], measurement_times, with_photon=False
            )
            
            sensing_contrast = prob_with - prob_without
            return -sensing_contrast  # Negative for minimization
        
        # Initialize optimization state
        opt_state = self._optimizer.init(params)
        
        # History tracking
        history = {
            'loss': [],
            'sensing_contrast': [],
            'gradients': [],
            'params': [],
            'theta1': [],
            'theta2': [],
            'learning_rates': [],
            'prob_with_photon': [],
            'prob_without_photon': []
        }
        
        if verbose:
            print(f"Starting optimization with {self.config.optimizer_type.value.upper()}")
            print(f"Configuration:")
            print(f"  • Learning rate: {self.config.learning_rate} {'(scheduled)' if self.config.use_lr_schedule else '(fixed)'}")
            print(f"  • Max iterations: {self.config.max_iterations}")
            print(f"  • Tolerance: {self.config.tolerance:.2e}")
            print(f"  • Initial params: θ₁={params[0]:.3f}, θ₂={params[1]:.3f}")
            print("=" * 80)
            print("Step\tTheta1\t\tTheta2\t\tContrast\tLoss\t\tGrad Norm\tLR")
            print("-" * 80)
        
        best_contrast = -np.inf
        best_params = params.copy()
        converged = False
        
        for step in range(self.config.max_iterations):
            try:
                # Compute loss and gradients
                loss_value, grads = jax.value_and_grad(objective_function)(params)
                
                # Calculate detailed metrics
                theta1, theta2 = params
                prob_with = self._simulate_detection(
                    experiment, [theta1, theta2], measurement_times, with_photon=True
                )
                prob_without = self._simulate_detection(
                    experiment, [theta1, theta2], measurement_times, with_photon=False
                )
                sensing_contrast = prob_with - prob_without
                
                # Track best parameters
                if sensing_contrast > best_contrast:
                    best_contrast = sensing_contrast
                    best_params = params.copy()
                
                # Store history
                history['loss'].append(float(loss_value))
                history['sensing_contrast'].append(float(sensing_contrast))
                history['gradients'].append([float(grads[0]), float(grads[1])])
                history['params'].append([float(params[0]), float(params[1])])
                history['theta1'].append(float(params[0]))
                history['theta2'].append(float(params[1]))
                history['prob_with_photon'].append(float(prob_with))
                history['prob_without_photon'].append(float(prob_without))
                
                # Learning rate tracking
                current_lr = self.config.learning_rate
                if self.config.use_lr_schedule and hasattr(self._lr_schedule, '__call__'):
                    current_lr = self._lr_schedule(step)
                history['learning_rates'].append(float(current_lr))
                
                # Calculate gradient norm for convergence
                grad_norm = jnp.linalg.norm(grads)
                
                # Progress reporting
                if verbose and (step % 20 == 0 or grad_norm < self.config.tolerance):
                    print(f"{step:3d}\t{params[0]:.6f}\t{params[1]:.6f}\t"
                          f"{sensing_contrast:.6f}\t{loss_value:.6f}\t"
                          f"{grad_norm:.2e}\t{current_lr:.2e}")
                
                # Convergence check
                if grad_norm < self.config.tolerance:
                    if verbose:
                        print(f"\nConverged after {step+1} iterations!")
                        print(f"Final gradient norm: {grad_norm:.2e}")
                    converged = True
                    break
                
                # Parameter update
                updates, opt_state = self._optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                
            except Exception as e:
                if verbose:
                    print(f"Step {step} failed: {e}")
                # Continue with previous parameters
                break
        
        if not converged and verbose:
            print(f"\nReached maximum iterations without convergence")
            print(f"Final gradient norm: {grad_norm:.2e}")
        
        if verbose:
            print(f"Best sensing contrast achieved: {best_contrast:.6f}")
        
        # Return best parameters found
        final_params = best_params if best_contrast > sensing_contrast else params
        
        return OptimizationResult(
            optimal_params=final_params,
            final_loss=float(-best_contrast),
            final_contrast=float(best_contrast),
            iterations=len(history['loss']),
            converged=converged,
            history=history,
            metadata={
                'optimizer_type': self.config.optimizer_type.value,
                'learning_rate': self.config.learning_rate,
                'use_lr_schedule': self.config.use_lr_schedule,
                'tolerance': self.config.tolerance
            }
        )
    
    def _simulate_detection(self, experiment, rotation_angles: List[float], 
                          measurement_times: Dict[float, int], with_photon: bool) -> float:
        """
        Simulate quantum detection protocol.
        
        Args:
            experiment: Initialized Experiment object
            rotation_angles: [theta1, theta2] rotation parameters
            measurement_times: Measurement protocol specification
            with_photon: Whether to include input photon interaction
            
        Returns:
            Detection probability
        """
        # Use the experiment's run_sensing_protocol method with theta1_theta2 protocol
        return experiment.run_sensing_protocol(
            rotation_angles=rotation_angles,
            with_photon=with_photon,
            protocol_type="theta1_theta2"
        )


def create_optimizer(optimizer_type: str = "adam", 
                    learning_rate: float = 0.1,
                    max_iterations: int = 100,
                    **kwargs) -> Optimizer:
    """
    Factory function to create optimizers with common configurations.
    
    Args:
        optimizer_type: Type of optimizer ("adam", "rmsprop", "sgd", etc.)
        learning_rate: Learning rate for optimization
        max_iterations: Maximum number of optimization steps
        **kwargs: Additional optimizer configuration parameters
        
    Returns:
        Configured Optimizer instance
    """
    optimizer_enum = OptimizerType(optimizer_type.lower())
    
    config = OptimizationConfig(
        optimizer_type=optimizer_enum,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        **kwargs
    )
    
    return Optimizer(config)

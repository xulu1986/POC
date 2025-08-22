


"""
Advanced Auto-Tuning Algorithms for FM Model Hyperparameter Optimization

This module provides multiple advanced optimization algorithms for tuning FM model hyperparameters,
including Bayesian Optimization, Optuna, and other state-of-the-art techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime

# Optional imports for different optimization libraries
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
    from scipy.optimize import minimize
    from scipy.stats import norm
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Bayesian Optimization will be limited.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

try:
    import hyperopt
    from hyperopt import fmin, tpe, hp, Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("Warning: Hyperopt not available. Install with: pip install hyperopt")

logger = logging.getLogger(__name__)

# Historical performance data
perf = [
    # [batch_size, epoch, learning_rate, emb_dim, performance]
    [256, 2, 0.01, 16, 0.1916],
    [256, 10, 0.01, 16, 0.1995],
    [256, 20, 0.01, 16, 0.2058],
    [256, 10, 0.01, 96, 0.2057],
    [2000, 10, 0.01, 96, 0.2157],
    [5000, 10, 0.01, 96, 0.2167],
    [10000, 10, 0.01, 96, 0.2141],
    [10000, 20, 0.01, 96, 0.2204],
    [10000, 25, 0.01, 96, 0.2197],
    [10000, 20, 0.1, 96, 0.1985],
    [15000, 20, 0.01, 256, 0.2154],
    [15000, 30, 0.01, 96, 0.2179],
    [15000, 50, 0.01, 96, 0.2173],
    [15000, 50, 0.1, 96, 0.2007]
]

@dataclass
class HyperparameterSpace:
    """Defines the hyperparameter search space for FM models."""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Union[Tuple[float, float], List[Any]]
    log_scale: bool = False
    description: str = ""

@dataclass
class OptimizationResult:
    """Stores optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    algorithm_used: str

class BaseOptimizer(ABC):
    """Base class for hyperparameter optimization algorithms."""
    
    def __init__(self, 
                 search_space: Dict[str, HyperparameterSpace],
                 objective_function: Callable[[Dict[str, Any]], float],
                 maximize: bool = True,
                 random_state: int = 42):
        self.search_space = search_space
        self.objective_function = objective_function
        self.maximize = maximize
        self.random_state = random_state
        self.optimization_history = []
        
    @abstractmethod
    def optimize(self, n_trials: int = 50) -> OptimizationResult:
        """Run the optimization algorithm."""
        pass
    
    def _evaluate_objective(self, params: Dict[str, Any]) -> float:
        """Evaluate the objective function and store results."""
        score = self.objective_function(params)
        
        # Store in history
        self.optimization_history.append({
            'params': params.copy(),
            'score': score,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Evaluated params: {params}, Score: {score:.6f}")
        return score

class BayesianOptimizer(BaseOptimizer):
    """Bayesian Optimization using Gaussian Process."""
    
    def __init__(self, *args, acquisition_function: str = 'ei', **kwargs):
        super().__init__(*args, **kwargs)
        self.acquisition_function = acquisition_function
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for Bayesian Optimization")
    
    def optimize(self, n_trials: int = 50, n_initial_points: int = 5) -> OptimizationResult:
        """Run Bayesian Optimization."""
        start_time = datetime.now()
        
        # Convert search space to bounds for GP
        bounds = []
        param_names = []
        
        for name, space in self.search_space.items():
            if space.param_type == 'continuous':
                bounds.append(space.bounds)
                param_names.append(name)
            else:
                # For discrete/categorical, we'll handle them separately
                # This is a simplified implementation
                if space.param_type == 'discrete':
                    bounds.append((min(space.bounds), max(space.bounds)))
                    param_names.append(name)
        
        bounds = np.array(bounds)
        
        # Initial random sampling
        X_init = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_initial_points, len(bounds)))
        y_init = []
        
        for x in X_init:
            params = {name: val for name, val in zip(param_names, x)}
            # Handle discrete parameters
            for name, space in self.search_space.items():
                if space.param_type == 'discrete' and name in params:
                    params[name] = int(round(params[name]))
            
            score = self._evaluate_objective(params)
            y_init.append(score if self.maximize else -score)
        
        X_sample = X_init
        y_sample = np.array(y_init)
        
        # Gaussian Process setup
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=self.random_state)
        
        # Bayesian optimization loop
        for i in range(n_trials - n_initial_points):
            # Fit GP
            gp.fit(X_sample, y_sample)
            
            # Find next point using acquisition function
            next_x = self._optimize_acquisition(gp, bounds)
            
            # Evaluate next point
            params = {name: val for name, val in zip(param_names, next_x)}
            for name, space in self.search_space.items():
                if space.param_type == 'discrete' and name in params:
                    params[name] = int(round(params[name]))
            
            next_y = self._evaluate_objective(params)
            if not self.maximize:
                next_y = -next_y
            
            # Update samples
            X_sample = np.vstack([X_sample, next_x])
            y_sample = np.append(y_sample, next_y)
        
        # Find best result
        best_idx = np.argmax(y_sample) if self.maximize else np.argmin(-y_sample)
        best_x = X_sample[best_idx]
        best_params = {name: val for name, val in zip(param_names, best_x)}
        
        # Handle discrete parameters in best result
        for name, space in self.search_space.items():
            if space.param_type == 'discrete' and name in best_params:
                best_params[name] = int(round(best_params[name]))
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=y_sample[best_idx] if self.maximize else -y_sample[best_idx],
            optimization_history=self.optimization_history,
            total_evaluations=len(self.optimization_history),
            optimization_time=optimization_time,
            algorithm_used="Bayesian Optimization"
        )
    
    def _optimize_acquisition(self, gp, bounds):
        """Optimize the acquisition function to find next point."""
        def acquisition(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            
            if self.acquisition_function == 'ei':
                # Expected Improvement
                best_f = np.max(gp.y_train_) if self.maximize else np.min(gp.y_train_)
                z = (mu - best_f) / (sigma + 1e-9)
                return -(mu + sigma * norm.pdf(z) + (mu - best_f) * norm.cdf(z))
            elif self.acquisition_function == 'ucb':
                # Upper Confidence Bound
                kappa = 2.576  # 99% confidence
                return -(mu + kappa * sigma)
            else:
                return -mu  # Greedy
        
        # Multi-start optimization
        best_x = None
        best_acq = float('inf')
        
        for _ in range(10):
            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
            result = minimize(acquisition, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        return best_x

class OptunaOptimizer(BaseOptimizer):
    """Optuna-based hyperparameter optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")
    
    def optimize(self, n_trials: int = 50) -> OptimizationResult:
        """Run Optuna optimization."""
        start_time = datetime.now()
        
        def objective(trial):
            params = {}
            for name, space in self.search_space.items():
                if space.param_type == 'continuous':
                    if space.log_scale:
                        params[name] = trial.suggest_loguniform(name, space.bounds[0], space.bounds[1])
                    else:
                        params[name] = trial.suggest_uniform(name, space.bounds[0], space.bounds[1])
                elif space.param_type == 'discrete':
                    params[name] = trial.suggest_int(name, space.bounds[0], space.bounds[1])
                elif space.param_type == 'categorical':
                    params[name] = trial.suggest_categorical(name, space.bounds)
            
            score = self._evaluate_objective(params)
            return score if self.maximize else -score
        
        # Create study
        direction = 'maximize' if self.maximize else 'minimize'
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler())
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value if self.maximize else -study.best_value,
            optimization_history=self.optimization_history,
            total_evaluations=len(study.trials),
            optimization_time=optimization_time,
            algorithm_used="Optuna TPE"
        )

class HyperoptOptimizer(BaseOptimizer):
    """Hyperopt-based hyperparameter optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt is required. Install with: pip install hyperopt")
    
    def optimize(self, n_trials: int = 50) -> OptimizationResult:
        """Run Hyperopt optimization."""
        start_time = datetime.now()
        
        # Convert search space to hyperopt format
        space = {}
        for name, param_space in self.search_space.items():
            if param_space.param_type == 'continuous':
                if param_space.log_scale:
                    space[name] = hp.loguniform(name, np.log(param_space.bounds[0]), np.log(param_space.bounds[1]))
                else:
                    space[name] = hp.uniform(name, param_space.bounds[0], param_space.bounds[1])
            elif param_space.param_type == 'discrete':
                space[name] = hp.choice(name, list(range(param_space.bounds[0], param_space.bounds[1] + 1)))
            elif param_space.param_type == 'categorical':
                space[name] = hp.choice(name, param_space.bounds)
        
        def objective(params):
            # Handle choice parameters (convert back to actual values)
            processed_params = {}
            for name, value in params.items():
                param_space = self.search_space[name]
                if param_space.param_type == 'discrete':
                    processed_params[name] = value + param_space.bounds[0]
                elif param_space.param_type == 'categorical':
                    processed_params[name] = param_space.bounds[value]
                else:
                    processed_params[name] = value
            
            score = self._evaluate_objective(processed_params)
            return -score if self.maximize else score
        
        # Run optimization
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=n_trials, trials=trials)
        
        # Process best result
        best_params = {}
        for name, value in best.items():
            param_space = self.search_space[name]
            if param_space.param_type == 'discrete':
                best_params[name] = int(value) + param_space.bounds[0]
            elif param_space.param_type == 'categorical':
                best_params[name] = param_space.bounds[int(value)]
            else:
                best_params[name] = value
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Get best score
        best_trial = min(trials.trials, key=lambda x: x['result']['loss'])
        best_score = -best_trial['result']['loss'] if self.maximize else best_trial['result']['loss']
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=self.optimization_history,
            total_evaluations=len(trials.trials),
            optimization_time=optimization_time,
            algorithm_used="Hyperopt TPE"
        )

class FMHyperparameterTuner:
    """Main class for FM model hyperparameter tuning."""
    
    def __init__(self, 
                 fm_training_function: Callable[[Dict[str, Any]], float],
                 maximize: bool = True,
                 random_state: int = 42):
        """
        Initialize the FM hyperparameter tuner.
        
        Args:
            fm_training_function: Function that takes hyperparameters and returns performance metric
            maximize: Whether to maximize (True) or minimize (False) the objective
            random_state: Random seed for reproducibility
        """
        self.fm_training_function = fm_training_function
        self.maximize = maximize
        self.random_state = random_state
        self.search_space = self._create_default_search_space()
        
    def _create_default_search_space(self) -> Dict[str, HyperparameterSpace]:
        """Create default search space for FM models based on user's historical data."""
        return {
            'batch_size': HyperparameterSpace(
                name='batch_size',
                param_type='discrete',
                bounds=[256, 32000],
                description='Training batch size'
            ),
            'epochs': HyperparameterSpace(
                name='epochs',
                param_type='discrete', 
                bounds=[5, 100],
                description='Number of training epochs'
            ),
            'learning_rate': HyperparameterSpace(
                name='learning_rate',
                param_type='continuous',
                bounds=[1e-5, 1e-1],
                log_scale=True,
                description='Learning rate for optimizer'
            ),
            'embedding_dim': HyperparameterSpace(
                name='embedding_dim',
                param_type='discrete',
                bounds=[8, 512],
                description='Embedding dimension for features'
            )
        }
    
    def add_hyperparameter(self, name: str, space: HyperparameterSpace):
        """Add a custom hyperparameter to the search space."""
        self.search_space[name] = space
    
    def remove_hyperparameter(self, name: str):
        """Remove a hyperparameter from the search space."""
        if name in self.search_space:
            del self.search_space[name]
    
    def optimize(self, 
                 algorithm: str = 'bayesian',
                 n_trials: int = 50,
                 **kwargs) -> OptimizationResult:
        """
        Run hyperparameter optimization.
        
        Args:
            algorithm: Optimization algorithm ('bayesian', 'optuna', 'hyperopt')
            n_trials: Number of optimization trials
            **kwargs: Additional arguments for the optimizer
        
        Returns:
            OptimizationResult containing the best parameters and optimization history
        """
        
        if algorithm == 'bayesian':
            optimizer = BayesianOptimizer(
                search_space=self.search_space,
                objective_function=self.fm_training_function,
                maximize=self.maximize,
                random_state=self.random_state,
                **kwargs
            )
        elif algorithm == 'optuna':
            optimizer = OptunaOptimizer(
                search_space=self.search_space,
                objective_function=self.fm_training_function,
                maximize=self.maximize,
                random_state=self.random_state
            )
        elif algorithm == 'hyperopt':
            optimizer = HyperoptOptimizer(
                search_space=self.search_space,
                objective_function=self.fm_training_function,
                maximize=self.maximize,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return optimizer.optimize(n_trials=n_trials)
    
    def load_historical_data(self, historical_data: List[List]) -> None:
        """Load historical performance data to warm-start optimization."""
        # This could be used to initialize the GP with prior knowledge
        self.historical_data = historical_data
    
    def save_results(self, result: OptimizationResult, filepath: str):
        """Save optimization results to file."""
        result_dict = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'optimization_history': result.optimization_history,
            'total_evaluations': result.total_evaluations,
            'optimization_time': result.optimization_time,
            'algorithm_used': result.algorithm_used,
            'search_space': {name: {
                'param_type': space.param_type,
                'bounds': space.bounds,
                'log_scale': space.log_scale,
                'description': space.description
            } for name, space in self.search_space.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")

def create_fm_objective_function(config_template: Dict[str, Any], 
                                data_path: str,
                                validation_metric: str = 'rmse') -> Callable:
    """
    Create an objective function for FM model training.
    
    This function should be customized based on your specific FM training setup.
    """
    def objective(params: Dict[str, Any]) -> float:
        # Update config with new hyperparameters
        config = config_template.copy()
        config.update(params)
        
        # Here you would integrate with your FM training pipeline
        # This is a placeholder - you should replace this with your actual training code
        
        # Example integration (you would replace this):
        try:
            # Import your FM training modules
            from fm_training.config import ConfigManager
            from fm_training.models.fm import FactorizationMachine
            from fm_training.data.s3_parquet_loader import S3ParquetDataLoader
            from fm_training.trainer import Trainer
            
            # Create temporary config
            temp_config = config.copy()
            
            # Initialize components
            data_loader = S3ParquetDataLoader(data_path, 'train', temp_config)
            feature_info = data_loader.get_feature_info()
            
            model_config = temp_config.copy()
            model_config.update(feature_info)
            model = FactorizationMachine(model_config)
            
            trainer = Trainer(model, data_loader, temp_config)
            
            # Train and evaluate
            history = trainer.train()
            
            # Return the metric you want to optimize
            # This should return the validation performance
            if validation_metric in history:
                return history[validation_metric]
            else:
                # Fallback to training loss
                return -history.get('train_loss', float('inf'))
                
        except Exception as e:
            logger.error(f"Training failed for params {params}: {e}")
            return -float('inf') if validation_metric != 'loss' else float('inf')
    
    return objective

# Example usage and demonstration
if __name__ == "__main__":
    # Example of how to use the hyperparameter tuner
    
    # Mock objective function for demonstration
    def mock_fm_objective(params):
        # Simulate FM training with realistic behavior based on user's 4 parameters only
        batch_size = params.get('batch_size', 1024)
        epochs = params.get('epochs', 10) 
        learning_rate = params.get('learning_rate', 0.001)
        embedding_dim = params.get('embedding_dim', 64)
        
        # Simulate performance based on hyperparameters (matching user's historical patterns)
        # This is just for demonstration - replace with actual training
        score = 0.15 + 0.05 * np.log(embedding_dim / 16) + 0.03 * np.log(batch_size / 1000)
        score += 0.02 * epochs / 50 - 0.1 * abs(learning_rate - 0.01)
        score += np.random.normal(0, 0.01)  # Add some noise
        
        return max(0.15, min(0.25, score))  # Clamp to reasonable range based on historical data
    
    # Create tuner
    tuner = FMHyperparameterTuner(
        fm_training_function=mock_fm_objective,
        maximize=True,
        random_state=42
    )
    
    # Load historical data
    tuner.load_historical_data(perf)
    
    print("Available optimization algorithms:")
    algorithms = ['bayesian']
    if OPTUNA_AVAILABLE:
        algorithms.append('optuna')
    if HYPEROPT_AVAILABLE:
        algorithms.append('hyperopt')
    
    print(f"Algorithms: {algorithms}")
    
    # Run optimization with the first available algorithm
    if algorithms:
        result = tuner.optimize(algorithm=algorithms[0], n_trials=20)
        
        print(f"\nOptimization Results ({result.algorithm_used}):")
        print(f"Best Score: {result.best_score:.6f}")
        print(f"Best Parameters: {result.best_params}")
        print(f"Total Evaluations: {result.total_evaluations}")
        print(f"Optimization Time: {result.optimization_time:.2f} seconds")
        
        # Save results
        tuner.save_results(result, 'optimization_results.json')
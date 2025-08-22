#!/usr/bin/env python3
"""
Practical example of using advanced auto-tuning algorithms for FM model hyperparameter optimization.

This script demonstrates how to integrate the hyperparameter tuning system with your existing FM training pipeline.
"""

import sys
import os
import logging
from typing import Dict, Any
import numpy as np

# Add the FM and BO directories to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FM'))
sys.path.append(os.path.dirname(__file__))

from bayesian_optimization import (
    FMHyperparameterTuner, 
    HyperparameterSpace, 
    create_fm_objective_function
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_real_fm_objective_function(base_config_path: str, data_path: str) -> callable:
    """
    Create a real objective function that integrates with your FM training pipeline.
    
    Args:
        base_config_path: Path to your base FM configuration file
        data_path: Path to your training data
    
    Returns:
        Callable that takes hyperparameters and returns validation performance
    """
    
    def objective(params: Dict[str, Any]) -> float:
        try:
            # Import FM training components
            from fm_training.config import ConfigManager
            from fm_training.models.fm import FactorizationMachine
            from fm_training.data.s3_parquet_loader import S3ParquetDataLoader
            from fm_training.trainer import Trainer
            
            # Load base configuration
            config_manager = ConfigManager(base_config_path)
            base_config = config_manager.get_full_config()
            
            # Update with hyperparameters from optimization
            config = base_config.copy()
            config.update(params)
            
            # Ensure some reasonable defaults and constraints
            config['early_stopping_patience'] = min(config.get('early_stopping_patience', 10), 
                                                   max(5, config.get('epochs', 50) // 5))
            
            logger.info(f"Training with params: {params}")
            
            # Initialize data loader
            data_loader = S3ParquetDataLoader(data_path, 'train', config)
            feature_info = data_loader.get_feature_info()
            
            # Create model
            model_config = config.copy()
            model_config.update(feature_info)
            model = FactorizationMachine(model_config)
            
            # Initialize trainer
            trainer = Trainer(model, data_loader, config)
            
            # Train model
            history = trainer.train()
            
            # Extract validation performance
            # Adjust this based on what metrics your trainer returns
            if 'val_rmse' in history:
                # For RMSE, lower is better, so we negate it for maximization
                return -history['val_rmse']
            elif 'val_loss' in history:
                return -history['val_loss']
            elif 'val_accuracy' in history:
                return history['val_accuracy']
            else:
                # Fallback to training metrics
                if 'train_loss' in history:
                    return -history['train_loss']
                else:
                    logger.warning("No suitable metric found in training history")
                    return -float('inf')
                    
        except Exception as e:
            logger.error(f"Training failed with params {params}: {str(e)}")
            # Return a very bad score for failed runs
            return -float('inf')
    
    return objective

def analyze_historical_performance():
    """Analyze your historical performance data to gain insights."""
    from bayesian_optimization import perf
    
    print("=== Historical Performance Analysis ===")
    
    # Convert to numpy array for easier analysis
    data = np.array(perf)
    
    print(f"Total experiments: {len(data)}")
    print(f"Performance range: {data[:, 4].min():.4f} - {data[:, 4].max():.4f}")
    print(f"Average performance: {data[:, 4].mean():.4f}")
    print(f"Performance std: {data[:, 4].std():.4f}")
    
    # Find best configuration
    best_idx = np.argmax(data[:, 4])
    best_config = data[best_idx]
    print(f"\nBest historical configuration:")
    print(f"  Batch Size: {int(best_config[0])}")
    print(f"  Epochs: {int(best_config[1])}")
    print(f"  Learning Rate: {best_config[2]}")
    print(f"  Embedding Dim: {int(best_config[3])}")
    print(f"  Performance: {best_config[4]:.4f}")
    
    # Analyze parameter correlations
    print(f"\n=== Parameter Analysis ===")
    
    # Batch size analysis
    batch_sizes = data[:, 0]
    performances = data[:, 4]
    print(f"Batch size correlation with performance: {np.corrcoef(batch_sizes, performances)[0,1]:.3f}")
    
    # Learning rate analysis
    learning_rates = data[:, 2]
    print(f"Learning rate correlation with performance: {np.corrcoef(learning_rates, performances)[0,1]:.3f}")
    
    # Embedding dimension analysis
    embedding_dims = data[:, 3]
    print(f"Embedding dim correlation with performance: {np.corrcoef(embedding_dims, performances)[0,1]:.3f}")
    
    # Insights
    print(f"\n=== Key Insights ===")
    
    # Best performing ranges
    top_25_percent = data[data[:, 4] >= np.percentile(data[:, 4], 75)]
    
    print(f"Top 25% performing configurations:")
    print(f"  Batch size range: {int(top_25_percent[:, 0].min())} - {int(top_25_percent[:, 0].max())}")
    print(f"  Learning rate range: {top_25_percent[:, 2].min():.5f} - {top_25_percent[:, 2].max():.5f}")
    print(f"  Embedding dim range: {int(top_25_percent[:, 3].min())} - {int(top_25_percent[:, 3].max())}")

def create_custom_search_space() -> Dict[str, HyperparameterSpace]:
    """
    Create a customized search space based on your historical data analysis.
    Only includes the 4 parameters from your original data.
    """
    
    # Based on historical data analysis, we can create a more focused search space
    search_space = {
        # Batch size: Historical data shows good performance with larger batches
        'batch_size': HyperparameterSpace(
            name='batch_size',
            param_type='discrete',
            bounds=[1000, 20000],  # Focused on higher batch sizes
            description='Training batch size - larger batches showed better performance'
        ),
        
        # Epochs: Reasonable range with early stopping
        'epochs': HyperparameterSpace(
            name='epochs',
            param_type='discrete', 
            bounds=[10, 80],
            description='Number of training epochs'
        ),
        
        # Learning rate: Focus around 0.01 which performed well
        'learning_rate': HyperparameterSpace(
            name='learning_rate',
            param_type='continuous',
            bounds=[0.001, 0.05],  # More focused range
            log_scale=True,
            description='Learning rate - focus around 0.01'
        ),
        
        # Embedding dimension: Good performance with 96-256
        'embedding_dim': HyperparameterSpace(
            name='embedding_dim',
            param_type='discrete',
            bounds=[64, 384],  # Focused on the sweet spot
            description='Embedding dimension - historical data favors 96-256'
        )
    }
    
    return search_space

def run_hyperparameter_optimization_example():
    """
    Run a complete example of hyperparameter optimization.
    
    This is a mock example - you should replace the objective function
    with your real FM training pipeline.
    """
    
    print("=== FM Hyperparameter Optimization Example ===\n")
    
    # Analyze historical performance first
    analyze_historical_performance()
    
    print(f"\n=== Setting Up Optimization ===")
    
    # Create a mock objective function (replace this with your real one)
    def mock_objective(params):
        # Simulate realistic FM training behavior using only the 4 user-provided parameters
        batch_size = params.get('batch_size', 1024)
        epochs = params.get('epochs', 20)
        learning_rate = params.get('learning_rate', 0.001)
        embedding_dim = params.get('embedding_dim', 64)
        
        # Simulate performance based on historical insights (4 parameters only)
        score = 0.18  # Base score
        
        # Batch size effect (larger is better, but with diminishing returns)
        score += 0.03 * np.log(batch_size / 1000) * (1 / (1 + batch_size / 15000))
        
        # Embedding dimension effect (sweet spot around 96-256)
        optimal_emb = 128
        score += 0.02 * (1 - abs(embedding_dim - optimal_emb) / optimal_emb)
        
        # Learning rate effect (optimal around 0.01)
        score -= 0.05 * abs(np.log10(learning_rate) + 2)  # log10(0.01) = -2
        
        # Epoch effect with diminishing returns
        score += 0.015 * np.log(epochs / 10)
        
        # Add some noise to simulate real training variance
        score += np.random.normal(0, 0.005)
        
        return max(0.15, min(0.25, score))  # Clamp to reasonable range based on historical data
    
    # Create tuner with custom search space
    tuner = FMHyperparameterTuner(
        fm_training_function=mock_objective,
        maximize=True,
        random_state=42
    )
    
    # Use custom search space based on historical analysis
    custom_search_space = create_custom_search_space()
    tuner.search_space = custom_search_space
    
    # Load historical data for warm start
    from bayesian_optimization import perf
    tuner.load_historical_data(perf)
    
    print(f"Search space configured with {len(tuner.search_space)} hyperparameters:")
    for name, space in tuner.search_space.items():
        print(f"  {name}: {space.param_type} {space.bounds} - {space.description}")
    
    # Run optimization with different algorithms
    algorithms_to_try = ['bayesian']
    
    # Check which algorithms are available
    try:
        import optuna
        algorithms_to_try.append('optuna')
    except ImportError:
        print("Optuna not available - install with: pip install optuna")
    
    try:
        import hyperopt
        algorithms_to_try.append('hyperopt')
    except ImportError:
        print("Hyperopt not available - install with: pip install hyperopt")
    
    print(f"\nAvailable algorithms: {algorithms_to_try}")
    
    # Run optimization
    best_results = {}
    
    for algorithm in algorithms_to_try:
        print(f"\n=== Running {algorithm.upper()} Optimization ===")
        
        try:
            result = tuner.optimize(
                algorithm=algorithm,
                n_trials=30,  # Reduce for demo
                acquisition_function='ei' if algorithm == 'bayesian' else None
            )
            
            best_results[algorithm] = result
            
            print(f"\n{algorithm.upper()} Results:")
            print(f"Best Score: {result.best_score:.6f}")
            print(f"Best Parameters:")
            for param, value in result.best_params.items():
                print(f"  {param}: {value}")
            print(f"Total Evaluations: {result.total_evaluations}")
            print(f"Optimization Time: {result.optimization_time:.2f} seconds")
            
            # Save results
            output_file = f'{algorithm}_optimization_results.json'
            tuner.save_results(result, output_file)
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error running {algorithm}: {e}")
    
    # Compare algorithms
    if len(best_results) > 1:
        print(f"\n=== Algorithm Comparison ===")
        for alg, result in best_results.items():
            print(f"{alg.upper()}: {result.best_score:.6f} (in {result.optimization_time:.1f}s)")
        
        # Find best overall
        best_algorithm = max(best_results.keys(), key=lambda k: best_results[k].best_score)
        print(f"\nBest performing algorithm: {best_algorithm.upper()}")
        
        best_result = best_results[best_algorithm]
        print(f"Recommended hyperparameters:")
        for param, value in best_result.best_params.items():
            print(f"  {param}: {value}")

def provide_hyperparameter_recommendations():
    """
    Provide specific recommendations for additional hyperparameters to explore.
    """
    
    print("\n=== Additional Hyperparameter Recommendations ===")
    
    recommendations = {
        "Model Architecture": [
            {
                "param": "hidden_layers",
                "type": "categorical", 
                "values": ["none", "single", "multiple"],
                "description": "Add hidden layers between linear and interaction terms",
                "rationale": "Can capture more complex patterns but increases overfitting risk"
            },
            {
                "param": "interaction_type",
                "type": "categorical",
                "values": ["all", "selected", "hierarchical"],
                "description": "Type of feature interactions to model",
                "rationale": "Different interaction patterns for different data types"
            },
            {
                "param": "field_aware",
                "type": "categorical",
                "values": [True, False],
                "description": "Use field-aware factorization (FFM)",
                "rationale": "Better for categorical features with different semantics"
            }
        ],
        
        "Regularization": [
            {
                "param": "l1_regularization",
                "type": "continuous",
                "range": [1e-6, 1e-2],
                "log_scale": True,
                "description": "L1 regularization for sparsity",
                "rationale": "Helps with feature selection in high-dimensional data"
            },
            {
                "param": "embedding_regularization",
                "type": "continuous", 
                "range": [1e-6, 1e-3],
                "log_scale": True,
                "description": "Separate regularization for embeddings",
                "rationale": "Embeddings often need different regularization than linear weights"
            },
            {
                "param": "batch_norm",
                "type": "categorical",
                "values": [True, False],
                "description": "Use batch normalization",
                "rationale": "Can help with training stability and speed"
            }
        ],
        
        "Training Dynamics": [
            {
                "param": "warmup_steps",
                "type": "discrete",
                "range": [0, 1000],
                "description": "Learning rate warmup steps",
                "rationale": "Helps with training stability, especially for large models"
            },
            {
                "param": "label_smoothing",
                "type": "continuous",
                "range": [0.0, 0.2],
                "description": "Label smoothing factor",
                "rationale": "Reduces overconfidence and improves generalization"
            },
            {
                "param": "mixed_precision",
                "type": "categorical",
                "values": [True, False],
                "description": "Use mixed precision training",
                "rationale": "Faster training with minimal accuracy loss"
            }
        ],
        
        "Data-Specific": [
            {
                "param": "feature_selection_ratio",
                "type": "continuous",
                "range": [0.7, 1.0],
                "description": "Ratio of features to use in each batch",
                "rationale": "Random feature dropout can improve generalization"
            },
            {
                "param": "negative_sampling_ratio", 
                "type": "continuous",
                "range": [1.0, 10.0],
                "description": "Ratio of negative to positive samples",
                "rationale": "Important for imbalanced datasets"
            },
            {
                "param": "embedding_init_std",
                "type": "continuous",
                "range": [0.01, 0.5],
                "description": "Standard deviation for embedding initialization",
                "rationale": "Proper initialization can significantly impact convergence"
            }
        ]
    }
    
    for category, params in recommendations.items():
        print(f"\n{category}:")
        for param in params:
            print(f"  • {param['param']} ({param['type']})")
            if param['type'] == 'continuous':
                print(f"    Range: {param['range']}")
                if param.get('log_scale'):
                    print(f"    Log scale: Yes")
            elif param['type'] == 'discrete':
                print(f"    Range: {param['range']}")
            elif param['type'] == 'categorical':
                print(f"    Values: {param['values']}")
            
            print(f"    Description: {param['description']}")
            print(f"    Rationale: {param['rationale']}")
            print()

if __name__ == "__main__":
    print("FM Model Hyperparameter Tuning with Advanced Auto-Tuning Algorithms")
    print("=" * 70)
    
    # Run the complete example
    run_hyperparameter_optimization_example()
    
    # Provide additional recommendations
    provide_hyperparameter_recommendations()
    
    print(f"\n=== Usage Instructions ===")
    print(f"""
To use this with your real FM training pipeline:

1. Replace the mock_objective function with create_real_fm_objective_function()
2. Provide your config file path and data path
3. Adjust the search space based on your specific needs
4. Install optional dependencies for better algorithms:
   - pip install optuna (recommended)
   - pip install hyperopt
   - pip install scikit-learn scipy (for Bayesian optimization)

Example integration:
```python
# Create real objective function
objective = create_real_fm_objective_function(
    base_config_path='path/to/your/config.yaml',
    data_path='path/to/your/data'
)

# Create tuner
tuner = FMHyperparameterTuner(objective, maximize=True)

# Run optimization
result = tuner.optimize(algorithm='optuna', n_trials=100)
```
    """)

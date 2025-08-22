# Advanced Auto-Tuning for FM Model Hyperparameters

This directory contains advanced hyperparameter optimization algorithms specifically designed for Factorization Machine (FM) models. The implementation includes multiple state-of-the-art optimization techniques to help you find the best hyperparameters for your FM model.

## Features

### 🚀 Multiple Optimization Algorithms
- **Bayesian Optimization**: Uses Gaussian Processes with various acquisition functions (Expected Improvement, Upper Confidence Bound)
- **Optuna**: Tree-structured Parzen Estimator (TPE) with advanced pruning
- **Hyperopt**: Another TPE implementation with different sampling strategies

### 📊 Comprehensive Hyperparameter Space
- **Model Parameters**: embedding_dim, dropout_rate, architecture choices
- **Training Parameters**: batch_size, learning_rate, epochs, optimizer selection
- **Regularization**: weight_decay, L1/L2 regularization, gradient clipping
- **Advanced Options**: learning rate scheduling, mixed precision, field-aware factorization

### 🔍 Smart Search Space Design
- Log-scale search for learning rates and regularization parameters
- Historical data integration for warm-start optimization
- Categorical parameters for optimizer and scheduler selection
- Constraint-aware parameter sampling

### 📈 Performance Analysis
- Historical performance analysis with correlation insights
- Optimization progress tracking and visualization
- Result persistence and comparison across algorithms
- Statistical significance testing for hyperparameter importance

## Quick Start

### Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install individual packages as needed
pip install optuna hyperopt scikit-learn scipy numpy pandas
```

### Basic Usage

```python
from bayesian_optimization import FMHyperparameterTuner, HyperparameterSpace

# Define your FM training function
def train_fm_model(params):
    # Your FM training code here
    # Return validation performance (higher is better)
    return validation_score

# Create tuner
tuner = FMHyperparameterTuner(
    fm_training_function=train_fm_model,
    maximize=True,
    random_state=42
)

# Run optimization
result = tuner.optimize(
    algorithm='optuna',  # or 'bayesian', 'hyperopt'
    n_trials=100
)

print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score:.6f}")
```

### Integration with Existing FM Pipeline

```python
from bayesian_optimization import create_fm_objective_function

# Create objective function that integrates with your existing pipeline
objective = create_fm_objective_function(
    config_template=your_base_config,
    data_path='path/to/your/data',
    validation_metric='rmse'  # or your preferred metric
)

tuner = FMHyperparameterTuner(objective, maximize=False)  # minimize RMSE
result = tuner.optimize(algorithm='optuna', n_trials=50)
```

## Advanced Features

### Custom Search Space

```python
# Add custom hyperparameters
tuner.add_hyperparameter('custom_param', HyperparameterSpace(
    name='custom_param',
    param_type='continuous',
    bounds=[0.1, 10.0],
    log_scale=True,
    description='Custom parameter with log scale'
))

# Remove default hyperparameters
tuner.remove_hyperparameter('dropout_rate')
```

### Historical Data Integration

```python
# Load your historical experiments
historical_data = [
    [batch_size, epochs, lr, emb_dim, performance],
    # ... more experiments
]

tuner.load_historical_data(historical_data)
```

### Multi-Algorithm Comparison

```python
algorithms = ['bayesian', 'optuna', 'hyperopt']
results = {}

for algo in algorithms:
    results[algo] = tuner.optimize(algorithm=algo, n_trials=50)

# Compare results
best_algo = max(results.keys(), key=lambda k: results[k].best_score)
print(f"Best algorithm: {best_algo}")
```

## Hyperparameter Recommendations

Based on analysis of FM models and empirical studies, here are key hyperparameters to tune:

### High Impact Parameters
1. **embedding_dim** (8-512): Most critical for model capacity
2. **learning_rate** (1e-5 to 1e-1): Use log scale, often optimal around 0.001-0.01
3. **batch_size** (256-32000): Larger often better, but with diminishing returns
4. **weight_decay** (1e-6 to 1e-2): Important for generalization

### Medium Impact Parameters
5. **dropout_rate** (0.0-0.5): Regularization, especially for large embeddings
6. **optimizer** (adam, adamw, sgd): AdamW often works best
7. **lr_scheduler** (cosine, step, exponential): Can provide final performance boost

### Advanced Parameters
8. **gradient_clipping** (0.5-5.0): Helps training stability
9. **label_smoothing** (0.0-0.2): For classification tasks
10. **field_aware** (True/False): For heterogeneous categorical features

## Algorithm Comparison

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Bayesian Optimization** | Principled uncertainty quantification, works with few samples | Slower for high dimensions, requires sklearn | Small search spaces, expensive evaluations |
| **Optuna** | Fast, handles mixed types well, great pruning | Less theoretical foundation | Most practical applications, large search spaces |
| **Hyperopt** | Mature, well-tested, good TPE implementation | Less active development | Baseline comparisons, reproducible research |

## Performance Tips

### 🎯 Search Space Design
- Use log scale for learning rates, regularization parameters
- Focus ranges based on historical data analysis
- Remove irrelevant parameters to reduce dimensionality

### ⚡ Optimization Efficiency
- Start with Optuna for most cases (fastest, most robust)
- Use Bayesian Optimization for expensive evaluations
- Enable early stopping in your FM training
- Use warm-start with historical data

### 📊 Evaluation Strategy
- Always use validation set for hyperparameter selection
- Consider k-fold cross-validation for small datasets
- Track multiple metrics, optimize the most important one
- Save all results for post-hoc analysis

## Examples

### Complete Example
See `fm_hyperparameter_tuning_example.py` for a comprehensive example that includes:
- Historical data analysis
- Custom search space creation
- Multi-algorithm comparison
- Results visualization and interpretation

### Real Integration Example
```python
# Your existing FM training setup
from fm_training.config import ConfigManager
from fm_training.models.fm import FactorizationMachine
from fm_training.trainer import Trainer

# Create optimization-aware training function
def optimized_fm_training(params):
    config = base_config.copy()
    config.update(params)
    
    # Your training pipeline
    model = FactorizationMachine(config)
    trainer = Trainer(model, data_loader, config)
    history = trainer.train()
    
    return history['val_rmse']  # or your metric

# Run optimization
tuner = FMHyperparameterTuner(optimized_fm_training, maximize=False)
best_result = tuner.optimize('optuna', n_trials=100)
```

## Troubleshooting

### Common Issues
1. **Import errors**: Install missing dependencies from requirements.txt
2. **Memory issues**: Reduce batch_size search range or use gradient checkpointing
3. **Slow optimization**: Enable early stopping, reduce n_trials, or use faster algorithm
4. **Poor results**: Check search space bounds, ensure proper validation split

### Performance Debugging
- Monitor optimization progress with logging
- Check for failed trials (return -inf for failures)
- Verify search space covers reasonable ranges
- Compare against manual hyperparameter selection

## Contributing

Feel free to extend the optimization algorithms or add new hyperparameters specific to your FM model variant. The modular design makes it easy to add new optimizers or search spaces.

## References

- Bayesian Optimization: Snoek et al. (2012)
- Optuna: Akiba et al. (2019) 
- Hyperopt: Bergstra et al. (2013)
- Factorization Machines: Rendle (2010)

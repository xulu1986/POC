# FM Model Training Framework

A professional, production-ready framework for training Factorization Machine (FM) models on large-scale datasets stored in S3 Parquet format, optimized for Databricks and PySpark environments.

## Features

- ✅ **Factorization Machine Implementation**: Complete FM model with numerical and categorical feature support
- ✅ **Design Patterns**: Extensible architecture using abstract base classes for easy model replacement
- ✅ **S3 Parquet Integration**: Efficient Spark-based data loading with caching
- ✅ **Large Dataset Support**: Handles large datasets using Spark DataFrames
- ✅ **Databricks Optimization**: Specialized runner for Databricks with PySpark integration
- ✅ **Professional Training Pipeline**: Focused training with metrics tracking, users control all saving
- ✅ **Configuration Management**: YAML-based configuration with validation
- ✅ **Memory Efficiency**: Optimized for large datasets with minimal memory footprint
- ✅ **Distributed Training**: Support for multi-GPU training on Databricks

## Architecture

The framework follows professional software design patterns:

```
fm_training/
├── models/
│   ├── base.py              # Abstract base model class
│   └── fm.py                # Factorization Machine implementation
├── data/
│   ├── base.py              # Abstract base data loader class
│   └── s3_parquet_loader.py # S3 Parquet data loader with Spark integration
├── config.py                # Configuration management
└── trainer.py               # Training pipeline (no automatic saving)
```

### Key Design Patterns

1. **Strategy Pattern**: Abstract base classes allow easy model and data loader replacement
2. **Factory Pattern**: Configuration-driven model and data loader creation
3. **Observer Pattern**: Comprehensive metrics tracking and logging
4. **Template Method**: Standardized training pipeline with customizable components

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For Databricks, ensure the following libraries are installed in your cluster:
   - PyTorch >= 2.0.0
   - PySpark >= 3.4.0
   - PyArrow >= 10.0.0

## Quick Start

### 1. Create Configuration

Generate a sample configuration file:

```bash
python train_fm.py --config config.yaml --create-sample-config
```

### 2. Update Configuration

Edit `config.yaml` with your specific settings:

```yaml
# Model Configuration
model:
  embedding_dim: 10
  dropout_rate: 0.1
  task_type: 'regression'  # or 'classification'

# Data Configuration
data:
  numerical_features:
    - 'feature1'
    - 'feature2'
  categorical_features:
    - 'category1'
    - 'category2'
  target_column: 'target'

# Training Configuration
training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.01  # L2 regularization
  batch_size: 1024
  early_stopping_patience: 10
```

### 3. Train Model

#### Local/Standard Environment:
```bash
python train_fm.py --config config.yaml
```

#### Databricks Environment:
```bash
python databricks_runner.py --config config.yaml
```

#### Databricks Notebook:
```python
from databricks_runner import run_fm_training

results = run_fm_training('/dbfs/path/to/config.yaml')
```

## Usage Examples

### Basic Training

```python
from fm_training.config import ConfigManager
from fm_training.models.fm import FactorizationMachine
from fm_training.data.s3_parquet_loader import S3ParquetDataLoader
from fm_training.trainer import Trainer

# Load configuration
config_manager = ConfigManager('config.yaml')
config = config_manager.get_full_config()

# Initialize data loaders with path and mode
train_loader = S3ParquetDataLoader('s3://bucket/train.parquet', 'train', config, spark)
val_loader = S3ParquetDataLoader('s3://bucket/val.parquet', 'val', config, spark)

# Get feature info from training data (only train mode builds encoders)
feature_info = train_loader.get_feature_info()

# Create model
model_config = config.copy()
model_config.update(feature_info)
model = FactorizationMachine(model_config)

# Note: You'll need to modify trainer to accept separate loaders
# For now, use the train_loader which has the backward compatibility methods
trainer = Trainer(model, train_loader, config)
history = trainer.train()
```

### Custom Model Implementation

To replace FM with your own model, simply inherit from `BaseModel`:

```python
from fm_training.models.base import BaseModel
import torch

class MyCustomModel(BaseModel):
    def _build_model(self):
        # Your model architecture here
        pass
    
    def forward(self, numerical_features, categorical_features):
        # Your forward pass here
        pass
    
    def get_loss_function(self):
        return torch.nn.MSELoss()  # or your custom loss
```

### Custom Data Loader

To use different data sources, inherit from `BaseDataLoader`:

```python
from fm_training.data.base import BaseDataLoader

class MyCustomDataLoader(BaseDataLoader):
    def get_train_loader(self):
        # Your data loading logic here
        pass
    
    def get_feature_info(self):
        # Return feature information
        pass
```

## Configuration Reference

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | int | 10 | Dimension of embedding vectors |
| `dropout_rate` | float | 0.1 | Dropout rate for regularization |
| `task_type` | str | 'regression' | Task type: 'regression' or 'classification' |

### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 100 | Maximum number of training epochs |
| `learning_rate` | float | 0.001 | Learning rate for optimizer |
| `weight_decay` | float | 0.0 | Weight decay (L2 regularization) for optimizer |
| `batch_size` | int | 1024 | Batch size for training |
| `early_stopping_patience` | int | 10 | Early stopping patience |
| `gradient_clipping` | float | null | Gradient clipping threshold |

### Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numerical_features` | list | [] | List of numerical feature columns |
| `categorical_features` | list | [] | List of categorical feature columns |
| `target_column` | str | 'target' | Target column name |
| `dataset_batch_size` | int | 10000 | Batch size for loading from Spark |


## Memory Efficiency

The framework is designed to handle large datasets efficiently:

1. **Spark-Based Loading**: Data is read with Spark and cached for reuse
2. **Batch Processing**: Loads data in configurable batches to avoid memory overflow
3. **No S3 Direct Access**: PyTorch datasets work with Spark DataFrames using sampling
4. **Memory Optimization**: Only small batches are converted to pandas/tensors at a time

## Databricks Integration

### Cluster Configuration

Recommended Databricks cluster configuration:

- **Runtime**: 13.3 LTS ML or later
- **Node Type**: GPU-enabled instances (e.g., g4dn.xlarge)
- **Workers**: 2-8 workers depending on data size
- **Libraries**: Install requirements.txt packages

### Environment Setup

The `DatabricksRunner` automatically configures:

- Spark session with ML optimizations
- GPU memory management
- Distributed training setup
- DBFS path handling
- Performance optimizations

### Example Databricks Notebook

```python
# Cell 1: Install dependencies (if needed)
%pip install torch torchvision pyarrow s3fs scikit-learn tqdm pyyaml

# Cell 2: Upload your code and config
# Upload the fm_training package and config.yaml to DBFS

# Cell 3: Run training
from databricks_runner import run_fm_training

config_path = '/dbfs/path/to/your/config.yaml'
results = run_fm_training(config_path)

# Cell 4: View results
print("Training completed!")
print(f"Final test metrics: {results['test_metrics']}")
```

## Training Results

The framework returns all results as Python objects, giving users full control over saving:

```python
# Training returns a dictionary with everything
training_history = trainer.train()

# Contains:
# - train_losses: List of training losses
# - config: Training configuration  
# - total_training_time: Training duration
# - model_dict: Serialized model for saving

# Users decide how to save:
import json
with open('my_model.json', 'w') as f:
    json.dump(training_history['model_dict'], f)

# Load model later:
with open('my_model.json', 'r') as f:
    model_dict = json.load(f)
loaded_model = FactorizationMachine.from_dict(model_dict)
```

## Performance Optimization

### For Large Datasets

1. **Increase chunk_size**: For datasets > 1GB, use chunk_size of 50000-100000
2. **Use Spark**: Enable `use_spark: true` for datasets > 10GB
3. **Adjust batch_size**: Increase batch size based on available GPU memory
4. **Enable gradient_clipping**: Use gradient clipping for stable training

### For Databricks

1. **Use GPU clusters**: Enable GPU acceleration for faster training
2. **Optimize Spark**: The framework automatically configures Spark for ML workloads
3. **Use DBFS**: Store outputs in DBFS for persistence across cluster restarts

## Monitoring and Logging

The framework provides comprehensive logging:

- Training progress with metrics
- Memory usage monitoring
- Error handling and recovery
- Model saving in JSON format
- Performance profiling

Logs are saved to both console and `fm_training.log` file.

## Extending the Framework

### Adding New Models

1. Inherit from `BaseModel`
2. Implement required abstract methods
3. Add model-specific configuration parameters
4. Update model factory in training script

### Adding New Data Sources

1. Inherit from `BaseDataLoader` and `BaseDataset`
2. Implement data loading logic
3. Ensure compatibility with feature preprocessing
4. Add data source configuration options

### Custom Metrics

Add custom metrics by extending the `_calculate_metrics` method in `FMTrainer`.

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `batch_size` and `chunk_size`
2. **S3 Access**: Ensure proper AWS credentials are configured
3. **Spark Errors**: Check Spark cluster configuration and resources
4. **GPU Issues**: Verify CUDA installation and GPU availability

### Debug Mode

Enable debug logging:

```yaml
system:
  log_level: 'DEBUG'
```

## Contributing

1. Follow the existing code structure and design patterns
2. Add comprehensive tests for new features
3. Update documentation for any new configuration options
4. Ensure compatibility with both local and Databricks environments

## License

This project is licensed under the MIT License.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the configuration reference
3. Examine the example usage patterns
4. Create an issue with detailed error information 
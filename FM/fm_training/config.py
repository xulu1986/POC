import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    embedding_dim: int = 10
    dropout_rate: float = 0.1
    task_type: str = 'regression'  # 'regression' or 'classification'


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 1024
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-6
    gradient_clipping: Optional[float] = None



@dataclass
class DataConfig:
    """Data configuration parameters."""
    numerical_features: list = None
    categorical_features: list = None
    target_column: str = 'target'
    dataset_batch_size: int = 10000
    num_workers: int = 4


@dataclass
class SystemConfig:
    """System configuration parameters."""
    output_dir: Optional[str] = None
    log_level: str = 'INFO'
    random_seed: int = 42


class ConfigManager:
    """
    Configuration manager for FM training.
    Handles loading, validation, and merging of configurations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.config = config
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def save_config(self, config_path: str, config: Optional[Dict[str, Any]] = None):
        """Save configuration to YAML file."""
        config_to_save = config or self.config
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        model_config = self.config.get('model', {})
        return ModelConfig(**model_config)
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        training_config = self.config.get('training', {})
        return TrainingConfig(**training_config)
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        data_config = self.config.get('data', {})
        return DataConfig(**data_config)
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        system_config = self.config.get('system', {})
        return SystemConfig(**system_config)
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        full_config = {}
        
        # Merge all configurations
        full_config.update(asdict(self.get_model_config()))
        full_config.update(asdict(self.get_training_config()))
        full_config.update(asdict(self.get_data_config()))
        full_config.update(asdict(self.get_system_config()))
        
        return full_config
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate data config
            data_config = self.get_data_config()
            
            # Validate model config
            model_config = self.get_model_config()
            if model_config.embedding_dim <= 0:
                raise ValueError("embedding_dim must be positive")
            
            if not 0 <= model_config.dropout_rate <= 1:
                raise ValueError("dropout_rate must be between 0 and 1")
            
            if model_config.task_type not in ['regression', 'classification']:
                raise ValueError("task_type must be 'regression' or 'classification'")
            
            # Validate training config
            training_config = self.get_training_config()
            if training_config.epochs <= 0:
                raise ValueError("epochs must be positive")
            
            if training_config.learning_rate <= 0:
                raise ValueError("learning_rate must be positive")
            
            if training_config.batch_size <= 0:
                raise ValueError("batch_size must be positive")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        default_config = {
            'model': asdict(ModelConfig()),
            'training': asdict(TrainingConfig()),
            'data': {
                'numerical_features': ['feature1', 'feature2'],
                'categorical_features': ['category1', 'category2'],
                'target_column': 'target',
                'dataset_batch_size': 10000,
                'num_workers': 4
            },
            'system': asdict(SystemConfig())
        }
        
        return default_config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        logger.info("Configuration updated")


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fm_training.log')
        ]
    )


def create_sample_config(output_path: str = 'config.yaml'):
    """Create a sample configuration file."""
    config_manager = ConfigManager()
    sample_config = config_manager.create_default_config()
    
    # Add comments to the sample config
    sample_config_with_comments = f"""# FM Model Training Configuration

# Model Configuration
model:
  embedding_dim: 10          # Dimension of embedding vectors
  dropout_rate: 0.1          # Dropout rate for regularization
  task_type: 'regression'    # 'regression' or 'classification'

# Training Configuration  
training:
  epochs: 100                # Maximum number of training epochs
  learning_rate: 0.001       # Learning rate for optimizer
  batch_size: 1024           # Batch size for training
  early_stopping_patience: 10    # Early stopping patience
  early_stopping_min_delta: 1e-6 # Minimum improvement for early stopping
  gradient_clipping: null    # Gradient clipping threshold (null to disable)
  

# Data Configuration
data:
  numerical_features:        # List of numerical feature column names
    - 'feature1'
    - 'feature2'
  categorical_features:      # List of categorical feature column names
    - 'category1'
    - 'category2'
  target_column: 'target'    # Target column name
  dataset_batch_size: 10000  # Batch size for loading data from Spark
  num_workers: 4             # Number of data loading workers

# System Configuration
system:
  output_dir: null           # Output directory (optional, for user convenience)
  log_level: 'INFO'          # Logging level
  random_seed: 42            # Random seed for reproducibility
"""
    
    with open(output_path, 'w') as f:
        f.write(sample_config_with_comments)
    
    print(f"Sample configuration created at {output_path}")
    return output_path 
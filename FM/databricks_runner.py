#!/usr/bin/env python3
"""
Databricks-specific runner for FM model training.
This script handles Databricks environment setup and distributed training considerations.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional
import torch
import torch.distributed as dist
from pyspark.sql import SparkSession

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fm_training.config import ConfigManager, setup_logging
from fm_training.models.fm import FactorizationMachine
from fm_training.data.s3_parquet_loader import S3ParquetDataLoader
from fm_training.trainer import Trainer

logger = logging.getLogger(__name__)


class DatabricksRunner:
    """
    Databricks-specific runner for FM model training.
    Handles Databricks environment setup and optimizations.
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.spark = None
        self.config_manager = None
        self.config = None
        
        self._setup_databricks_environment()
        self._load_configuration()
    
    def _setup_databricks_environment(self):
        """Setup Databricks-specific environment configurations."""
        logger.info("Setting up Databricks environment...")
        
        # Use existing Spark session
        self.spark = SparkSession.getActiveSession()
        if self.spark is None:
            raise ValueError("No active Spark session found in Databricks environment")
        
        # Set Spark log level to reduce noise
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Configure PyTorch for Databricks
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
            # Set CUDA device if available
            torch.cuda.set_device(0)
        else:
            logger.info("CUDA not available, using CPU")
        
        # Set environment variables for optimal performance
        os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        
        logger.info("Databricks environment setup completed")
    
    def _load_configuration(self):
        """Load and validate configuration."""
        self.config_manager = ConfigManager(self.config_path)
        
        if not self.config_manager.validate_config():
            raise ValueError("Configuration validation failed")
        
        self.config = self.config_manager.get_full_config()
        
        # Databricks-specific configuration adjustments
        self._adjust_config_for_databricks()
    
    def _adjust_config_for_databricks(self):
        """Adjust configuration for Databricks environment."""
        # Adjust batch size based on available memory
        if torch.cuda.is_available():
            # GPU memory considerations
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 8 * 1024**3:  # Less than 8GB
                self.config['batch_size'] = min(self.config['batch_size'], 512)
        
        # Adjust number of workers based on cluster configuration
        try:
            # Get number of executor cores
            executor_cores = int(self.spark.conf.get("spark.executor.cores", "2"))
            self.config['num_workers'] = min(self.config['num_workers'], executor_cores)
        except:
            pass
        
        # Set output directory to DBFS if not already set
        if not self.config['output_dir'].startswith('/dbfs/'):
            self.config['output_dir'] = f"/dbfs/tmp/fm_training/{self.config['output_dir']}"
        
        logger.info(f"Adjusted configuration for Databricks: {self.config}")
    
    def _setup_distributed_training(self) -> bool:
        """Setup distributed training if multiple GPUs are available."""
        if not torch.cuda.is_available():
            return False
        
        gpu_count = torch.cuda.device_count()
        if gpu_count <= 1:
            return False
        
        try:
            # Initialize distributed training
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=gpu_count,
                    rank=0
                )
            
            logger.info(f"Distributed training initialized with {gpu_count} GPUs")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize distributed training: {e}")
            return False
    
    def run_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        try:
            logger.info("Starting FM model training on Databricks")
            
            # Setup logging
            setup_logging(self.config.get('log_level', 'INFO'))
            
            # Create output directory if specified (users might want it for their own saving)
            if self.config.get('output_dir'):
                os.makedirs(self.config['output_dir'], exist_ok=True)
            
            # Initialize data loader with Spark session
            logger.info("Initializing data loader...")
            
            # Create data loader
            data_loader = S3ParquetDataLoader(
                self.config['data_path'], self.config['mode'], self.config, self.spark
            )
            feature_info = data_loader.get_feature_info()
            
            logger.info(f"Feature info: {feature_info}")
            
            # Create model
            logger.info("Creating FM model...")
            model_config = self.config.copy()
            model_config.update(feature_info)
            model = FactorizationMachine(model_config)
            
            logger.info(f"Created FM model with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Setup distributed training if available
            is_distributed = self._setup_distributed_training()
            if is_distributed:
                model = torch.nn.parallel.DistributedDataParallel(model)
            
            # Initialize trainer
            logger.info("Initializing trainer...")
            trainer = Trainer(model, data_loader, self.config)
            
            # Check mode
            if self.config['mode'] == 'train':
                # Train model
                logger.info("Starting training...")
                training_history = trainer.train()
                
                # Prepare results
                results = {
                    'training_history': training_history,
                    'feature_info': feature_info,
                    'config': self.config
                }
                
                logger.info("Training completed successfully!")
                logger.info(f"Training time: {training_history['total_training_time']:.2f} seconds")
                logger.info("Training results returned. Users can save model and history as needed.")
                
            else:
                # Evaluation mode
                logger.info("Starting evaluation...")
                metrics = trainer.evaluate()
                
                # Prepare results
                results = {
                    'metrics': metrics,
                    'feature_info': feature_info,
                    'config': self.config
                }
                
                logger.info("Evaluation completed successfully!")
                if metrics:
                    logger.info("Evaluation results:")
                    for metric, value in metrics.items():
                        logger.info(f"  {metric}: {value:.6f}")
                
                logger.info("Evaluation results returned. Users can save as needed.")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # No cleanup needed since we're using existing Spark session
            pass
    
    def run_evaluation_only(self, model_path: str) -> Dict[str, float]:
        """Run evaluation only on a trained model."""
        try:
            logger.info("Running evaluation only...")
            
            # Setup logging
            setup_logging(self.config.get('log_level', 'INFO'))
            
            # Initialize data loader
            data_loader = S3ParquetDataLoader(
                self.config['data_path'], self.config['mode'], self.config, self.spark
            )
            
            # Load trained model
            logger.info(f"Loading trained model from {model_path}")
            with open(model_path, 'r') as f:
                model_dict = json.load(f)
            model = FactorizationMachine.from_dict(model_dict)
            
            # Create evaluator and evaluate
            from .evaluator import Evaluator
            evaluator = Evaluator(model, self.config)
            metrics = evaluator.evaluate(data_loader.get_data_loader())
            
            logger.info("Evaluation completed!")
            if metrics:
                logger.info("Evaluation results:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value:.6f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
        
        finally:
            # No cleanup needed since we're using existing Spark session
            pass


def main():
    """Main function for Databricks execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FM training on Databricks')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--evaluate-only', type=str, default=None,
                       help='Path to trained model for evaluation only')
    
    args = parser.parse_args()
    
    try:
        runner = DatabricksRunner(args.config)
        
        if args.evaluate_only:
            results = runner.run_evaluation_only(args.evaluate_only)
        else:
            results = runner.run_training()
        
        print("Training/Evaluation completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        raise


# For Databricks notebook execution
def run_fm_training(config_path: str, evaluate_only: Optional[str] = None):
    """
    Function to run FM training from Databricks notebook.
    
    Args:
        config_path: Path to configuration YAML file
        evaluate_only: Path to trained model for evaluation only (optional)
    
    Returns:
        Training results dictionary
    """
    runner = DatabricksRunner(config_path)
    
    if evaluate_only:
        return runner.run_evaluation_only(evaluate_only)
    else:
        return runner.run_training()


if __name__ == '__main__':
    main() 
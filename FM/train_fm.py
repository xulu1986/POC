#!/usr/bin/env python3
"""
Main training script for FM model.
This script provides a command-line interface for training FM models on S3 Parquet data.
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
import logging
import json
from typing import Dict, Any

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fm_training.config import ConfigManager, setup_logging, create_sample_config
from fm_training.models.fm import FactorizationMachine
from fm_training.data.s3_parquet_loader import S3ParquetDataLoader
from fm_training.trainer import Trainer

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config: Dict[str, Any], feature_info: Dict[str, Any]) -> FactorizationMachine:
    """Create FM model with proper configuration."""
    model_config = config.copy()
    model_config.update(feature_info)
    
    model = FactorizationMachine(model_config)
    
    logger.info(f"Created FM model with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Model configuration: {model_config}")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train FM model on S3 Parquet data')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--create-sample-config', action='store_true',
                       help='Create a sample configuration file and exit')

    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate the model (requires trained model)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory from config')
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_sample_config:
        create_sample_config(args.config)
        return
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        
        if not config_manager.validate_config():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        config = config_manager.get_full_config()
        
        # Override output directory if specified
        if args.output_dir:
            config['output_dir'] = args.output_dir
        
        # Setup logging
        setup_logging(config.get('log_level', 'INFO'))
        
        # Set random seeds
        set_random_seeds(config.get('random_seed', 42))
        
        logger.info("Starting FM model training")
        logger.info(f"Configuration: {config}")
        
        # Create output directory if specified (users might want it for their own saving)
        if config.get('output_dir'):
            os.makedirs(config['output_dir'], exist_ok=True)
        
        # Initialize data loader
        logger.info("Initializing data loader...")
        
        # Create data loader
        data_loader = S3ParquetDataLoader(
            config['data_path'], config['mode'], config
        )
        feature_info = data_loader.get_feature_info()
        
        logger.info(f"Feature info: {feature_info}")
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config, feature_info)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(model, data_loader, config)
        
        # Evaluate only mode
        if args.evaluate_only:
            logger.info("Evaluation mode - skipping training")
            logger.error("--evaluate-only requires a pre-trained model. Please train a model first or provide model loading logic.")
            sys.exit(1)
        
        # Check mode
        if config['mode'] == 'train':
            # Train model
            logger.info("Starting training...")
            training_history = trainer.train()
            
            # Log final results
            logger.info("Training completed successfully!")
            logger.info(f"Training time: {training_history['total_training_time']:.2f} seconds")
            logger.info("Training results returned. Users can save model and history as needed.")
            logger.info("Example: model_dict = training_history['model_dict']")
            logger.info("Example: with open('my_model.json', 'w') as f: json.dump(model_dict, f)")
            
        else:
            # Evaluation mode
            logger.info("Starting evaluation...")
            metrics = trainer.evaluate()
            
            if metrics:
                logger.info("Evaluation results:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value:.6f}")
            
            logger.info("Evaluation completed. Results returned for user to save as needed.")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main() 
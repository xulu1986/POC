import time
import logging
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

from .models.base import BaseModel
from .data.base import BaseDataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """
    Professional training pipeline for models with focused features:
    - Configuration management
    - Training loop management
    """
    
    def __init__(self, 
                 model: BaseModel,
                 data_loader: BaseDataLoader,
                 config: Dict[str, Any]):
        
        self.model = model
        self.data_loader = data_loader
        self.config = config
        
        # Training configuration
        self.epochs = config.get('epochs', 100)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = model.get_optimizer(self.learning_rate, self.weight_decay)
        self.criterion = model.get_loss_function()
        
        # Metrics tracking
        self.train_losses = []
        
        logger.info(f"Initialized Trainer with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (numerical_features, categorical_features, targets) in enumerate(train_loader):
            # Move to device
            numerical_features = numerical_features.to(self.device)
            categorical_features = categorical_features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(numerical_features, categorical_features)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clipping'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clipping']
                )
            
            self.optimizer.step()
            
            # Track loss only
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    

    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        
        # Get data loader
        data_loader = self.data_loader.get_data_loader()
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self.train_epoch(data_loader)
            self.train_losses.append(train_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            # Logging
            log_msg = f"Epoch {epoch+1}/{self.epochs} - "
            log_msg += f"Loss: {train_loss:.6f}"
            log_msg += f" - Time: {epoch_time:.2f}s"
            logger.info(log_msg)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Return training history and model - let users decide how to save them
        history = {
            'train_losses': self.train_losses,
            'config': self.config,
            'total_training_time': total_time,
            'model_dict': self.model.to_dict()  # Include model for user to save
        }
        
        return history
    
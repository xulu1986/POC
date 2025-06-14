import logging
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, accuracy_score

from .models.base import BaseModel

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Model evaluator focused on evaluation tasks.
    Separated from Trainer to follow single responsibility principle.
    """
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Initialized Evaluator with device: {self.device}")
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader containing evaluation data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting model evaluation...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for numerical_features, categorical_features, targets in data_loader:
                numerical_features = numerical_features.to(self.device)
                categorical_features = categorical_features.to(self.device)
                
                predictions = self.model(numerical_features, categorical_features)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        logger.info("Evaluation completed!")
        logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
        
        return metrics
    
    def evaluate_batch(self, numerical_features: torch.Tensor, 
                      categorical_features: torch.Tensor, 
                      targets: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on a single batch.
        
        Args:
            numerical_features: Numerical features tensor
            categorical_features: Categorical features tensor
            targets: Target values tensor
            
        Returns:
            Dictionary containing evaluation metrics for this batch
        """
        self.model.eval()
        
        with torch.no_grad():
            numerical_features = numerical_features.to(self.device)
            categorical_features = categorical_features.to(self.device)
            targets = targets.to(self.device)
            
            predictions = self.model(numerical_features, categorical_features)
            
            predictions_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()
        
        return self._calculate_metrics(predictions_np, targets_np)
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Generate predictions for a dataset.
        
        Args:
            data_loader: DataLoader containing input data
            
        Returns:
            Array of predictions
        """
        logger.info("Generating predictions...")
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for numerical_features, categorical_features, _ in data_loader:
                numerical_features = numerical_features.to(self.device)
                categorical_features = categorical_features.to(self.device)
                
                predictions = self.model(numerical_features, categorical_features)
                
                if self.config.get('task_type') == 'classification':
                    predictions = torch.sigmoid(predictions)
                
                all_predictions.extend(predictions.cpu().numpy())
        
        return np.array(all_predictions)
    
    def predict_batch(self, numerical_features: torch.Tensor, 
                     categorical_features: torch.Tensor) -> np.ndarray:
        """
        Generate predictions for a single batch.
        
        Args:
            numerical_features: Numerical features tensor
            categorical_features: Categorical features tensor
            
        Returns:
            Array of predictions for this batch
        """
        self.model.eval()
        
        with torch.no_grad():
            numerical_features = numerical_features.to(self.device)
            categorical_features = categorical_features.to(self.device)
            
            predictions = self.model(numerical_features, categorical_features)
            
            if self.config.get('task_type') == 'classification':
                predictions = torch.sigmoid(predictions)
            
            return predictions.cpu().numpy()
    
    def _calculate_metrics(self, predictions: list, targets: list) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        metrics = {}
        
        task_type = self.config.get('task_type', 'regression')
        
        if task_type == 'regression':
            metrics['mse'] = mean_squared_error(targets, predictions)
            metrics['mae'] = mean_absolute_error(targets, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # R² score (handle edge cases)
            try:
                metrics['r2'] = r2_score(targets, predictions)
            except:
                metrics['r2'] = 0.0
                
        elif task_type == 'classification':
            # Convert logits to probabilities
            probabilities = 1 / (1 + np.exp(-predictions))  # sigmoid
            binary_predictions = (probabilities > 0.5).astype(int)
            
            metrics['accuracy'] = accuracy_score(targets, binary_predictions)
            
            # AUC (handle edge cases)
            try:
                metrics['auc'] = roc_auc_score(targets, probabilities)
            except:
                metrics['auc'] = 0.5
        
        return metrics
    
    def compare_models(self, other_model: BaseModel, data_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Compare this model with another model on the same dataset.
        
        Args:
            other_model: Another model to compare against
            data_loader: DataLoader containing evaluation data
            
        Returns:
            Dictionary containing metrics for both models
        """
        logger.info("Comparing models...")
        
        # Evaluate current model
        current_metrics = self.evaluate(data_loader)
        
        # Evaluate other model
        other_evaluator = Evaluator(other_model, self.config)
        other_metrics = other_evaluator.evaluate(data_loader)
        
        comparison = {
            'current_model': current_metrics,
            'other_model': other_metrics
        }
        
        logger.info("Model comparison completed!")
        return comparison 
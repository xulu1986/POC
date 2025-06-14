#!/usr/bin/env python3
"""
Example demonstrating the new Evaluator class.
Shows how evaluation is now decoupled from training.
"""

import torch
import numpy as np
from fm_training.models.fm import FactorizationMachine
from fm_training.trainer import Trainer
from fm_training.evaluator import Evaluator

def create_sample_data(n_samples=1000):
    """Create sample data for demonstration."""
    
    # Create features
    numerical_features = torch.randn(n_samples, 3)
    categorical_features = torch.randint(0, 5, (n_samples, 2))
    
    # Create target
    target = (
        0.5 * numerical_features[:, 0] + 
        0.3 * numerical_features[:, 1] * numerical_features[:, 2] +
        0.1 * torch.randn(n_samples)
    ).unsqueeze(1)
    
    return numerical_features, categorical_features, target

def main():
    """Demonstrate the Evaluator class."""
    
    print("Evaluator Class Demonstration")
    print("=" * 40)
    
    # Create sample data
    print("1. Creating sample data...")
    numerical, categorical, target = create_sample_data(1000)
    
    # Create dataset and data loader
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, numerical, categorical, target):
            self.numerical = numerical
            self.categorical = categorical
            self.target = target
        
        def __len__(self):
            return len(self.target)
        
        def __getitem__(self, idx):
            return self.numerical[idx], self.categorical[idx], self.target[idx]
    
    dataset = SimpleDataset(numerical, categorical, target)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Create model configuration
    config = {
        'num_numerical_features': 3,
        'num_categorical_features': 2,
        'categorical_vocab_sizes': [5, 5],
        'embedding_dim': 4,
        'dropout_rate': 0.1,
        'task_type': 'regression',
        'epochs': 10,
        'learning_rate': 0.01,
        'batch_size': 64,
        'output_dir': None  # No automatic saving
    }
    
    # Create and train model
    print("2. Training model...")
    model = FactorizationMachine(config)
    trainer = Trainer(model, None, config)  # No data_loader needed for trainer now
    
    # Simple training loop (since we removed evaluation from trainer)
    optimizer = model.get_optimizer(config['learning_rate'])
    criterion = model.get_loss_function()
    
    for epoch in range(config['epochs']):
        total_loss = 0.0
        for numerical_batch, categorical_batch, target_batch in data_loader:
            optimizer.zero_grad()
            predictions = model(numerical_batch, categorical_batch)
            loss = criterion(predictions, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    print("3. Creating Evaluator...")
    evaluator = Evaluator(model, config)
    
    # Demonstrate different evaluation methods
    print("\n4. Full dataset evaluation:")
    metrics = evaluator.evaluate(data_loader)
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   R²: {metrics['r2']:.4f}")
    
    print("\n5. Single batch evaluation:")
    # Get first batch
    numerical_batch, categorical_batch, target_batch = next(iter(data_loader))
    batch_metrics = evaluator.evaluate_batch(numerical_batch, categorical_batch, target_batch)
    print(f"   Batch RMSE: {batch_metrics['rmse']:.4f}")
    
    print("\n6. Prediction generation:")
    predictions = evaluator.predict(data_loader)
    print(f"   Generated {len(predictions)} predictions")
    print(f"   Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    print("\n7. Single batch prediction:")
    batch_predictions = evaluator.predict_batch(numerical_batch, categorical_batch)
    print(f"   Batch predictions shape: {batch_predictions.shape}")
    
    # Demonstrate model comparison
    print("\n8. Model comparison:")
    
    # Create a second model for comparison
    model2 = FactorizationMachine(config)
    
    # Train it slightly differently
    optimizer2 = model2.get_optimizer(config['learning_rate'] * 0.5)  # Different learning rate
    for epoch in range(5):  # Fewer epochs
        for numerical_batch, categorical_batch, target_batch in data_loader:
            optimizer2.zero_grad()
            predictions = model2(numerical_batch, categorical_batch)
            loss = criterion(predictions, target_batch)
            loss.backward()
            optimizer2.step()
    
    # Compare models
    comparison = evaluator.compare_models(model2, data_loader)
    
    print("   Model 1 (current):")
    for metric, value in comparison['current_model'].items():
        print(f"     {metric}: {value:.4f}")
    
    print("   Model 2 (other):")
    for metric, value in comparison['other_model'].items():
        print(f"     {metric}: {value:.4f}")
    
    # Show the clean separation of concerns
    print("\n" + "=" * 40)
    print("CLEAN ARCHITECTURE BENEFITS")
    print("=" * 40)
    print("✅ Trainer: Focused only on training")
    print("✅ Evaluator: Focused only on evaluation")
    print("✅ Memory efficient: No accumulation during training")
    print("✅ Flexible: Evaluate any model on any dataset")
    print("✅ Reusable: Same evaluator for different models")
    print("✅ Testable: Easy to unit test evaluation logic")

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
Example demonstrating the effect of weight decay on FM model training.
Weight decay (L2 regularization) helps prevent overfitting by penalizing large weights.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from fm_training.models.fm import FactorizationMachine
from fm_training.trainer import Trainer
from fm_training.data.s3_parquet_loader import S3ParquetDataLoader

def create_synthetic_data(n_samples=10000, n_numerical=5, n_categorical=3, noise_level=0.1):
    """Create synthetic data for testing weight decay effects."""
    
    # Create numerical features
    numerical_features = np.random.randn(n_samples, n_numerical)
    
    # Create categorical features
    categorical_features = np.random.randint(0, 10, (n_samples, n_categorical))
    
    # Create target with some interaction effects
    target = (
        0.5 * numerical_features[:, 0] +  # Linear effect
        0.3 * numerical_features[:, 1] * numerical_features[:, 2] +  # Interaction
        0.2 * (categorical_features[:, 0] == 5).astype(float) +  # Categorical effect
        noise_level * np.random.randn(n_samples)  # Noise
    )
    
    return numerical_features, categorical_features, target

def train_with_weight_decay(weight_decay_values, n_epochs=50):
    """Train models with different weight decay values and compare results."""
    
    print("Creating synthetic dataset...")
    numerical_features, categorical_features, target = create_synthetic_data()
    
    # Split into train/validation
    split_idx = int(0.8 * len(target))
    
    train_numerical = numerical_features[:split_idx]
    train_categorical = categorical_features[:split_idx]
    train_target = target[:split_idx]
    
    val_numerical = numerical_features[split_idx:]
    val_categorical = categorical_features[split_idx:]
    val_target = target[split_idx:]
    
    results = {}
    
    for weight_decay in weight_decay_values:
        print(f"\nTraining with weight_decay = {weight_decay}")
        
        # Create model configuration
        config = {
            'num_numerical_features': 5,
            'num_categorical_features': 3,
            'categorical_vocab_sizes': [10, 10, 10],
            'embedding_dim': 8,
            'dropout_rate': 0.1,
            'task_type': 'regression',
            'epochs': n_epochs,
            'learning_rate': 0.001,
            'weight_decay': weight_decay,
            'batch_size': 256,
            'output_dir': None  # No automatic saving
        }
        
        # Create model
        model = FactorizationMachine(config)
        
        # Create simple dataset class for this example
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, numerical, categorical, target):
                self.numerical = torch.tensor(numerical, dtype=torch.float32)
                self.categorical = torch.tensor(categorical, dtype=torch.long)
                self.target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
            
            def __len__(self):
                return len(self.target)
            
            def __getitem__(self, idx):
                return self.numerical[idx], self.categorical[idx], self.target[idx]
        
        # Create data loaders
        train_dataset = SimpleDataset(train_numerical, train_categorical, train_target)
        val_dataset = SimpleDataset(val_numerical, val_categorical, val_target)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # Training loop
        optimizer = model.get_optimizer(config['learning_rate'], config['weight_decay'])
        criterion = model.get_loss_function()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for numerical, categorical, targets in train_loader:
                optimizer.zero_grad()
                predictions = model(numerical, categorical)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for numerical, categorical, targets in val_loader:
                    predictions = model(numerical, categorical)
                    loss = criterion(predictions, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Calculate weight norms
        total_weight_norm = 0.0
        for param in model.parameters():
            total_weight_norm += torch.norm(param).item() ** 2
        total_weight_norm = np.sqrt(total_weight_norm)
        
        results[weight_decay] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'weight_norm': total_weight_norm,
            'overfitting': val_losses[-1] - train_losses[-1]
        }
        
        print(f"  Final - Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}")
        print(f"  Weight Norm: {total_weight_norm:.4f}")
        print(f"  Overfitting (Val-Train): {val_losses[-1] - train_losses[-1]:.4f}")
    
    return results

def plot_results(results):
    """Plot training curves and weight decay effects."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training curves
    ax1 = axes[0, 0]
    for weight_decay, result in results.items():
        epochs = range(len(result['train_losses']))
        ax1.plot(epochs, result['train_losses'], '--', label=f'Train (wd={weight_decay})')
        ax1.plot(epochs, result['val_losses'], '-', label=f'Val (wd={weight_decay})')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves with Different Weight Decay')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Final losses
    ax2 = axes[0, 1]
    weight_decays = list(results.keys())
    train_losses = [results[wd]['final_train_loss'] for wd in weight_decays]
    val_losses = [results[wd]['final_val_loss'] for wd in weight_decays]
    
    x = np.arange(len(weight_decays))
    width = 0.35
    
    ax2.bar(x - width/2, train_losses, width, label='Train Loss', alpha=0.8)
    ax2.bar(x + width/2, val_losses, width, label='Val Loss', alpha=0.8)
    
    ax2.set_xlabel('Weight Decay')
    ax2.set_ylabel('Final Loss')
    ax2.set_title('Final Losses vs Weight Decay')
    ax2.set_xticks(x)
    ax2.set_xticklabels(weight_decays)
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Weight norms
    ax3 = axes[1, 0]
    weight_norms = [results[wd]['weight_norm'] for wd in weight_decays]
    
    ax3.bar(weight_decays, weight_norms, alpha=0.8, color='green')
    ax3.set_xlabel('Weight Decay')
    ax3.set_ylabel('Total Weight Norm')
    ax3.set_title('Model Weight Norms vs Weight Decay')
    ax3.grid(True)
    
    # Plot 4: Overfitting measure
    ax4 = axes[1, 1]
    overfitting = [results[wd]['overfitting'] for wd in weight_decays]
    
    ax4.bar(weight_decays, overfitting, alpha=0.8, color='red')
    ax4.set_xlabel('Weight Decay')
    ax4.set_ylabel('Overfitting (Val Loss - Train Loss)')
    ax4.set_title('Overfitting vs Weight Decay')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('weight_decay_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to demonstrate weight decay effects."""
    
    print("Weight Decay Demonstration for FM Model")
    print("=" * 50)
    
    # Test different weight decay values
    weight_decay_values = [0.0, 0.001, 0.01, 0.1]
    
    print(f"Testing weight decay values: {weight_decay_values}")
    print("This will train 4 models and compare their performance...")
    
    # Train models
    results = train_with_weight_decay(weight_decay_values, n_epochs=30)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    print(f"{'Weight Decay':<12} {'Train Loss':<12} {'Val Loss':<12} {'Weight Norm':<12} {'Overfitting':<12}")
    print("-" * 60)
    
    for wd in weight_decay_values:
        result = results[wd]
        print(f"{wd:<12.3f} {result['final_train_loss']:<12.4f} {result['final_val_loss']:<12.4f} "
              f"{result['weight_norm']:<12.4f} {result['overfitting']:<12.4f}")
    
    # Find best weight decay
    best_wd = min(weight_decay_values, key=lambda wd: results[wd]['final_val_loss'])
    print(f"\nBest weight decay: {best_wd} (lowest validation loss)")
    
    # Plot results
    try:
        plot_results(results)
        print("\nPlots saved as 'weight_decay_comparison.png'")
    except ImportError:
        print("\nMatplotlib not available, skipping plots")
    
    print("\nKey Observations:")
    print("- Higher weight decay typically reduces overfitting")
    print("- Weight decay shrinks model weights (regularization)")
    print("- Too much weight decay can hurt training performance")
    print("- Optimal weight decay balances bias-variance tradeoff")

if __name__ == '__main__':
    main() 
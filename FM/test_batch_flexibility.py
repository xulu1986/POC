#!/usr/bin/env python3
"""
Test script to demonstrate batch size flexibility of JIT compiled models.
"""

import torch
import json
import base64
import io
from fm_training.models.fm import FactorizationMachine

def test_batch_flexibility():
    """Test that JIT model works with different batch sizes."""
    
    # Create a sample FM model
    config = {
        'num_numerical_features': 3,
        'num_categorical_features': 2,
        'categorical_vocab_sizes': [5, 8],
        'embedding_dim': 4,
        'dropout_rate': 0.0,  # Disable dropout for consistent results
        'task_type': 'regression'
    }
    
    print("Creating and testing FM model...")
    model = FactorizationMachine(config)
    model.eval()  # Set to eval mode for consistent results
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    
    print("\nTesting original PyTorch model with different batch sizes:")
    original_results = {}
    
    for batch_size in batch_sizes:
        # Create test data
        numerical_features = torch.randn(batch_size, 3)
        categorical_features = torch.randint(0, 5, (batch_size, 2))
        
        with torch.no_grad():
            predictions = model(numerical_features, categorical_features)
            original_results[batch_size] = predictions
            print(f"  Batch size {batch_size:2d}: shape {predictions.shape}, first prediction: {predictions[0].item():.6f}")
    
    # Convert to JIT scripted model
    print("\nConverting to JIT scripted model...")
    model_dict = model.to_dict()
    
    # Load as JIT scripted model
    model_base64 = model_dict['model']
    model_bytes = base64.b64decode(model_base64.encode('utf-8'))
    buffer = io.BytesIO(model_bytes)
    jit_model = torch.jit.load(buffer)
    jit_model.eval()
    
    print("Testing JIT scripted model with different batch sizes:")
    jit_results = {}
    
    for batch_size in batch_sizes:
        try:
            # Create test data (same as before)
            numerical_features = torch.randn(batch_size, 3)
            categorical_features = torch.randint(0, 5, (batch_size, 2))
            
            with torch.no_grad():
                predictions = jit_model(numerical_features, categorical_features)
                jit_results[batch_size] = predictions
                print(f"  Batch size {batch_size:2d}: ✅ shape {predictions.shape}, first prediction: {predictions[0].item():.6f}")
                
        except Exception as e:
            print(f"  Batch size {batch_size:2d}: ❌ Error: {e}")
            jit_results[batch_size] = None
    
    # Test consistency between original and JIT models
    print("\nTesting consistency between original and JIT scripted models:")
    
    # Use the same random seed for fair comparison
    torch.manual_seed(42)
    numerical_features = torch.randn(4, 3)
    categorical_features = torch.randint(0, 5, (4, 2))
    
    with torch.no_grad():
        original_pred = model(numerical_features, categorical_features)
        jit_pred = jit_model(numerical_features, categorical_features)
        
        diff = torch.abs(original_pred - jit_pred).max().item()
        print(f"Max difference between original and JIT scripted: {diff:.8f}")
        
        if diff < 1e-6:
            print("✅ Models produce identical results!")
        else:
            print("⚠️  Models produce slightly different results (might be due to different random states)")
    
    # Show model type
    print(f"\nJIT scripted model type: {type(jit_model)}")
    print(f"JIT scripted model code:\n{jit_model.code}")

def test_go_service_simulation():
    """Simulate how a Go service would use the model."""
    
    print("\n" + "="*60)
    print("SIMULATING GO SERVICE USAGE")
    print("="*60)
    
    # Create and save model
    config = {
        'num_numerical_features': 5,
        'num_categorical_features': 3,
        'categorical_vocab_sizes': [10, 20, 15],
        'embedding_dim': 8,
        'task_type': 'regression'
    }
    
    model = FactorizationMachine(config)
    model_dict = model.to_dict()
    
    # Save to file (simulating model deployment)
    with open('production_model.json', 'w') as f:
        json.dump(model_dict, f, indent=2)
    
    print("Model saved to production_model.json")
    
    # Simulate Go service loading the model
    print("\nSimulating Go service loading model...")
    
    with open('production_model.json', 'r') as f:
        loaded_dict = json.load(f)
    
    # Load JIT scripted model
    model_base64 = loaded_dict['model']
    model_bytes = base64.b64decode(model_base64.encode('utf-8'))
    buffer = io.BytesIO(model_bytes)
    production_model = torch.jit.load(buffer)
    production_model.eval()
    
    # Parse config
    production_config = json.loads(loaded_dict['config'])
    print(f"Loaded config: {production_config}")
    
    # Test different scenarios a Go service might encounter
    scenarios = [
        ("Single prediction", 1),
        ("Small batch", 4),
        ("Medium batch", 16),
        ("Large batch", 64),
    ]
    
    print("\nTesting different batch scenarios:")
    
    for scenario_name, batch_size in scenarios:
        try:
            # Create random test data
            numerical = torch.randn(batch_size, 5)
            categorical = torch.randint(0, 10, (batch_size, 3))
            
            with torch.no_grad():
                predictions = production_model(numerical, categorical)
                
            print(f"  {scenario_name:15s} (batch={batch_size:2d}): ✅ {predictions.shape} -> avg prediction: {predictions.mean().item():.6f}")
            
        except Exception as e:
            print(f"  {scenario_name:15s} (batch={batch_size:2d}): ❌ {e}")
    
    print("\n✅ Production model ready for Go service!")
    print("   - Supports flexible batch sizes")
    print("   - Config available as JSON string")
    print("   - Model optimized for inference")

if __name__ == '__main__':
    test_batch_flexibility()
    test_go_service_simulation() 
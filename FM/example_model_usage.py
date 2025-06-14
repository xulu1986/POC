#!/usr/bin/env python3
"""
Example usage of the new model save/load functionality with base64-encoded JIT buffer.
"""

import torch
import json
from fm_training.models.fm import FactorizationMachine

def main():
    """Demonstrate model save/load with base64-encoded JIT buffer."""
    
    # Create a sample FM model configuration
    config = {
        'num_numerical_features': 5,
        'num_categorical_features': 3,
        'categorical_vocab_sizes': [10, 20, 15],
        'embedding_dim': 8,
        'dropout_rate': 0.1,
        'task_type': 'regression'
    }
    
    # Create and train a model (simplified example)
    print("Creating FM model...")
    model = FactorizationMachine(config)
    
    # Create some dummy data
    batch_size = 32
    numerical_features = torch.randn(batch_size, 5)
    categorical_features = torch.randint(0, 10, (batch_size, 3))
    
    # Forward pass to ensure model works
    print("Testing forward pass...")
    with torch.no_grad():
        predictions = model(numerical_features, categorical_features)
        print(f"Predictions shape: {predictions.shape}")
    
    # Method 1: Save to file (user handles file I/O)
    print("\nSaving model to file...")
    model_dict = model.to_dict()
    with open('example_model.json', 'w') as f:
        json.dump(model_dict, f, indent=2)
    print("Model saved to 'example_model.json'")
    
    # Method 2: Convert to dictionary (for in-memory usage)
    print("\nConverting model to dictionary...")
    model_dict = model.to_dict()
    print(f"Dictionary keys: {list(model_dict.keys())}")
    print(f"Config (JSON string): {model_dict['config']}")
    print(f"Config type: {type(model_dict['config'])}")
    print(f"Model (base64) length: {len(model_dict['model'])} characters")
    
    # Method 3: Load from file (user handles file I/O)
    print("\nLoading model from file...")
    with open('example_model.json', 'r') as f:
        loaded_model_dict = json.load(f)
    loaded_model = FactorizationMachine.from_dict(loaded_model_dict)
    
    # Test loaded model
    with torch.no_grad():
        loaded_predictions = loaded_model(numerical_features, categorical_features)
        print(f"Loaded model predictions shape: {loaded_predictions.shape}")
        
        # Check if predictions are the same (they should be)
        diff = torch.abs(predictions - loaded_predictions).max().item()
        print(f"Max difference between original and loaded model: {diff}")
        
        if diff < 1e-6:
            print("✓ Models produce identical results!")
        else:
            print("✗ Models produce different results")
    
    # Method 4: Load from dictionary
    print("\nLoading model from dictionary...")
    dict_loaded_model = FactorizationMachine.from_dict(model_dict)
    
    with torch.no_grad():
        dict_predictions = dict_loaded_model(numerical_features, categorical_features)
        diff = torch.abs(predictions - dict_predictions).max().item()
        print(f"Max difference with dict-loaded model: {diff}")
        
        if diff < 1e-6:
            print("✓ Dictionary-loaded model produces identical results!")
        else:
            print("✗ Dictionary-loaded model produces different results")
    
    # Method 5: Load as JIT scripted model (for inference only)
    print("\nLoading as JIT scripted model...")
    with open('example_model.json', 'r') as f:
        jit_model_dict = json.load(f)
    
    # Decode base64 model
    import base64
    import io
    model_base64 = jit_model_dict['model']
    model_bytes = base64.b64decode(model_base64.encode('utf-8'))
    
    # Load JIT scripted model from bytes
    buffer = io.BytesIO(model_bytes)
    jit_model = torch.jit.load(buffer)
    
    # Parse config from JSON string
    jit_config = json.loads(jit_model_dict['config'])
    
    print(f"JIT scripted model type: {type(jit_model)}")
    print(f"JIT config: {jit_config}")
    
    with torch.no_grad():
        jit_predictions = jit_model(numerical_features, categorical_features)
        diff = torch.abs(predictions - jit_predictions).max().item()
        print(f"Max difference with JIT scripted model: {diff}")
        
        if diff < 1e-6:
            print("✓ JIT scripted model produces identical results!")
        else:
            print("✗ JIT scripted model produces different results")
    
    # Show the JSON structure (first 500 characters)
    print("\nJSON file structure (first 500 chars):")
    with open('example_model.json', 'r') as f:
        content = f.read()
        print(content[:500] + "..." if len(content) > 500 else content)
    
    print("\nExample completed successfully!")

if __name__ == '__main__':
    main() 
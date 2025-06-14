#!/usr/bin/env python3
"""
Demo showing the difference between config as dict vs JSON string format.
"""

import json
from fm_training.models.fm import FactorizationMachine

def main():
    """Demonstrate the JSON string config format."""
    
    # Create a sample FM model configuration
    config = {
        'num_numerical_features': 3,
        'num_categorical_features': 2,
        'categorical_vocab_sizes': [5, 8],
        'embedding_dim': 4,
        'dropout_rate': 0.1,
        'task_type': 'regression'
    }
    
    print("Original config:")
    print(f"Type: {type(config)}")
    print(f"Content: {config}")
    print()
    
    # Create model
    model = FactorizationMachine(config)
    
    # Convert to dictionary format
    model_dict = model.to_dict()
    
    print("Model dictionary structure:")
    print(f"Keys: {list(model_dict.keys())}")
    print()
    
    print("Config in model dictionary:")
    print(f"Type: {type(model_dict['config'])}")
    print(f"Content: {model_dict['config']}")
    print()
    
    # Show that it's a valid JSON string
    print("Parsing config back to dict:")
    parsed_config = json.loads(model_dict['config'])
    print(f"Type: {type(parsed_config)}")
    print(f"Content: {parsed_config}")
    print()
    
    # Verify they're the same
    print("Verification:")
    print(f"Original == Parsed: {config == parsed_config}")
    
    # Show JSON file structure
    print("\nSaving and showing JSON file structure:")
    model_dict = model.to_dict()
    with open('demo_model.json', 'w') as f:
        json.dump(model_dict, f, indent=2)
    
    with open('demo_model.json', 'r') as f:
        file_content = f.read()
    
    # Show first part of the file
    lines = file_content.split('\n')
    print("First few lines of saved JSON file:")
    for i, line in enumerate(lines[:10]):
        print(f"{i+1:2d}: {line}")
    
    if len(lines) > 10:
        print("    ...")
        print(f"{len(lines):2d}: {lines[-1]}")
    
    print(f"\nTotal file size: {len(file_content)} characters")

if __name__ == '__main__':
    main() 
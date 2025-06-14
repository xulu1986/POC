#!/usr/bin/env python3
"""
Utility functions for saving and loading models.
Users can customize these functions based on their needs.
"""

import json
import base64
import io
import torch
from typing import Dict, Any, Tuple, Optional
from fm_training.models.base import BaseModel


def save_model_to_file(model: BaseModel, filepath: str) -> None:
    """
    Save model to a JSON file.
    
    Args:
        model: The model to save
        filepath: Path to save the model (should end with .json)
    """
    model_dict = model.to_dict()
    with open(filepath, 'w') as f:
        json.dump(model_dict, f, indent=2)
    print(f"Model saved to {filepath}")


def load_model_from_file(model_class, filepath: str, device: Optional[torch.device] = None):
    """
    Load model from a JSON file.
    
    Args:
        model_class: The model class (e.g., FactorizationMachine)
        filepath: Path to the saved model file
        device: Device to load the model to
        
    Returns:
        Loaded model instance
    """
    with open(filepath, 'r') as f:
        model_dict = json.load(f)
    
    return model_class.from_dict(model_dict, device)


def load_jit_model_from_file(filepath: str, device: Optional[torch.device] = None) -> Tuple[torch.jit.ScriptModule, Dict[str, Any]]:
    """
    Load JIT scripted model directly from file (for inference only).
    
    Args:
        filepath: Path to the saved model file
        device: Device to load the model to
        
    Returns:
        Tuple of (jit_scripted_model, config_dict)
    """
    with open(filepath, 'r') as f:
        model_dict = json.load(f)
    
    # Decode base64 model
    model_base64 = model_dict['model']
    model_bytes = base64.b64decode(model_base64.encode('utf-8'))
    
    # Load JIT scripted model from bytes
    buffer = io.BytesIO(model_bytes)
    jit_model = torch.jit.load(buffer, map_location=device)
    
    # Parse config from JSON string
    config = json.loads(model_dict['config'])
    
    return jit_model, config


def save_model_to_database(model: BaseModel, db_connection, model_id: str) -> None:
    """
    Example: Save model to database.
    
    Args:
        model: The model to save
        db_connection: Database connection object
        model_id: Unique identifier for the model
    """
    model_dict = model.to_dict()
    
    # Example SQL (adapt to your database)
    query = """
    INSERT INTO models (id, model_data, config) 
    VALUES (?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET 
        model_data = excluded.model_data,
        config = excluded.config
    """
    
    db_connection.execute(query, (
        model_id,
        model_dict['model'],  # Base64 string
        model_dict['config']  # JSON string
    ))
    db_connection.commit()
    print(f"Model saved to database with ID: {model_id}")


def load_model_from_database(model_class, db_connection, model_id: str, device: Optional[torch.device] = None):
    """
    Example: Load model from database.
    
    Args:
        model_class: The model class
        db_connection: Database connection object
        model_id: Unique identifier for the model
        device: Device to load the model to
        
    Returns:
        Loaded model instance
    """
    query = "SELECT model_data, config FROM models WHERE id = ?"
    cursor = db_connection.execute(query, (model_id,))
    row = cursor.fetchone()
    
    if row is None:
        raise ValueError(f"Model with ID {model_id} not found in database")
    
    model_dict = {
        'model': row[0],  # Base64 string
        'config': row[1]  # JSON string
    }
    
    return model_class.from_dict(model_dict, device)


def save_model_to_s3(model: BaseModel, bucket: str, key: str, s3_client) -> None:
    """
    Example: Save model to AWS S3.
    
    Args:
        model: The model to save
        bucket: S3 bucket name
        key: S3 object key
        s3_client: boto3 S3 client
    """
    model_dict = model.to_dict()
    model_json = json.dumps(model_dict, indent=2)
    
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=model_json.encode('utf-8'),
        ContentType='application/json'
    )
    print(f"Model saved to s3://{bucket}/{key}")


def load_model_from_s3(model_class, bucket: str, key: str, s3_client, device: Optional[torch.device] = None):
    """
    Example: Load model from AWS S3.
    
    Args:
        model_class: The model class
        bucket: S3 bucket name
        key: S3 object key
        s3_client: boto3 S3 client
        device: Device to load the model to
        
    Returns:
        Loaded model instance
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    model_json = response['Body'].read().decode('utf-8')
    model_dict = json.loads(model_json)
    
    return model_class.from_dict(model_dict, device)


# Example usage
if __name__ == '__main__':
    from fm_training.models.fm import FactorizationMachine
    
    # Create a sample model
    config = {
        'num_numerical_features': 3,
        'num_categorical_features': 2,
        'categorical_vocab_sizes': [5, 8],
        'embedding_dim': 4,
        'task_type': 'regression'
    }
    
    model = FactorizationMachine(config)
    
    # Save to file
    save_model_to_file(model, 'my_model.json')
    
    # Load from file
    loaded_model = load_model_from_file(FactorizationMachine, 'my_model.json')
    
    # Load as JIT scripted model
    jit_model, config = load_jit_model_from_file('my_model.json')
    
    print("All operations completed successfully!") 
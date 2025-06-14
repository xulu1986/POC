import base64
import io
import json
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import torch


class BaseModel(ABC, torch.nn.Module):
    """
    Abstract base class for all models.
    This enables easy model replacement by implementing this interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._build_model()
    
    @abstractmethod
    def _build_model(self) -> None:
        """Build the model architecture based on config."""
        pass
    
    @abstractmethod
    def forward(self, numerical_features: torch.Tensor, 
                categorical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            numerical_features: Tensor of shape (batch_size, num_numerical_features)
            categorical_features: Tensor of shape (batch_size, num_categorical_features)
            
        Returns:
            Model predictions of shape (batch_size, 1)
        """
        pass
    
    @abstractmethod
    def get_loss_function(self) -> torch.nn.Module:
        """Return the appropriate loss function for this model."""
        pass
    
    def get_optimizer(self, learning_rate: float = 0.001, weight_decay: float = 0.0) -> torch.optim.Optimizer:
        """Return the optimizer for training. Can be overridden by subclasses."""
        return torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary with base64-encoded JIT buffer and config."""
        # Create a traced/scripted version of the model for JIT
        self.eval()
        
        # Use torch.jit.script for JIT compilation
        # Scripting preserves control flow and supports dynamic shapes
        try:
            scripted_model = torch.jit.script(self)
        except Exception as e:
            raise RuntimeError(f"Failed to script model for JIT compilation: {e}")
        
        # Save scripted model to buffer
        buffer = io.BytesIO()
        torch.jit.save(scripted_model, buffer)
        buffer.seek(0)
        
        # Encode buffer as base64
        model_bytes = buffer.getvalue()
        model_base64 = base64.b64encode(model_bytes).decode('utf-8')
        
        # Create the dictionary with config as JSON string
        return {
            'model': model_base64,
            'config': json.dumps(self.config)
        }
    
    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any], device: Optional[torch.device] = None):
        """Load model from dictionary format."""
        # Decode base64 model
        model_base64 = model_dict['model']
        model_bytes = base64.b64decode(model_base64.encode('utf-8'))
        
        # Load JIT model from bytes
        buffer = io.BytesIO(model_bytes)
        scripted_model = torch.jit.load(buffer, map_location=device)
        
        # For compatibility, we still return the original model class instance
        # but with the scripted model's state
        config_str = model_dict['config']
        config = json.loads(config_str)  # Parse JSON string back to dict
        model = cls(config)
        
        # Copy parameters from scripted model to the new instance
        scripted_state_dict = scripted_model.state_dict()
        model.load_state_dict(scripted_state_dict)
        
        if device:
            model.to(device)
            
        return model 
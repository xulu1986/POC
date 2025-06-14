import torch
import torch.nn.functional as F
from typing import Dict, Any
from .base import BaseModel


class FactorizationMachine(BaseModel):
    """
    Factorization Machine model implementation.
    
    FM models feature interactions through factorized parameters:
    y = w0 + Σ(wi * xi) + Σ(Σ(<vi, vj> * xi * xj))
    
    Where:
    - w0 is the global bias
    - wi are feature weights
    - vi are embedding vectors for feature interactions
    """
    
    def _build_model(self) -> None:
        """Build FM model components."""
        self.num_numerical = self.config['num_numerical_features']
        self.num_categorical = self.config['num_categorical_features']
        self.categorical_vocab_sizes = self.config['categorical_vocab_sizes']
        self.embedding_dim = self.config.get('embedding_dim', 16)
        self.dropout_rate = self.config.get('dropout_rate', 0.0)
        
        # Global bias
        self.bias = torch.nn.Parameter(torch.zeros(1))
        
        # Linear weights for numerical features
        if self.num_numerical > 0:
            self.numerical_linear = torch.nn.Linear(self.num_numerical, 1, bias=False)
            self.numerical_embeddings = torch.nn.Linear(self.num_numerical, self.embedding_dim, bias=False)
        
        # Embeddings for categorical features
        if self.num_categorical > 0:
            self.categorical_embeddings = torch.nn.ModuleList([
                torch.nn.Embedding(vocab_size, self.embedding_dim)
                for vocab_size in self.categorical_vocab_sizes
            ])
            self.categorical_linear = torch.nn.ModuleList([
                torch.nn.Embedding(vocab_size, 1)
                for vocab_size in self.categorical_vocab_sizes
            ])
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        if hasattr(self, 'numerical_linear'):
            torch.nn.init.xavier_uniform_(self.numerical_linear.weight)
            torch.nn.init.xavier_uniform_(self.numerical_embeddings.weight)
        
        if hasattr(self, 'categorical_embeddings'):
            for embedding in self.categorical_embeddings:
                torch.nn.init.xavier_uniform_(embedding.weight)
            for linear in self.categorical_linear:
                torch.nn.init.xavier_uniform_(linear.weight)
    
    def forward(self, numerical_features: torch.Tensor, 
                categorical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FM model.
        
        Args:
            numerical_features: (batch_size, num_numerical_features)
            categorical_features: (batch_size, num_categorical_features)
            
        Returns:
            predictions: (batch_size, 1)
        """
        batch_size = numerical_features.size(0) if numerical_features is not None else categorical_features.size(0)
        
        # Start with global bias
        output = self.bias.expand(batch_size, 1)
        
        # Collect all embeddings for interaction computation
        all_embeddings = []
        
        # Process numerical features
        if numerical_features is not None and self.num_numerical > 0:
            # Linear part
            numerical_linear = self.numerical_linear(numerical_features)
            output = output + numerical_linear
            
            # Embedding part for interactions
            numerical_emb = self.numerical_embeddings(numerical_features)  # (batch_size, embedding_dim)
            all_embeddings.append(numerical_emb)
        
        # Process categorical features
        if categorical_features is not None and self.num_categorical > 0:
            categorical_linear_sum = torch.zeros(batch_size, 1, device=categorical_features.device)
            
            for i, (embedding_layer, linear_layer) in enumerate(zip(self.categorical_embeddings, self.categorical_linear)):
                cat_indices = categorical_features[:, i]  # (batch_size,)
                
                # Linear part
                cat_linear = linear_layer(cat_indices)  # (batch_size, 1)
                categorical_linear_sum = categorical_linear_sum + cat_linear
                
                # Embedding part for interactions
                cat_emb = embedding_layer(cat_indices)  # (batch_size, embedding_dim)
                all_embeddings.append(cat_emb)
            
            output = output + categorical_linear_sum
        
        # Compute feature interactions using FM formula
        if len(all_embeddings) > 1:
            # Stack all embeddings
            stacked_embeddings = torch.stack(all_embeddings, dim=1)  # (batch_size, num_features, embedding_dim)
            
            # FM interaction: 0.5 * (sum^2 - sum_of_squares)
            sum_of_embeddings = torch.sum(stacked_embeddings, dim=1)  # (batch_size, embedding_dim)
            sum_of_squares = torch.sum(stacked_embeddings ** 2, dim=1)  # (batch_size, embedding_dim)
            
            interactions = 0.5 * (sum_of_embeddings ** 2 - sum_of_squares)  # (batch_size, embedding_dim)
            interactions = torch.sum(interactions, dim=1, keepdim=True)  # (batch_size, 1)
            
            output = output + interactions
        
        # Apply dropout
        output = self.dropout(output)
        
        return output
    
    def get_loss_function(self) -> torch.nn.Module:
        """Return appropriate loss function for FM (typically MSE for regression, BCE for classification)."""
        task_type = self.config.get('task_type', 'regression')
        if task_type == 'regression':
            return torch.nn.MSELoss()
        elif task_type == 'classification':
            return torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def predict_proba(self, numerical_features: torch.Tensor, 
                      categorical_features: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities for classification tasks."""
        with torch.no_grad():
            logits = self.forward(numerical_features, categorical_features)
            if self.config.get('task_type') == 'classification':
                return torch.sigmoid(logits)
            else:
                return logits 
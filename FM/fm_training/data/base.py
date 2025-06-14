from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Dict, Any, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    This enables easy data source replacement by implementing this interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get('batch_size', 1024)
        self.num_workers = config.get('num_workers', 4)
        
    @abstractmethod
    def get_data_loader(self) -> DataLoader:
        """Return data loader for the specified mode."""
        pass
    
    @abstractmethod
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Return feature information including:
        - num_numerical_features
        - num_categorical_features  
        - categorical_vocab_sizes
        - feature_names
        """
        pass


class BaseDataset(ABC, Dataset):
    """
    Abstract base class for datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.numerical_features: Optional[List[str]] = config.get('numerical_features')
        self.categorical_features: Optional[List[str]] = config.get('categorical_features')
        self.target_column: str = config.get('target_column', 'target')
        
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return a single sample.
        
        Returns:
            numerical_features: Tensor of numerical features
            categorical_features: Tensor of categorical feature indices
            target: Target value
        """
        pass 
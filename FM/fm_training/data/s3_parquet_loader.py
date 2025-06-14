import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, monotonically_increasing_id
import logging

from .base import BaseDataLoader, BaseDataset

logger = logging.getLogger(__name__)


class SparkDataset(BaseDataset):
    """
    Memory-efficient Spark-based dataset with local buffering.
    Uses a local in-memory buffer to minimize Spark access overhead.
    Loads multiple batches at once into local memory for efficient access.
    """
    
    def __init__(self, 
                 spark_df,
                 config: Dict[str, Any],
                 categorical_encoders: Optional[Dict[str, LabelEncoder]] = None,
                 numerical_scaler: Optional[StandardScaler] = None):
        super().__init__(config)
        
        self.categorical_encoders = categorical_encoders or {}
        self.numerical_scaler = numerical_scaler
        
        # Add row numbers once during initialization for efficient batch access
        logger.info("Adding row numbers to DataFrame for efficient batch access...")
        
        # Use monotonically_increasing_id() for ordering - clean and maintainable
        # It creates a unique, monotonically increasing ID for each row
        self.spark_df = spark_df.withColumn("temp_id", monotonically_increasing_id())
        window_spec = Window.orderBy("temp_id")
        self.spark_df = self.spark_df.withColumn("row_num", row_number().over(window_spec))
        self.spark_df = self.spark_df.drop("temp_id")  # Clean up temporary column
        self.spark_df.persist()  # Cache the DataFrame with row numbers
        
        # Get total count
        self.total_rows = self.spark_df.count()
        
        # Determine batch size for efficient processing
        self.batch_size = config.get('dataset_batch_size', 128)
        self.num_batches = (self.total_rows + self.batch_size - 1) // self.batch_size
        
        # Local buffer configuration
        self.buffer_size = config.get('buffer_size', 100_000)  # Number of rows to buffer at once
        self.buffer_data = None  # Current buffer: (numerical, categorical, target) tensors
        self.buffer_start_row = 0  # Starting row index of current buffer
        self.buffer_end_row = 0    # Ending row index of current buffer
        
        logger.info(f"Initialized SparkDataset with {self.total_rows} rows in {self.num_batches} batches")
        logger.info(f"Using local buffer for {self.buffer_size} rows per buffer")
        
        # Load initial buffer
        self._load_next_buffer()
    
    def __len__(self) -> int:
        return self.total_rows
    
    def _load_next_buffer(self):
        """Load the next buffer chunk from Spark DataFrame."""
        # Calculate start and end row indices for this buffer
        start_row = self.buffer_end_row + 1  # row_number() is 1-indexed
        end_row = min(start_row + self.buffer_size - 1, self.total_rows)
        
        logger.info(f"Loading buffer: rows {start_row} to {end_row}")
        
        # Filter using the pre-computed row numbers
        buffer_df = self.spark_df.filter(
            (self.spark_df.row_num >= start_row) & 
            (self.spark_df.row_num <= end_row)
        ).drop("row_num")  # Remove the row number column
        
        # Convert this buffer to pandas
        buffer_pandas = buffer_df.toPandas()
        
        # Verify we got the expected number of rows
        expected_size = end_row - start_row + 1
        if len(buffer_pandas) != expected_size:
            logger.warning(f"Expected {expected_size} rows but got {len(buffer_pandas)} for buffer")
        
        # Preprocess and store in buffer
        self.buffer_data = self._preprocess_batch(buffer_pandas)
        self.buffer_start_row = start_row - 1  # Convert to 0-indexed for dataset access
        self.buffer_end_row = self.buffer_start_row + len(buffer_pandas)
        
        logger.info(f"Buffer loaded: dataset indices {self.buffer_start_row} to {self.buffer_end_row-1}")
    
    def _load_buffer_for_index(self, idx: int):
        """Load buffer that contains the specified index."""
        # Calculate which buffer chunk this index should be in
        buffer_chunk = idx // self.buffer_size
        start_row = buffer_chunk * self.buffer_size + 1  # Spark row numbers are 1-indexed
        end_row = min(start_row + self.buffer_size - 1, self.total_rows)
        
        logger.info(f"Loading buffer for index {idx}: rows {start_row} to {end_row}")
        
        # Filter using the pre-computed row numbers
        buffer_df = self.spark_df.filter(
            (self.spark_df.row_num >= start_row) & 
            (self.spark_df.row_num <= end_row)
        ).drop("row_num")  # Remove the row number column
        
        # Convert this buffer to pandas
        buffer_pandas = buffer_df.toPandas()
        
        # Preprocess and store in buffer
        self.buffer_data = self._preprocess_batch(buffer_pandas)
        self.buffer_start_row = start_row - 1  # Convert to 0-indexed for dataset access
        self.buffer_end_row = self.buffer_start_row + len(buffer_pandas)
        
        logger.info(f"Buffer loaded: dataset indices {self.buffer_start_row} to {self.buffer_end_row-1}")
    
    def _preprocess_batch(self, batch_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess a batch of data."""
        batch_size = len(batch_df)
        
        # Handle numerical features
        if self.numerical_features:
            numerical_data = batch_df[self.numerical_features].values.astype(np.float32)
            if self.numerical_scaler:
                numerical_data = self.numerical_scaler.transform(numerical_data)
            numerical_data = torch.tensor(numerical_data, dtype=torch.float32)
        else:
            numerical_data = torch.empty(batch_size, 0, dtype=torch.float32)
        
        # Handle categorical features
        if self.categorical_features:
            categorical_values = []
            for feature in self.categorical_features:
                if feature in self.categorical_encoders:
                    encoded = self.categorical_encoders[feature].transform(
                        batch_df[feature].fillna('unknown')
                    )
                else:
                    # If encoder not available, use simple integer encoding
                    encoded = pd.Categorical(batch_df[feature].fillna('unknown')).codes
                categorical_values.append(encoded)
            
            categorical_data = torch.tensor(np.column_stack(categorical_values), dtype=torch.long)
        else:
            categorical_data = torch.empty(batch_size, 0, dtype=torch.long)
        
        # Handle target
        target_data = torch.tensor(
            batch_df[self.target_column].values, dtype=torch.float32
        ).unsqueeze(1)
        
        return numerical_data, categorical_data, target_data
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item by index - uses local buffer for efficiency."""
        # Check if the requested index is outside current buffer
        if idx < self.buffer_start_row or idx >= self.buffer_end_row:
            self._load_buffer_for_index(idx)
        
        # Calculate local index within the buffer
        local_idx = idx - self.buffer_start_row
        
        numerical_data, categorical_data, target_data = self.buffer_data
        
        return (
            numerical_data[local_idx],
            categorical_data[local_idx],
            target_data[local_idx]
        )


class S3ParquetDataLoader(BaseDataLoader):
    """
    Data loader for S3 Parquet files using Spark DataFrames.
    Takes a single data path and mode ('train', 'val', 'test').
    No direct S3 access from PyTorch - much faster!
    """
    
    def __init__(self, data_path: str, mode: str, config: Dict[str, Any], spark_session=None):
        super().__init__(config)
        
        self.data_path = data_path
        self.mode = mode.lower()
        
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got: {mode}")
        
        self.numerical_features = config.get('numerical_features', [])
        self.categorical_features = config.get('categorical_features', [])
        self.target_column = config.get('target_column', 'target')
        
        # Feature preprocessing objects
        self.categorical_encoders = {}
        self.numerical_scaler = None
        self.feature_info = {}
        
        # Use provided Spark session or get existing one
        if spark_session:
            self.spark = spark_session
        else:
            self.spark = SparkSession.getActiveSession()
            if self.spark is None:
                raise ValueError("No active Spark session found. Please provide spark_session parameter.")
        
        # Load the data
        self._load_data()
    
    def _load_data(self):
        """Load data from the specified path."""
        logger.info(f"Loading {self.mode} data from {self.data_path}")
        
        # Read data with Spark and cache it
        self.data_df = self.spark.read.parquet(self.data_path)
        self.data_df.cache()
        
        # Create a view for SQL operations (only for train mode to build encoders)
        if self.mode == 'train':
            self.data_df.createOrReplaceTempView("train_data")
            self._build_feature_info()
    
    def _build_feature_info(self):
        """Build feature info and preprocessors (only called for train mode)."""
        logger.info("Building feature information and preprocessors...")
        
        # Handle categorical features
        categorical_vocab_sizes = []
        if self.categorical_features:
            for feature in self.categorical_features:
                # Get unique values count using SQL
                unique_count = self.spark.sql(f"SELECT COUNT(DISTINCT {feature}) as count FROM train_data").collect()[0]['count']
                categorical_vocab_sizes.append(unique_count + 1)  # +1 for unknown values
                
                # Build label encoder
                unique_values = [row[feature] for row in self.spark.sql(f"SELECT DISTINCT {feature} FROM train_data").collect()]
                encoder = LabelEncoder()
                encoder.fit(unique_values + ['unknown'])  # Include unknown for unseen values
                self.categorical_encoders[feature] = encoder
        
        # Handle numerical features scaling
        if self.numerical_features:
            # Sample data for fitting scaler (to avoid memory issues)
            numerical_cols = ', '.join(self.numerical_features)
            sample_df = self.spark.sql(f"SELECT {numerical_cols} FROM train_data TABLESAMPLE (10 PERCENT)").toPandas()
            self.numerical_scaler = StandardScaler()
            self.numerical_scaler.fit(sample_df.values)
        
        self.feature_info = {
            'num_numerical_features': len(self.numerical_features),
            'num_categorical_features': len(self.categorical_features),
            'categorical_vocab_sizes': categorical_vocab_sizes,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'target_column': self.target_column
        }
        
        logger.info(f"Feature info: {self.feature_info}")
    

    
    def get_data_loader(self) -> DataLoader:
        """Return data loader for the specified mode."""
        dataset = SparkDataset(
            self.data_df,
            self.config,
            self.categorical_encoders,
            self.numerical_scaler
        )
        
        # Shuffle only for training
        shuffle = (self.mode == 'train')
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    

    
    def get_feature_info(self) -> Dict[str, Any]:
        """Return feature information."""
        return self.feature_info
    
    def save_preprocessors(self, path: str):
        """Save preprocessing objects."""
        preprocessors = {
            'categorical_encoders': self.categorical_encoders,
            'numerical_scaler': self.numerical_scaler,
            'feature_info': self.feature_info
        }
        
        with open(path, 'wb') as f:
            pickle.dump(preprocessors, f)
    
    def load_preprocessors(self, path: str):
        """Load preprocessing objects."""
        with open(path, 'rb') as f:
            preprocessors = pickle.load(f)
        
        self.categorical_encoders = preprocessors['categorical_encoders']
        self.numerical_scaler = preprocessors['numerical_scaler']
        self.feature_info = preprocessors['feature_info'] 
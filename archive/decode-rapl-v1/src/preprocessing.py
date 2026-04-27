"""
DECODE-RAPL Preprocessing
Time-delay embedding, normalization, and sequence windowing
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Optional
import warnings


class DelayEmbedding:
    """
    Time-delay embedding for reconstructing attractor from partial observations

    Based on Takens' embedding theorem:
    h(t) = [y(t), y(t-τ), y(t-2τ), ..., y(t-(d-1)τ)]

    Where:
    - y(t) is the observed variable (CPU usage)
    - τ (tau) is the time delay
    - d is the number of delays (embedding dimension)
    """

    def __init__(self, tau: int = 1, d: int = 25, n_features: int = 1):
        """
        Args:
            tau: Time delay in timesteps
            d: Number of delays (embedding dimension)
            n_features: Number of input features (default: 1 for single variable)
        """
        self.tau = tau
        self.d = d
        self.n_features = n_features
        self.embedding_dim = d * n_features  # Total embedding dimension

    def embed(self, time_series: np.ndarray) -> np.ndarray:
        """
        Create delay-embedded representation from time series

        Args:
            time_series: 1D array of shape (T,) or 2D array of shape (T, n_vars)

        Returns:
            Embedded array of shape (T_valid, d * n_vars)
            where T_valid = T - (d-1) * tau
        """
        if time_series.ndim == 1:
            time_series = time_series.reshape(-1, 1)

        T, n_vars = time_series.shape
        valid_length = T - (self.d - 1) * self.tau

        if valid_length <= 0:
            raise ValueError(
                f"Time series too short for embedding. "
                f"Need at least {(self.d - 1) * self.tau + 1} samples, "
                f"got {T}"
            )

        # Create Hankel matrix
        # For each time point t, create vector [y(t), y(t-τ), ..., y(t-(d-1)τ)]
        embedded = np.zeros((valid_length, self.d * n_vars))

        for i in range(valid_length):
            for j in range(self.d):
                offset = i + (self.d - 1 - j) * self.tau
                embedded[i, j*n_vars:(j+1)*n_vars] = time_series[offset]

        return embedded

    def get_output_dim(self, input_dim: int = 1) -> int:
        """Get output dimension after embedding"""
        return self.d * input_dim


class RAPLDataset(Dataset):
    """PyTorch Dataset for RAPL data with delay embedding"""

    def __init__(
        self,
        data: pd.DataFrame,
        config: dict,
        mode: str = 'train',
        usage_scaler: Optional[MinMaxScaler] = None,
        power_scaler: Optional[MinMaxScaler] = None
    ):
        """
        Args:
            data: DataFrame with columns [timestamp, machine_id, cpu_usage, power]
                  OR [timestamp, machine_id, user_percent, system_percent, context_switches, power]
            config: Configuration dictionary
            mode: 'train', 'val', or 'test'
            usage_scaler: Fitted scaler for CPU usage features (for val/test)
            power_scaler: Fitted scaler for power (for val/test)
        """
        self.data = data.copy()
        self.config = config
        self.mode = mode

        # Extract config parameters
        self.tau = config['embedding']['tau']
        self.d = config['embedding']['d']
        self.window_size = config['preprocessing']['window_size']
        self.stride = config['preprocessing']['stride']

        # Determine feature columns
        self.feature_columns = config['preprocessing'].get('feature_columns', ['cpu_usage'])
        self.n_features = len(self.feature_columns)

        # Initialize delay embedding with multiple features
        self.embedder = DelayEmbedding(tau=self.tau, d=self.d, n_features=self.n_features)

        # Normalize data
        if mode == 'train':
            # Fit scalers on training data
            self.usage_scaler = MinMaxScaler() if config['preprocessing']['normalize_usage'] else None
            self.power_scaler = MinMaxScaler() if config['preprocessing']['normalize_power'] else None

            if self.usage_scaler:
                # Fit and transform all feature columns
                self.data[self.feature_columns] = self.usage_scaler.fit_transform(
                    self.data[self.feature_columns]
                )
            if self.power_scaler:
                self.data['power'] = self.power_scaler.fit_transform(
                    self.data[['power']]
                )
        else:
            # Use provided scalers for val/test
            self.usage_scaler = usage_scaler
            self.power_scaler = power_scaler

            if self.usage_scaler:
                # Transform all feature columns
                self.data[self.feature_columns] = self.usage_scaler.transform(
                    self.data[self.feature_columns]
                )
            if self.power_scaler:
                self.data['power'] = self.power_scaler.transform(
                    self.data[['power']]
                )

        # Group by machine_id to maintain temporal coherence
        self.machine_groups = {}
        for machine_id in self.data['machine_id'].unique():
            machine_data = self.data[self.data['machine_id'] == machine_id].copy()
            machine_data = machine_data.sort_values('timestamp').reset_index(drop=True)
            self.machine_groups[machine_id] = machine_data

        # Create sequences with delay embedding
        self.sequences = []
        self._create_sequences()

    def _create_sequences(self):
        """Create windowed sequences with delay embedding"""
        for machine_id, machine_data in self.machine_groups.items():
            # Extract feature time series (single or multi-feature)
            features = machine_data[self.feature_columns].values  # Shape: (T, n_features)
            power = machine_data['power'].values

            # Apply delay embedding
            # Input shape: (T, n_features) or (T, 1) if single feature
            # Output shape: (T_valid, d * n_features)
            embedded_features = self.embedder.embed(features)

            # Number of valid samples after embedding
            n_embedded = len(embedded_features)

            # Account for LSTM window size
            if n_embedded < self.window_size:
                warnings.warn(
                    f"Machine {machine_id} has {n_embedded} embedded samples, "
                    f"less than window_size {self.window_size}. Skipping."
                )
                continue

            # Create sliding windows
            for i in range(0, n_embedded - self.window_size + 1, self.stride):
                # Embedded feature sequence
                # Shape: (window_size, d * n_features)
                embedding_seq = embedded_features[i:i + self.window_size]

                # Corresponding power targets (align with end of embedding)
                # Offset accounts for samples lost during embedding
                power_offset = (self.d - 1) * self.tau
                power_seq = power[power_offset + i:power_offset + i + self.window_size]

                # Target is the final power value in the window (next-step prediction)
                target_power = power_seq[-1]

                self.sequences.append({
                    'embedding': embedding_seq,  # Shape: (window_size, d * n_features)
                    'power': target_power,        # Scalar
                    'machine_id': machine_id
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        return {
            'embedding': torch.FloatTensor(seq['embedding']),
            'power': torch.FloatTensor([seq['power']]),
            'machine_id': seq['machine_id']
        }


def load_and_split_data(
    csv_path: str,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CSV data and split into train/val/test

    Ensures cross-machine validation: val set has at least one complete machine
    that's not in training

    Args:
        csv_path: Path to CSV file
        config: Configuration dictionary

    Returns:
        (train_df, val_df, test_df)
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Parse timestamp if it's a string
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded {len(df)} samples from {df['machine_id'].nunique()} machines")

    # Get unique machines
    machines = df['machine_id'].unique()
    n_machines = len(machines)

    if n_machines < 3:
        warnings.warn(
            f"Only {n_machines} machines available. "
            f"Cross-machine validation may not be meaningful."
        )

    # Shuffle machines for random split
    np.random.shuffle(machines)

    # Split machines for cross-machine validation
    train_split = config['preprocessing']['train_split']
    val_split = config['preprocessing']['val_split']

    n_train = max(1, int(n_machines * train_split))
    n_val = max(1, int(n_machines * val_split))

    train_machines = machines[:n_train]
    val_machines = machines[n_train:n_train + n_val]
    test_machines = machines[n_train + n_val:]

    # If not enough machines, fall back to time-based split on all data
    if len(test_machines) == 0:
        warnings.warn("Not enough machines for cross-machine split. Using time-based split.")
        return _time_based_split(df, config)

    # Create splits
    train_df = df[df['machine_id'].isin(train_machines)].copy()
    val_df = df[df['machine_id'].isin(val_machines)].copy()
    test_df = df[df['machine_id'].isin(test_machines)].copy()

    print(f"Split: Train={len(train_df)} ({len(train_machines)} machines), "
          f"Val={len(val_df)} ({len(val_machines)} machines), "
          f"Test={len(test_df)} ({len(test_machines)} machines)")

    return train_df, val_df, test_df


def _time_based_split(
    df: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fallback: time-based split when not enough machines"""
    df = df.sort_values('timestamp').reset_index(drop=True)

    train_split = config['preprocessing']['train_split']
    val_split = config['preprocessing']['val_split']

    n = len(df)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_df = df[:n_train].copy()
    val_df = df[n_train:n_train + n_val].copy()
    test_df = df[n_train + n_val:].copy()

    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test

    Args:
        train_df, val_df, test_df: Data splits
        config: Configuration dictionary

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Create datasets
    train_dataset = RAPLDataset(train_df, config, mode='train')
    val_dataset = RAPLDataset(
        val_df, config, mode='val',
        usage_scaler=train_dataset.usage_scaler,
        power_scaler=train_dataset.power_scaler
    )
    test_dataset = RAPLDataset(
        test_df, config, mode='test',
        usage_scaler=train_dataset.usage_scaler,
        power_scaler=train_dataset.power_scaler
    )

    print(f"Dataset sizes: Train={len(train_dataset)}, "
          f"Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Create dataloaders
    batch_size = config['training']['batch_size']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues with CUDA
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Store scalers in config for later use
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'scalers': {
            'usage': train_dataset.usage_scaler,
            'power': train_dataset.power_scaler
        },
        'embedder': train_dataset.embedder
    }

    return dataloaders


if __name__ == "__main__":
    # Test preprocessing pipeline
    import sys
    sys.path.append('.')
    from src.utils import load_config, generate_synthetic_data, set_seed

    print("Testing preprocessing pipeline...")

    # Load config
    config = load_config()
    set_seed(config['seed'])

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(
        num_machines=3,
        duration_hours=0.05,
        sampling_rate_ms=1,
        output_path="data/test_synthetic.csv"
    )

    # Test delay embedding
    print("\n2. Testing delay embedding...")
    embedder = DelayEmbedding(tau=1, d=25)
    usage = df[df['machine_id'] == 'machine_0']['cpu_usage'].values[:1000]
    embedded = embedder.embed(usage)
    print(f"Original shape: {usage.shape}, Embedded shape: {embedded.shape}")

    # Test data loading and splitting
    print("\n3. Testing data loading and splitting...")
    train_df, val_df, test_df = load_and_split_data("data/test_synthetic.csv", config)

    # Test dataset creation
    print("\n4. Testing dataset creation...")
    dataloaders = create_dataloaders(train_df, val_df, test_df, config)

    # Test batch retrieval
    print("\n5. Testing batch retrieval...")
    batch = next(iter(dataloaders['train']))
    print(f"Batch embedding shape: {batch['embedding'].shape}")
    print(f"Batch power shape: {batch['power'].shape}")
    print(f"Machine IDs: {batch['machine_id'][:5]}")

    print("\nPreprocessing pipeline test completed successfully!")

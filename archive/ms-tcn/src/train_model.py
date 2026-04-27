#!/usr/bin/env python3
"""
MS-TCN Model Training for CPU Power Prediction

Trains a Multi-Scale Temporal Convolutional Network to predict CPU power consumption
from system metrics using normalized, scale-independent features.

Usage:
    # Step 1: Preprocess raw data to add normalized features
    python scripts/preprocess_data.py data/training_raw.csv data/training_normalized.csv

    # Step 2: Train model on preprocessed data
    python train_model.py --data data/training_normalized.csv --output models/model.pth
    python train_model.py --data data1.csv data2.csv data3.csv --output models/model.pth
    python train_model.py --data data/training_normalized.csv --batch-size 512 --epochs 30

Note:
    - Data must be preprocessed with scripts/preprocess_data.py first to add normalized features
    - The best model during training is saved as 'best_model.pth' in the same
      directory as --output. For example, if --output is models/my_model.pth,
      the best model will be saved to models/best_model.pth.
    - Model trains on all available RAPL targets (typically package + dram)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tqdm import tqdm


# Feature columns for VM portability using CPU time
# CPU time features naturally scale with core count (0-4 on 4c, 0-20 on 20c)
FEATURE_COLUMNS = [
    # CPU time (seconds/second) - naturally encodes system scale
    # 50% on 20c = 10 sec/sec, 50% on 4c = 2 sec/sec
    'cpu_user_sec', 'cpu_system_sec', 'cpu_idle_sec',
    'cpu_iowait_sec', 'cpu_irq_sec', 'cpu_softirq_sec',

    # System activity (absolute counts per second)
    'interrupts_sec',
    'context_switches_sec',
    'page_faults_sec',
    'running_processes',

    # Memory ratios (0-1, scale-independent)
    'memory_used_ratio',
    'memory_cached_ratio',
    'memory_free_ratio',
    'swap_used_ratio',

    # Removed features:
    # - num_cores, memory_total_gb, swap_total_gb: zero variance when training on single system
    # - CPU percentages: replaced with absolute CPU time for VM portability
    # - Per-core normalized metrics: replaced with absolute counts
]

# RAPL power targets - auto-detected from preprocessed data
# Model will train on whatever power domains are available in the dataset
# Currently training package power only (DRAM requires memory bandwidth features)
# Note: rapl_core_power may not be available on all CPUs
AVAILABLE_RAPL_TARGETS = ['rapl_package_power']  # DRAM removed - see docs/joint_package_dram_model_plan.md

# Feature categories for selective scaling
# CPU time and absolute metrics are unbounded, need StandardScaler
SCALE_FEATURES = [
    # CPU time (0-20 range on 20-core, needs normalization)
    'cpu_user_sec', 'cpu_system_sec', 'cpu_idle_sec',
    'cpu_iowait_sec', 'cpu_irq_sec', 'cpu_softirq_sec',

    # System activity (unbounded counts)
    'interrupts_sec',
    'context_switches_sec',
    'page_faults_sec',
    'running_processes'
]

# Features that should NOT be scaled (already bounded 0-1)
NO_SCALE_FEATURES = [
    # Memory ratios (0-1, already bounded)
    'memory_used_ratio', 'memory_cached_ratio',
    'memory_free_ratio', 'swap_used_ratio'
]


class DilatedTemporalBlock(nn.Module):
    """Dilated temporal convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class MultiScaleConv(nn.Module):
    """Multi-scale convolutional layer with different kernel sizes."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Distribute channels evenly across 3 branches
        branch1_channels = out_channels // 3
        branch2_channels = out_channels // 3
        branch3_channels = out_channels - branch1_channels - branch2_channels  # Handles remainder

        self.branch3 = nn.Conv1d(in_channels, branch1_channels, kernel_size=3, padding=1)
        self.branch5 = nn.Conv1d(in_channels, branch2_channels, kernel_size=5, padding=2)
        self.branch7 = nn.Conv1d(in_channels, branch3_channels, kernel_size=7, padding=3)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)

        out = torch.cat([b3, b5, b7], dim=1)
        out = self.bn(out)
        out = self.relu(out)

        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        # Transpose for attention: (batch, seq_len, channels)
        x = x.transpose(1, 2)

        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        out = self.norm(x + attn_out)

        # Transpose back: (batch, channels, seq_len)
        out = out.transpose(1, 2)

        return out


class AttentionPooling(nn.Module):
    """Learnable attention-based pooling to replace global average pooling.

    Instead of uniform averaging across all timesteps, this learns which
    timesteps are important for the final prediction. This provides a gradient
    signal that encourages the attention mechanism to maintain meaningful
    temporal structure.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Learn attention weights for each timestep
        # Linear layer maps each timestep's feature vector to a scalar importance score
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, channels, timesteps)
        # Transpose to (batch, timesteps, channels) for attention computation
        x_t = x.transpose(1, 2)  # (batch, timesteps, channels)

        # Compute importance score for each timestep
        scores = self.attention_weights(x_t)  # (batch, timesteps, 1)

        # Normalize scores across timesteps with softmax
        # This ensures weights sum to 1 for each batch
        weights = F.softmax(scores, dim=1)  # (batch, timesteps, 1)

        # Apply weighted sum: multiply each timestep by its importance weight
        # and sum across the temporal dimension
        weighted_features = (x_t * weights).sum(dim=1)  # (batch, channels)

        return weighted_features


class MSTCN(nn.Module):
    """Multi-Scale Temporal Convolutional Network for power prediction."""

    def __init__(self, num_features: int, num_targets: int, hidden_dim: int = 128,
                 num_heads: int = 8, dropout: float = 0.2):
        super().__init__()

        self.num_targets = num_targets

        # Multi-scale input convolution
        self.multiscale_conv = MultiScaleConv(num_features, hidden_dim)

        # Dilated temporal blocks with increasing dilation
        dilations = [1, 2, 4, 8, 16, 32]
        self.temporal_blocks = nn.ModuleList([
            DilatedTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=d, dropout=dropout)
            for d in dilations
        ])

        # Multi-head attention (num_heads must divide hidden_dim evenly)
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.attention = MultiHeadAttention(hidden_dim, num_heads=num_heads, dropout=dropout)

        # Attention-based pooling (learnable weighted sum to replace uniform averaging)
        self.attention_pool = AttentionPooling(hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        # Output heads for each power domain
        self.output_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(num_targets)
        ])

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Transpose for convolution: (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Multi-scale convolution
        x = self.multiscale_conv(x)

        # Dilated temporal blocks
        for block in self.temporal_blocks:
            x = block(x)

        # Multi-head attention
        x = self.attention(x)

        # Attention-based pooling (no squeeze needed - already returns (batch, channels))
        x = self.attention_pool(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)

        # Output heads
        outputs = [head(x) for head in self.output_heads]

        # Concatenate outputs
        out = torch.cat(outputs, dim=1)

        return out


class PowerDataset(Dataset):
    """Dataset for power prediction sequences."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class DataPreprocessor:
    """Preprocesses data for MS-TCN training."""

    def __init__(self, sequence_length: int = 64, stride: int = 1):
        self.sequence_length = sequence_length
        self.stride = stride
        # Selective feature scaler - only scales unbounded features
        self.feature_scaler = None  # Will be initialized in prepare_data
        self.target_scaler = StandardScaler()
        self.feature_columns = None
        self.target_columns = None

    def load_csv_files(self, csv_files: List[str]) -> pd.DataFrame:
        """Load and concatenate multiple CSV files."""
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"Loaded {len(df)} samples from {csv_file}")

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total samples: {len(combined_df)}")

        return combined_df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets from dataframe."""

        # Identify available feature columns
        available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
        self.feature_columns = available_features

        # Auto-detect available RAPL power targets
        available_targets = []
        print("\nDetecting available RAPL power targets:")
        for target in AVAILABLE_RAPL_TARGETS:
            if target in df.columns:
                # Check if column has actual data (not all NaN/empty)
                non_null_count = df[target].notna().sum()
                if non_null_count > 0:
                    available_targets.append(target)
                    print(f"  ✓ {target}: {non_null_count} valid samples")
                else:
                    print(f"  ⚠ {target}: column exists but contains no data (all NaN)")
            else:
                print(f"  ✗ {target}: not found in dataset")

        if not available_targets:
            raise ValueError("No valid power targets found in data. Cannot train without power measurements.")

        self.target_columns = available_targets
        print(f"\nTraining configuration:")
        print(f"  Features: {len(available_features)} columns")
        print(f"  Targets: {len(available_targets)} power domain(s): {available_targets}")

        # Filter out initialization rows (where all detected power domains are 0)
        df_filtered = df.copy()
        if len(available_targets) > 0:
            # Create mask for rows where ALL power targets are 0
            zero_mask = (df_filtered[available_targets[0]] == 0)
            for target in available_targets[1:]:
                zero_mask = zero_mask & (df_filtered[target] == 0)

            num_zeros = zero_mask.sum()
            if num_zeros > 0:
                print(f"\nFiltering out {num_zeros} initialization rows (all power domains = 0)")
                df_filtered = df_filtered[~zero_mask].reset_index(drop=True)

        # Extract features and targets
        X = df_filtered[available_features].values
        y = df_filtered[available_targets].values

        # Print raw data statistics BEFORE normalization
        print("\nRaw Power Statistics (Watts):")
        for i, target_name in enumerate(available_targets):
            target_data = y[:, i]
            print(f"  {target_name}:")
            print(f"    Min:  {np.min(target_data):.3f}W")
            print(f"    Max:  {np.max(target_data):.3f}W")
            print(f"    Mean: {np.mean(target_data):.3f}W")
            print(f"    Std:  {np.std(target_data):.3f}W")

            # Count very low values
            low_count = np.sum(target_data < 1.0)
            low_pct = (low_count / len(target_data)) * 100
            print(f"    Samples <1W: {low_count} ({low_pct:.1f}%)")

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        # Selective scaling: only scale unbounded features
        # Build column indices for each feature category
        scale_indices = [available_features.index(f) for f in SCALE_FEATURES if f in available_features]
        no_scale_indices = [available_features.index(f) for f in NO_SCALE_FEATURES if f in available_features]

        print(f"\nFeature scaling strategy:")
        print(f"  Features to scale (unbounded): {len(scale_indices)}")
        print(f"    {[available_features[i] for i in scale_indices]}")
        print(f"  Features preserved (bounded/critical): {len(no_scale_indices)}")
        print(f"    {[available_features[i] for i in no_scale_indices]}")

        # Create ColumnTransformer for selective scaling
        if len(scale_indices) > 0:
            transformers = [
                ('scaler', StandardScaler(), scale_indices),
                ('passthrough', 'passthrough', no_scale_indices)
            ]
        else:
            # No features need scaling - just passthrough all
            transformers = [('passthrough', 'passthrough', list(range(len(available_features))))]

        self.feature_scaler = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop any features not in either category
        )

        X = self.feature_scaler.fit_transform(X)
        y = self.target_scaler.fit_transform(y)

        # Verify critical features are preserved
        num_cores_idx_in_no_scale = [f for f in NO_SCALE_FEATURES if f in available_features].index('num_cores') if 'num_cores' in available_features else -1
        if num_cores_idx_in_no_scale >= 0:
            # num_cores is in the passthrough section (after scaled features)
            num_cores_col_in_X = len(scale_indices) + num_cores_idx_in_no_scale
            num_cores_values = X[:, num_cores_col_in_X]
            print(f"\n✓ Verification: num_cores preserved correctly")
            print(f"  Value range: {num_cores_values.min():.1f} - {num_cores_values.max():.1f}")
            print(f"  Mean: {num_cores_values.mean():.1f}")
            if num_cores_values.std() < 0.01:
                print(f"  (Constant value - this is expected for single-system training)")

        return X, y

    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from time series data."""

        sequences = []
        targets = []

        for i in range(0, len(X) - self.sequence_length, self.stride):
            seq = X[i:i + self.sequence_length]
            target = y[i + self.sequence_length - 1]  # Predict the last timestep

            sequences.append(seq)
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)

        print(f"Created {len(sequences)} sequences of length {self.sequence_length}")

        return sequences, targets

    def split_data(self, sequences: np.ndarray, targets: np.ndarray,
                   train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Split data into train/val/test sets maintaining temporal order."""

        n = len(sequences)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_seq = sequences[:train_end]
        train_tgt = targets[:train_end]

        val_seq = sequences[train_end:val_end]
        val_tgt = targets[train_end:val_end]

        test_seq = sequences[val_end:]
        test_tgt = targets[val_end:]

        print(f"Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")

        # Print variance statistics for validation set (normalized values)
        print("\nValidation Set Statistics (normalized):")
        for i, target_name in enumerate(self.target_columns):
            train_var = np.var(train_tgt[:, i])
            val_var = np.var(val_tgt[:, i])
            val_mean = np.mean(val_tgt[:, i])
            val_std = np.std(val_tgt[:, i])

            print(f"  {target_name}:")
            print(f"    Train variance: {train_var:.4f}")
            print(f"    Val variance:   {val_var:.4f}")
            print(f"    Val mean:       {val_mean:.4f}")
            print(f"    Val std:        {val_std:.4f}")

            if val_var < 0.01:
                print(f"    WARNING: Very low variance in validation set!")

        return (train_seq, train_tgt), (val_seq, val_tgt), (test_seq, test_tgt)

    def split_and_sequence(self, X: np.ndarray, y: np.ndarray,
                          train_ratio: float = 0.7, val_ratio: float = 0.15,
                          split_mode: str = "random"):
        """
        Split raw data first, then create sequences separately for each split.
        This prevents sequence overlap between train/val/test and ensures
        better distribution of workload patterns across splits.

        Args:
            X: Raw features (samples x features)
            y: Raw targets (samples x targets)
            train_ratio: Fraction for training (default 0.7)
            val_ratio: Fraction for validation (default 0.15)
            split_mode: "random" or "temporal"

        Returns:
            (train_seq, train_tgt), (val_seq, val_tgt), (test_seq, test_tgt)
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split_mode == "random":
            # Random split: shuffle indices first
            print(f"Using random split (better distribution across workload phases)")
            indices = np.random.permutation(n)
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

            # Sort back to maintain temporal order within each split
            train_indices = np.sort(train_indices)
            val_indices = np.sort(val_indices)
            test_indices = np.sort(test_indices)

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            X_test, y_test = X[test_indices], y[test_indices]
        else:
            # Temporal split: maintain time order
            print(f"Using temporal split (testing on future data)")
            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]

        # Create sequences separately for each split
        print(f"  Creating training sequences...")
        train_seq, train_tgt = self.create_sequences(X_train, y_train)

        print(f"  Creating validation sequences...")
        val_seq, val_tgt = self.create_sequences(X_val, y_val)

        print(f"  Creating test sequences...")
        test_seq, test_tgt = self.create_sequences(X_test, y_test)

        print(f"Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")

        # Print variance statistics for validation set (normalized values)
        print("\nValidation Set Statistics (normalized):")
        for i, target_name in enumerate(self.target_columns):
            train_var = np.var(train_tgt[:, i])
            val_var = np.var(val_tgt[:, i])
            val_mean = np.mean(val_tgt[:, i])
            val_std = np.std(val_tgt[:, i])

            print(f"  {target_name}:")
            print(f"    Train variance: {train_var:.4f}")
            print(f"    Val variance:   {val_var:.4f}")
            print(f"    Val mean:       {val_mean:.4f}")
            print(f"    Val std:        {val_std:.4f}")

            if val_var < 0.01:
                print(f"    WARNING: Very low variance in validation set!")

        return (train_seq, train_tgt), (val_seq, val_tgt), (test_seq, test_tgt)


class MultiDomainHuberLoss(nn.Module):
    """Multi-domain Huber loss with configurable weights."""

    def __init__(self, weights: Optional[List[float]] = None, delta: float = 1.0):
        super().__init__()
        self.weights = weights or [0.5, 0.3, 0.2]  # package, core, dram
        self.delta = delta

    def forward(self, predictions, targets):
        losses = []

        for i in range(predictions.shape[1]):
            pred = predictions[:, i]
            target = targets[:, i]

            diff = torch.abs(pred - target)
            loss = torch.where(diff < self.delta,
                              0.5 * diff ** 2,
                              self.delta * (diff - 0.5 * self.delta))

            weight = self.weights[i] if i < len(self.weights) else 1.0 / predictions.shape[1]
            losses.append(weight * loss.mean())

        return sum(losses)


class Trainer:
    """Trainer for MS-TCN model."""

    def __init__(self, model: nn.Module, device: torch.device, config: Dict, preprocessor, best_model_path: str = 'best_model.pth'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.preprocessor = preprocessor
        self.best_model_path = best_model_path

        # Loss and optimizer
        loss_weights = config.get('loss_weights', [0.5, 0.5])  # package, core
        self.criterion = MultiDomainHuberLoss(weights=loss_weights)

        self.optimizer = AdamW(model.parameters(),
                              lr=config['learning_rate'],
                              weight_decay=config.get('weight_decay', 1e-5))

        self.scheduler = CosineAnnealingLR(self.optimizer,
                                          T_max=config['epochs'],
                                          eta_min=1e-6)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_r2': []
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc="Training")
        for sequences, targets in pbar:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(sequences)
            loss = self.criterion(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(sequences)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        # Denormalize predictions and targets to actual Watts
        all_preds_watts = self.preprocessor.target_scaler.inverse_transform(all_preds)
        all_targets_watts = self.preprocessor.target_scaler.inverse_transform(all_targets)

        # Calculate metrics on actual power values (Watts)
        metrics = {}
        for i in range(all_preds.shape[1]):
            mae = mean_absolute_error(all_targets_watts[:, i], all_preds_watts[:, i])
            mape = mean_absolute_percentage_error(all_targets_watts[:, i], all_preds_watts[:, i])
            r2 = r2_score(all_targets_watts[:, i], all_preds_watts[:, i])

            metrics[f'mae_{i}'] = mae
            metrics[f'mape_{i}'] = mape
            metrics[f'r2_{i}'] = r2

        avg_loss = total_loss / len(val_loader)

        return avg_loss, metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""

        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, metrics = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(metrics.get('mae_0', 0))
            self.history['val_r2'].append(metrics.get('r2_0', 0))

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Print per-target metrics (package, core)
            target_names = self.preprocessor.target_columns
            avg_r2 = 0.0
            for i, target_name in enumerate(target_names):
                mae = metrics.get(f'mae_{i}', 0)
                r2 = metrics.get(f'r2_{i}', 0)
                avg_r2 += r2

                # Add improvement indicator
                indicator = ""
                if epoch > 0:
                    prev_mae = self.history['val_mae'][-2] if len(self.history['val_mae']) > 1 else mae
                    if mae < prev_mae:
                        indicator = " ↓"
                    elif mae > prev_mae:
                        indicator = " ↑"

                print(f"  {target_name}: MAE={mae:.3f}W{indicator}, R²={r2:.4f}")

            avg_r2 /= len(target_names)
            print(f"  Average R²: {avg_r2:.4f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model
                self.save_checkpoint(self.best_model_path)
                print("Saved best model!")
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.config.get('patience', 20):
                    print(f"\nEarly stopping after {epoch + 1} epochs")
                    break

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'preprocessor': {
                # Serialize entire feature_scaler (ColumnTransformer or StandardScaler)
                'feature_scaler': pickle.dumps(self.preprocessor.feature_scaler),
                'target_scaler_mean': self.preprocessor.target_scaler.mean_,
                'target_scaler_scale': self.preprocessor.target_scaler.scale_,
                'feature_columns': self.preprocessor.feature_columns,
                'target_columns': self.preprocessor.target_columns,
                'sequence_length': self.preprocessor.sequence_length
            }
        }, path)


def plot_training_history(history: Dict, output_dir: str):
    """Plot training history."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MAE
    axes[0, 1].plot(history['val_mae'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].grid(True, alpha=0.3)

    # R2
    axes[1, 0].plot(history['val_r2'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Validation R² Score')
    axes[1, 0].grid(True, alpha=0.3)

    # Clear unused subplot
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {output_dir}/training_history.png")


def main():
    parser = argparse.ArgumentParser(
        description="Train MS-TCN model for CPU power prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data', nargs='+', required=True, help='CSV data files')
    parser.add_argument('--output', type=str, default='model.pth', help='Output model path')
    parser.add_argument('--sequence-length', type=int, default=64, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (increased for better convergence)')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--split-mode', type=str, default='random', choices=['random', 'temporal'],
                       help='Data split mode: random (better distribution) or temporal (test on future data)')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    print("\nLoading data...")
    preprocessor = DataPreprocessor(sequence_length=args.sequence_length)
    df = preprocessor.load_csv_files(args.data)

    print("\nPreparing features and targets...")
    X, y = preprocessor.prepare_data(df)

    print(f"\nSplitting data and creating sequences (mode: {args.split_mode})...")
    train_data, val_data, test_data = preprocessor.split_and_sequence(X, y, split_mode=args.split_mode)

    # Create datasets and loaders
    train_dataset = PowerDataset(*train_data)
    val_dataset = PowerDataset(*val_data)
    test_dataset = PowerDataset(*test_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    num_features = len(preprocessor.feature_columns)
    num_targets = len(preprocessor.target_columns)

    print(f"\nCreating model with {num_features} features and {num_targets} targets...")
    model = MSTCN(num_features=num_features, num_targets=num_targets, hidden_dim=args.hidden_dim)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training configuration
    config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'loss_weights': [0.5, 0.5] if num_targets == 2 else [0.5, 0.3, 0.2]
    }

    # Determine output paths
    output_path = Path(args.output_dir) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Best model goes to same directory as final model
    best_model_path = output_path.parent / 'best_model.pth'

    # Train model
    trainer = Trainer(model, device, config, preprocessor, best_model_path=str(best_model_path))
    trainer.train(train_loader, val_loader)

    # Save final model and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'preprocessor': {
            # Serialize entire feature_scaler (ColumnTransformer or StandardScaler)
            'feature_scaler': pickle.dumps(preprocessor.feature_scaler),
            'target_scaler_mean': preprocessor.target_scaler.mean_,
            'target_scaler_scale': preprocessor.target_scaler.scale_,
            'feature_columns': preprocessor.feature_columns,
            'target_columns': preprocessor.target_columns,
            'sequence_length': args.sequence_length
        },
        'config': config,
        'history': trainer.history
    }, output_path)

    print(f"\nFinal model saved to: {output_path}")
    print(f"Best model saved to: {best_model_path}")

    # Plot training history
    plot_training_history(trainer.history, args.output_dir)

    # Save training summary
    summary = {
        'best_val_loss': float(trainer.best_val_loss),
        'final_train_loss': float(trainer.history['train_loss'][-1]),
        'final_val_loss': float(trainer.history['val_loss'][-1]),
        'final_val_mae': float(trainer.history['val_mae'][-1]),
        'final_val_r2': float(trainer.history['val_r2'][-1]),
        'num_features': num_features,
        'num_targets': num_targets,
        'feature_columns': preprocessor.feature_columns,
        'target_columns': preprocessor.target_columns
    }

    with open(f"{args.output_dir}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary saved to {args.output_dir}/training_summary.json")
    print("\nTraining Summary:")
    print(f"  Best Val Loss: {summary['best_val_loss']:.4f}")
    print(f"  Final Val MAE: {summary['final_val_mae']:.4f}")
    print(f"  Final Val R²: {summary['final_val_r2']:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LSTM-based Power Prediction Model for VMs

This module provides end-to-end training and inference for predicting per-VM
power consumption from CPU usage time series and vCPU count.

Features:
- Multivariate LSTM with temporal dependencies
- Handles variable VM sizes (vCPU count as static feature)
- Train on multi-VM bare metal data, inference in VMs
- ONNX export support for deployment
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Error: PyTorch not installed. Run: pip install torch")
    sys.exit(1)

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except ImportError:
    print("Error: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Generation (for testing)
# ============================================================================

def generate_synthetic_data(num_samples: int = 10000, output_file: str = 'multi_vm_training_data.csv'):
    """
    Generate synthetic multi-VM power data for testing

    Simulates non-linear power-CPU relationship with temporal dependencies:
    - Power = base_power + (cpu_usage^1.5) * scaling_factor + noise + thermal_lag
    - Different VM sizes (2, 4, 10 vCPUs) with scaled base power

    Args:
        num_samples: Total number of samples to generate
        output_file: Output CSV file path
    """
    logger.info("Generating synthetic multi-VM data with %d samples...", num_samples)

    vm_configs = [
        {'vm_id': 'vm-2vcpu', 'vcpus': 2},
        {'vm_id': 'vm-4vcpu', 'vcpus': 4},
        {'vm_id': 'vm-10vcpu', 'vcpus': 10},
    ]

    data = []
    samples_per_vm = num_samples // len(vm_configs)

    for vm_config in vm_configs:
        vm_id = vm_config['vm_id']
        vcpus = vm_config['vcpus']

        # Base power scales with vCPUs (idle power)
        base_power = 10 + vcpus * 2.5

        # Scaling factor for CPU-dependent power
        scaling_factor = vcpus * 1.8

        # Generate CPU usage with temporal correlation
        cpu_usage = []
        current_usage = np.random.uniform(10, 50)

        for i in range(samples_per_vm):
            # Random walk with drift
            delta = np.random.randn() * 5
            current_usage = np.clip(current_usage + delta, 5, 95)
            cpu_usage.append(current_usage)

        cpu_usage = np.array(cpu_usage)

        # Simulate non-linear power with thermal lag
        power = []
        thermal_state = 0  # Simulates thermal accumulation

        for i, usage in enumerate(cpu_usage):
            # Non-linear CPU-power relationship (power grows faster at high usage)
            instant_power = base_power + (usage / 100) ** 1.5 * scaling_factor

            # Thermal lag: power depends on recent history
            thermal_state = 0.8 * thermal_state + 0.2 * instant_power

            # Add noise
            noise = np.random.randn() * 2

            final_power = thermal_state + noise
            power.append(max(base_power * 0.8, final_power))  # Min power threshold

        # Create dataframe for this VM
        for i in range(samples_per_vm):
            data.append({
                'timestamp': pd.Timestamp('2025-01-01') + pd.Timedelta(seconds=i),
                'vm_id': vm_id,
                'vcpus': vcpus,
                'cpu_usage': cpu_usage[i],
                'power': power[i]
            })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logger.info("Generated synthetic data saved to %s", output_file)
    logger.info("  Total samples: %d", len(df))
    logger.info("  VMs: %s", df['vm_id'].unique())
    logger.info("  Power range: %.2f - %.2f W", df['power'].min(), df['power'].max())

    return df


# ============================================================================
# Dataset
# ============================================================================

class PowerDataset(Dataset):
    """
    PyTorch Dataset for time series power prediction

    Creates sliding windows of (cpu_usage, vcpus) -> power
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Array of shape (num_samples, seq_len, num_features)
            targets: Array of shape (num_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ============================================================================
# Data Preprocessing
# ============================================================================

class DataPreprocessor:
    """Handle data loading, normalization, and sequence creation"""

    def __init__(self, seq_length: int = 60, max_vcpus: int = 64):
        """
        Args:
            seq_length: Length of input sequences (time steps)
            max_vcpus: Maximum expected vCPUs for normalization
        """
        self.seq_length = seq_length
        self.max_vcpus = max_vcpus

        # Scalers (fit on training data only)
        self.power_scaler = MinMaxScaler()
        self.fitted = False

    def load_data(self, csv_file: str) -> pd.DataFrame:
        """
        Load data from CSV

        Args:
            csv_file: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        logger.info("Loading data from %s...", csv_file)
        df = pd.read_csv(csv_file)

        # Validate columns
        required_cols = ['timestamp', 'vm_id', 'vcpus', 'cpu_usage', 'power']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # Sort by VM and timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['vm_id', 'timestamp']).reset_index(drop=True)

        logger.info("Loaded %d samples from %d VMs", len(df), df['vm_id'].nunique())
        logger.info("  vCPU counts: %s", sorted(df['vcpus'].unique()))

        return df

    def create_sequences(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences from dataframe

        Args:
            df: Input dataframe
            is_training: If True, fit scalers on this data

        Returns:
            Tuple of (sequences, targets)
            - sequences: (num_samples, seq_length, 2) for [cpu_usage, vcpus]
            - targets: (num_samples,) for power
        """
        sequences = []
        targets = []

        # Process each VM separately to maintain temporal order
        for vm_id in df['vm_id'].unique():
            vm_data = df[df['vm_id'] == vm_id].reset_index(drop=True)

            # Normalize features
            cpu_usage_norm = vm_data['cpu_usage'].values / 100.0  # [0, 1]
            vcpus_norm = vm_data['vcpus'].values / self.max_vcpus  # [0, 1]
            power = vm_data['power'].values.reshape(-1, 1)

            # Fit power scaler on training data
            if is_training and not self.fitted:
                if len(sequences) == 0:  # First VM
                    self.power_scaler.fit(power)
                else:
                    # Incrementally fit (accumulate data)
                    self.power_scaler.fit(
                        np.vstack([self.power_scaler.inverse_transform(
                            self.power_scaler.transform(power[:1])), power])
                    )

            # Normalize power
            power_norm = self.power_scaler.transform(power).flatten()

            # Create sliding windows
            for i in range(len(vm_data) - self.seq_length):
                # Input: [cpu_usage, vcpus] over seq_length time steps
                seq = np.column_stack([
                    cpu_usage_norm[i:i + self.seq_length],
                    vcpus_norm[i:i + self.seq_length]
                ])

                # Target: power at next time step
                target = power_norm[i + self.seq_length]

                sequences.append(seq)
                targets.append(target)

        if is_training:
            self.fitted = True

        sequences = np.array(sequences)
        targets = np.array(targets)

        logger.info("Created %d sequences of length %d", len(sequences), self.seq_length)

        return sequences, targets

    def denormalize_power(self, power_norm: np.ndarray) -> np.ndarray:
        """
        Denormalize predicted power values

        Args:
            power_norm: Normalized power values

        Returns:
            Original scale power values
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call create_sequences with is_training=True first.")

        power_norm = np.array(power_norm).reshape(-1, 1)
        return self.power_scaler.inverse_transform(power_norm).flatten()

    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test by time (chronological split per VM)

        Args:
            df: Input dataframe
            train_ratio: Fraction for training
            val_ratio: Fraction for validation (test = 1 - train - val)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_data = []
        val_data = []
        test_data = []

        for vm_id in df['vm_id'].unique():
            vm_data = df[df['vm_id'] == vm_id].reset_index(drop=True)
            n = len(vm_data)

            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_data.append(vm_data[:train_end])
            val_data.append(vm_data[train_end:val_end])
            test_data.append(vm_data[val_end:])

        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)

        logger.info("Data split: train=%d, val=%d, test=%d",
                   len(train_df), len(val_df), len(test_df))

        return train_df, val_df, test_df


# ============================================================================
# Model Architecture
# ============================================================================

class PowerLSTM(nn.Module):
    """
    LSTM model for power prediction

    Architecture:
    - Input: (batch_size, seq_length, 2) for [cpu_usage, vcpus]
    - LSTM layer(s)
    - Dropout
    - Linear layer for regression
    - Output: (batch_size, 1) for power
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2):
        """
        Args:
            input_size: Number of input features (default: 2 for cpu_usage + vcpus)
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(PowerLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward
        # out: (batch_size, seq_length, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        out, (h_n, c_n) = self.lstm(x)

        # Use last time step output
        # out[:, -1, :] has shape (batch_size, hidden_size)
        out = out[:, -1, :]

        # Dropout
        out = self.dropout(out)

        # Linear layer
        out = self.fc(out)

        return out


# ============================================================================
# Training
# ============================================================================

class Trainer:
    """Handle model training with early stopping"""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 0.001):
        """
        Args:
            model: PyTorch model
            device: Device to train on (cpu/cuda)
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch

        Args:
            dataloader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0

        for sequences, targets in dataloader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(sequences)
            loss = self.criterion(outputs.squeeze(), targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate model

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for sequences, targets in dataloader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(sequences)
                loss = self.criterion(outputs.squeeze(), targets)

                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             num_epochs: int = 100, patience: int = 10,
             save_path: str = 'power_lstm_model.pth'):
        """
        Train model with early stopping

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save best model
        """
        logger.info("Starting training for up to %d epochs...", num_epochs)
        logger.info("Device: %s", self.device)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            logger.info("Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f",
                       epoch + 1, num_epochs, train_loss, val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                logger.info("  Saved best model to %s", save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info("Early stopping triggered after %d epochs", epoch + 1)
                    break

        logger.info("Training complete. Best val loss: %.6f", self.best_val_loss)

    def plot_losses(self, save_path: str = 'training_loss.png'):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Loss plot saved to %s", save_path)
        plt.close()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model: nn.Module, dataloader: DataLoader,
                  preprocessor: DataPreprocessor, device: torch.device,
                  test_df: pd.DataFrame) -> Dict:
    """
    Evaluate model on test set

    Args:
        model: Trained model
        dataloader: Test data loader
        preprocessor: Data preprocessor for denormalization
        device: Device
        test_df: Original test dataframe (for stratification)

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            outputs = model(sequences)

            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Denormalize
    predictions_denorm = preprocessor.denormalize_power(predictions)
    actuals_denorm = preprocessor.denormalize_power(actuals)

    # Calculate metrics
    mse = mean_squared_error(actuals_denorm, predictions_denorm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_denorm, predictions_denorm)
    mape = np.mean(np.abs((actuals_denorm - predictions_denorm) / actuals_denorm)) * 100

    logger.info("Test Metrics:")
    logger.info("  MSE:  %.4f W²", mse)
    logger.info("  RMSE: %.4f W", rmse)
    logger.info("  MAE:  %.4f W", mae)
    logger.info("  MAPE: %.2f %%", mape)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions_denorm,
        'actuals': actuals_denorm
    }


def plot_predictions(results: Dict, test_df: pd.DataFrame,
                    seq_length: int, save_path: str = 'predictions.png'):
    """
    Plot predictions vs actuals, stratified by vCPU count

    Args:
        results: Results from evaluate_model
        test_df: Original test dataframe
        seq_length: Sequence length (for offset)
        save_path: Path to save plot
    """
    predictions = results['predictions']
    actuals = results['actuals']

    # Get vCPU info (offset by seq_length to match predictions)
    vcpus_list = []
    for vm_id in test_df['vm_id'].unique():
        vm_data = test_df[test_df['vm_id'] == vm_id]
        if len(vm_data) > seq_length:
            vcpus_list.extend(vm_data['vcpus'].values[seq_length:])

    vcpus_list = np.array(vcpus_list[:len(predictions)])

    # Plot overall
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Overall scatter
    axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=1)
    axes[0, 0].plot([actuals.min(), actuals.max()],
                    [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Power (W)')
    axes[0, 0].set_ylabel('Predicted Power (W)')
    axes[0, 0].set_title('Overall Predictions vs Actuals')
    axes[0, 0].grid(True)

    # Time series sample
    sample_size = min(500, len(predictions))
    axes[0, 1].plot(actuals[:sample_size], label='Actual', alpha=0.7)
    axes[0, 1].plot(predictions[:sample_size], label='Predicted', alpha=0.7)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Power (W)')
    axes[0, 1].set_title('Time Series (First 500 samples)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Stratified by vCPU count
    unique_vcpus = sorted(np.unique(vcpus_list))
    for vcpu in unique_vcpus:
        mask = vcpus_list == vcpu
        axes[1, 0].scatter(actuals[mask], predictions[mask],
                          label=f'{int(vcpu)} vCPUs', alpha=0.5, s=1)

    axes[1, 0].plot([actuals.min(), actuals.max()],
                    [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Power (W)')
    axes[1, 0].set_ylabel('Predicted Power (W)')
    axes[1, 0].set_title('Predictions by vCPU Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Error distribution
    errors = predictions - actuals
    axes[1, 1].hist(errors, bins=50, edgecolor='black')
    axes[1, 1].axvline(0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Prediction Error (W)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Predictions plot saved to %s", save_path)
    plt.close()


# ============================================================================
# Inference
# ============================================================================

class PowerPredictor:
    """
    Lightweight inference class for real-time VM power prediction
    Can be deployed in VMs with only CPU usage monitoring
    """

    def __init__(self, model_path: str, seq_length: int = 60,
                 max_vcpus: int = 64, device: str = 'cpu'):
        """
        Args:
            model_path: Path to saved model
            seq_length: Input sequence length
            max_vcpus: Max vCPUs for normalization
            device: Device for inference (default: 'cpu')
        """
        self.seq_length = seq_length
        self.max_vcpus = max_vcpus
        self.device = torch.device(device)

        # Load model
        self.model = PowerLSTM()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load scaler info (assume saved separately or use dummy)
        # In production, save scaler with model
        self.power_scaler = MinMaxScaler()
        # TODO: Load scaler params from file

        logger.info("Loaded model from %s", model_path)
        logger.info("Inference device: %s", self.device)

    def set_power_scaler(self, scaler: MinMaxScaler):
        """Set the power scaler (must be done after loading)"""
        self.power_scaler = scaler

    def predict(self, cpu_usage_history: List[float], vcpus: int) -> float:
        """
        Predict power consumption from CPU usage history

        Args:
            cpu_usage_history: List of recent CPU usage values (%)
                              Length must be >= seq_length
            vcpus: Number of vCPUs for this VM

        Returns:
            Predicted power in Watts
        """
        if len(cpu_usage_history) < self.seq_length:
            raise ValueError(f"Need at least {self.seq_length} CPU usage samples")

        # Take last seq_length samples
        recent_usage = cpu_usage_history[-self.seq_length:]

        # Normalize
        cpu_norm = np.array(recent_usage) / 100.0
        vcpu_norm = np.full(self.seq_length, vcpus / self.max_vcpus)

        # Create input sequence
        sequence = np.column_stack([cpu_norm, vcpu_norm])
        sequence = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dim
        sequence = sequence.to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(sequence)
            power_norm = output.item()

        # Denormalize
        power_norm_arr = np.array([[power_norm]])
        power_watts = self.power_scaler.inverse_transform(power_norm_arr)[0, 0]

        return power_watts

    def export_onnx(self, output_path: str = 'power_lstm_model.onnx'):
        """
        Export model to ONNX format for easier deployment

        Args:
            output_path: Path to save ONNX model
        """
        try:
            # Dummy input
            dummy_input = torch.randn(1, self.seq_length, 2).to(self.device)

            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info("Exported model to ONNX: %s", output_path)
        except Exception as e:
            logger.error("Failed to export ONNX: %s", e)


# ============================================================================
# Main Program
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train/Inference LSTM model for VM power prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', choices=['train', 'inference', 'generate'],
                       default='train', help='Operation mode')
    parser.add_argument('--data', type=str, default='multi_vm_training_data.csv',
                       help='Path to training data CSV')
    parser.add_argument('--model', type=str, default='power_lstm_model.pth',
                       help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--seq_length', type=int, default=60,
                       help='Input sequence length (seconds)')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='LSTM hidden size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples for synthetic data generation')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    # ========================================================================
    # Mode: Generate synthetic data
    # ========================================================================
    if args.mode == 'generate':
        generate_synthetic_data(args.num_samples, args.data)
        return 0

    # ========================================================================
    # Mode: Training
    # ========================================================================
    if args.mode == 'train':
        # Check if data exists
        if not os.path.exists(args.data):
            logger.warning("Data file not found. Generating synthetic data...")
            generate_synthetic_data(args.num_samples, args.data)

        # Load and preprocess data
        preprocessor = DataPreprocessor(seq_length=args.seq_length)
        df = preprocessor.load_data(args.data)

        # Split data
        train_df, val_df, test_df = preprocessor.split_data(df)

        # Create sequences
        train_seq, train_targets = preprocessor.create_sequences(train_df, is_training=True)
        val_seq, val_targets = preprocessor.create_sequences(val_df, is_training=False)
        test_seq, test_targets = preprocessor.create_sequences(test_df, is_training=False)

        # Create datasets and loaders
        train_dataset = PowerDataset(train_seq, train_targets)
        val_dataset = PowerDataset(val_seq, val_targets)
        test_dataset = PowerDataset(test_seq, test_targets)

        # shuffeling is safe because each sample is a complete temporal window
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Create model
        model = PowerLSTM(hidden_size=args.hidden_size)
        logger.info("Model architecture:\n%s", model)

        # Train
        trainer = Trainer(model, device, learning_rate=args.lr)
        trainer.train(train_loader, val_loader, num_epochs=args.epochs,
                     patience=args.patience, save_path=args.model)

        # Plot losses
        trainer.plot_losses('training_loss.png')

        # Load best model for evaluation
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        results = evaluate_model(model, test_loader, preprocessor, device, test_df)
        plot_predictions(results, test_df, args.seq_length, 'predictions.png')

        # Save preprocessor for inference
        import pickle
        with open('preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        logger.info("Saved preprocessor to preprocessor.pkl")

        logger.info("Training complete!")
        return 0

    # ========================================================================
    # Mode: Inference
    # ========================================================================
    if args.mode == 'inference':
        # Load preprocessor
        import pickle
        try:
            with open('preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
        except FileNotFoundError:
            logger.error("preprocessor.pkl not found. Train model first.")
            return 1

        # Create predictor
        predictor = PowerPredictor(args.model, seq_length=args.seq_length, device='cpu')
        predictor.set_power_scaler(preprocessor.power_scaler)

        # Simulate inference with dummy data
        logger.info("\n" + "="*60)
        logger.info("Simulating real-time inference in VM")
        logger.info("="*60)

        # Simulate: Get vCPU count (in real VM: use psutil.cpu_count())
        vcpus = 4
        logger.info("VM vCPUs: %d", vcpus)

        # Simulate: Buffer of recent CPU usage (in real VM: collect from psutil)
        cpu_history = [random.uniform(20, 80) for _ in range(args.seq_length)]
        logger.info("CPU usage history (last %d seconds): [%.1f ... %.1f]",
                   args.seq_length, cpu_history[0], cpu_history[-1])

        # Predict
        predicted_power = predictor.predict(cpu_history, vcpus)
        logger.info("Predicted power: %.2f W", predicted_power)

        # Export ONNX
        predictor.export_onnx('power_lstm_model.onnx')

        logger.info("\nInference complete!")
        return 0


if __name__ == '__main__':
    sys.exit(main())

"""
DECODE-RAPL Utilities
Synthetic data generation, metrics, and visualization functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List, Optional
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_synthetic_data(
    num_machines: int = 3,
    duration_hours: float = 0.1,  # 0.1 hours = 6 minutes for testing
    sampling_rate_ms: int = 1,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate synthetic CPU usage and power data for testing

    Simulates realistic patterns:
    - CPU usage: Random walk + periodic component + noise
    - Power: Nonlinear function of usage with machine-specific baseline

    Args:
        num_machines: Number of machines to simulate
        duration_hours: Duration of data collection in hours
        sampling_rate_ms: Sampling rate in milliseconds
        output_path: Optional path to save CSV

    Returns:
        DataFrame with columns: timestamp, machine_id, cpu_usage, power
    """
    print(f"Generating synthetic data for {num_machines} machines...")

    # Calculate number of samples
    samples_per_machine = int(duration_hours * 3600 * 1000 / sampling_rate_ms)
    total_samples = num_machines * samples_per_machine

    data = []

    for machine_idx in range(num_machines):
        machine_id = f"machine_{machine_idx}"

        # Machine-specific parameters
        baseline_power = 50 + machine_idx * 10  # Different baseline per machine
        power_scale = 1.0 + machine_idx * 0.1   # Slight variation in scaling

        # Generate CPU usage (0-100%)
        # Start with random walk
        usage = np.zeros(samples_per_machine)
        usage[0] = np.random.uniform(20, 80)

        for i in range(1, samples_per_machine):
            # Random walk with mean reversion
            drift = 0.1 * (50 - usage[i-1])  # Pull towards 50%
            noise = np.random.normal(0, 5)
            usage[i] = np.clip(usage[i-1] + drift + noise, 5, 95)

        # Add periodic component (simulates workload patterns)
        t = np.arange(samples_per_machine) * sampling_rate_ms / 1000  # Time in seconds
        periodic = 10 * np.sin(2 * np.pi * t / 30)  # 30-second period
        usage = np.clip(usage + periodic, 0, 100)

        # Generate power based on usage (nonlinear relationship)
        # Power model: P = baseline + scale * (a * usage + b * usage^2 + c * usage^3)
        usage_norm = usage / 100.0
        power = baseline_power + power_scale * (
            30 * usage_norm +           # Linear term
            50 * usage_norm ** 2 +      # Quadratic term (dominant)
            20 * usage_norm ** 3        # Cubic term (for high loads)
        )

        # Add measurement noise
        power += np.random.normal(0, 2, samples_per_machine)
        power = np.maximum(power, baseline_power * 0.8)  # Minimum power floor

        # Create timestamps
        timestamps = pd.date_range(
            start='2024-01-01',
            periods=samples_per_machine,
            freq=f'{sampling_rate_ms}ms'
        )

        # Create DataFrame for this machine
        machine_data = pd.DataFrame({
            'timestamp': timestamps,
            'machine_id': machine_id,
            'cpu_usage': usage,
            'power': power
        })

        data.append(machine_data)

    # Combine all machines
    df = pd.concat(data, ignore_index=True)

    # Shuffle to mix machines (realistic training scenario)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Generated {len(df)} samples ({total_samples} total)")
    print(f"CPU usage range: [{df['cpu_usage'].min():.2f}, {df['cpu_usage'].max():.2f}]")
    print(f"Power range: [{df['power'].min():.2f}, {df['power'].max():.2f}] W")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero

    Returns:
        MAPE in percentage
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate multiple evaluation metrics

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with MSE, RMSE, MAE, MAPE
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = calculate_mape(y_true, y_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.close()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    machine_ids: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    max_samples: int = 1000
):
    """
    Plot predicted vs actual power consumption

    Args:
        y_true: Ground truth power values
        y_pred: Predicted power values
        machine_ids: Optional machine identifiers for color coding
        save_path: Optional path to save plot
        max_samples: Maximum number of samples to plot (for clarity)
    """
    # Subsample if too many points
    if len(y_true) > max_samples:
        indices = np.random.choice(len(y_true), max_samples, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        if machine_ids is not None:
            machine_ids = machine_ids[indices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    if machine_ids is not None:
        unique_machines = np.unique(machine_ids)
        for machine in unique_machines:
            mask = machine_ids == machine
            axes[0].scatter(y_true[mask], y_pred[mask], alpha=0.5, label=machine, s=20)
        axes[0].legend()
    else:
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    axes[0].set_xlabel('Actual Power (W)', fontsize=12)
    axes[0].set_ylabel('Predicted Power (W)', fontsize=12)
    axes[0].set_title('Predicted vs Actual Power', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Time series plot (first N samples)
    n_plot = min(500, len(y_true))
    x = np.arange(n_plot)
    axes[1].plot(x, y_true[:n_plot], label='Actual', linewidth=2, alpha=0.7)
    axes[1].plot(x, y_pred[:n_plot], label='Predicted', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Power (W)', fontsize=12)
    axes[1].set_title('Time Series Comparison', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions plot to {save_path}")

    plt.close()


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """Plot distribution of prediction errors"""
    errors = y_pred - y_true
    percent_errors = 100 * errors / (y_true + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Absolute errors
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error (W)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Error Distribution\nMean: {errors.mean():.2f}, Std: {errors.std():.2f}', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Percentage errors
    axes[1].hist(percent_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Percentage Error Distribution\nMean: {percent_errors.mean():.2f}%, Std: {percent_errors.std():.2f}%',
                     fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved error distribution to {save_path}")

    plt.close()


def ensure_directories(config: dict):
    """Create necessary directories if they don't exist"""
    dirs = [
        config['data']['output_dir'],
        config['data']['plots_dir'],
        config['data']['checkpoint_dir'],
        Path(config['logging']['log_file']).parent,
        'data'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test synthetic data generation
    print("Testing synthetic data generation...")
    df = generate_synthetic_data(
        num_machines=3,
        duration_hours=0.05,  # 3 minutes
        sampling_rate_ms=1,
        output_path="data/test_synthetic.csv"
    )

    print("\nData summary:")
    print(df.groupby('machine_id').agg({
        'cpu_usage': ['mean', 'std', 'min', 'max'],
        'power': ['mean', 'std', 'min', 'max']
    }).round(2))

    # Test metrics
    print("\nTesting metrics calculation...")
    y_true = df['power'].values[:1000]
    y_pred = y_true + np.random.normal(0, 5, len(y_true))
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Metrics: {metrics}")

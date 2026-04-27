"""
DECODE-RAPL v2 Utilities

Helper functions for config loading, metrics, plotting, and reproducibility
"""

import yaml
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_directories(config: dict):
    """
    Create output directories if they don't exist

    Args:
        config: Configuration dictionary
    """
    dirs_to_create = [
        config['output']['checkpoint_dir'],
        config['output']['results_dir'],
        config['output']['plots_dir']
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics

    Args:
        predictions: Predicted values (N,)
        targets: Ground truth values (N,)

    Returns:
        Dictionary with metrics: MSE, RMSE, MAE, R²
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    mask = targets != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        mape = 0

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }


def plot_training_curves(
    train_power: list,
    train_recon: list,
    val_power: list,
    val_recon: list,
    output_path: str
):
    """
    Plot power and reconstruction loss curves separately

    Args:
        train_power: List of training power losses per epoch
        train_recon: List of training reconstruction losses per epoch
        val_power: List of validation power losses per epoch
        val_recon: List of validation reconstruction losses per epoch
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    epochs = range(1, len(train_power) + 1)

    # Top panel: Power Loss (what we optimize for!)
    ax1.plot(epochs, train_power, 'b-', label='Training Power Loss', linewidth=2)
    ax1.plot(epochs, val_power, 'r-', label='Validation Power Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Power Loss (MSE)', fontsize=12)
    ax1.set_title('Power Prediction Loss (Primary Objective)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Reconstruction Loss (regularization)
    ax2.plot(epochs, train_recon, 'b-', label='Training Reconstruction Loss', linewidth=2)
    ax2.plot(epochs, val_recon, 'r-', label='Validation Reconstruction Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
    ax2.set_title('Autoencoder Reconstruction Loss (Regularization)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves_v3(
    train_power: list,
    val_power: list,
    output_path: str,
    version: str = 'v3'
):
    """
    Plot power loss curve for v3/v4 (single panel, no reconstruction)

    Args:
        train_power: List of training power losses per epoch
        val_power: List of validation power losses per epoch
        output_path: Path to save plot
        version: Model version ('v3' or 'v4') for labeling
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_power) + 1)

    # Determine loss type and architecture description
    if version == 'v4':
        loss_label = 'Power Loss (MAE)'
        arch_desc = 'v4: 1D-CNN Encoder'
    else:  # v3
        loss_label = 'Power Loss (MSE)'
        arch_desc = 'v3: MLP Encoder'

    # Single panel: Power Loss (what we optimize for!)
    plt.plot(epochs, train_power, 'b-', label='Training Power Loss', linewidth=2)
    plt.plot(epochs, val_power, 'r-', label='Validation Power Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(loss_label, fontsize=12)
    plt.title(f'Power Prediction Loss ({arch_desc})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    title: str = "Power Predictions",
    split_name: str = "Test"
):
    """
    Plot predicted vs actual power values

    Args:
        predictions: Predicted values (N,)
        targets: Ground truth values (N,)
        output_path: Path to save plot
        title: Plot title
        split_name: Name of data split (Train/Val/Test)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot: Predicted vs Actual
    ax1 = axes[0]
    ax1.scatter(targets, predictions, alpha=0.3, s=1)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax1.set_xlabel('Actual Power (W)', fontsize=11)
    ax1.set_ylabel('Predicted Power (W)', fontsize=11)
    ax1.set_title(f'{split_name} Set: Predicted vs Actual', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Calculate metrics for annotation
    metrics = calculate_metrics(predictions, targets)
    textstr = '\n'.join([
        f'MSE: {metrics["mse"]:.2f}',
        f'RMSE: {metrics["rmse"]:.2f} W',
        f'MAE: {metrics["mae"]:.2f} W',
        f'R²: {metrics["r2"]:.4f}',
        f'MAPE: {metrics["mape"]:.2f}%'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props, family='monospace')

    # Error distribution
    ax2 = axes[1]
    errors = predictions - targets
    ax2.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')

    ax2.set_xlabel('Prediction Error (W)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'{split_name} Set: Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add mean and std to error plot
    error_mean = errors.mean()
    error_std = errors.std()
    textstr = f'Mean: {error_mean:.2f} W\nStd: {error_std:.2f} W'
    ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=props, family='monospace')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_space(
    latent_vectors: np.ndarray,
    power_values: np.ndarray,
    output_path: str,
    title: str = "Latent Space Visualization",
    method: str = "tsne"
):
    """
    Visualize 64-dim latent space in 2D using t-SNE or UMAP

    Args:
        latent_vectors: Latent representations (N, 64)
        power_values: Corresponding power values (N,)
        output_path: Path to save plot
        title: Plot title
        method: 'tsne' or 'umap'
    """
    try:
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "umap":
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Reduce to 2D
        latent_2d = reducer.fit_transform(latent_vectors)

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            latent_2d[:, 0], latent_2d[:, 1],
            c=power_values, cmap='viridis',
            alpha=0.5, s=1
        )

        plt.colorbar(scatter, label='Power (W)')
        plt.xlabel(f'{method.upper()} Dimension 1', fontsize=11)
        plt.ylabel(f'{method.upper()} Dimension 2', fontsize=11)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Latent space visualization saved to {output_path}")

    except ImportError as e:
        print(f"Warning: Could not create latent space visualization: {e}")
        print(f"Install required package: pip install {method}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_path: str,
    is_best: bool = False,
    config: dict = None
):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best model so far
        config: Configuration dictionary (needed for inference)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }

    # Add config for inference (tau, d, model architecture, etc.)
    if config is not None:
        checkpoint['config'] = config

    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = str(Path(checkpoint_path).parent / 'best_model.pt')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Tuple[int, float, float]:
    """
    Load model checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: PyTorch optimizer to load state into (optional)

    Returns:
        (epoch, train_loss, val_loss)
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def get_device(config: dict) -> torch.device:
    """
    Get torch device based on config and availability

    Args:
        config: Configuration dictionary

    Returns:
        torch.device
    """
    device_str = config['training'].get('device', 'cuda')

    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


if __name__ == '__main__':
    # Test utilities
    print("Testing DECODE-RAPL v2 utilities...")

    # Test metrics
    predictions = np.array([25.0, 30.0, 35.0, 40.0, 45.0])
    targets = np.array([24.0, 32.0, 34.0, 42.0, 46.0])
    metrics = calculate_metrics(predictions, targets)
    print("\nMetrics test:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test seed setting
    set_seed(42)
    print("\nRandom seed set to 42")

    print("\nUtilities test complete!")

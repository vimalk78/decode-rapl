"""
DECODE-RAPL Training Pipeline
End-to-end training with combined loss (MSE + reconstruction + adversarial)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import time
import logging
import argparse

from src.model import create_model, GradientReversalLayer
from src.preprocessing import load_and_split_data, create_dataloaders
from src.utils import (
    load_config, set_seed, ensure_directories,
    calculate_metrics, plot_training_curves, plot_predictions, plot_error_distribution
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined loss for DECODE-RAPL training:
    - Power MSE: Prediction accuracy
    - Reconstruction MSE: Autoencoder quality
    - Adversarial CE: Machine-invariance
    """

    def __init__(self, config: dict):
        super(CombinedLoss, self).__init__()

        self.weights = config['training']['loss_weights']

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # Gradient reversal for adversarial training
        self.grad_reversal = GradientReversalLayer(lambda_=self.weights['adversarial'])

    def forward(
        self,
        outputs: Dict,
        targets: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            outputs: Model outputs dict with keys:
                - x_recon: Reconstructed input
                - power_pred: Power prediction
                - machine_logits: Machine classification logits
            targets: Target dict with keys:
                - x: Original input
                - power: Ground truth power
                - machine_id: Machine ID (integer encoded)

        Returns:
            (total_loss, loss_dict)
        """
        # Power prediction loss (main objective)
        power_loss = self.mse_loss(outputs['power_pred'], targets['power'])

        # Reconstruction loss (autoencoder quality)
        recon_loss = self.mse_loss(outputs['x_recon'], targets['x'])

        # Adversarial loss (fool discriminator for generalization)
        # Apply gradient reversal to encoder during backprop
        z_reversed = self.grad_reversal(outputs['z'])
        machine_logits_reversed = outputs['machine_logits']

        adv_loss = self.ce_loss(machine_logits_reversed, targets['machine_id'])

        # Combined loss
        total_loss = (
            self.weights['power_mse'] * power_loss +
            self.weights['reconstruction'] * recon_loss +
            self.weights['adversarial'] * adv_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'power': power_loss.item(),
            'reconstruction': recon_loss.item(),
            'adversarial': adv_loss.item()
        }

        return total_loss, loss_dict


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    machine_id_map: Dict
) -> Dict:
    """Train for one epoch"""
    model.train()

    epoch_losses = {
        'total': 0.0,
        'power': 0.0,
        'reconstruction': 0.0,
        'adversarial': 0.0
    }

    num_batches = 0

    for batch in dataloader:
        # Move data to device
        x = batch['embedding'].to(device)  # (batch, seq_len, input_dim)
        power_target = batch['power'].to(device)  # (batch, 1)

        # Encode machine IDs to integers
        machine_ids = torch.tensor(
            [machine_id_map[mid] for mid in batch['machine_id']],
            dtype=torch.long,
            device=device
        )

        # Forward pass
        outputs = model(x)

        # Compute loss
        targets = {
            'x': x,
            'power': power_target,
            'machine_id': machine_ids
        }

        loss, loss_dict = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += loss_dict[key]

        num_batches += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return epoch_losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    machine_id_map: Dict
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Validate model"""
    model.eval()

    epoch_losses = {
        'total': 0.0,
        'power': 0.0,
        'reconstruction': 0.0,
        'adversarial': 0.0
    }

    all_predictions = []
    all_targets = []

    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            x = batch['embedding'].to(device)
            power_target = batch['power'].to(device)

            machine_ids = torch.tensor(
                [machine_id_map[mid] for mid in batch['machine_id']],
                dtype=torch.long,
                device=device
            )

            # Forward pass
            outputs = model(x)

            # Compute loss
            targets = {
                'x': x,
                'power': power_target,
                'machine_id': machine_ids
            }

            loss, loss_dict = criterion(outputs, targets)

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]

            # Store predictions
            all_predictions.append(outputs['power_pred'].cpu().numpy())
            all_targets.append(power_target.cpu().numpy())

            num_batches += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    # Concatenate predictions
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    return epoch_losses, all_predictions, all_targets


def train_model(config: dict, dataloaders: Dict, machine_id_map: Dict) -> nn.Module:
    """
    Main training loop

    Args:
        config: Configuration dictionary
        dataloaders: Dict with 'train', 'val', 'test' DataLoaders
        machine_id_map: Mapping from machine_id to integer

    Returns:
        Trained model
    """
    # Setup device
    if config['training']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # Create model
    num_machines = len(machine_id_map)
    model = create_model(config, num_machines)
    model = model.to(device)

    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Setup loss
    criterion = CombinedLoss(config)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mape': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience = config['training']['early_stopping_patience']
    patience_counter = 0

    checkpoint_dir = Path(config['data']['checkpoint_dir'])
    best_model_path = checkpoint_dir / 'best_model.pth'

    # Training loop
    num_epochs = config['training']['epochs']
    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        train_losses = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, machine_id_map
        )

        # Validate
        val_losses, val_preds, val_targets = validate(
            model, dataloaders['val'], criterion, device, machine_id_map
        )

        # Denormalize predictions if power was normalized
        power_scaler = dataloaders['scalers']['power']
        if power_scaler is not None:
            val_preds_denorm = power_scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()
            val_targets_denorm = power_scaler.inverse_transform(val_targets.reshape(-1, 1)).flatten()
        else:
            val_preds_denorm = val_preds
            val_targets_denorm = val_targets

        # Calculate metrics
        val_metrics = calculate_metrics(val_targets_denorm, val_preds_denorm)

        # Store history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        history['val_mape'].append(val_metrics['mape'])

        # Logging
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - "
            f"Train Loss: {train_losses['total']:.4f} - "
            f"Val Loss: {val_losses['total']:.4f} - "
            f"Val MAPE: {val_metrics['mape']:.2f}%"
        )

        # Early stopping and checkpointing
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'machine_id_map': machine_id_map,
                'scalers': dataloaders['scalers'],
                'embedder': dataloaders['embedder'],
                'history': history
            }, best_model_path)

            logger.info(f"Saved best model to {best_model_path}")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Plot training curves
    plot_training_curves(
        history['train_loss'],
        history['val_loss'],
        save_path=Path(config['data']['plots_dir']) / 'training_curves.png'
    )

    # Load best model
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Loaded best model for final evaluation")

    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    machine_id_map: Dict,
    power_scaler,
    split_name: str = 'Test'
):
    """Evaluate model on test set"""
    model.eval()

    all_predictions = []
    all_targets = []
    all_machine_ids = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['embedding'].to(device)
            power_target = batch['power'].to(device)

            outputs = model(x)

            all_predictions.append(outputs['power_pred'].cpu().numpy())
            all_targets.append(power_target.cpu().numpy())
            all_machine_ids.extend(batch['machine_id'])

    # Concatenate
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    # Denormalize
    if power_scaler is not None:
        predictions = power_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        targets = power_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

    # Calculate metrics
    metrics = calculate_metrics(targets, predictions)

    logger.info(f"\n{split_name} Set Metrics:")
    logger.info(f"  MSE: {metrics['mse']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")

    # Plot predictions
    plots_dir = Path(config['data']['plots_dir'])

    # Encode machine IDs for plotting
    unique_machines = sorted(set(all_machine_ids))
    machine_colors = np.array([unique_machines.index(mid) for mid in all_machine_ids])

    plot_predictions(
        targets, predictions,
        machine_ids=machine_colors,
        save_path=plots_dir / f'{split_name.lower()}_predictions.png'
    )

    plot_error_distribution(
        targets, predictions,
        save_path=plots_dir / f'{split_name.lower()}_errors.png'
    )

    return metrics


def main():
    """Main training script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DECODE-RAPL Training Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file (default: config.yaml)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Ensure directories exist
    ensure_directories(config)

    logger.info("=" * 60)
    logger.info("DECODE-RAPL Training Pipeline")
    logger.info("=" * 60)

    # Load and split data
    logger.info("\nLoading and splitting data...")
    csv_path = config['data']['train_csv']

    train_df, val_df, test_df = load_and_split_data(csv_path, config)

    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    dataloaders = create_dataloaders(train_df, val_df, test_df, config)

    # Create machine ID mapping
    all_machine_ids = sorted(
        set(train_df['machine_id'].unique()) |
        set(val_df['machine_id'].unique()) |
        set(test_df['machine_id'].unique())
    )
    machine_id_map = {mid: idx for idx, mid in enumerate(all_machine_ids)}
    logger.info(f"Machine ID mapping: {machine_id_map}")

    # Train model
    logger.info("\n" + "=" * 60)
    logger.info("Training")
    logger.info("=" * 60)

    model = train_model(config, dataloaders, machine_id_map)

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    power_scaler = dataloaders['scalers']['power']

    # Evaluate on validation set
    val_metrics = evaluate_model(
        model, dataloaders['val'], config, device,
        machine_id_map, power_scaler, split_name='Validation'
    )

    # Evaluate on test set
    test_metrics = evaluate_model(
        model, dataloaders['test'], config, device,
        machine_id_map, power_scaler, split_name='Test'
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)

    # Print final results
    logger.info(f"\nFinal Test MAPE: {test_metrics['mape']:.2f}%")
    if test_metrics['mape'] < 5.0:
        logger.info("✓ Target accuracy (<5% MAPE) achieved!")
    else:
        logger.info(f"✗ Target accuracy not achieved. Current: {test_metrics['mape']:.2f}%, Target: <5%")


if __name__ == "__main__":
    main()

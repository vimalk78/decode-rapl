"""
DECODE-RAPL v2 Training Pipeline

Train the model with support for:
- Background execution (nohup compatible)
- Resume from checkpoint
- Early stopping
- Progress logging to file
- Visualization

Usage:
    # Foreground
    python src/train.py --config config/v2_default.yaml

    # Background
    nohup python src/train.py --config config/v2_default.yaml > logs/train.out 2>&1 &

    # Resume
    python src/train.py --config config/v2_default.yaml --resume checkpoints/best_model.pt
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import create_model, CombinedLoss, count_parameters
from utils import (
    load_config, set_seed, ensure_directories, calculate_metrics,
    plot_training_curves, plot_predictions, save_checkpoint, get_device
)


class RAPLDataset(Dataset):
    """PyTorch Dataset for loading NPZ files"""

    def __init__(self, npz_path: str):
        """
        Args:
            npz_path: Path to NPZ file (train.npz, val.npz, or test.npz)
        """
        data = np.load(npz_path)
        self.X = torch.from_numpy(data['X']).float()  # (N, 100)
        self.y = torch.from_numpy(data['y']).float()  # (N,)

        print(f"Loaded {npz_path}: {len(self.X):,} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Trainer:
    """Training manager with background execution support"""

    def __init__(self, config: dict, resume_path: str = None):
        self.config = config
        self.device = get_device(config)
        self.start_epoch = 0
        self.best_val_power_loss = float('inf')
        self.epochs_without_improvement = 0

        # Setup directories
        ensure_directories(config)

        # Setup logging
        self.log_file = Path(config['output']['results_dir']) / 'training.log'
        self.progress_file = Path(config['output']['results_dir']) / 'training_progress.txt'
        self.history_file = Path(config['output']['results_dir']) / 'training_history.json'

        # Training history
        self.history = {
            'train_loss': [],
            'train_power_loss': [],
            'train_recon_loss': [],
            'val_loss': [],
            'val_power_loss': [],
            'val_recon_loss': [],
            'learning_rates': []
        }

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Load data
        self._load_data()

        # Create model
        self.model = create_model(config['model']).to(self.device)
        self.log(f"Model created: {count_parameters(self.model):,} parameters")

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            min_lr=config['training']['scheduler']['min_lr']
        )

        # Setup loss
        self.model_version = config.get('version', 'v2')
        if self.model_version == 'v4':
            # v4: L1Loss (MAE) - more robust to outliers
            self.criterion = nn.L1Loss()
            self.is_v3 = True
        elif self.model_version == 'v3':
            # v3: MSELoss (kept for comparison with v4)
            self.criterion = nn.MSELoss()
            self.is_v3 = True
        else:
            # v2: Combined power + reconstruction loss
            self.criterion = CombinedLoss(
                power_weight=config['training']['loss_weights']['power'],
                reconstruction_weight=config['training']['loss_weights']['reconstruction']
            )
            self.is_v3 = False

        # Resume from checkpoint if provided
        if resume_path:
            self._load_checkpoint(resume_path)

    def _load_data(self):
        """Load train/val/test datasets and normalization parameters"""
        data_dir = Path(self.config['data']['processed_dir'])

        # Load metadata with normalization parameters
        metadata_path = data_dir / 'metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Add normalization parameters to config for checkpoint
            if 'normalization' in metadata:
                self.config['normalization'] = metadata['normalization']
                self.log("✓ Loaded normalization parameters from metadata")
                self.log(f"  Type: {metadata['normalization']['type']}")
                self.log(f"  Feature range: [{min(metadata['normalization']['feature_min']):.4f}, "
                        f"{max(metadata['normalization']['feature_min']):.4f}] (min)")
                self.log(f"  Feature range: [{min(metadata['normalization']['feature_range']):.4f}, "
                        f"{max(metadata['normalization']['feature_range']):.4f}] (range)")
            else:
                self.log("WARNING: No normalization parameters found in metadata!")
        else:
            self.log(f"WARNING: Metadata file not found: {metadata_path}")

        train_path = data_dir / self.config['data']['train_file']
        val_path = data_dir / self.config['data']['val_file']
        test_path = data_dir / self.config['data']['test_file']

        self.train_dataset = RAPLDataset(str(train_path))
        self.val_dataset = RAPLDataset(str(val_path))
        self.test_dataset = RAPLDataset(str(test_path))

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.log(f"Train batches: {len(self.train_loader)}")
        self.log(f"Val batches: {len(self.val_loader)}")
        self.log(f"Test batches: {len(self.test_loader)}")

    def log(self, message: str, flush: bool = True):
        """Log to both file and stdout"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"

        # Print to stdout
        print(log_msg)

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
            if flush:
                f.flush()

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully"""
        self.log(f"Received signal {signum}, saving checkpoint and exiting...")
        checkpoint_path = Path(self.config['output']['checkpoint_dir']) / 'interrupted.pt'
        save_checkpoint(
            self.model, self.optimizer, self.start_epoch,
            self.history['train_loss'][-1] if self.history['train_loss'] else 0,
            self.best_val_power_loss, str(checkpoint_path), config=self.config
        )
        self.log("Checkpoint saved. Exiting.")
        sys.exit(0)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""
        self.log(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_power_loss = checkpoint.get('val_loss', float('inf'))  # Legacy checkpoints use 'val_loss'

        self.log(f"Resumed from epoch {self.start_epoch}, best_val_power_loss={self.best_val_power_loss:.4f}")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_power_loss = 0
        total_recon_loss = 0

        # Use tqdm only if output is a terminal
        use_tqdm = sys.stdout.isatty()
        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch}") if use_tqdm else self.train_loader

        for batch_idx, (x, y) in enumerate(iterator):
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x)

            # Compute loss
            if self.is_v3:
                # v3: Simple power loss only
                # Ensure y has correct shape
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                loss = self.criterion(outputs['power_pred'], y)
                total_loss += loss.item()
                total_power_loss += loss.item()  # Same as total for v3
                total_recon_loss += 0  # No reconstruction in v3
            else:
                # v2: Combined loss
                loss, loss_dict = self.criterion(outputs, x, y)
                total_loss += loss.item()
                total_power_loss += loss_dict['power']
                total_recon_loss += loss_dict['reconstruction']

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Log progress
            if (batch_idx + 1) % self.config['logging']['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                if self.is_v3:
                    msg = f"Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] " \
                          f"Power Loss: {avg_loss:.4f}"
                else:
                    msg = f"Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] " \
                          f"Loss: {avg_loss:.4f} (Power: {total_power_loss/(batch_idx+1):.4f}, " \
                          f"Recon: {total_recon_loss/(batch_idx+1):.4f})"
                if not use_tqdm:
                    self.log(msg, flush=True)

        avg_loss = total_loss / len(self.train_loader)
        avg_power_loss = total_power_loss / len(self.train_loader)
        avg_recon_loss = total_recon_loss / len(self.train_loader)
        return avg_loss, avg_power_loss, avg_recon_loss

    def validate(self, epoch: int) -> Tuple[float, float, float, Dict]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        total_power_loss = 0
        total_recon_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                outputs = self.model(x)

                # Compute loss
                if self.is_v3:
                    # v3: Simple power loss only
                    if y.dim() == 1:
                        y_pred_target = y.unsqueeze(1)
                    else:
                        y_pred_target = y
                    loss = self.criterion(outputs['power_pred'], y_pred_target)
                    total_loss += loss.item()
                    total_power_loss += loss.item()
                    total_recon_loss += 0
                else:
                    # v2: Combined loss
                    loss, loss_dict = self.criterion(outputs, x, y)
                    total_loss += loss.item()
                    total_power_loss += loss_dict['power']
                    total_recon_loss += loss_dict['reconstruction']

                # Collect predictions for metrics
                all_predictions.append(outputs['power_pred'].cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        avg_power_loss = total_power_loss / len(self.val_loader)
        avg_recon_loss = total_recon_loss / len(self.val_loader)

        # Calculate metrics
        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()
        metrics = calculate_metrics(predictions, targets)

        return avg_loss, avg_power_loss, avg_recon_loss, metrics

    def evaluate_test(self) -> Tuple[float, float, float, Dict]:
        """Evaluate on held-out test set"""
        self.model.eval()
        total_loss = 0
        total_power_loss = 0
        total_recon_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                outputs = self.model(x)

                # Compute loss
                if self.is_v3:
                    # v3: Simple power loss only
                    if y.dim() == 1:
                        y_pred_target = y.unsqueeze(1)
                    else:
                        y_pred_target = y
                    loss = self.criterion(outputs['power_pred'], y_pred_target)
                    total_loss += loss.item()
                    total_power_loss += loss.item()
                    total_recon_loss += 0
                else:
                    # v2: Combined loss
                    loss, loss_dict = self.criterion(outputs, x, y)
                    total_loss += loss.item()
                    total_power_loss += loss_dict['power']
                    total_recon_loss += loss_dict['reconstruction']

                # Collect predictions for metrics
                all_predictions.append(outputs['power_pred'].cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / len(self.test_loader)
        avg_power_loss = total_power_loss / len(self.test_loader)
        avg_recon_loss = total_recon_loss / len(self.test_loader)

        # Calculate metrics
        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()
        metrics = calculate_metrics(predictions, targets)

        return avg_loss, avg_power_loss, avg_recon_loss, metrics

    def save_progress(self, epoch: int, train_loss: float, val_loss: float):
        """Save training progress to file"""
        with open(self.progress_file, 'a') as f:
            f.write(f"Epoch {epoch}/{self.config['training']['epochs']}: "
                   f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}\n")
            f.flush()

    def save_history(self):
        """Save training history to JSON"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def train(self):
        """Main training loop"""
        self.log("=" * 70)
        self.log("Starting training")
        self.log("=" * 70)

        epochs = self.config['training']['epochs']

        for epoch in range(self.start_epoch, epochs):
            epoch_start_time = time.time()

            # Train
            train_loss, train_power_loss, train_recon_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_power_loss, val_recon_loss, val_metrics = self.validate(epoch)

            # Update scheduler based on POWER LOSS only
            self.scheduler.step(val_power_loss)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_power_loss'].append(train_power_loss)
            self.history['train_recon_loss'].append(train_recon_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_power_loss'].append(val_power_loss)
            self.history['val_recon_loss'].append(val_recon_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            self.log(f"\nEpoch {epoch}/{epochs} Summary:")
            if self.is_v3:
                # v3: Only power loss
                self.log(f"  Train Power Loss: {train_power_loss:.4f}")
                self.log(f"  Val Power Loss:   {val_power_loss:.4f}")
            else:
                # v2: Combined + breakdown
                self.log(f"  Train Loss: {train_loss:.4f} (Power: {train_power_loss:.4f}, Recon: {train_recon_loss:.4f})")
                self.log(f"  Val Loss:   {val_loss:.4f} (Power: {val_power_loss:.4f}, Recon: {val_recon_loss:.4f})")
            self.log(f"  Val MAE:    {val_metrics['mae']:.2f}W")
            self.log(f"  Val R²:     {val_metrics['r2']:.4f}")
            self.log(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            self.log(f"  Time:       {epoch_time:.1f}s")

            # Save progress
            self.save_progress(epoch, train_loss, val_loss)
            self.save_history()

            # Save checkpoint - use POWER LOSS for best model selection
            checkpoint_path = Path(self.config['output']['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pt'
            is_best = val_power_loss < self.best_val_power_loss

            save_checkpoint(
                self.model, self.optimizer, epoch, train_loss, val_loss,
                str(checkpoint_path), is_best=is_best, config=self.config
            )

            if is_best:
                self.best_val_power_loss = val_power_loss
                self.epochs_without_improvement = 0
                self.log(f"  ✓ New best model! Val power loss: {val_power_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Plot training curves
            if (epoch + 1) % self.config['logging']['plot_interval'] == 0:
                plot_path = Path(self.config['output']['plots_dir']) / 'training_curves.png'
                if self.is_v3:
                    # v3/v4: Single panel (power loss only)
                    from utils import plot_training_curves_v3
                    plot_training_curves_v3(
                        self.history['train_power_loss'],
                        self.history['val_power_loss'],
                        str(plot_path),
                        version=self.model_version
                    )
                else:
                    # v2: Two panels (power + reconstruction)
                    plot_training_curves(
                        self.history['train_power_loss'],
                        self.history['train_recon_loss'],
                        self.history['val_power_loss'],
                        self.history['val_recon_loss'],
                        str(plot_path)
                    )

            # Early stopping based on POWER LOSS only
            if self.epochs_without_improvement >= self.config['training']['early_stopping_patience']:
                self.log(f"\nEarly stopping triggered after {epoch + 1} epochs")
                self.log(f"Best validation power loss: {self.best_val_power_loss:.4f}")
                break

        self.log("=" * 70)
        self.log("Training completed!")
        self.log("=" * 70)

        # Final evaluation and plots
        self.final_evaluation()

    def final_evaluation(self):
        """Final evaluation on validation and test sets with plots"""
        self.log("\nRunning final evaluation...")

        # Load best model
        best_model_path = Path(self.config['output']['checkpoint_dir']) / 'best_model.pt'
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate on validation set
        val_loss, val_power_loss, val_recon_loss, val_metrics = self.validate(epoch=-1)

        self.log("\nFinal Validation Metrics:")
        for key, value in val_metrics.items():
            self.log(f"  {key.upper()}: {value:.4f}")

        # Generate validation predictions for plotting
        self.model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                outputs = self.model(x)
                val_predictions.append(outputs['power_pred'].cpu().numpy())
                val_targets.append(y.numpy())

        val_predictions = np.concatenate(val_predictions).flatten()
        val_targets = np.concatenate(val_targets).flatten()

        # Plot validation predictions
        plot_path = Path(self.config['output']['plots_dir']) / 'final_predictions_val.png'
        plot_predictions(val_predictions, val_targets, str(plot_path), split_name="Validation")

        # Evaluate on TEST set (held-out, never seen during training)
        self.log("\n" + "="*70)
        self.log("TEST SET EVALUATION (Unbiased Performance)")
        self.log("="*70)

        test_loss, test_power_loss, test_recon_loss, test_metrics = self.evaluate_test()

        self.log("\nFinal Test Metrics:")
        for key, value in test_metrics.items():
            self.log(f"  {key.upper()}: {value:.4f}")

        # Generate test predictions for plotting
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                outputs = self.model(x)
                test_predictions.append(outputs['power_pred'].cpu().numpy())
                test_targets.append(y.numpy())

        test_predictions = np.concatenate(test_predictions).flatten()
        test_targets = np.concatenate(test_targets).flatten()

        # Plot test predictions
        plot_path = Path(self.config['output']['plots_dir']) / 'final_predictions_test.png'
        plot_predictions(test_predictions, test_targets, str(plot_path), split_name="Test")

        self.log(f"\nFinal plots saved to: {self.config['output']['plots_dir']}")


def main():
    parser = argparse.ArgumentParser(description='Train DECODE-RAPL v2 model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(config['seed'])

    # Create trainer
    trainer = Trainer(config, resume_path=args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()

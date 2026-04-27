#!/usr/bin/env python3
"""
Fine-tune existing MS-TCN model with attention-based pooling.

This script loads a pre-trained model that used global average pooling and
fine-tunes it with the new attention-based pooling mechanism. This should
allow the attention mechanism to learn meaningful temporal patterns instead
of collapsing to focus only on the final timestep.

Usage:
    python scripts/finetune_attention_pooling.py \
        --model models/best_model_diverse_2hr_17f.pth \
        --data data/training_diverse_2hr.csv \
        --output models/best_model_17f_attention_pooling.pth \
        --epochs 50 \
        --learning-rate 1e-5

The fine-tuning process:
1. Loads pre-trained model weights
2. Replaces global_pool with AttentionPooling (randomly initialized)
3. Optionally freezes early layers to preserve learned features
4. Fine-tunes with smaller learning rate to adapt attention pooling
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from train_model import (
    MSTCN, DataPreprocessor, PowerDataset, Trainer,
    plot_training_history
)


def load_pretrained_model(model_path: str, device: torch.device):
    """Load pre-trained model and extract configuration."""
    print(f"\nLoading pre-trained model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model configuration
    preprocessor_data = checkpoint['preprocessor']
    num_features = len(preprocessor_data['feature_columns'])
    num_targets = len(preprocessor_data['target_columns'])

    # Get hidden_dim from temporal blocks (reliable for any hidden_dim value)
    # temporal_blocks always use the full hidden_dim regardless of remainder from division
    hidden_dim = checkpoint['model_state_dict']['temporal_blocks.0.conv1.weight'].shape[0]

    # Find compatible num_heads for the hidden_dim
    # num_heads must divide hidden_dim evenly for MultiheadAttention
    # Try to use 8 heads if possible, otherwise find largest valid value
    preferred_heads = [8, 7, 6, 5, 4, 3, 2, 1]
    num_heads = next((h for h in preferred_heads if hidden_dim % h == 0), 1)

    print(f"  Model architecture: {num_features} features → {num_targets} targets")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Attention heads: {num_heads} (auto-detected for compatibility)")
    print(f"  Feature columns: {preprocessor_data['feature_columns']}")
    print(f"  Target columns: {preprocessor_data['target_columns']}")

    return checkpoint, num_features, num_targets, hidden_dim, num_heads


def create_finetuned_model(checkpoint, num_features: int, num_targets: int,
                           hidden_dim: int, num_heads: int, freeze_early_layers: bool = False):
    """Create new model with attention pooling and load pre-trained weights."""

    # Create new model with AttentionPooling
    model = MSTCN(num_features=num_features, num_targets=num_targets,
                  hidden_dim=hidden_dim, num_heads=num_heads)

    # Load state dict, excluding the old global_pool layer
    pretrained_state = checkpoint['model_state_dict']
    model_state = model.state_dict()

    # Copy all weights except global_pool (which doesn't exist in new model)
    loaded_keys = []
    skipped_keys = []

    for key in pretrained_state.keys():
        if 'global_pool' in key:
            skipped_keys.append(key)
            continue
        if key in model_state:
            model_state[key] = pretrained_state[key]
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state)

    print(f"\n  Loaded {len(loaded_keys)} weight tensors from pre-trained model")
    if skipped_keys:
        print(f"  Skipped {len(skipped_keys)} keys: {skipped_keys}")

    # AttentionPooling is randomly initialized - this is what we're fine-tuning
    print(f"  AttentionPooling layer: randomly initialized (will be trained)")

    # Optionally freeze early layers to preserve learned features
    if freeze_early_layers:
        print("\n  Freezing early layers (multiscale_conv, temporal_blocks, attention)...")
        frozen_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            if any(layer in name for layer in ['multiscale_conv', 'temporal_blocks', 'attention']):
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                trainable_params += param.numel()

        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  All layers trainable: {trainable_params:,} parameters")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MS-TCN with attention-based pooling",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--data', nargs='+', required=True,
                       help='CSV data files for fine-tuning')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of fine-tuning epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                       help='Learning rate for fine-tuning (default: 1e-5, much smaller than initial training)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--freeze-early-layers', action='store_true',
                       help='Freeze early layers (multiscale_conv, temporal_blocks, attention) and only train pooling + FC layers')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--split-mode', type=str, default='random', choices=['random', 'temporal'],
                       help='Data split mode: random or temporal (default: random)')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-trained model
    checkpoint, num_features, num_targets, hidden_dim, num_heads = load_pretrained_model(args.model, device)

    # Create model with attention pooling
    print("\nCreating model with attention-based pooling...")
    model = create_finetuned_model(checkpoint, num_features, num_targets, hidden_dim, num_heads,
                                   freeze_early_layers=args.freeze_early_layers)

    # Load and preprocess data using saved preprocessor configuration
    print("\nLoading training data...")
    preprocessor_data = checkpoint['preprocessor']
    preprocessor = DataPreprocessor(sequence_length=preprocessor_data['sequence_length'])
    df = preprocessor.load_csv_files(args.data)

    print("\nPreparing features and targets...")
    X, y = preprocessor.prepare_data(df)

    # Override the fitted scalers with the saved ones from the pre-trained model
    print("\nLoading saved scalers from pre-trained model...")
    preprocessor.feature_scaler = pickle.loads(preprocessor_data['feature_scaler'])
    preprocessor.target_scaler.mean_ = preprocessor_data['target_scaler_mean']
    preprocessor.target_scaler.scale_ = preprocessor_data['target_scaler_scale']
    preprocessor.feature_columns = preprocessor_data['feature_columns']
    preprocessor.target_columns = preprocessor_data['target_columns']

    # Re-transform with saved scalers
    X = preprocessor.feature_scaler.transform(df[preprocessor.feature_columns].values)
    y = preprocessor.target_scaler.transform(df[preprocessor.target_columns].values)

    print(f"  Using saved scalers with {len(preprocessor.feature_columns)} features")

    print(f"\nSplitting data and creating sequences (mode: {args.split_mode})...")
    train_data, val_data, test_data = preprocessor.split_and_sequence(X, y, split_mode=args.split_mode)

    # Create datasets and loaders
    train_dataset = PowerDataset(*train_data)
    val_dataset = PowerDataset(*val_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Fine-tuning configuration with smaller learning rate
    config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'loss_weights': [0.5, 0.5] if num_targets == 2 else [0.5, 0.3, 0.2]
    }

    print(f"\nFine-tuning configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate} (smaller than initial training)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Patience: {args.patience}")
    print(f"  Freeze early layers: {args.freeze_early_layers}")

    # Determine output paths
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_path.parent

    best_model_path = output_path.parent / f"best_{output_path.name}"

    # Fine-tune model
    trainer = Trainer(model, device, config, preprocessor, best_model_path=str(best_model_path))

    print("\n" + "="*60)
    print("FINE-TUNING WITH ATTENTION-BASED POOLING")
    print("="*60)
    print("\nThis replaces uniform global average pooling with learnable")
    print("attention-based pooling. The model will learn which timesteps")
    print("are most important for the final prediction, providing gradient")
    print("signals that encourage the multi-head attention to maintain")
    print("meaningful temporal structure.")
    print("="*60)

    trainer.train(train_loader, val_loader)

    # Save final fine-tuned model
    torch.save({
        'model_state_dict': model.state_dict(),
        'preprocessor': {
            'feature_scaler': pickle.dumps(preprocessor.feature_scaler),
            'target_scaler_mean': preprocessor.target_scaler.mean_,
            'target_scaler_scale': preprocessor.target_scaler.scale_,
            'feature_columns': preprocessor.feature_columns,
            'target_columns': preprocessor.target_columns,
            'sequence_length': preprocessor.sequence_length
        },
        'config': config,
        'history': trainer.history,
        'finetune_info': {
            'original_model': args.model,
            'frozen_layers': args.freeze_early_layers
        }
    }, output_path)

    print(f"\nFinal fine-tuned model saved to: {output_path}")
    print(f"Best fine-tuned model saved to: {best_model_path}")

    # Plot training history
    plot_training_history(trainer.history, str(output_dir))

    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print(f"  1. Analyze attention patterns with scripts/analyze_attention.py")
    print(f"  2. Compare to original collapsed attention patterns")
    print(f"  3. Verify attention now focuses on meaningful timesteps")
    print("="*60)


if __name__ == "__main__":
    main()

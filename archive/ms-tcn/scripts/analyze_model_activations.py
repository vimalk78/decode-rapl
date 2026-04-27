#!/usr/bin/env python3
"""
Model Activation Analyzer

Hooks into the MS-TCN model's intermediate layers to analyze activations
and detect dead/saturated neurons that might cause constant predictions.

This helps diagnose if the model architecture has issues like:
- Dead neurons (always output zero)
- Saturated neurons (always output same value)
- Layers that don't respond to input changes
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from train_model import MSTCN


class ActivationHook:
    """Hook to capture layer activations."""

    def __init__(self, name):
        self.name = name
        self.activations = []

    def __call__(self, module, input, output):
        # Store activation
        self.activations.append(output.detach().cpu().numpy())

    def clear(self):
        self.activations = []

    def get_stats(self):
        """Get statistics about captured activations."""
        if not self.activations:
            return None

        # Concatenate all activations
        all_acts = np.concatenate([a.reshape(-1) for a in self.activations])

        return {
            'mean': np.mean(all_acts),
            'std': np.std(all_acts),
            'min': np.min(all_acts),
            'max': np.max(all_acts),
            'median': np.median(all_acts),
            'num_zero': np.sum(all_acts == 0),
            'num_saturated_pos': np.sum(all_acts >= 0.99),  # For ReLU
            'num_saturated_neg': np.sum(all_acts <= -0.99),
            'total': len(all_acts),
        }


def load_model(model_path):
    """Load trained model and preprocessing parameters."""

    if not Path(model_path).exists():
        print(f"Error: Model file '{model_path}' not found")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if 'preprocessor' not in checkpoint:
        print("Error: Model file missing preprocessor parameters")
        sys.exit(1)

    preprocessor_params = checkpoint['preprocessor']

    # Setup scalers
    feature_scaler = StandardScaler()
    feature_scaler.mean_ = preprocessor_params['feature_scaler_mean']
    feature_scaler.scale_ = preprocessor_params['feature_scaler_scale']

    target_scaler = StandardScaler()
    target_scaler.mean_ = preprocessor_params['target_scaler_mean']
    target_scaler.scale_ = preprocessor_params['target_scaler_scale']

    # Create model
    num_features = len(preprocessor_params['feature_columns'])
    num_targets = len(preprocessor_params['target_columns'])

    model = MSTCN(num_features=num_features, num_targets=num_targets)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded")

    return model, feature_scaler, target_scaler, preprocessor_params


def register_hooks(model):
    """Register activation hooks on model layers."""

    hooks = {}
    handles = []

    # Hook into each TCN stage
    for i, stage in enumerate([model.stage1, model.stage2, model.stage3]):
        hook = ActivationHook(f'stage{i+1}')
        hooks[f'stage{i+1}'] = hook
        handle = stage.register_forward_hook(hook)
        handles.append(handle)

    # Hook into final layers
    fc_hook = ActivationHook('fc')
    hooks['fc'] = fc_hook
    handles.append(model.fc.register_forward_hook(fc_hook))

    return hooks, handles


def create_test_sequences(feature_cols, seq_len=64):
    """Create test sequences with different CPU loads."""

    # Create idle and high-load sequences
    sequences = {}

    # Idle sequence
    idle_features = {
        'cpu_user_percent': 0.0,
        'cpu_system_percent': 0.0,
        'cpu_idle_percent': 100.0,
        'cpu_iowait_percent': 0.0,
        'cpu_irq_percent': 0.0,
        'cpu_softirq_percent': 0.0,
        'context_switches_sec': 1000.0,
        'interrupts_sec': 500.0,
        'memory_used_mb': 4000.0,
        'memory_cached_mb': 2000.0,
        'memory_buffers_mb': 500.0,
        'memory_free_mb': 8000.0,
        'swap_used_mb': 0.0,
        'page_faults_sec': 100.0,
        'load_1min': 0.0,
        'load_5min': 0.0,
        'load_15min': 0.0,
        'running_processes': 1.0,
        'blocked_processes': 0.0,
    }

    # High load sequence
    high_load_features = idle_features.copy()
    high_load_features['cpu_user_percent'] = 90.0
    high_load_features['cpu_system_percent'] = 10.0
    high_load_features['cpu_idle_percent'] = 0.0
    high_load_features['context_switches_sec'] = 10000.0
    high_load_features['interrupts_sec'] = 5000.0
    high_load_features['load_1min'] = 8.0
    high_load_features['load_5min'] = 6.0
    high_load_features['load_15min'] = 4.0
    high_load_features['running_processes'] = 8.0

    # Build sequences
    for name, features_dict in [('idle', idle_features), ('high_load', high_load_features)]:
        sequence = []
        for _ in range(seq_len):
            row = [features_dict.get(col, 0.0) for col in feature_cols]
            sequence.append(row)
        sequences[name] = np.array(sequence)

    return sequences


def analyze_activations(model_path):
    """Analyze model activations for different inputs."""

    # Load model
    model, feature_scaler, target_scaler, params = load_model(model_path)

    feature_cols = params['feature_columns']
    seq_len = params['sequence_length']

    # Register hooks
    hooks, handles = register_hooks(model)

    print("\n" + "="*80)
    print("Analyzing Model Activations")
    print("="*80)

    # Create test sequences
    sequences = create_test_sequences(feature_cols, seq_len)

    results = {}

    with torch.no_grad():
        for seq_name, sequence in sequences.items():
            print(f"\nTesting with {seq_name} sequence...")

            # Clear previous activations
            for hook in hooks.values():
                hook.clear()

            # Normalize
            sequence_normalized = feature_scaler.transform(sequence)

            # Predict
            seq_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0)
            prediction_normalized = model(seq_tensor)
            prediction = target_scaler.inverse_transform(prediction_normalized.numpy())[0]

            print(f"  Prediction: {prediction[0]:.2f}W")

            # Get activation statistics
            results[seq_name] = {}
            for hook_name, hook in hooks.items():
                stats = hook.get_stats()
                results[seq_name][hook_name] = stats

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Print comparison
    print("\n" + "="*80)
    print("Activation Statistics Comparison")
    print("="*80)

    for hook_name in hooks.keys():
        print(f"\n{hook_name}:")
        print(f"  {'Metric':<20} {'Idle':<20} {'High Load':<20} {'Difference'}")
        print("  " + "-"*76)

        idle_stats = results['idle'][hook_name]
        high_stats = results['high_load'][hook_name]

        metrics = ['mean', 'std', 'min', 'max']
        for metric in metrics:
            idle_val = idle_stats[metric]
            high_val = high_stats[metric]
            diff = abs(high_val - idle_val)

            print(f"  {metric:<20} {idle_val:<20.6f} {high_val:<20.6f} {diff:.6f}")

        # Check for dead neurons
        idle_zero_pct = (idle_stats['num_zero'] / idle_stats['total']) * 100
        high_zero_pct = (high_stats['num_zero'] / high_stats['total']) * 100

        print(f"\n  Dead neurons (always 0):")
        print(f"    Idle: {idle_zero_pct:.1f}% ({idle_stats['num_zero']}/{idle_stats['total']})")
        print(f"    High: {high_zero_pct:.1f}% ({high_stats['num_zero']}/{high_stats['total']})")

        if idle_zero_pct > 50 or high_zero_pct > 50:
            print(f"    ⚠️  WARNING: >50% neurons always output zero!")

    # Diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    issues = []

    for hook_name in hooks.keys():
        idle_stats = results['idle'][hook_name]
        high_stats = results['high_load'][hook_name]

        # Check if activations change between idle and high load
        mean_diff = abs(high_stats['mean'] - idle_stats['mean'])
        std_diff = abs(high_stats['std'] - idle_stats['std'])

        if mean_diff < 0.01 and std_diff < 0.01:
            issues.append(f"{hook_name}: Activations don't change between idle and high load")

        # Check for too many dead neurons
        idle_zero_pct = (idle_stats['num_zero'] / idle_stats['total']) * 100
        if idle_zero_pct > 70:
            issues.append(f"{hook_name}: {idle_zero_pct:.0f}% of neurons always output zero (dead)")

    if issues:
        print("\n⚠️  ISSUES DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\nThis suggests:")
        print("  - Model layers are not responding to input changes")
        print("  - Many neurons are dead (always output zero)")
        print("  - Model architecture may need redesign")
        print("  - Or training process failed to activate neurons properly")
    else:
        print("\n✓ Activations look healthy")
        print("  - Layers respond to input changes")
        print("  - Neurons are active (not dead)")
        print("  - Architecture appears functional")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model activations to detect dead/saturated neurons"
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth)')

    args = parser.parse_args()

    analyze_activations(args.model)


if __name__ == '__main__':
    main()

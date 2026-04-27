#!/usr/bin/env python3
"""
Test if the trained model actually responds to different CPU loads.

This diagnostic script creates synthetic test cases with varying CPU utilization
and checks if the model's predictions change accordingly. If predictions are
constant regardless of input, the model didn't learn the relationship.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from train_model import MSTCN
from sklearn.preprocessing import StandardScaler


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

    print(f"✓ Model loaded successfully")
    print(f"  Features: {preprocessor_params['feature_columns']}")
    print(f"  Targets: {preprocessor_params['target_columns']}")
    print(f"  Sequence length: {preprocessor_params['sequence_length']}\n")

    return model, feature_scaler, target_scaler, preprocessor_params


def create_synthetic_sequence(cpu_percent, feature_cols, seq_len=64):
    """Create synthetic feature sequence for given CPU utilization."""

    # Create baseline features (typical idle values)
    baseline = {
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

    # Adjust CPU metrics based on load
    features = baseline.copy()
    features['cpu_user_percent'] = cpu_percent * 0.9  # Mostly user time
    features['cpu_system_percent'] = cpu_percent * 0.1  # Some system time
    features['cpu_idle_percent'] = 100.0 - cpu_percent

    # Higher CPU = more context switches and interrupts
    features['context_switches_sec'] = 1000.0 + cpu_percent * 100.0
    features['interrupts_sec'] = 500.0 + cpu_percent * 50.0

    # Higher CPU = higher load average
    features['load_1min'] = cpu_percent / 100.0 * 4.0
    features['load_5min'] = cpu_percent / 100.0 * 3.0
    features['load_15min'] = cpu_percent / 100.0 * 2.0
    features['running_processes'] = 1.0 + int(cpu_percent / 25.0)

    # Create sequence (all timesteps have same values for simplicity)
    sequence = []
    for _ in range(seq_len):
        row = [features.get(col, 0.0) for col in feature_cols]
        sequence.append(row)

    return np.array(sequence)


def test_model_responsiveness(model_path):
    """Test if model responds to different CPU loads."""

    # Load model
    model, feature_scaler, target_scaler, params = load_model(model_path)

    feature_cols = params['feature_columns']
    target_cols = params['target_columns']
    seq_len = params['sequence_length']

    # Test cases: different CPU loads
    test_cases = [0, 10, 25, 50, 75, 90, 100]

    print("="*80)
    print("Model Responsiveness Test")
    print("="*80)
    print("\nTesting model predictions at different CPU utilization levels...")
    print("If model is working, predictions should increase with CPU load.\n")

    print(f"{'CPU Load':<12} {'Predicted Power':<20} {'Expected Behavior'}")
    print("-" * 70)

    predictions = []

    with torch.no_grad():
        for cpu_load in test_cases:
            # Create synthetic sequence
            sequence = create_synthetic_sequence(cpu_load, feature_cols, seq_len)

            # Normalize
            sequence_normalized = feature_scaler.transform(sequence)

            # Predict
            seq_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0)
            prediction_normalized = model(seq_tensor)
            prediction = target_scaler.inverse_transform(prediction_normalized.numpy())[0]

            predictions.append((cpu_load, prediction[0]))

            # Expected behavior
            if cpu_load == 0:
                expected = "Idle power (lowest)"
            elif cpu_load == 100:
                expected = "Max power (highest)"
            else:
                expected = f"~{cpu_load}% of range"

            # Print result
            power_str = f"{prediction[0]:.2f}W"
            if len(target_cols) > 1:
                power_str += f" (pkg), {prediction[1]:.2f}W (core)"

            print(f"{cpu_load:>3}%         {power_str:<20} {expected}")

    print("\n" + "="*80)
    print("Analysis")
    print("="*80)

    # Analyze predictions
    pred_values = [p[1] for p in predictions]
    min_pred = min(pred_values)
    max_pred = max(pred_values)
    range_pred = max_pred - min_pred
    mean_pred = np.mean(pred_values)
    std_pred = np.std(pred_values)

    print(f"\nPrediction Statistics:")
    print(f"  Minimum:  {min_pred:.2f}W (at {predictions[pred_values.index(min_pred)][0]}% CPU)")
    print(f"  Maximum:  {max_pred:.2f}W (at {predictions[pred_values.index(max_pred)][0]}% CPU)")
    print(f"  Range:    {range_pred:.2f}W")
    print(f"  Mean:     {mean_pred:.2f}W")
    print(f"  Std Dev:  {std_pred:.2f}W")

    # Diagnosis
    print(f"\n{'Diagnosis:':}")

    if range_pred < 5.0:
        print(f"  ❌ FAILED - Model is NOT responsive to CPU load")
        print(f"  Predictions vary by only {range_pred:.2f}W across 0-100% CPU")
        print(f"  Model appears to be predicting a constant value (~{mean_pred:.1f}W)")
        print(f"\n  This means the model didn't learn the relationship between")
        print(f"  CPU metrics and power consumption. It just learned to predict")
        print(f"  the average/mode of the training data distribution.")
        print(f"\n  Recommendation: Retrain with better data or different architecture")
    elif range_pred < 15.0:
        print(f"  ⚠️  WEAK - Model shows limited response to CPU load")
        print(f"  Predictions vary by {range_pred:.2f}W across 0-100% CPU")
        print(f"  Model learned some relationship but underfitted")
        print(f"\n  Recommendation: Check training data quality and training duration")
    else:
        print(f"  ✓ GOOD - Model responds to CPU load changes")
        print(f"  Predictions vary by {range_pred:.2f}W across 0-100% CPU")

        # Check if monotonic (predictions increase with load)
        is_monotonic = all(predictions[i][1] <= predictions[i+1][1] for i in range(len(predictions)-1))

        if is_monotonic:
            print(f"  ✓ Predictions increase monotonically with CPU load")
        else:
            print(f"  ⚠️  Predictions are not monotonic (may indicate issues)")

    print("\n" + "="*80)

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Test if trained model responds to different CPU loads"
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth)')

    args = parser.parse_args()

    test_model_responsiveness(args.model)


if __name__ == '__main__':
    main()

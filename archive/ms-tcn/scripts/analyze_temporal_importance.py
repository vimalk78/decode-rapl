#!/usr/bin/env python3
"""
Temporal Feature Importance Analyzer

Tests if the MS-TCN model actually uses temporal context from the 64-sample
history or if it only looks at the most recent sample.

This is critical because:
- MS-TCN is designed to capture temporal patterns
- If it ignores history, it's essentially a feed-forward network
- This could explain constant predictions (no temporal dynamics)

The script tests:
1. Varying only recent samples vs only old samples
2. Measuring prediction sensitivity to each
3. Checking if temporal receptive field works
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


def create_baseline_sequence(feature_cols, seq_len=64, cpu_load=0):
    """Create baseline sequence with specific CPU load."""

    features = {
        'cpu_user_percent': cpu_load * 0.9,
        'cpu_system_percent': cpu_load * 0.1,
        'cpu_idle_percent': 100.0 - cpu_load,
        'cpu_iowait_percent': 0.0,
        'cpu_irq_percent': 0.0,
        'cpu_softirq_percent': 0.0,
        'context_switches_sec': 1000.0 + cpu_load * 100.0,
        'interrupts_sec': 500.0 + cpu_load * 50.0,
        'memory_used_mb': 4000.0,
        'memory_cached_mb': 2000.0,
        'memory_buffers_mb': 500.0,
        'memory_free_mb': 8000.0,
        'swap_used_mb': 0.0,
        'page_faults_sec': 100.0,
        'load_1min': cpu_load / 100.0 * 4.0,
        'load_5min': cpu_load / 100.0 * 3.0,
        'load_15min': cpu_load / 100.0 * 2.0,
        'running_processes': 1.0 + int(cpu_load / 25.0),
        'blocked_processes': 0.0,
    }

    sequence = []
    for _ in range(seq_len):
        row = [features.get(col, 0.0) for col in feature_cols]
        sequence.append(row)

    return np.array(sequence)


def predict_sequence(model, feature_scaler, target_scaler, sequence):
    """Make prediction for a sequence."""

    with torch.no_grad():
        # Normalize
        sequence_normalized = feature_scaler.transform(sequence)

        # Predict
        seq_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0)
        prediction_normalized = model(seq_tensor)
        prediction = target_scaler.inverse_transform(prediction_normalized.numpy())[0]

        return prediction[0]


def analyze_temporal_importance(model_path):
    """Analyze if model uses temporal context."""

    # Load model
    model, feature_scaler, target_scaler, params = load_model(model_path)

    feature_cols = params['feature_columns']
    seq_len = params['sequence_length']

    print("\n" + "="*80)
    print("Temporal Context Importance Analysis")
    print("="*80)

    print(f"\nSequence length: {seq_len} samples")
    print(f"Testing if model uses temporal history or only recent samples...\n")

    # Test 1: All idle vs only recent samples are high-load
    print("="*80)
    print("Test 1: Recent samples vs Full history")
    print("="*80)

    # Scenario A: All 64 samples are idle
    seq_all_idle = create_baseline_sequence(feature_cols, seq_len, cpu_load=0)
    pred_all_idle = predict_sequence(model, feature_scaler, target_scaler, seq_all_idle)

    # Scenario B: First 60 samples idle, last 4 samples high-load
    seq_recent_high = create_baseline_sequence(feature_cols, seq_len, cpu_load=0)
    seq_recent_high[-4:] = create_baseline_sequence(feature_cols, 4, cpu_load=100)
    pred_recent_high = predict_sequence(model, feature_scaler, target_scaler, seq_recent_high)

    # Scenario C: All 64 samples are high-load
    seq_all_high = create_baseline_sequence(feature_cols, seq_len, cpu_load=100)
    pred_all_high = predict_sequence(model, feature_scaler, target_scaler, seq_all_high)

    print(f"A. All 64 samples idle (0% CPU):")
    print(f"   Prediction: {pred_all_idle:.2f}W")

    print(f"\nB. First 60 idle, last 4 high-load (100% CPU):")
    print(f"   Prediction: {pred_recent_high:.2f}W")
    print(f"   Change from all-idle: {pred_recent_high - pred_all_idle:+.2f}W")

    print(f"\nC. All 64 samples high-load (100% CPU):")
    print(f"   Prediction: {pred_all_high:.2f}W")
    print(f"   Change from all-idle: {pred_all_high - pred_all_idle:+.2f}W")

    # Analysis
    recent_sensitivity = abs(pred_recent_high - pred_all_idle)
    full_sensitivity = abs(pred_all_high - pred_all_idle)

    if recent_sensitivity > 5.0:
        print("\n✓ Model responds to recent samples (last 4)")
    else:
        print("\n✗ Model does NOT respond to recent samples")

    if full_sensitivity > 10.0:
        print("✓ Model responds strongly to full history")
    else:
        print("✗ Model does NOT respond to full history")

    # Test 2: Vary position of high-load window
    print("\n" + "="*80)
    print("Test 2: Temporal Receptive Field (position sensitivity)")
    print("="*80)
    print("\nInserting 4-sample high-load window at different positions...")
    print("If model uses temporal context, earlier positions should matter.\n")

    positions = [0, 16, 32, 48, 60]  # Start positions for 4-sample window
    results = []

    for pos in positions:
        # Create sequence: all idle except 4 samples at position
        seq = create_baseline_sequence(feature_cols, seq_len, cpu_load=0)
        high_window = create_baseline_sequence(feature_cols, 4, cpu_load=100)
        seq[pos:pos+4] = high_window

        pred = predict_sequence(model, feature_scaler, target_scaler, seq)
        change = pred - pred_all_idle

        results.append({
            'position': pos,
            'end_position': pos + 4,
            'prediction': pred,
            'change_from_idle': change,
        })

        print(f"Position {pos:>2}-{pos+4:<2} (samples {pos:>2} to {pos+4:<2}): {pred:>6.2f}W  (Δ {change:+.2f}W)")

    # Check if position matters
    changes = [r['change_from_idle'] for r in results]
    position_variance = np.var(changes)

    print(f"\nVariance in prediction changes: {position_variance:.4f}")

    if position_variance > 5.0:
        print("✓ Model is sensitive to window position (uses temporal context)")
    elif position_variance > 1.0:
        print("⚠️  Model shows some position sensitivity (weak temporal context)")
    else:
        print("✗ Model ignores window position (NOT using temporal context)")

    # Test 3: Gradual transition vs sudden transition
    print("\n" + "="*80)
    print("Test 3: Temporal Dynamics (gradual vs sudden changes)")
    print("="*80)

    # Scenario A: Sudden jump from idle to high
    seq_sudden = create_baseline_sequence(feature_cols, seq_len, cpu_load=0)
    seq_sudden[32:] = create_baseline_sequence(feature_cols, 32, cpu_load=100)
    pred_sudden = predict_sequence(model, feature_scaler, target_scaler, seq_sudden)

    # Scenario B: Gradual ramp from idle to high
    seq_gradual = np.zeros((seq_len, len(feature_cols)))
    for i in range(seq_len):
        cpu_load = (i / seq_len) * 100.0  # 0% to 100% gradually
        seq_gradual[i] = create_baseline_sequence(feature_cols, 1, cpu_load)
    pred_gradual = predict_sequence(model, feature_scaler, target_scaler, seq_gradual)

    print(f"\nA. Sudden transition (32 idle, then 32 high-load):")
    print(f"   Prediction: {pred_sudden:.2f}W")

    print(f"\nB. Gradual ramp (0% → 100% linearly):")
    print(f"   Prediction: {pred_gradual:.2f}W")

    print(f"\nDifference: {abs(pred_sudden - pred_gradual):.2f}W")

    if abs(pred_sudden - pred_gradual) > 5.0:
        print("✓ Model captures temporal dynamics (transition pattern matters)")
    else:
        print("✗ Model ignores temporal dynamics (only cares about average)")

    # Overall conclusion
    print("\n" + "="*80)
    print("OVERALL DIAGNOSIS")
    print("="*80)

    issues = []

    if full_sensitivity < 10.0:
        issues.append("Model does NOT respond to CPU load changes")

    if position_variance < 1.0:
        issues.append("Model does NOT use temporal context (ignores sample position)")

    if abs(pred_sudden - pred_gradual) < 3.0:
        issues.append("Model does NOT capture temporal dynamics")

    if issues:
        print("\n⚠️  CRITICAL ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\nThis suggests:")
        print("  - The MS-TCN architecture is NOT working as intended")
        print("  - Temporal convolutions may be ineffective")
        print("  - Model might be collapsing to average/constant prediction")
        print("  - Architecture needs redesign or training process is broken")

        print("\nPossible causes:")
        print("  - Dilated convolutions have zero-valued kernels")
        print("  - Gradient flow issues during training")
        print("  - Learning rate too high (weights didn't converge)")
        print("  - Or too low (weights never moved from initialization)")
    else:
        print("\n✓ Temporal context appears to work correctly")
        print("  - Model uses history from multiple time positions")
        print("  - Temporal dynamics are captured")
        print("  - Architecture is functional")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze if model uses temporal context"
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth)')

    args = parser.parse_args()

    analyze_temporal_importance(args.model)


if __name__ == '__main__':
    main()

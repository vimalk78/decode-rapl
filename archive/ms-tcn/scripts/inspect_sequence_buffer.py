#!/usr/bin/env python3
"""
Sequence Buffer Inspector

Inspects the 64-sample sequence buffer used by the MS-TCN model
to understand if buffer initialization or contamination causes
incorrect predictions at startup.

The model uses a sliding window of 64 samples. This script:
1. Shows how the buffer evolves from startup to steady state
2. Tracks prediction changes as buffer fills
3. Identifies if predictions stabilize after certain number of samples
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
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
    print(f"  Sequence length: {preprocessor_params['sequence_length']}")
    print(f"  Features: {num_features}")
    print(f"  Targets: {num_targets}")

    return model, feature_scaler, target_scaler, preprocessor_params


def inspect_buffer_evolution(model_path, test_csv, output_csv=None):
    """Inspect how buffer evolves during live prediction."""

    # Load model
    model, feature_scaler, target_scaler, params = load_model(model_path)

    feature_cols = params['feature_columns']
    target_cols = params['target_columns']
    seq_len = params['sequence_length']

    # Load test data
    print(f"\nLoading test data from {test_csv}...")
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} samples")

    # Extract features and targets
    features = df[feature_cols].values
    if 'rapl_package_power' in df.columns:
        actual_power = df['rapl_package_power'].values
    else:
        actual_power = None

    print(f"\nSimulating live prediction (filling 64-sample buffer)...")
    print("="*80)

    # Initialize buffer (like in live prediction)
    buffer = np.zeros((seq_len, len(feature_cols)))
    buffer_fill_count = 0

    results = []

    with torch.no_grad():
        for i in range(min(len(features), 200)):  # First 200 samples
            # Add sample to buffer (like live prediction does)
            buffer = np.roll(buffer, -1, axis=0)
            buffer[-1] = features[i]
            buffer_fill_count = min(buffer_fill_count + 1, seq_len)

            # Normalize buffer
            buffer_normalized = feature_scaler.transform(buffer)

            # Predict
            seq_tensor = torch.FloatTensor(buffer_normalized).unsqueeze(0)
            prediction_normalized = model(seq_tensor)
            prediction = target_scaler.inverse_transform(prediction_normalized.numpy())[0]

            # Calculate buffer statistics
            buffer_mean = np.mean(buffer)
            buffer_std = np.std(buffer)
            buffer_nonzero_pct = (np.count_nonzero(buffer) / buffer.size) * 100

            # Store result
            result = {
                'sample': i + 1,
                'buffer_fill_count': buffer_fill_count,
                'buffer_fill_pct': (buffer_fill_count / seq_len) * 100,
                'predicted_power': prediction[0],
                'buffer_mean': buffer_mean,
                'buffer_std': buffer_std,
                'buffer_nonzero_pct': buffer_nonzero_pct,
            }

            if actual_power is not None:
                result['actual_power'] = actual_power[i]
                result['error_pct'] = abs(prediction[0] - actual_power[i]) / actual_power[i] * 100

            results.append(result)

            # Print periodic updates
            if i < 65 or i % 20 == 0:
                status = f"Sample {i+1:>3}: "
                status += f"Buffer={buffer_fill_count:>2}/{seq_len} ({buffer_fill_count/seq_len*100:>3.0f}%) "
                status += f"Pred={prediction[0]:>5.1f}W "
                if actual_power is not None:
                    status += f"Actual={actual_power[i]:>5.1f}W "
                    status += f"Err={abs(prediction[0] - actual_power[i])/actual_power[i]*100:>4.1f}%"
                print(status)

    results_df = pd.DataFrame(results)

    # Save results
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to {output_csv}")

    # Analysis
    print("\n" + "="*80)
    print("Buffer Evolution Analysis")
    print("="*80)

    # Check if predictions stabilize after buffer fills
    if len(results_df) >= seq_len:
        pre_fill = results_df[results_df['buffer_fill_count'] < seq_len]
        post_fill = results_df[results_df['buffer_fill_count'] == seq_len]

        print(f"\nBefore buffer fully filled (samples 1-{seq_len-1}):")
        print(f"  Mean prediction: {pre_fill['predicted_power'].mean():.2f}W")
        print(f"  Std deviation: {pre_fill['predicted_power'].std():.2f}W")
        print(f"  Range: {pre_fill['predicted_power'].min():.2f}W - {pre_fill['predicted_power'].max():.2f}W")

        print(f"\nAfter buffer fully filled (samples {seq_len}+):")
        print(f"  Mean prediction: {post_fill['predicted_power'].mean():.2f}W")
        print(f"  Std deviation: {post_fill['predicted_power'].std():.2f}W")
        print(f"  Range: {post_fill['predicted_power'].min():.2f}W - {post_fill['predicted_power'].max():.2f}W")

        # Check if there's a significant change
        diff = abs(post_fill['predicted_power'].mean() - pre_fill['predicted_power'].mean())
        if diff > 5.0:
            print(f"\n⚠️  Predictions changed significantly after buffer filled ({diff:.1f}W difference)")
            print("   This suggests buffer initialization affects predictions.")
        else:
            print(f"\n✓ Predictions remained stable after buffer filled ({diff:.1f}W difference)")

        # Check if predictions are constant after buffer fills
        post_fill_range = post_fill['predicted_power'].max() - post_fill['predicted_power'].min()
        if post_fill_range < 5.0:
            print(f"\n⚠️  Predictions are nearly constant after buffer fills (range: {post_fill_range:.2f}W)")
            print("   This suggests the model is not responding to changing inputs.")
        else:
            print(f"\n✓ Predictions vary after buffer fills (range: {post_fill_range:.2f}W)")

    # Check for startup artifacts
    first_10 = results_df.head(10)
    print(f"\nFirst 10 predictions:")
    for _, row in first_10.iterrows():
        print(f"  Sample {int(row['sample']):>3}: {row['predicted_power']:>6.2f}W (buffer {int(row['buffer_fill_count'])}/{seq_len})")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if len(results_df) < seq_len:
        print("\n⚠️  Not enough samples to fully analyze buffer behavior")
    else:
        post_fill = results_df[results_df['buffer_fill_count'] == seq_len]
        post_fill_range = post_fill['predicted_power'].max() - post_fill['predicted_power'].min()

        if post_fill_range < 5.0:
            print("\n⚠️  ISSUE DETECTED: Buffer filling is NOT the problem")
            print("   Even after buffer is fully filled with real data,")
            print("   predictions remain constant. This points to:")
            print("   - Model architecture issue (not learning temporal patterns)")
            print("   - Or training issue (model learned to predict constant)")
        else:
            print("\n✓ Buffer filling works correctly")
            print("  Predictions vary appropriately after buffer is filled.")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Inspect sequence buffer evolution during prediction"
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth)')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--output', type=str,
                       help='Output CSV for buffer evolution data')

    args = parser.parse_args()

    results = inspect_buffer_evolution(
        args.model,
        args.test_data,
        args.output
    )


if __name__ == '__main__':
    main()

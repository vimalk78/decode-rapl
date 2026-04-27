#!/usr/bin/env python3
"""
Prediction Sensitivity Analyzer

Tests which features the model actually pays attention to by:
1. Starting with baseline idle features
2. Perturbing each feature individually
3. Measuring how much predictions change

This helps identify if the model ignores certain features that should
be important (like cpu_user_percent for idle detection).
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

    return model, feature_scaler, target_scaler, preprocessor_params


def create_baseline_sequence(feature_cols, seq_len=64, scenario='idle'):
    """Create baseline feature sequence."""

    if scenario == 'idle':
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
    else:  # high_load
        baseline = {
            'cpu_user_percent': 90.0,
            'cpu_system_percent': 10.0,
            'cpu_idle_percent': 0.0,
            'cpu_iowait_percent': 0.0,
            'cpu_irq_percent': 0.0,
            'cpu_softirq_percent': 0.0,
            'context_switches_sec': 10000.0,
            'interrupts_sec': 5000.0,
            'memory_used_mb': 4000.0,
            'memory_cached_mb': 2000.0,
            'memory_buffers_mb': 500.0,
            'memory_free_mb': 8000.0,
            'swap_used_mb': 0.0,
            'page_faults_sec': 100.0,
            'load_1min': 8.0,
            'load_5min': 6.0,
            'load_15min': 4.0,
            'running_processes': 8.0,
            'blocked_processes': 0.0,
        }

    sequence = []
    for _ in range(seq_len):
        row = [baseline.get(col, 0.0) for col in feature_cols]
        sequence.append(row)

    return np.array(sequence), baseline


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


def analyze_feature_sensitivity(model_path, scenario='idle'):
    """Analyze how sensitive predictions are to each feature."""

    # Load model
    model, feature_scaler, target_scaler, params = load_model(model_path)

    feature_cols = params['feature_columns']
    seq_len = params['sequence_length']

    print("\n" + "="*80)
    print(f"Feature Sensitivity Analysis ({scenario.upper()})")
    print("="*80)

    # Create baseline sequence
    baseline_sequence, baseline_features = create_baseline_sequence(
        feature_cols, seq_len, scenario
    )

    # Get baseline prediction
    baseline_pred = predict_sequence(
        model, feature_scaler, target_scaler, baseline_sequence
    )

    print(f"\nBaseline prediction: {baseline_pred:.2f}W")
    print(f"\nTesting sensitivity to each feature...")
    print("(Perturbing each feature by ±10%, ±50%, ±100%)\n")

    # Test each feature
    results = []

    for i, feature_name in enumerate(feature_cols):
        baseline_value = baseline_features.get(feature_name, 0.0)

        # Perturbation amounts
        perturbations = []

        if abs(baseline_value) > 1e-6:  # Not zero
            perturbations = [
                ('−50%', baseline_value * 0.5),
                ('−10%', baseline_value * 0.9),
                ('+10%', baseline_value * 1.1),
                ('+50%', baseline_value * 1.5),
                ('+100%', baseline_value * 2.0),
            ]
        else:  # Zero baseline - add absolute values
            perturbations = [
                ('+10', 10.0),
                ('+50', 50.0),
                ('+100', 100.0),
            ]

        max_change = 0.0
        best_perturbation = None

        for perturb_name, perturb_value in perturbations:
            # Create perturbed sequence
            perturbed_sequence = baseline_sequence.copy()
            perturbed_sequence[:, i] = perturb_value

            # Predict
            pred = predict_sequence(
                model, feature_scaler, target_scaler, perturbed_sequence
            )

            change = abs(pred - baseline_pred)

            if change > max_change:
                max_change = change
                best_perturbation = perturb_name

        results.append({
            'feature': feature_name,
            'baseline_value': baseline_value,
            'max_prediction_change': max_change,
            'best_perturbation': best_perturbation,
        })

    # Sort by sensitivity (largest change first)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('max_prediction_change', ascending=False)

    # Print results
    print(f"{'Feature':<30} {'Baseline':<12} {'Max Δ Pred':<12} {'Status'}")
    print("-" * 80)

    sensitive_features = []
    ignored_features = []

    for _, row in results_df.iterrows():
        feature = row['feature']
        baseline_val = row['baseline_value']
        max_change = row['max_prediction_change']

        # Classify sensitivity
        if max_change > 5.0:
            status = "✓ HIGH sensitivity"
            sensitive_features.append(feature)
        elif max_change > 1.0:
            status = "⚠️  LOW sensitivity"
        else:
            status = "✗ IGNORED by model"
            ignored_features.append(feature)

        print(f"{feature:<30} {baseline_val:>11.2f} {max_change:>11.2f}W  {status}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print(f"\nSensitive features ({len(sensitive_features)}):")
    if sensitive_features:
        for feature in sensitive_features:
            print(f"  ✓ {feature}")
    else:
        print("  (none)")

    print(f"\nIgnored features ({len(ignored_features)}):")
    if ignored_features:
        for feature in ignored_features:
            print(f"  ✗ {feature}")
    else:
        print("  (none)")

    # Check if important features are ignored
    important_for_idle = ['cpu_user_percent', 'cpu_idle_percent', 'cpu_system_percent']
    important_for_load = ['cpu_user_percent', 'load_1min', 'context_switches_sec']

    if scenario == 'idle':
        check_features = important_for_idle
    else:
        check_features = important_for_load

    ignored_important = [f for f in check_features if f in ignored_features]

    if ignored_important:
        print("\n⚠️  CRITICAL ISSUE DETECTED:")
        print(f"   Model ignores these important features for {scenario}:")
        for feature in ignored_important:
            print(f"     - {feature}")
        print("\n   This explains why predictions are wrong!")
        print("   The model cannot distinguish different states if it")
        print("   ignores the features that define those states.")
    else:
        print(f"\n✓ Model pays attention to important features for {scenario}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze which features the model is sensitive to"
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth)')
    parser.add_argument('--scenario', type=str, default='idle',
                       choices=['idle', 'high_load'],
                       help='Scenario to test (idle or high_load)')

    args = parser.parse_args()

    results = analyze_feature_sensitivity(args.model, args.scenario)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Training Loss Distribution Analyzer

Analyzes why the model might predict ~40W by calculating what the
training loss would be for different constant predictions.

If predicting 40W minimizes loss better than predicting actual values,
this explains the model's behavior.

The model was trained with MSE (Mean Squared Error) loss, which means:
- Large errors are heavily penalized (squared)
- Model may prefer "safe" predictions that avoid large errors
- Predicting median/mode might minimize loss better than being accurate
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_training_data(training_csv):
    """Load training data."""

    print(f"Loading training data from {training_csv}...")
    df = pd.read_csv(training_csv)
    print(f"Loaded {len(df)} samples")

    return df


def analyze_loss_distribution(training_csv, output_plot=None):
    """Analyze training loss distribution for different constant predictions."""

    # Load data
    df = load_training_data(training_csv)

    if 'rapl_package_power' not in df.columns:
        print("ERROR: Training data missing 'rapl_package_power' column")
        sys.exit(1)

    power_values = df['rapl_package_power'].values

    print("\n" + "="*80)
    print("Training Data Power Distribution")
    print("="*80)

    print(f"\nStatistics:")
    print(f"  Mean:   {np.mean(power_values):.2f}W")
    print(f"  Median: {np.median(power_values):.2f}W")
    mode_result = stats.mode(power_values.round(0), keepdims=True)
    print(f"  Mode:   {mode_result.mode[0]:.0f}W")
    print(f"  Std:    {np.std(power_values):.2f}W")
    print(f"  Min:    {np.min(power_values):.2f}W")
    print(f"  Max:    {np.max(power_values):.2f}W")

    # Power distribution by bins
    bins = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]
    print(f"\nPower distribution:")
    for low, high in bins:
        count = np.sum((power_values >= low) & (power_values < high))
        pct = (count / len(power_values)) * 100
        print(f"  {low:>3}-{high:>3}W: {count:>7} samples ({pct:>5.1f}%)")

    print("\n" + "="*80)
    print("Loss Analysis for Constant Predictions")
    print("="*80)

    # Test different constant predictions
    test_predictions = [27, 30, 35, 40, 44.4, 50, 55, 60, 70]

    loss_results = []

    for const_pred in test_predictions:
        # Calculate MSE loss if model always predicted this value
        errors = power_values - const_pred
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(mse)

        loss_results.append({
            'prediction': const_pred,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
        })

    loss_df = pd.DataFrame(loss_results)
    loss_df = loss_df.sort_values('mse')

    print(f"\nIf model always predicted constant value:")
    print(f"{'Constant Pred':<15} {'MSE':<15} {'MAE':<15} {'RMSE':<15}")
    print("-" * 60)

    best_mse = loss_df.iloc[0]['mse']

    for _, row in loss_df.iterrows():
        pred = row['prediction']
        mse = row['mse']
        mae = row['mae']
        rmse = row['rmse']

        marker = ""
        if mse == best_mse:
            marker = "  ← MINIMUM MSE"
        elif pred == 40.0:
            marker = "  ← Model's actual prediction!"
        elif pred == 44.4:
            marker = "  ← Training mean"

        print(f"{pred:>14.1f}W {mse:>14.2f} {mae:>14.2f} {rmse:>14.2f}{marker}")

    # Analysis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    best_constant = loss_df.iloc[0]['prediction']
    best_mse_value = loss_df.iloc[0]['mse']

    model_pred_40_mse = loss_df[loss_df['prediction'] == 40.0]['mse'].values[0]

    print(f"\nBest constant prediction to minimize MSE: {best_constant:.1f}W")
    print(f"  MSE: {best_mse_value:.2f}")

    print(f"\nIf model predicts 40W for everything:")
    print(f"  MSE: {model_pred_40_mse:.2f}")
    print(f"  Difference from optimal: {model_pred_40_mse - best_mse_value:.2f}")

    if abs(best_constant - 40.0) < 5.0:
        print("\n⚠️  CRITICAL FINDING:")
        print(f"   Predicting ~40W is NEAR-OPTIMAL for minimizing MSE loss!")
        print(f"   The model learned to predict a value close to {best_constant:.1f}W")
        print(f"   because it minimizes overall squared error.")
        print("\n   This is a fundamental issue with using MSE loss on imbalanced data:")
        print("   - Model prefers 'safe' predictions that avoid large errors")
        print("   - Being wrong by 15W on idle (40W vs 27W) is acceptable")
        print("   - Because it avoids being wrong by 30W+ on high loads")
        print("\n   Solutions:")
        print("   1. Use weighted MSE loss (higher weight for idle samples)")
        print("   2. Use balanced sampling during training")
        print("   3. Use different loss function (Huber loss, quantile loss)")
    elif best_constant < 35.0:
        print("\n✓ Optimal constant prediction is low (~27-30W)")
        print("  MSE loss alone doesn't explain why model predicts 40W.")
        print("  The issue is likely elsewhere (architecture, features, etc.)")
    else:
        print(f"\n⚠️  Optimal constant is {best_constant:.1f}W")
        print("  MSE loss may be biasing model toward higher predictions.")

    # Plot loss landscape
    if output_plot:
        plt.figure(figsize=(12, 8))

        # Plot 1: Loss vs constant prediction
        plt.subplot(2, 1, 1)
        pred_range = np.linspace(25, 75, 100)
        mse_values = []
        for pred in pred_range:
            errors = power_values - pred
            mse = np.mean(errors ** 2)
            mse_values.append(mse)

        plt.plot(pred_range, mse_values, 'b-', linewidth=2)
        plt.axvline(x=40.0, color='r', linestyle='--', linewidth=2, label='Model predicts ~40W')
        plt.axvline(x=best_constant, color='g', linestyle='--', linewidth=2, label=f'Optimal: {best_constant:.1f}W')
        plt.axvline(x=np.mean(power_values), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(power_values):.1f}W')
        plt.xlabel('Constant Prediction (W)', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('MSE Loss for Different Constant Predictions', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Power distribution histogram
        plt.subplot(2, 1, 2)
        plt.hist(power_values, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=40.0, color='r', linestyle='--', linewidth=2, label='Model predicts ~40W')
        plt.axvline(x=np.mean(power_values), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(power_values):.1f}W')
        plt.axvline(x=np.median(power_values), color='purple', linestyle='--', linewidth=2, label=f'Median: {np.median(power_values):.1f}W')
        plt.xlabel('Power (W)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Training Data Power Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {output_plot}")

    return loss_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training loss distribution to understand constant predictions"
    )
    parser.add_argument('--training-data', type=str, required=True,
                       help='Path to training CSV')
    parser.add_argument('--output-plot', type=str,
                       help='Output plot file (PNG)')

    args = parser.parse_args()

    analyze_loss_distribution(args.training_data, args.output_plot)


if __name__ == '__main__':
    main()

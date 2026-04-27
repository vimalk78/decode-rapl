#!/usr/bin/env python3
"""
Test trained model on CSV data
Useful for testing model predictions on training data to diagnose issues
"""

import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_model(model_path):
    """Load trained model"""
    from train_model import MSTCN

    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    preprocessor = model_data['preprocessor']

    # Reconstruct model
    num_features = len(preprocessor['feature_columns'])
    num_targets = len(preprocessor['target_columns'])

    model = MSTCN(num_features=num_features, num_targets=num_targets, dropout=0.3)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    return model, preprocessor

def prepare_sequences(df, preprocessor):
    """Prepare data sequences for prediction"""
    feature_columns = preprocessor['feature_columns']
    sequence_length = preprocessor['sequence_length']

    # Extract features
    features = df[feature_columns].values

    # Normalize
    mean = preprocessor['feature_scaler_mean']
    scale = preprocessor['feature_scaler_scale']
    features_normalized = (features - mean) / scale

    # Create sequences
    sequences = []
    targets = []

    for i in range(len(features_normalized) - sequence_length + 1):
        seq = features_normalized[i:i+sequence_length]
        target = df[preprocessor['target_columns'][0]].iloc[i+sequence_length-1]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

def predict(model, sequences, preprocessor):
    """Run predictions on sequences"""
    with torch.no_grad():
        sequences_tensor = torch.FloatTensor(sequences)
        predictions_normalized = model(sequences_tensor).numpy()

    # Denormalize - take only first target if multiple
    target_mean = preprocessor['target_scaler_mean'][0]
    target_std = preprocessor['target_scaler_scale'][0]

    # If model outputs multiple targets, take first one
    if predictions_normalized.shape[1] > 1:
        predictions_normalized = predictions_normalized[:, 0:1]

    predictions = predictions_normalized * target_std + target_mean

    return predictions.flatten()

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 test_on_csv.py <model_path> <csv_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    csv_path = sys.argv[2]

    print(f"Loading model from {model_path}...")
    model, preprocessor = load_model(model_path)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    # Check for actual power values
    target_col = preprocessor['target_columns'][0]
    if target_col not in df.columns:
        print(f"ERROR: Target column '{target_col}' not found in CSV")
        sys.exit(1)

    # Prepare sequences
    print(f"\nPreparing sequences (length={preprocessor['sequence_length']})...")
    sequences, actual = prepare_sequences(df, preprocessor)
    print(f"Created {len(sequences)} sequences")

    # Run predictions
    print(f"\nRunning predictions...")
    predicted = predict(model, sequences, preprocessor)

    # Calculate metrics
    errors = predicted - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)

    # R²
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (ss_res / ss_tot)

    # Print results
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {len(predicted)}")
    print()
    print(f"Actual Power:")
    print(f"  Mean:  {actual.mean():.2f}W")
    print(f"  Std:   {actual.std():.2f}W")
    print(f"  Range: {actual.min():.2f}W - {actual.max():.2f}W")
    print()
    print(f"Predicted Power:")
    print(f"  Mean:  {predicted.mean():.2f}W")
    print(f"  Std:   {predicted.std():.2f}W")
    print(f"  Range: {predicted.min():.2f}W - {predicted.max():.2f}W")
    print()
    print(f"Error Metrics:")
    print(f"  MAE:        {mae:.2f}W")
    print(f"  RMSE:       {rmse:.2f}W")
    print(f"  Mean Error: {mean_error:+.2f}W")
    print(f"  R²:         {r2:.4f}")
    print()

    # Diagnosis
    if abs(mean_error) > 5:
        print(f"⚠️  SYSTEMATIC BIAS: {mean_error:+.2f}W")
        print(f"   Model {'over' if mean_error > 0 else 'under'}-predicts by {abs(mean_error):.2f}W")
    else:
        print(f"✓ No systematic bias")

    print(f"{'='*60}\n")

    # Show first 10 predictions
    print("Sample Predictions:")
    print(f"{'Actual':>8}  {'Predicted':>10}  {'Error':>8}")
    print("-" * 30)
    for i in range(min(10, len(predicted))):
        print(f"{actual[i]:>8.2f}  {predicted[i]:>10.2f}  {errors[i]:>+8.2f}")

if __name__ == '__main__':
    main()

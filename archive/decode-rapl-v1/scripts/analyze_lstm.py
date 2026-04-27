#!/usr/bin/env python3
"""
Analyze LSTM Layer Behavior in DECODE-RAPL Models

Investigates:
1. What LSTM layers learn
2. Whether LSTM helps or hurts predictions
3. Why multi-feature model fails
4. If delay embedding + LSTM is redundant
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')

from src.model import create_model
from src.preprocessing import RAPLDataset, DelayEmbedding
from torch.utils.data import DataLoader


class LSTMAnalyzer:
    """Analyze LSTM behavior in trained models"""

    def __init__(self, checkpoint_path, config_path=None):
        """Load model and extract components"""
        print(f"Loading model from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.config = checkpoint['config']
        self.scalers = checkpoint['scalers']
        self.machine_id_map = checkpoint['machine_id_map']

        # Create model
        num_machines = len(self.machine_id_map)
        self.model = create_model(self.config, num_machines)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Extract components
        self.encoder = self.model.encoder
        self.lstm_module = self.model.lstm  # PowerLSTM module (contains lstm + fc)

        # Model info
        self.is_multifeature = len(self.config['preprocessing'].get('feature_columns', ['cpu_usage'])) > 1
        self.feature_columns = self.config['preprocessing'].get('feature_columns', ['cpu_usage'])

        print(f"Model type: {'Multi-feature' if self.is_multifeature else 'Single-feature'}")
        print(f"Features: {self.feature_columns}")
        print(f"LSTM hidden size: {self.lstm_module.lstm.hidden_size}")

    def predict_with_lstm_states(self, dataloader):
        """
        Run predictions and capture LSTM hidden states

        Returns:
            predictions: Power predictions
            actuals: Actual power values
            lstm_states: LSTM hidden states for each sample
            latent_codes: Encoder outputs (LSTM inputs)
        """
        predictions = []
        actuals = []
        lstm_states = []
        latent_codes = []

        with torch.no_grad():
            for batch in dataloader:
                x, y_power, machine_ids = batch

                # Encoder
                latent = self.encoder(x)  # (batch, seq_len, latent_dim)
                latent_codes.append(latent.cpu().numpy())

                # LSTM (PowerLSTM handles both LSTM and FC)
                lstm_out, (hidden, cell) = self.lstm_module.lstm(latent)
                lstm_states.append(hidden.cpu().numpy())  # (num_layers, batch, hidden_size)

                # Power prediction from final LSTM hidden state
                final_hidden = hidden[-1]  # Last layer
                power_pred = self.lstm_module.fc(final_hidden)

                # Denormalize
                if self.scalers['power'] is not None:
                    power_pred_denorm = self.scalers['power'].inverse_transform(power_pred.cpu().numpy())
                    y_power_denorm = self.scalers['power'].inverse_transform(y_power.cpu().numpy())
                else:
                    power_pred_denorm = power_pred.cpu().numpy()
                    y_power_denorm = y_power.cpu().numpy()

                predictions.append(power_pred_denorm)
                actuals.append(y_power_denorm)

        return {
            'predictions': np.concatenate(predictions),
            'actuals': np.concatenate(actuals),
            'lstm_states': np.concatenate(lstm_states, axis=1),  # (num_layers, total_samples, hidden_size)
            'latent_codes': np.concatenate(latent_codes, axis=0)  # (total_samples, seq_len, latent_dim)
        }

    def predict_without_lstm(self, dataloader):
        """
        Predict by skipping LSTM (direct encoder → power)

        This tests if LSTM adds value or is redundant
        """
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in dataloader:
                x, y_power, machine_ids = batch

                # Encoder
                latent = self.encoder(x)  # (batch, seq_len, latent_dim)

                # Skip LSTM, use last timestep of encoder output directly
                # Feed into the FC layer (as if LSTM output was just the encoder output)
                power_pred = self.lstm_module.fc(latent[:, -1, :])

                # Denormalize
                if self.scalers['power'] is not None:
                    power_pred_denorm = self.scalers['power'].inverse_transform(power_pred.cpu().numpy())
                    y_power_denorm = self.scalers['power'].inverse_transform(y_power.cpu().numpy())
                else:
                    power_pred_denorm = power_pred.cpu().numpy()
                    y_power_denorm = y_power.cpu().numpy()

                predictions.append(power_pred_denorm)
                actuals.append(y_power_denorm)

        return {
            'predictions': np.concatenate(predictions),
            'actuals': np.concatenate(actuals)
        }

    def predict_shuffled_time(self, dataloader):
        """
        Predict with shuffled temporal order

        Tests if LSTM uses temporal patterns or just treats each timestep independently
        """
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in dataloader:
                x, y_power, machine_ids = batch

                # Shuffle time dimension
                batch_size, seq_len, feat_dim = x.shape
                shuffled_indices = torch.randperm(seq_len)
                x_shuffled = x[:, shuffled_indices, :]

                # Run through model
                outputs = self.model(x_shuffled, machine_ids)
                power_pred = outputs['power_pred']

                # Denormalize
                if self.scalers['power'] is not None:
                    power_pred_denorm = self.scalers['power'].inverse_transform(power_pred.cpu().numpy())
                    y_power_denorm = self.scalers['power'].inverse_transform(y_power.cpu().numpy())
                else:
                    power_pred_denorm = power_pred.cpu().numpy()
                    y_power_denorm = y_power.cpu().numpy()

                predictions.append(power_pred_denorm)
                actuals.append(y_power_denorm)

        return {
            'predictions': np.concatenate(predictions),
            'actuals': np.concatenate(actuals)
        }

    def calculate_metrics(self, predictions, actuals):
        """Calculate prediction metrics"""
        predictions = predictions.flatten()
        actuals = actuals.flatten()

        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100

        # R²
        ss_res = np.sum((actuals - predictions)**2)
        ss_tot = np.sum((actuals - np.mean(actuals))**2)
        r2 = 1 - (ss_res / ss_tot)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }


def load_test_data(data_path, config, scalers, machine_id_map):
    """Load test dataset"""
    print(f"\nLoading test data from {data_path}...")

    # Determine feature columns
    feature_columns = config['preprocessing'].get('feature_columns', ['cpu_usage'])
    is_multifeature = len(feature_columns) > 1

    # Read data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")

    if is_multifeature:
        print(f"Multi-feature mode: {feature_columns}")
        # Check columns exist
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    else:
        print("Single-feature mode: cpu_usage")
        if 'cpu_usage' not in df.columns and 'cpu_total' in df.columns:
            df['cpu_usage'] = df['cpu_total']

    # Create dataset
    dataset = RAPLDataset(
        data=df,
        config=config,
        mode='test',
        usage_scaler=scalers['usage'],
        power_scaler=scalers['power']
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    return dataloader


def run_ablation_study(model_path, test_data_path, output_dir):
    """
    Run ablation study to understand LSTM contribution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    analyzer = LSTMAnalyzer(model_path)

    # Load test data
    dataloader = load_test_data(
        test_data_path,
        analyzer.config,
        analyzer.scalers,
        analyzer.machine_id_map
    )

    print("\n" + "="*60)
    print("ABLATION STUDY: LSTM Contribution")
    print("="*60)

    # 1. Full model (with LSTM)
    print("\n1. Full model (Encoder → LSTM → Power)...")
    results_full = analyzer.predict_with_lstm_states(dataloader)
    metrics_full = analyzer.calculate_metrics(results_full['predictions'], results_full['actuals'])

    print(f"   MAE:  {metrics_full['MAE']:.2f}W")
    print(f"   RMSE: {metrics_full['RMSE']:.2f}W")
    print(f"   MAPE: {metrics_full['MAPE']:.2f}%")
    print(f"   R²:   {metrics_full['R2']:.4f}")

    # 2. Without LSTM (direct encoder → power)
    print("\n2. Without LSTM (Encoder → Power directly)...")
    results_no_lstm = analyzer.predict_without_lstm(dataloader)
    metrics_no_lstm = analyzer.calculate_metrics(results_no_lstm['predictions'], results_no_lstm['actuals'])

    print(f"   MAE:  {metrics_no_lstm['MAE']:.2f}W")
    print(f"   RMSE: {metrics_no_lstm['RMSE']:.2f}W")
    print(f"   MAPE: {metrics_no_lstm['MAPE']:.2f}%")
    print(f"   R²:   {metrics_no_lstm['R2']:.4f}")

    # 3. Shuffled time order
    print("\n3. Shuffled time order (test temporal dependency)...")
    results_shuffled = analyzer.predict_shuffled_time(dataloader)
    metrics_shuffled = analyzer.calculate_metrics(results_shuffled['predictions'], results_shuffled['actuals'])

    print(f"   MAE:  {metrics_shuffled['MAE']:.2f}W")
    print(f"   RMSE: {metrics_shuffled['RMSE']:.2f}W")
    print(f"   MAPE: {metrics_shuffled['MAPE']:.2f}%")
    print(f"   R²:   {metrics_shuffled['R2']:.4f}")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    lstm_improvement = metrics_full['R2'] - metrics_no_lstm['R2']
    print(f"\nLSTM contribution to R²: {lstm_improvement:+.4f}")
    if abs(lstm_improvement) < 0.01:
        print("→ LSTM provides minimal benefit (< 1% R² change)")
        print("→ Delay embedding alone may be sufficient")
    elif lstm_improvement > 0:
        print("→ LSTM improves predictions")
    else:
        print("→ LSTM HURTS predictions!")
        print("→ Consider removing LSTM layer")

    temporal_degradation = metrics_full['R2'] - metrics_shuffled['R2']
    print(f"\nTemporal order importance: {temporal_degradation:+.4f}")
    if abs(temporal_degradation) < 0.01:
        print("→ LSTM does NOT use temporal patterns")
        print("→ Treating each timestep independently")
    else:
        print("→ LSTM uses temporal patterns")

    # Visualize
    print(f"\nCreating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'LSTM Ablation Study - {"Multi-feature" if analyzer.is_multifeature else "Single-feature"} Model',
                 fontsize=14, fontweight='bold')

    # Plot 1: Full model predictions
    ax = axes[0, 0]
    ax.scatter(results_full['actuals'], results_full['predictions'], alpha=0.3, s=5)
    ax.plot([results_full['actuals'].min(), results_full['actuals'].max()],
            [results_full['actuals'].min(), results_full['actuals'].max()],
            'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Power (W)')
    ax.set_ylabel('Predicted Power (W)')
    ax.set_title(f'Full Model (R²={metrics_full["R2"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Without LSTM
    ax = axes[0, 1]
    ax.scatter(results_no_lstm['actuals'], results_no_lstm['predictions'], alpha=0.3, s=5)
    ax.plot([results_no_lstm['actuals'].min(), results_no_lstm['actuals'].max()],
            [results_no_lstm['actuals'].min(), results_no_lstm['actuals'].max()],
            'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Power (W)')
    ax.set_ylabel('Predicted Power (W)')
    ax.set_title(f'Without LSTM (R²={metrics_no_lstm["R2"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Comparison bar chart
    ax = axes[1, 0]
    models = ['Full\nModel', 'No\nLSTM', 'Shuffled\nTime']
    r2_scores = [metrics_full['R2'], metrics_no_lstm['R2'], metrics_shuffled['R2']]
    colors = ['green' if r2 == max(r2_scores) else 'orange' for r2 in r2_scores]
    ax.bar(models, r2_scores, color=colors, alpha=0.7)
    ax.set_ylabel('R² Score')
    ax.set_title('Model Comparison')
    ax.set_ylim([min(r2_scores) - 0.1, 1.0])
    ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(r2_scores):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

    # Plot 4: LSTM hidden states (first layer, first 100 samples)
    ax = axes[1, 1]
    lstm_hidden = results_full['lstm_states'][0, :100, :]  # (100 samples, hidden_size)
    im = ax.imshow(lstm_hidden.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('LSTM Hidden Unit')
    ax.set_title('LSTM Hidden States (Layer 1)')
    plt.colorbar(im, ax=ax, label='Activation')

    plt.tight_layout()
    plot_path = output_dir / 'lstm_ablation_study.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_path}")

    # Save results
    results_df = pd.DataFrame({
        'Model': ['Full', 'No LSTM', 'Shuffled Time'],
        'MAE': [metrics_full['MAE'], metrics_no_lstm['MAE'], metrics_shuffled['MAE']],
        'RMSE': [metrics_full['RMSE'], metrics_no_lstm['RMSE'], metrics_shuffled['RMSE']],
        'MAPE': [metrics_full['MAPE'], metrics_no_lstm['MAPE'], metrics_shuffled['MAPE']],
        'R2': [metrics_full['R2'], metrics_no_lstm['R2'], metrics_shuffled['R2']]
    })

    csv_path = output_dir / 'lstm_ablation_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved results to {csv_path}")

    return results_df


def main():
    """Main analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze LSTM behavior in DECODE-RAPL models')
    parser.add_argument('--model', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--test-data', required=True, help='Path to test data CSV')
    parser.add_argument('--output', default='results/lstm_analysis', help='Output directory')

    args = parser.parse_args()

    print("="*60)
    print("DECODE-RAPL LSTM Analysis")
    print("="*60)

    results = run_ablation_study(args.model, args.test_data, args.output)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(results.to_string(index=False))

    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()

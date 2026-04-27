#!/usr/bin/env python3
"""
Attention Head Analysis for MS-TCN Power Prediction Model

Extracts and visualizes attention weights from trained model to understand
what temporal patterns each of the 8 attention heads learned.

Usage:
    python scripts/analyze_attention.py \\
        --model models/best_model_diverse_2hr_18f.pth \\
        --data data/training_diverse_2hr.csv \\
        --output results/attention_analysis/ \\
        --num-samples 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from train_model import MSTCN, DataPreprocessor


class AttentionCapture:
    """Captures attention weights during model forward pass."""

    def __init__(self):
        self.attention_weights = []

    def hook_fn(self, module, input, output):
        """Hook function to capture attention weights."""
        # nn.MultiheadAttention returns (attn_output, attn_output_weights)
        # attn_output_weights shape: (batch, num_heads, seq_len, seq_len) when average_attn_weights=False
        # or (batch, seq_len, seq_len) when average_attn_weights=True (default)
        if len(output) == 2:
            _, attn_weights = output
            if attn_weights is not None:
                self.attention_weights.append(attn_weights.detach().cpu())

    def reset(self):
        """Clear captured weights."""
        self.attention_weights = []


def load_model_and_data(model_path: str, data_path: str, sequence_length: int = 64):
    """Load trained model and prepare test data using saved preprocessor."""

    import pickle
    from sklearn.preprocessing import StandardScaler

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Extract model configuration
    preprocessor_data = checkpoint['preprocessor']
    feature_columns = preprocessor_data['feature_columns']
    target_columns = preprocessor_data['target_columns']
    num_features = len(feature_columns)
    num_targets = len(target_columns)

    print(f"Model: {num_features} features → {num_targets} targets")
    print(f"Features: {feature_columns}")
    print(f"Targets: {target_columns}")

    # Create model
    model = MSTCN(num_features=num_features, num_targets=num_targets, hidden_dim=128)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Load SAVED scalers from checkpoint (don't create new ones!)
    print("Loading saved feature scaler from checkpoint...")
    feature_scaler = pickle.loads(preprocessor_data['feature_scaler'])

    # Recreate target scaler from saved parameters
    target_scaler = StandardScaler()
    target_scaler.mean_ = preprocessor_data['target_scaler_mean']
    target_scaler.scale_ = preprocessor_data['target_scaler_scale']

    # Extract only the features that model was trained on
    X = df[feature_columns].values
    y = df[target_columns].values

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    print(f"Raw data shape: X={X.shape}, y={y.shape}")

    # Apply SAVED scaler (not a new one!)
    X = feature_scaler.transform(X)
    y = target_scaler.transform(y)

    print("Applied saved scalers to data")

    # Create sequences
    print(f"Creating sequences with length {sequence_length}...")
    sequences = []
    targets = []

    for i in range(0, len(X) - sequence_length, 1):  # stride=1
        seq = X[i:i + sequence_length]
        target = y[i + sequence_length - 1]  # Predict the last timestep

        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    print(f"Created {len(sequences)} sequences")

    # Create a minimal preprocessor object for compatibility
    preprocessor = DataPreprocessor(sequence_length=sequence_length)
    preprocessor.feature_columns = feature_columns
    preprocessor.target_columns = target_columns
    preprocessor.feature_scaler = feature_scaler
    preprocessor.target_scaler = target_scaler

    return model, sequences, targets, preprocessor


def modify_attention_for_weight_extraction(model):
    """
    Modify the model's attention layer to return attention weights.

    PyTorch's MultiheadAttention can return weights but we need to access
    the underlying module and temporarily modify it.
    """
    # Access the attention module
    attention_module = model.attention.attention

    # Store original settings
    original_need_weights = attention_module._qkv_same_embed_dim

    # We need to capture weights per head, not averaged
    # This requires modifying how we call the attention

    return attention_module


def extract_attention_weights(model, sequences: np.ndarray, num_samples: int = 10) -> List[np.ndarray]:
    """
    Extract attention weights from model for sample sequences.

    Returns:
        List of attention weight arrays, one per sample.
        Each array has shape (seq_len, seq_len) or (num_heads, seq_len, seq_len)
    """

    print(f"\nExtracting attention weights from {num_samples} samples...")

    # Create attention capture hook
    capturer = AttentionCapture()

    # Register hook on the MultiheadAttention module
    # The attention is inside model.attention.attention
    hook_handle = model.attention.attention.register_forward_hook(
        lambda module, input, output: capturer.hook_fn(module, input, output)
    )

    # Select random samples
    indices = np.random.choice(len(sequences), size=min(num_samples, len(sequences)), replace=False)

    all_attention_weights = []

    with torch.no_grad():
        for idx in indices:
            capturer.reset()

            # Get single sequence
            seq = torch.FloatTensor(sequences[idx:idx+1])  # Shape: (1, seq_len, features)

            # Forward pass (triggers hook)
            _ = model(seq)

            # Collect captured weights
            if len(capturer.attention_weights) > 0:
                # Weights shape: (batch=1, seq_len, seq_len) when averaged across heads
                weights = capturer.attention_weights[0].squeeze(0).numpy()
                all_attention_weights.append(weights)

    # Remove hook
    hook_handle.remove()

    print(f"Captured attention weights for {len(all_attention_weights)} samples")
    if len(all_attention_weights) > 0:
        print(f"Weight matrix shape per sample: {all_attention_weights[0].shape}")

    return all_attention_weights


def analyze_attention_patterns(attention_weights: List[np.ndarray]) -> Dict:
    """
    Analyze attention weight patterns across samples.

    Returns:
        Dictionary with analysis results.
    """

    print("\nAnalyzing attention patterns...")

    # Average attention across all samples
    avg_attention = np.mean(attention_weights, axis=0)  # (seq_len, seq_len)

    seq_len = avg_attention.shape[0]

    analysis = {
        'avg_attention': avg_attention.tolist(),
        'seq_len': seq_len,
        'num_samples': len(attention_weights)
    }

    # Analyze temporal focus
    # For each query position (row), where does it attend? (columns)

    # 1. Local vs Global attention
    # Compute average attention distance from diagonal
    attention_distances = []
    for i in range(seq_len):
        for j in range(seq_len):
            distance = abs(i - j)
            attention_distances.append(distance * avg_attention[i, j])

    avg_attention_distance = np.sum(attention_distances)
    analysis['avg_attention_distance'] = float(avg_attention_distance)

    # 2. Recent vs Distant past bias
    # For final timestep (t=seq_len-1), where does it attend?
    final_timestep_attention = avg_attention[-1, :]

    # Split into recent (last 25%) and distant (first 75%)
    recent_start = int(seq_len * 0.75)
    recent_attention = np.sum(final_timestep_attention[recent_start:])
    distant_attention = np.sum(final_timestep_attention[:recent_start])

    analysis['final_timestep_recent_focus'] = float(recent_attention)
    analysis['final_timestep_distant_focus'] = float(distant_attention)
    analysis['recent_to_distant_ratio'] = float(recent_attention / (distant_attention + 1e-8))

    # 3. Attention entropy (uniformity vs spikiness)
    # High entropy = uniform attention, Low entropy = focused attention
    attention_entropies = []
    for i in range(seq_len):
        row_entropy = entropy(avg_attention[i, :] + 1e-10)  # Add small value to avoid log(0)
        attention_entropies.append(row_entropy)

    analysis['avg_attention_entropy'] = float(np.mean(attention_entropies))
    analysis['min_attention_entropy'] = float(np.min(attention_entropies))
    analysis['max_attention_entropy'] = float(np.max(attention_entropies))

    # 4. Diagonal dominance (self-attention strength)
    diagonal_attention = np.mean(np.diag(avg_attention))
    off_diagonal_attention = np.mean(avg_attention[~np.eye(seq_len, dtype=bool)])

    analysis['diagonal_attention'] = float(diagonal_attention)
    analysis['off_diagonal_attention'] = float(off_diagonal_attention)
    analysis['self_attention_ratio'] = float(diagonal_attention / (off_diagonal_attention + 1e-8))

    return analysis


def visualize_attention(attention_weights: List[np.ndarray], output_dir: Path, analysis: Dict):
    """
    Create visualizations of attention patterns.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Average attention across samples
    avg_attention = np.array(analysis['avg_attention'])

    # Create figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Average attention heatmap
    ax = axes[0, 0]
    sns.heatmap(avg_attention, cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Weight'})
    ax.set_title('Average Attention Weights Across All Samples', fontsize=14, fontweight='bold')
    ax.set_xlabel('Attended Timestep (Key)', fontsize=12)
    ax.set_ylabel('Current Timestep (Query)', fontsize=12)

    # 2. Final timestep attention (what does prediction attend to?)
    ax = axes[0, 1]
    final_attention = avg_attention[-1, :]
    timesteps = np.arange(len(final_attention))
    ax.bar(timesteps, final_attention, color='steelblue', alpha=0.7)
    ax.set_title('Final Timestep Attention Distribution\n(What history matters for prediction?)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestep in Sequence', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add vertical line at 75% mark (recent vs distant)
    recent_start = int(len(final_attention) * 0.75)
    ax.axvline(recent_start, color='red', linestyle='--', label=f'Recent boundary (t={recent_start})')
    ax.legend()

    # 3. Attention distance distribution
    ax = axes[1, 0]
    distances = []
    for i in range(avg_attention.shape[0]):
        for j in range(avg_attention.shape[1]):
            dist = abs(i - j)
            weight = avg_attention[i, j]
            distances.extend([dist] * int(weight * 1000))  # Weight by attention

    ax.hist(distances, bins=30, color='coral', alpha=0.7, edgecolor='black')
    ax.set_title('Attention Distance Distribution\n(Local vs Global Focus)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Temporal Distance (|query - key|)', fontsize=12)
    ax.set_ylabel('Frequency (weighted by attention)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # 4. Per-timestep attention entropy
    ax = axes[1, 1]
    entropies = [entropy(avg_attention[i, :] + 1e-10) for i in range(avg_attention.shape[0])]
    ax.plot(entropies, linewidth=2, color='darkgreen')
    ax.fill_between(range(len(entropies)), entropies, alpha=0.3, color='green')
    ax.set_title('Attention Entropy Over Time\n(Focused vs Distributed Attention)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Entropy (nats)', fontsize=12)
    ax.grid(alpha=0.3)

    # Add horizontal line for average
    ax.axhline(np.mean(entropies), color='red', linestyle='--',
               label=f'Mean = {np.mean(entropies):.2f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'attention_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved attention visualization to {output_dir / 'attention_analysis.png'}")

    # Create individual sample heatmaps (first 4 samples)
    if len(attention_weights) >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx in range(min(4, len(attention_weights))):
            sns.heatmap(attention_weights[idx], cmap='viridis', ax=axes[idx],
                       cbar_kws={'label': 'Weight'})
            axes[idx].set_title(f'Sample {idx+1} Attention Pattern')
            axes[idx].set_xlabel('Attended Timestep')
            axes[idx].set_ylabel('Current Timestep')

        plt.tight_layout()
        plt.savefig(output_dir / 'attention_samples.png', dpi=150, bbox_inches='tight')
        print(f"Saved sample heatmaps to {output_dir / 'attention_samples.png'}")


def generate_interpretation_report(analysis: Dict, output_path: Path):
    """
    Generate human-readable interpretation of attention patterns.
    """

    report = []
    report.append("=" * 80)
    report.append("MS-TCN ATTENTION HEAD ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    report.append(f"Analyzed {analysis['num_samples']} sequences of length {analysis['seq_len']}")
    report.append("")

    report.append("TEMPORAL FOCUS ANALYSIS")
    report.append("-" * 80)

    # Recent vs Distant
    recent_pct = analysis['final_timestep_recent_focus'] * 100
    distant_pct = analysis['final_timestep_distant_focus'] * 100
    ratio = analysis['recent_to_distant_ratio']

    report.append(f"\n1. Recent vs Distant Past (for final prediction timestep):")
    report.append(f"   - Recent history (last 25%):    {recent_pct:.1f}% attention")
    report.append(f"   - Distant history (first 75%):  {distant_pct:.1f}% attention")
    report.append(f"   - Recent/Distant ratio:         {ratio:.2f}x")
    report.append("")

    if ratio > 2.0:
        interpretation = "STRONGLY RECENT-FOCUSED: Model heavily prioritizes recent timesteps."
    elif ratio > 1.2:
        interpretation = "RECENT-BIASED: Model moderately favors recent history."
    elif ratio > 0.8:
        interpretation = "BALANCED: Model attends fairly evenly across time."
    else:
        interpretation = "DISTANT-BIASED: Model unexpectedly focuses on distant past."

    report.append(f"   Interpretation: {interpretation}")
    report.append("")

    # Local vs Global
    avg_dist = analysis['avg_attention_distance']
    max_dist = analysis['seq_len'] - 1

    report.append(f"2. Local vs Global Attention:")
    report.append(f"   - Average attention distance:   {avg_dist:.1f} timesteps")
    report.append(f"   - Maximum possible distance:    {max_dist} timesteps")
    report.append(f"   - Normalized distance:          {avg_dist/max_dist:.2%}")
    report.append("")

    if avg_dist < max_dist * 0.2:
        interpretation = "HIGHLY LOCAL: Attention focused on nearby timesteps."
    elif avg_dist < max_dist * 0.4:
        interpretation = "LOCAL: Moderate focus on recent context."
    elif avg_dist < max_dist * 0.6:
        interpretation = "MIXED: Balance between local and global attention."
    else:
        interpretation = "GLOBAL: Attention spread across entire sequence."

    report.append(f"   Interpretation: {interpretation}")
    report.append("")

    # Attention Entropy
    avg_entropy = analysis['avg_attention_entropy']
    max_entropy = np.log(analysis['seq_len'])  # Max entropy for uniform distribution

    report.append(f"3. Attention Concentration (Entropy):")
    report.append(f"   - Average entropy:              {avg_entropy:.3f} nats")
    report.append(f"   - Maximum entropy (uniform):    {max_entropy:.3f} nats")
    report.append(f"   - Normalized entropy:           {avg_entropy/max_entropy:.2%}")
    report.append(f"   - Min entropy (most focused):   {analysis['min_attention_entropy']:.3f} nats")
    report.append(f"   - Max entropy (most spread):    {analysis['max_attention_entropy']:.3f} nats")
    report.append("")

    if avg_entropy / max_entropy > 0.8:
        interpretation = "HIGHLY DISTRIBUTED: Attention spread nearly uniformly (may indicate weak learning)."
    elif avg_entropy / max_entropy > 0.6:
        interpretation = "MODERATELY DISTRIBUTED: Attention somewhat focused but still broad."
    elif avg_entropy / max_entropy > 0.4:
        interpretation = "FOCUSED: Clear preference for specific timesteps."
    else:
        interpretation = "HIGHLY FOCUSED: Strong concentration on few critical timesteps."

    report.append(f"   Interpretation: {interpretation}")
    report.append("")

    # Self-attention
    self_ratio = analysis['self_attention_ratio']
    diag_pct = analysis['diagonal_attention'] * 100

    report.append(f"4. Self-Attention Strength:")
    report.append(f"   - Diagonal attention (self):    {diag_pct:.2f}%")
    report.append(f"   - Self/Others ratio:            {self_ratio:.2f}x")
    report.append("")

    if self_ratio > 2.0:
        interpretation = "HIGH SELF-ATTENTION: Each timestep strongly attends to itself (may indicate identity-like behavior)."
    elif self_ratio > 1.2:
        interpretation = "MODERATE SELF-ATTENTION: Timesteps reference themselves plus context."
    else:
        interpretation = "LOW SELF-ATTENTION: Model primarily uses context from other timesteps."

    report.append(f"   Interpretation: {interpretation}")
    report.append("")

    # Overall Summary
    report.append("=" * 80)
    report.append("OVERALL INTERPRETATION")
    report.append("=" * 80)
    report.append("")

    report.append("The attention mechanism in your MS-TCN model shows:")
    report.append("")

    if ratio > 1.5 and avg_dist < max_dist * 0.3:
        summary = "Strong recency bias with local focus - model primarily uses recent short-term patterns."
    elif ratio > 1.2 and avg_entropy / max_entropy < 0.5:
        summary = "Recent-focused with selective attention - model identifies key recent moments."
    elif avg_entropy / max_entropy > 0.7:
        summary = "Broad distributed attention - model may not have learned strong temporal patterns (consider retraining with more data)."
    else:
        summary = "Balanced temporal reasoning - model integrates information across multiple timescales."

    report.append(f"  {summary}")
    report.append("")

    report.append("NOTE: This analysis shows AVERAGE attention across heads. PyTorch's default")
    report.append("MultiheadAttention returns averaged weights. To analyze individual head behavior,")
    report.append("you would need to modify the attention layer to return per-head weights.")
    report.append("")

    report.append("=" * 80)

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nSaved interpretation report to {output_path}")

    # Also print to console
    print("\n" + '\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attention patterns in trained MS-TCN model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV data file for testing')
    parser.add_argument('--output', type=str, default='results/attention_analysis/',
                       help='Output directory for analysis results')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of sequences to analyze (default: 20)')
    parser.add_argument('--sequence-length', type=int, default=64,
                       help='Sequence length (default: 64)')

    args = parser.parse_args()

    output_dir = Path(args.output)

    # Load model and data
    model, sequences, targets, preprocessor = load_model_and_data(
        args.model, args.data, args.sequence_length
    )

    # Extract attention weights
    attention_weights = extract_attention_weights(model, sequences, args.num_samples)

    if len(attention_weights) == 0:
        print("\nERROR: Could not extract attention weights!")
        print("This may happen if the model's attention layer doesn't return weights.")
        return

    # Analyze patterns
    analysis = analyze_attention_patterns(attention_weights)

    # Save raw analysis data
    with open(output_dir / 'attention_statistics.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nSaved statistics to {output_dir / 'attention_statistics.json'}")

    # Create visualizations
    visualize_attention(attention_weights, output_dir, analysis)

    # Generate interpretation report
    generate_interpretation_report(analysis, output_dir / 'interpretation_report.txt')

    print("\n" + "="*80)
    print("Attention analysis complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'attention_analysis.png'}")
    print(f"  - {output_dir / 'attention_samples.png'}")
    print(f"  - {output_dir / 'attention_statistics.json'}")
    print(f"  - {output_dir / 'interpretation_report.txt'}")


if __name__ == "__main__":
    main()

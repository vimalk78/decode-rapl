#!/usr/bin/env python3
"""
Analyze attention pooling weights in fine-tuned MS-TCN model.

This script extracts and visualizes the learned attention pooling weights
to understand which timesteps the model considers most important for the
final power prediction. This helps verify that the attention-based pooling
fix resolved the collapsed attention problem.

Usage:
    python scripts/analyze_pooling_attention.py \
        --model models/best_model_17f_attention_pooling.pth \
        --data data/training_diverse_2hr.csv \
        --output results/pooling_attention_analysis \
        --num-samples 20

The analysis includes:
1. Pooling attention weights across timesteps
2. Temporal importance distribution
3. Comparison with multi-head attention patterns
4. Interpretation of learned temporal focus
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from train_model import MSTCN, DataPreprocessor


class PoolingAttentionCapture:
    """Captures attention pooling weights during forward pass."""

    def __init__(self):
        self.pooling_weights = []
        self.attention_weights = []

    def pooling_hook(self, module, input, output):
        """Hook to capture attention pooling weights."""
        # Input is x after attention: (batch, channels, timesteps)
        x = input[0]
        x_t = x.transpose(1, 2)  # (batch, timesteps, channels)

        # Compute scores and weights (same as forward pass)
        scores = module.attention_weights(x_t)  # (batch, timesteps, 1)
        weights = F.softmax(scores, dim=1)  # (batch, timesteps, 1)

        # Store weights
        self.pooling_weights.append(weights.detach().cpu().numpy())

    def attention_hook(self, module, input, output):
        """Hook to capture multi-head attention weights (if available)."""
        # PyTorch's MultiheadAttention doesn't return weights by default
        # This would require modifying the attention module
        pass


def load_model_and_data(model_path: str, data_path: str, sequence_length: int = 64):
    """Load fine-tuned model and prepare data using saved scalers."""

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Extract preprocessor configuration
    preprocessor_data = checkpoint['preprocessor']
    num_features = len(preprocessor_data['feature_columns'])
    num_targets = len(preprocessor_data['target_columns'])

    # Get hidden_dim from temporal blocks (reliable for any hidden_dim value)
    # temporal_blocks always use the full hidden_dim regardless of remainder from division
    hidden_dim = checkpoint['model_state_dict']['temporal_blocks.0.conv1.weight'].shape[0]

    # Find compatible num_heads for the hidden_dim
    preferred_heads = [8, 7, 6, 5, 4, 3, 2, 1]
    num_heads = next((h for h in preferred_heads if hidden_dim % h == 0), 1)

    print(f"  Model: {num_features} features → {num_targets} targets")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Attention heads: {num_heads} (auto-detected)")

    # Create model and load weights
    model = MSTCN(num_features=num_features, num_targets=num_targets,
                  hidden_dim=hidden_dim, num_heads=num_heads)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Model loaded successfully")

    # Check if model has attention pooling
    if not hasattr(model, 'attention_pool'):
        raise ValueError("Model does not have attention_pool layer. "
                        "This script requires a fine-tuned model with AttentionPooling.")

    # Load data
    print(f"\nLoading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples")

    # Load saved scalers
    feature_scaler = pickle.loads(preprocessor_data['feature_scaler'])
    feature_columns = preprocessor_data['feature_columns']
    target_columns = preprocessor_data['target_columns']

    # Extract and transform features
    X = df[feature_columns].values
    X = feature_scaler.transform(X)

    # Create sequences
    sequences = []
    for i in range(0, len(X) - sequence_length, sequence_length // 2):  # 50% overlap
        seq = X[i:i + sequence_length]
        sequences.append(seq)

    sequences = np.array(sequences)
    print(f"  Created {len(sequences)} sequences")

    return model, torch.FloatTensor(sequences), feature_columns, target_columns


def extract_pooling_weights(model: MSTCN, sequences: torch.Tensor,
                            num_samples: int = 20) -> np.ndarray:
    """Extract attention pooling weights from model."""

    print("\nExtracting attention pooling weights...")

    # Register hook to capture pooling weights
    capture = PoolingAttentionCapture()
    hook = model.attention_pool.register_forward_hook(capture.pooling_hook)

    # Run inference
    num_samples = min(num_samples, len(sequences))
    with torch.no_grad():
        _ = model(sequences[:num_samples])

    # Remove hook
    hook.remove()

    # Concatenate all captured weights
    pooling_weights = np.concatenate(capture.pooling_weights, axis=0)  # (num_samples, timesteps, 1)
    pooling_weights = pooling_weights.squeeze(-1)  # (num_samples, timesteps)

    print(f"  Extracted weights for {pooling_weights.shape[0]} samples")
    print(f"  Sequence length: {pooling_weights.shape[1]}")

    return pooling_weights


def analyze_pooling_patterns(pooling_weights: np.ndarray) -> Dict:
    """Analyze temporal importance patterns in pooling weights."""

    print("\nAnalyzing pooling attention patterns...")

    num_samples, seq_len = pooling_weights.shape

    # Average weights across all samples
    avg_weights = pooling_weights.mean(axis=0)

    # Find timesteps with highest importance
    top_k = 5
    top_indices = np.argsort(avg_weights)[-top_k:][::-1]
    top_weights = avg_weights[top_indices]

    print(f"\n  Top {top_k} most important timesteps:")
    for idx, weight in zip(top_indices, top_weights):
        print(f"    Timestep {idx}: {weight:.4f} ({weight*100:.2f}%)")

    # Temporal distribution analysis
    quarter_len = seq_len // 4

    early_weight = avg_weights[:quarter_len].sum()
    mid_early_weight = avg_weights[quarter_len:2*quarter_len].sum()
    mid_late_weight = avg_weights[2*quarter_len:3*quarter_len].sum()
    late_weight = avg_weights[3*quarter_len:].sum()

    print(f"\n  Temporal distribution:")
    print(f"    Early quarter (0-25%):       {early_weight:.4f} ({early_weight*100:.1f}%)")
    print(f"    Mid-early quarter (25-50%):  {mid_early_weight:.4f} ({mid_early_weight*100:.1f}%)")
    print(f"    Mid-late quarter (50-75%):   {mid_late_weight:.4f} ({mid_late_weight*100:.1f}%)")
    print(f"    Late quarter (75-100%):      {late_weight:.4f} ({late_weight*100:.1f}%)")

    # Recent vs distant analysis
    recent_threshold = int(0.75 * seq_len)
    recent_weight = avg_weights[recent_threshold:].sum()
    distant_weight = avg_weights[:recent_threshold].sum()

    print(f"\n  Recent vs distant past:")
    print(f"    Recent (last 25%):   {recent_weight:.4f} ({recent_weight*100:.1f}%)")
    print(f"    Distant (first 75%): {distant_weight:.4f} ({distant_weight*100:.1f}%)")
    print(f"    Recent/Distant ratio: {recent_weight/distant_weight:.2f}x")

    # Concentration analysis (entropy)
    # Entropy measures how concentrated vs spread out the weights are
    epsilon = 1e-10
    entropy = -np.sum(avg_weights * np.log(avg_weights + epsilon))
    max_entropy = np.log(seq_len)  # Maximum entropy (uniform distribution)
    normalized_entropy = entropy / max_entropy

    print(f"\n  Attention concentration:")
    print(f"    Entropy: {entropy:.3f} nats")
    print(f"    Max entropy (uniform): {max_entropy:.3f} nats")
    print(f"    Normalized entropy: {normalized_entropy*100:.1f}%")

    if normalized_entropy < 0.3:
        concentration = "HIGHLY FOCUSED"
    elif normalized_entropy < 0.6:
        concentration = "FOCUSED"
    elif normalized_entropy < 0.8:
        concentration = "MODERATELY SPREAD"
    else:
        concentration = "HIGHLY SPREAD (nearly uniform)"

    print(f"    Interpretation: {concentration}")

    # Peak detection - multiple peaks suggest multi-modal importance
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(avg_weights, height=avg_weights.mean() + avg_weights.std())

    print(f"\n  Peak detection:")
    print(f"    Number of significant peaks: {len(peaks)}")
    if len(peaks) > 0:
        print(f"    Peak locations: {peaks.tolist()}")
        if len(peaks) == 1:
            print(f"    Interpretation: SINGLE FOCUS - model concentrates on one key timestep")
        elif len(peaks) <= 3:
            print(f"    Interpretation: MULTI-FOCUS - model identifies {len(peaks)} key moments")
        else:
            print(f"    Interpretation: DISTRIBUTED - model uses many timesteps")

    return {
        'avg_weights': avg_weights.tolist(),
        'top_timesteps': top_indices.tolist(),
        'top_weights': top_weights.tolist(),
        'temporal_distribution': {
            'early': float(early_weight),
            'mid_early': float(mid_early_weight),
            'mid_late': float(mid_late_weight),
            'late': float(late_weight)
        },
        'recent_vs_distant': {
            'recent': float(recent_weight),
            'distant': float(distant_weight),
            'ratio': float(recent_weight / distant_weight)
        },
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'concentration': concentration,
        'num_peaks': int(len(peaks)),
        'peak_locations': peaks.tolist() if len(peaks) > 0 else []
    }


def visualize_pooling_attention(pooling_weights: np.ndarray, analysis: Dict,
                                output_dir: str):
    """Create comprehensive visualization of pooling attention patterns."""

    print("\nCreating visualizations...")

    num_samples, seq_len = pooling_weights.shape
    avg_weights = np.array(analysis['avg_weights'])

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Average pooling weights across all samples
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(avg_weights, linewidth=2, color='#2E86AB')
    ax1.fill_between(range(seq_len), avg_weights, alpha=0.3, color='#2E86AB')

    # Highlight top timesteps
    top_indices = analysis['top_timesteps'][:5]
    ax1.scatter(top_indices, avg_weights[top_indices], color='#A23B72',
               s=100, zorder=5, label='Top 5 important timesteps')

    ax1.set_xlabel('Timestep in Sequence', fontsize=12)
    ax1.set_ylabel('Attention Weight', fontsize=12)
    ax1.set_title('Learned Attention Pooling Weights (Average Across All Samples)',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Heatmap of individual samples
    ax2 = fig.add_subplot(gs[1, :])
    sns.heatmap(pooling_weights[:min(20, num_samples)], cmap='viridis',
               cbar_kws={'label': 'Weight'}, ax=ax2)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Sample Index', fontsize=12)
    ax2.set_title('Pooling Attention Weights for Individual Samples', fontsize=14, fontweight='bold')

    # 3. Temporal distribution (quarters)
    ax3 = fig.add_subplot(gs[2, 0])
    quarters = ['Early\n(0-25%)', 'Mid-Early\n(25-50%)', 'Mid-Late\n(50-75%)', 'Late\n(75-100%)']
    quarter_weights = [
        analysis['temporal_distribution']['early'],
        analysis['temporal_distribution']['mid_early'],
        analysis['temporal_distribution']['mid_late'],
        analysis['temporal_distribution']['late']
    ]
    colors = ['#F18F01', '#C73E1D', '#6A994E', '#2E86AB']
    bars = ax3.bar(quarters, quarter_weights, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Total Weight', fontsize=12)
    ax3.set_title('Temporal Distribution by Quarter', fontsize=13, fontweight='bold')
    ax3.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Uniform (25%)')
    ax3.legend()

    # Add percentage labels on bars
    for bar, weight in zip(bars, quarter_weights):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight*100:.1f}%', ha='center', va='bottom', fontsize=10)

    # 4. Distribution statistics
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    stats_text = f"""
    POOLING ATTENTION ANALYSIS

    Concentration:
      • Entropy: {analysis['entropy']:.3f} nats
      • Normalized: {analysis['normalized_entropy']*100:.1f}%
      • {analysis['concentration']}

    Temporal Focus:
      • Recent (last 25%): {analysis['recent_vs_distant']['recent']*100:.1f}%
      • Distant (first 75%): {analysis['recent_vs_distant']['distant']*100:.1f}%
      • Ratio: {analysis['recent_vs_distant']['ratio']:.2f}x

    Peak Analysis:
      • Number of peaks: {analysis['num_peaks']}
      • Top timestep: {analysis['top_timesteps'][0]}
      • Max weight: {analysis['top_weights'][0]*100:.2f}%

    Interpretation:
      The model {"focuses heavily on specific timesteps" if analysis['normalized_entropy'] < 0.5 else "distributes attention broadly"}
      and {"prioritizes recent history" if analysis['recent_vs_distant']['ratio'] > 2 else "balances recent and distant past"}.
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(f"{output_dir}/pooling_attention_analysis.png", dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to {output_dir}/pooling_attention_analysis.png")


def generate_interpretation_report(analysis: Dict, output_dir: str):
    """Generate human-readable interpretation report."""

    print("\nGenerating interpretation report...")

    report = []
    report.append("="*80)
    report.append("ATTENTION POOLING ANALYSIS REPORT")
    report.append("="*80)
    report.append("")

    # Concentration
    report.append("ATTENTION CONCENTRATION")
    report.append("-" * 80)
    report.append(f"  Entropy: {analysis['entropy']:.3f} nats")
    report.append(f"  Normalized entropy: {analysis['normalized_entropy']*100:.1f}%")
    report.append(f"  Interpretation: {analysis['concentration']}")
    report.append("")

    # Temporal focus
    report.append("TEMPORAL FOCUS")
    report.append("-" * 80)
    report.append(f"  Recent history (last 25%):   {analysis['recent_vs_distant']['recent']*100:.1f}%")
    report.append(f"  Distant history (first 75%): {analysis['recent_vs_distant']['distant']*100:.1f}%")
    report.append(f"  Recent/Distant ratio: {analysis['recent_vs_distant']['ratio']:.2f}x")
    report.append("")

    ratio = analysis['recent_vs_distant']['ratio']
    if ratio > 3:
        report.append("  Interpretation: STRONGLY RECENT-FOCUSED")
        report.append("  The model heavily prioritizes recent timesteps for predictions.")
    elif ratio > 1.5:
        report.append("  Interpretation: MODERATELY RECENT-FOCUSED")
        report.append("  The model prefers recent timesteps but still uses historical context.")
    else:
        report.append("  Interpretation: BALANCED TEMPORAL ATTENTION")
        report.append("  The model balances recent and distant past equally.")
    report.append("")

    # Top timesteps
    report.append("TOP IMPORTANT TIMESTEPS")
    report.append("-" * 80)
    for i, (idx, weight) in enumerate(zip(analysis['top_timesteps'][:5],
                                          analysis['top_weights'][:5]), 1):
        report.append(f"  {i}. Timestep {idx}: {weight*100:.2f}% importance")
    report.append("")

    # Temporal distribution
    report.append("TEMPORAL DISTRIBUTION BY QUARTER")
    report.append("-" * 80)
    dist = analysis['temporal_distribution']
    report.append(f"  Early (0-25%):       {dist['early']*100:.1f}%")
    report.append(f"  Mid-Early (25-50%):  {dist['mid_early']*100:.1f}%")
    report.append(f"  Mid-Late (50-75%):   {dist['mid_late']*100:.1f}%")
    report.append(f"  Late (75-100%):      {dist['late']*100:.1f}%")
    report.append("")

    # Peak analysis
    report.append("PEAK ANALYSIS")
    report.append("-" * 80)
    report.append(f"  Number of significant peaks: {analysis['num_peaks']}")
    if analysis['num_peaks'] > 0:
        report.append(f"  Peak locations: {analysis['peak_locations']}")
        if analysis['num_peaks'] == 1:
            report.append("  Interpretation: SINGLE FOCUS - model concentrates on one key timestep")
        elif analysis['num_peaks'] <= 3:
            report.append(f"  Interpretation: MULTI-FOCUS - model identifies {analysis['num_peaks']} key moments")
        else:
            report.append("  Interpretation: DISTRIBUTED - model uses many timesteps")
    report.append("")

    # Overall interpretation
    report.append("="*80)
    report.append("OVERALL INTERPRETATION")
    report.append("="*80)
    report.append("")

    if analysis['normalized_entropy'] < 0.3:
        focus_type = "highly focused on specific timesteps"
    elif analysis['normalized_entropy'] < 0.6:
        focus_type = "selectively focused with clear preferences"
    else:
        focus_type = "broadly distributed across the sequence"

    report.append(f"  The attention pooling mechanism is {focus_type}.")
    report.append("")

    if analysis['num_peaks'] == 1 and analysis['top_weights'][0] > 0.15:
        report.append("  WARNING: Attention may still be collapsing to a single timestep!")
        report.append("  Consider additional fine-tuning or architectural changes.")
    elif analysis['num_peaks'] >= 2:
        report.append("  SUCCESS: Model identifies multiple important temporal moments.")
        report.append("  This suggests the attention pooling fix is working as intended.")
    else:
        report.append("  The model shows reasonable attention distribution.")

    report.append("")
    report.append("="*80)

    report_text = "\n".join(report)
    with open(f"{output_dir}/pooling_interpretation_report.txt", 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  Report saved to {output_dir}/pooling_interpretation_report.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attention pooling weights in fine-tuned MS-TCN",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='CSV data file for analysis')
    parser.add_argument('--output', type=str, default='results/pooling_attention_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to analyze (default: 20)')
    parser.add_argument('--sequence-length', type=int, default=64,
                       help='Sequence length (default: 64)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ATTENTION POOLING ANALYSIS")
    print("="*80)

    # Load model and data
    model, sequences, feature_columns, target_columns = load_model_and_data(
        args.model, args.data, args.sequence_length
    )

    # Extract pooling weights
    pooling_weights = extract_pooling_weights(model, sequences, args.num_samples)

    # Analyze patterns
    analysis = analyze_pooling_patterns(pooling_weights)

    # Visualize
    visualize_pooling_attention(pooling_weights, analysis, str(output_dir))

    # Generate report
    generate_interpretation_report(analysis, str(output_dir))

    # Save statistics
    with open(f"{output_dir}/pooling_statistics.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Statistics saved to {output_dir}/pooling_statistics.json")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

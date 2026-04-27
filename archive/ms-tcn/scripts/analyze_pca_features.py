#!/usr/bin/env python3
"""
PCA Feature Analysis

Performs Principal Component Analysis on training data to determine:
1. Which features actually matter for power prediction
2. How many features are redundant
3. Feature correlations with power consumption
4. Dimensionality reduction opportunities

This helps understand if the model has too many irrelevant features.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(csv_path):
    """Load training data and separate features from target."""

    print(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    # Define feature columns
    feature_cols = [
        'cpu_user_percent', 'cpu_system_percent', 'cpu_idle_percent',
        'cpu_iowait_percent', 'cpu_irq_percent', 'cpu_softirq_percent',
        'context_switches_sec', 'interrupts_sec',
        'memory_used_mb', 'memory_cached_mb', 'memory_buffers_mb',
        'memory_free_mb', 'swap_used_mb', 'page_faults_sec',
        'load_1min', 'load_5min', 'load_15min',
        'running_processes', 'blocked_processes'
    ]

    # Check which features exist
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]

    if missing_features:
        print(f"\nWarning: Missing features: {missing_features}")

    print(f"\nAvailable features: {len(available_features)}")

    # Extract features and target
    X = df[available_features].values

    if 'rapl_package_power' in df.columns:
        y = df['rapl_package_power'].values
    else:
        print("Warning: Target column 'rapl_package_power' not found")
        y = None

    return X, y, available_features


def analyze_feature_correlations(X, y, feature_names):
    """Analyze direct correlation between each feature and target."""

    print("\n" + "="*80)
    print("Feature-Target Correlations")
    print("="*80)

    if y is None:
        print("No target variable available, skipping correlation analysis")
        return None

    correlations = []
    for i, feature in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr),
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)

    print(f"\n{'Feature':<30} {'Correlation':<15} {'|Correlation|':<15}")
    print("-" * 60)

    for _, row in corr_df.iterrows():
        feature = row['feature']
        corr = row['correlation']
        abs_corr = row['abs_correlation']

        marker = ""
        if abs_corr > 0.7:
            marker = "✓ STRONG"
        elif abs_corr > 0.4:
            marker = "⚠️  MODERATE"
        elif abs_corr > 0.2:
            marker = "⚠️  WEAK"
        else:
            marker = "✗ NEGLIGIBLE"

        print(f"{feature:<30} {corr:>14.3f} {abs_corr:>14.3f}  {marker}")

    return corr_df


def perform_pca_analysis(X, y, feature_names):
    """Perform PCA and analyze results."""

    print("\n" + "="*80)
    print("Principal Component Analysis")
    print("="*80)

    # Check for NaN values
    print(f"\nOriginal samples: {len(X)}")
    nan_mask = np.isnan(X).any(axis=1)
    n_nan = nan_mask.sum()

    if n_nan > 0:
        print(f"⚠️  Found {n_nan} samples with NaN values ({n_nan/len(X)*100:.1f}%)")
        print("Dropping samples with NaN values...")
        X = X[~nan_mask]
        y = y[~nan_mask]
        print(f"Clean samples: {len(X)}")
    else:
        print("✓ No NaN values found")

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    print("Running PCA...")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"\nExplained variance by component:")
    print(f"{'Component':<12} {'Variance %':<15} {'Cumulative %':<15}")
    print("-" * 42)

    for i in range(min(10, len(explained_var))):
        print(f"PC{i+1:<10} {explained_var[i]*100:>13.2f}% {cumulative_var[i]*100:>13.2f}%")

    # Find components for 95% and 99% variance
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_99 = np.argmax(cumulative_var >= 0.99) + 1

    print(f"\nComponents needed:")
    print(f"  95% variance: {n_95}/{len(feature_names)} components")
    print(f"  99% variance: {n_99}/{len(feature_names)} components")
    print(f"  Dimensionality reduction: {len(feature_names)} → {n_95} features")

    # Feature contributions to top PCs
    components = pca.components_

    print("\n" + "="*80)
    print("Feature Contributions to Top Principal Components")
    print("="*80)

    # Top 3 PCs
    for pc_idx in range(min(3, len(components))):
        print(f"\nPC{pc_idx+1} (explains {explained_var[pc_idx]*100:.1f}% variance):")
        print(f"{'Feature':<30} {'Contribution':<15}")
        print("-" * 45)

        # Get contributions for this PC
        pc_contributions = []
        for i, feature in enumerate(feature_names):
            contrib = components[pc_idx, i]
            pc_contributions.append((feature, contrib, abs(contrib)))

        # Sort by absolute contribution
        pc_contributions.sort(key=lambda x: x[2], reverse=True)

        # Print top 5
        for feature, contrib, abs_contrib in pc_contributions[:5]:
            print(f"{feature:<30} {contrib:>14.3f}")

    # Feature importance (sum of absolute contributions across top PCs)
    print("\n" + "="*80)
    print("Overall Feature Importance (Top 95% PCs)")
    print("="*80)

    feature_importance = np.sum(np.abs(components[:n_95, :]), axis=0)
    feature_importance = feature_importance / feature_importance.sum()  # Normalize

    importance_list = []
    for i, feature in enumerate(feature_names):
        importance_list.append({
            'feature': feature,
            'importance': feature_importance[i],
        })

    importance_df = pd.DataFrame(importance_list)
    importance_df = importance_df.sort_values('importance', ascending=False)

    print(f"\n{'Rank':<6} {'Feature':<30} {'Importance %':<15}")
    print("-" * 51)

    for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
        feature = row['feature']
        importance = row['importance'] * 100

        marker = ""
        if importance > 10:
            marker = "✓ HIGH"
        elif importance > 5:
            marker = "⚠️  MEDIUM"
        else:
            marker = "✗ LOW"

        print(f"{rank:<6} {feature:<30} {importance:>13.2f}%  {marker}")

    return pca, X_scaled, explained_var, cumulative_var, importance_df


def create_visualizations(pca, explained_var, cumulative_var, importance_df,
                         corr_df, feature_names, output_path):
    """Create visualization plots."""

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Explained variance
    ax1 = plt.subplot(2, 2, 1)
    n_components = min(15, len(explained_var))
    x = np.arange(1, n_components + 1)
    ax1.bar(x, explained_var[:n_components] * 100, alpha=0.7, color='steelblue')
    ax1.plot(x, cumulative_var[:n_components] * 100, 'ro-', linewidth=2, markersize=6)
    ax1.axhline(y=95, color='g', linestyle='--', linewidth=2, label='95% threshold')
    ax1.axhline(y=99, color='orange', linestyle='--', linewidth=2, label='99% threshold')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Variance Explained (%)', fontsize=12)
    ax1.set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Feature importance
    ax2 = plt.subplot(2, 2, 2)
    top_n = min(15, len(importance_df))
    top_features = importance_df.head(top_n)
    colors = ['green' if x > 0.1 else 'orange' if x > 0.05 else 'red'
              for x in top_features['importance'].values]
    ax2.barh(range(top_n), top_features['importance'].values * 100, color=colors, alpha=0.7)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_features['feature'].values, fontsize=10)
    ax2.set_xlabel('Importance (%)', fontsize=12)
    ax2.set_title('Feature Importance (PCA-based)', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Feature-target correlation
    if corr_df is not None:
        ax3 = plt.subplot(2, 2, 3)
        top_n = min(15, len(corr_df))
        top_corr = corr_df.head(top_n)
        colors = ['green' if x > 0.7 else 'orange' if x > 0.4 else 'red'
                  for x in top_corr['abs_correlation'].values]
        ax3.barh(range(top_n), top_corr['correlation'].values, color=colors, alpha=0.7)
        ax3.set_yticks(range(top_n))
        ax3.set_yticklabels(top_corr['feature'].values, fontsize=10)
        ax3.set_xlabel('Correlation with Power', fontsize=12)
        ax3.set_title('Feature-Power Correlations', fontsize=14, fontweight='bold')
        ax3.axvline(x=0, color='black', linewidth=0.5)
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')

    # Plot 4: Component loadings heatmap (top 5 PCs, all features)
    ax4 = plt.subplot(2, 2, 4)
    n_pcs = min(5, pca.n_components_)
    loadings = pca.components_[:n_pcs, :].T

    sns.heatmap(loadings,
                xticklabels=[f'PC{i+1}' for i in range(n_pcs)],
                yticklabels=feature_names,
                cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Loading'},
                ax=ax4,
                vmin=-1, vmax=1)
    ax4.set_title('Feature Loadings on Top PCs', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Principal Component', fontsize=12)
    ax4.set_ylabel('Feature', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PCA analysis to determine feature importance"
    )
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--output', type=str,
                       default='results/plots/pca_feature_analysis.png',
                       help='Output plot file')

    args = parser.parse_args()

    # Load data
    X, y, feature_names = load_and_prepare_data(args.data)

    # Correlation analysis
    corr_df = analyze_feature_correlations(X, y, feature_names)

    # PCA analysis
    pca, X_scaled, explained_var, cumulative_var, importance_df = perform_pca_analysis(
        X, y, feature_names
    )

    # Create visualizations
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    create_visualizations(pca, explained_var, cumulative_var, importance_df,
                         corr_df, feature_names, output_path)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    n_95 = np.argmax(cumulative_var >= 0.95) + 1

    print(f"\nData: {len(X)} samples, {len(feature_names)} features")
    print(f"Dimensionality: Can reduce from {len(feature_names)} to {n_95} features")

    if corr_df is not None:
        strong_features = corr_df[corr_df['abs_correlation'] > 0.7]
        weak_features = corr_df[corr_df['abs_correlation'] < 0.2]

        print(f"\nStrongly correlated features ({len(strong_features)}):")
        for _, row in strong_features.iterrows():
            print(f"  - {row['feature']}: r={row['correlation']:.3f}")

        print(f"\nWeakly correlated features ({len(weak_features)}):")
        for _, row in weak_features.iterrows():
            print(f"  - {row['feature']}: r={row['correlation']:.3f}")

    print(f"\nRecommendation:")
    if n_95 < len(feature_names) * 0.7:
        print(f"  ✓ Use PCA with {n_95} components for dimensionality reduction")
        print(f"  ✓ Or remove low-importance features")
    else:
        print(f"  ⚠️  Most features contribute to variance")
        print(f"  ⚠️  Consider feature engineering instead")


if __name__ == '__main__':
    main()

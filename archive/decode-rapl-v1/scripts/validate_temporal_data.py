#!/usr/bin/env python3
"""
Temporal Data Validation for DECODE-RAPL

Validates time-series data quality before training:
- Temporal autocorrelation structure
- Stationarity properties
- Multi-scale predictability
- Frequency spectrum analysis
- CPU-Power cross-correlation
- Regime diversity
- Pattern diversity in temporal windows
- Delay embedding quality

Usage:
    python scripts/validate_temporal_data.py [--data-path PATH] [--output-dir DIR]
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Statistical tests
from statsmodels.tsa.stattools import acf, adfuller
from scipy import signal
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class TemporalDataValidator:
    """Validates temporal time-series data quality"""

    def __init__(self, df: pd.DataFrame, output_dir: str = 'results/validation'):
        """
        Args:
            df: DataFrame with columns [timestamp, machine_id, cpu_usage, power]
            output_dir: Directory to save validation plots and reports
        """
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}
        self.passed_checks = 0
        self.total_checks = 0

    def validate_all(self):
        """Run all validation checks"""
        print("=" * 80)
        print("DECODE-RAPL Temporal Data Validation")
        print("=" * 80)
        print(f"\nDataset info:")
        print(f"  Total samples: {len(self.df):,}")
        print(f"  Machines: {self.df['machine_id'].nunique()}")
        print(f"  Duration per machine: {len(self.df) / self.df['machine_id'].nunique() / 1000:.1f} seconds")
        print()

        # Run all checks
        self.check_basic_quality()
        self.check_autocorrelation()
        self.check_stationarity()
        self.check_temporal_predictability()
        self.check_frequency_spectrum()
        self.check_cross_correlation()
        self.check_regime_diversity()
        self.check_pattern_diversity()
        self.check_delay_embedding()

        # Generate summary report
        self.generate_report()

        return self.results

    def _add_check(self, name: str, passed: bool, value: float, threshold: str, message: str):
        """Record validation check result"""
        self.total_checks += 1
        if passed:
            self.passed_checks += 1

        self.results[name] = {
            'passed': passed,
            'value': value,
            'threshold': threshold,
            'message': message
        }

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        print(f"       Value: {value:.4f}, Threshold: {threshold}")
        print(f"       {message}")
        print()

    def check_basic_quality(self):
        """Check basic data quality"""
        print("1. Basic Data Quality")
        print("-" * 80)

        # Missing values
        missing = self.df.isnull().sum().sum()
        self._add_check(
            "No missing values",
            missing == 0,
            missing,
            "= 0",
            f"Found {missing} missing values" if missing > 0 else "No missing values detected"
        )

        # Value ranges
        cpu_valid = ((self.df['cpu_usage'] >= 0) & (self.df['cpu_usage'] <= 100)).all()
        self._add_check(
            "CPU usage in [0, 100]",
            cpu_valid,
            1.0 if cpu_valid else 0.0,
            "= True",
            "All CPU values valid" if cpu_valid else "Invalid CPU values detected"
        )

        power_valid = (self.df['power'] > 0).all()
        self._add_check(
            "Power > 0",
            power_valid,
            1.0 if power_valid else 0.0,
            "= True",
            "All power values positive" if power_valid else "Non-positive power detected"
        )

        # CPU-Power correlation
        corr = self.df['cpu_usage'].corr(self.df['power'])
        self._add_check(
            "CPU-Power correlation",
            corr > 0.7,
            corr,
            "> 0.7",
            "Strong positive correlation" if corr > 0.7 else "Weak correlation - check data quality"
        )

    def check_autocorrelation(self):
        """Check temporal autocorrelation structure"""
        print("2. Temporal Autocorrelation")
        print("-" * 80)

        # Compute ACF
        cpu_acf = acf(self.df['cpu_usage'][:50000], nlags=200, fft=True)
        power_acf = acf(self.df['power'][:50000], nlags=200, fft=True)

        # Check decay pattern
        cpu_lag50 = cpu_acf[50]
        cpu_lag100 = cpu_acf[100]

        # Good: ACF decays gradually (0.2-0.8 at lag=50ms)
        self._add_check(
            "CPU ACF decay (lag=50ms)",
            0.2 < cpu_lag50 < 0.9,
            cpu_lag50,
            "0.2 < x < 0.9",
            "Good temporal structure" if 0.2 < cpu_lag50 < 0.9 else
            "Too noisy" if cpu_lag50 < 0.2 else "Too smooth/static"
        )

        self._add_check(
            "CPU ACF decay (lag=100ms)",
            0.1 < cpu_lag100 < 0.7,
            cpu_lag100,
            "0.1 < x < 0.7",
            "Good long-range correlation" if 0.1 < cpu_lag100 < 0.7 else
            "Insufficient long-range structure" if cpu_lag100 < 0.1 else "Too predictable"
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        lags = np.arange(len(cpu_acf))
        ax.plot(lags, cpu_acf, label='CPU Usage', linewidth=2)
        ax.plot(lags, power_acf, label='Power', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(0.5, color='r', linestyle='--', alpha=0.3, label='Target decay range')
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Temporal Autocorrelation Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'autocorrelation.png', dpi=150)
        plt.close()

    def check_stationarity(self):
        """Check stationarity properties"""
        print("3. Stationarity")
        print("-" * 80)

        # Augmented Dickey-Fuller test
        adf_result = adfuller(self.df['cpu_usage'][:50000])
        is_stationary = adf_result[1] < 0.05

        self._add_check(
            "ADF stationarity test",
            is_stationary,
            adf_result[1],
            "< 0.05 (p-value)",
            "Data is stationary" if is_stationary else "Data is non-stationary (may need differencing)"
        )

        # Rolling statistics
        window = 10000  # 10 second windows
        rolling_mean = self.df['cpu_usage'].rolling(window).mean()
        rolling_std = self.df['cpu_usage'].rolling(window).std()

        # Check if rolling stats vary
        mean_variation = rolling_mean.std() / self.df['cpu_usage'].std()
        std_variation = rolling_std.std() / rolling_std.mean()

        self._add_check(
            "Rolling mean variation",
            0.1 < mean_variation < 0.5,
            mean_variation,
            "0.1 < x < 0.5",
            "Good temporal variation" if 0.1 < mean_variation < 0.5 else
            "Too static" if mean_variation < 0.1 else "Too volatile"
        )

        # Plot rolling statistics
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        ax = axes[0]
        ax.plot(rolling_mean, label='Rolling Mean', linewidth=1)
        ax.axhline(self.df['cpu_usage'].mean(), color='r', linestyle='--', alpha=0.5, label='Global Mean')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('Rolling Mean (10s window)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(rolling_std, label='Rolling Std', linewidth=1, color='orange')
        ax.axhline(self.df['cpu_usage'].std(), color='r', linestyle='--', alpha=0.5, label='Global Std')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Std Dev')
        ax.set_title('Rolling Standard Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'stationarity.png', dpi=150)
        plt.close()

    def check_temporal_predictability(self):
        """Check predictability at multiple time scales"""
        print("4. Temporal Predictability")
        print("-" * 80)

        def calc_lag_r2(series, lag):
            """R² for predicting series[t] from series[t-lag]"""
            x = series[:-lag].values
            y = series[lag:].values
            mean_y = y.mean()
            ss_res = np.sum((y - x)**2)
            ss_tot = np.sum((y - mean_y)**2)
            return max(0, 1 - ss_res/ss_tot)

        series = self.df['cpu_usage'][:50000]
        lags = [1, 5, 10, 25, 50, 100, 200, 500]
        pred_scores = [calc_lag_r2(series, lag) for lag in lags]

        # Check short-term (lag=10ms) and medium-term (lag=100ms)
        r2_10ms = pred_scores[2]  # lag=10
        r2_100ms = pred_scores[5]  # lag=100

        self._add_check(
            "Short-term predictability (10ms)",
            0.7 < r2_10ms < 0.99,
            r2_10ms,
            "0.7 < x < 0.99",
            "Good short-term structure" if 0.7 < r2_10ms < 0.99 else
            "Too noisy" if r2_10ms < 0.7 else "Too deterministic"
        )

        self._add_check(
            "Medium-term predictability (100ms)",
            0.2 < r2_100ms < 0.7,
            r2_100ms,
            "0.2 < x < 0.7",
            "Good temporal complexity" if 0.2 < r2_100ms < 0.7 else
            "Too random" if r2_100ms < 0.2 else "Too predictable"
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(lags, pred_scores, 'o-', linewidth=2, markersize=8)
        ax.fill_between([0, 500], [0.7, 0.7], [0.99, 0.99], alpha=0.2, color='green', label='Target range (short)')
        ax.fill_between([0, 500], [0.2, 0.2], [0.7, 0.7], alpha=0.2, color='blue', label='Target range (medium)')
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('Predictability (R²)')
        ax.set_title('Multi-Scale Temporal Predictability')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'predictability.png', dpi=150)
        plt.close()

    def check_frequency_spectrum(self):
        """Check frequency spectrum for multi-scale dynamics"""
        print("5. Frequency Spectrum")
        print("-" * 80)

        # Power spectral density
        freqs, psd = signal.welch(
            self.df['cpu_usage'][:50000],
            fs=1000,  # 1ms sampling = 1000 Hz
            nperseg=2048
        )

        # Check for single dominant frequency (bad: synthetic periodic data)
        psd_normalized = psd / psd.sum()
        max_power_ratio = psd_normalized.max()

        self._add_check(
            "Frequency diversity",
            max_power_ratio < 0.3,
            max_power_ratio,
            "< 0.3",
            "Good multi-scale dynamics" if max_power_ratio < 0.3 else
            "Dominated by single frequency (likely synthetic)"
        )

        # Check power at different frequency bands
        # Low: 0.1-1 Hz, Mid: 1-10 Hz, High: 10-100 Hz
        low_band = psd[(freqs > 0.1) & (freqs < 1)].sum()
        mid_band = psd[(freqs > 1) & (freqs < 10)].sum()
        high_band = psd[(freqs > 10) & (freqs < 100)].sum()
        total_power = low_band + mid_band + high_band

        has_multi_scale = (low_band > 0.1 * total_power and
                          mid_band > 0.1 * total_power and
                          high_band > 0.1 * total_power)

        self._add_check(
            "Multi-scale frequency content",
            has_multi_scale,
            min(low_band, mid_band, high_band) / total_power,
            "> 0.1 for each band",
            "Power distributed across time scales" if has_multi_scale else
            "Missing dynamics at some time scales"
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.loglog(freqs, psd, linewidth=2)
        ax.axvline(0.1, color='r', linestyle='--', alpha=0.3, label='Band boundaries')
        ax.axvline(1, color='r', linestyle='--', alpha=0.3)
        ax.axvline(10, color='r', linestyle='--', alpha=0.3)
        ax.axvline(100, color='r', linestyle='--', alpha=0.3)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('Frequency Spectrum Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frequency_spectrum.png', dpi=150)
        plt.close()

    def check_cross_correlation(self):
        """Check CPU-Power cross-correlation"""
        print("6. CPU-Power Cross-Correlation")
        print("-" * 80)

        # Normalize signals
        sample_size = 10000
        cpu_norm = (self.df['cpu_usage'][:sample_size] - self.df['cpu_usage'][:sample_size].mean()) / self.df['cpu_usage'][:sample_size].std()
        power_norm = (self.df['power'][:sample_size] - self.df['power'][:sample_size].mean()) / self.df['power'][:sample_size].std()

        # Cross-correlation
        xcorr = correlate(cpu_norm, power_norm, mode='same')
        lags = np.arange(-sample_size//2, sample_size//2)

        # Find peak
        center = len(lags) // 2
        search_range = 50  # Search within ±50ms
        search_slice = slice(center - search_range, center + search_range)
        peak_idx = np.argmax(xcorr[search_slice]) + center - search_range
        peak_lag = lags[peak_idx]
        peak_value = xcorr[peak_idx] / sample_size

        self._add_check(
            "CPU-Power lag",
            abs(peak_lag) < 20,
            abs(peak_lag),
            "< 20ms",
            f"Power lags CPU by {peak_lag}ms" if abs(peak_lag) < 20 else
            "Unusual lag - check data collection timing"
        )

        self._add_check(
            "CPU-Power correlation strength",
            peak_value > 0.7,
            peak_value,
            "> 0.7",
            "Strong correlation" if peak_value > 0.7 else "Weak correlation"
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_range = 100  # Plot ±100ms around zero
        plot_slice = slice(center - plot_range, center + plot_range)
        ax.plot(lags[plot_slice], xcorr[plot_slice] / sample_size, linewidth=2)
        ax.axvline(peak_lag, color='r', linestyle='--', label=f'Peak at {peak_lag}ms')
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Lag (ms, positive = power lags CPU)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title('CPU-Power Cross-Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_correlation.png', dpi=150)
        plt.close()

    def check_regime_diversity(self):
        """Check operating regime diversity"""
        print("7. Operating Regime Diversity")
        print("-" * 80)

        # Define regimes based on rolling mean
        window = 1000  # 1 second
        rolling_mean = self.df['cpu_usage'].rolling(window, center=True).mean()

        # Quantize into regimes
        regimes = pd.cut(rolling_mean, bins=[-1, 30, 70, 101], labels=['low', 'medium', 'high'])
        regime_counts = regimes.value_counts(normalize=True)

        # Check balance
        min_regime_fraction = regime_counts.min()

        self._add_check(
            "Regime balance",
            min_regime_fraction > 0.1,
            min_regime_fraction,
            "> 0.1",
            "All regimes well represented" if min_regime_fraction > 0.1 else
            "Some regimes underrepresented - collect more diverse workload data"
        )

        # Count transitions
        transitions = (regimes != regimes.shift()).sum()
        transition_rate = transitions / len(regimes) * 1000  # Per 1000 samples

        self._add_check(
            "Regime transition rate",
            1 < transition_rate < 20,
            transition_rate,
            "1 < x < 20 per 1000 samples",
            "Good regime diversity" if 1 < transition_rate < 20 else
            "Too static" if transition_rate < 1 else "Too volatile"
        )

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        ax = axes[0]
        regime_counts.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
        ax.axhline(0.1, color='k', linestyle='--', alpha=0.5, label='Min threshold (0.1)')
        ax.set_ylabel('Fraction')
        ax.set_title('Operating Regime Distribution')
        ax.set_xlabel('Regime')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1]
        sample_range = slice(0, 20000)
        ax.plot(self.df['cpu_usage'][sample_range], alpha=0.7, label='CPU Usage')
        # Color background by regime
        regime_colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
        for regime, color in regime_colors.items():
            mask = regimes[sample_range] == regime
            ax.fill_between(range(len(mask)), 0, 100, where=mask, alpha=0.2, color=color)
        ax.set_xlabel('Sample')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('Regime Transitions (first 20s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'regime_diversity.png', dpi=150)
        plt.close()

    def check_pattern_diversity(self):
        """Check diversity of temporal windows"""
        print("8. Temporal Pattern Diversity")
        print("-" * 80)

        # Extract windows
        window_size = 120  # Model window size
        stride = 100  # Subsample for speed

        windows = []
        for i in range(0, len(self.df) - window_size, stride):
            windows.append(self.df['cpu_usage'].iloc[i:i+window_size].values)

        windows = np.array(windows)

        # PCA to measure diversity
        pca = PCA(n_components=10)
        pca.fit(windows)

        # Check variance explained
        first_pc_variance = pca.explained_variance_ratio_[0]
        first_5_pc_variance = pca.explained_variance_ratio_[:5].sum()

        self._add_check(
            "Pattern diversity (1st PC)",
            first_pc_variance < 0.7,
            first_pc_variance,
            "< 0.7",
            "Diverse temporal patterns" if first_pc_variance < 0.7 else
            "Repetitive patterns - all windows look similar"
        )

        self._add_check(
            "Pattern complexity (5 PCs)",
            0.6 < first_5_pc_variance < 0.9,
            first_5_pc_variance,
            "0.6 < x < 0.9",
            "Good pattern complexity" if 0.6 < first_5_pc_variance < 0.9 else
            "Too simple" if first_5_pc_variance > 0.9 else "Too noisy"
        )

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.bar(range(1, 11), pca.explained_variance_ratio_, color='steelblue')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Explained')
        ax.set_title('PCA of Temporal Windows')
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1]
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, 11), cumsum, 'o-', linewidth=2, markersize=8)
        ax.axhline(0.9, color='r', linestyle='--', alpha=0.5, label='90% variance')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Variance Explained')
        ax.set_title('Cumulative Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_diversity.png', dpi=150)
        plt.close()

    def check_delay_embedding(self):
        """Check delay embedding quality"""
        print("9. Delay Embedding Quality")
        print("-" * 80)

        # Simplified false nearest neighbors
        tau = 1
        d_max = 30
        sample_size = 10000
        series = self.df['cpu_usage'][:sample_size].values

        fnn_ratios = []

        for d in range(2, d_max):
            # Create embedding
            embedded = []
            for i in range(len(series) - d*tau):
                embedded.append([series[i + j*tau] for j in range(d)])
            embedded = np.array(embedded)

            if len(embedded) < 10:
                break

            # Find nearest neighbors
            try:
                nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(embedded[:-1])
                distances, indices = nbrs.kneighbors(embedded[:-1])

                # Check false nearest neighbors
                nn_idx = indices[:, 1]
                d_dist = distances[:, 1] + 1e-10  # Avoid division by zero

                # Distance in d+1 dimension
                next_coord_diff = np.abs(series[np.arange(len(embedded)-1) + d*tau] -
                                        series[nn_idx + d*tau])

                # False nearest neighbor ratio
                threshold = 15
                fnn = np.sum(next_coord_diff / d_dist > threshold) / len(d_dist)
                fnn_ratios.append(fnn)
            except:
                break

        # Check if FNN drops below threshold at d=25
        if len(fnn_ratios) >= 24:
            fnn_at_25 = fnn_ratios[23]  # d=25 is index 23

            self._add_check(
                "Embedding quality (d=25)",
                fnn_at_25 < 0.1,
                fnn_at_25,
                "< 0.1",
                "Good embedding at d=25" if fnn_at_25 < 0.1 else
                "May need higher embedding dimension"
            )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(2, len(fnn_ratios) + 2), fnn_ratios, 'o-', linewidth=2, markersize=6)
        ax.axhline(0.1, color='r', linestyle='--', alpha=0.5, label='Threshold (0.1)')
        ax.axvline(25, color='g', linestyle='--', alpha=0.5, label='Model d=25')
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('False Nearest Neighbor Ratio')
        ax.set_title('Delay Embedding Quality (Takens Theorem)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'delay_embedding.png', dpi=150)
        plt.close()

    def generate_report(self):
        """Generate final summary report"""
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"\nPassed: {self.passed_checks}/{self.total_checks} checks")
        print(f"Success Rate: {100 * self.passed_checks / self.total_checks:.1f}%")
        print()

        # Categorize results
        failed_critical = []
        failed_warnings = []

        critical_checks = [
            'No missing values',
            'CPU usage in [0, 100]',
            'Power > 0',
            'CPU-Power correlation',
        ]

        for name, result in self.results.items():
            if not result['passed']:
                if name in critical_checks:
                    failed_critical.append(name)
                else:
                    failed_warnings.append(name)

        if failed_critical:
            print("CRITICAL ISSUES:")
            for name in failed_critical:
                print(f"  ✗ {name}")
            print()

        if failed_warnings:
            print("WARNINGS (temporal quality issues):")
            for name in failed_warnings:
                print(f"  ⚠ {name}")
            print()

        # Overall assessment
        if self.passed_checks == self.total_checks:
            verdict = "✓ EXCELLENT - Data is ready for training"
            color = "green"
        elif self.passed_checks / self.total_checks > 0.8:
            verdict = "✓ GOOD - Data should work, minor issues detected"
            color = "yellow"
        elif self.passed_checks / self.total_checks > 0.6:
            verdict = "⚠ FAIR - Some quality issues, training may be suboptimal"
            color = "orange"
        else:
            verdict = "✗ POOR - Significant data quality issues detected"
            color = "red"

        print("=" * 80)
        print(verdict)
        print("=" * 80)
        print()
        print(f"Validation plots saved to: {self.output_dir}")
        print()

        # Save text report
        with open(self.output_dir / 'validation_report.txt', 'w') as f:
            f.write("DECODE-RAPL Temporal Data Validation Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Passed: {self.passed_checks}/{self.total_checks} checks\n")
            f.write(f"Success Rate: {100 * self.passed_checks / self.total_checks:.1f}%\n\n")

            for name, result in self.results.items():
                status = "PASS" if result['passed'] else "FAIL"
                f.write(f"[{status}] {name}\n")
                f.write(f"  Value: {result['value']:.4f}\n")
                f.write(f"  Threshold: {result['threshold']}\n")
                f.write(f"  {result['message']}\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write(verdict + "\n")
            f.write("=" * 80 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate temporal data quality for DECODE-RAPL')
    parser.add_argument('--data-path', type=str, default='data/synthetic_data.csv',
                       help='Path to CSV data file')
    parser.add_argument('--output-dir', type=str, default='results/validation',
                       help='Directory to save validation results')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Run validation
    validator = TemporalDataValidator(df, args.output_dir)
    validator.validate_all()


if __name__ == '__main__':
    main()

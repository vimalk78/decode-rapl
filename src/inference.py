"""
DECODE-RAPL Inference Module (v2 and v3)

Real-time power prediction with dynamic tau support.

Features:
- Auto-detect tau from checkpoint (tau=1, 4, or 8)
- Dynamic buffer sizing based on tau
- Feature-grouped delay embedding
- Real-time prediction from buffered CPU metrics
- Batch prediction from CSV files
- Supports both v2 (autoencoder) and v3 (direct predictor) models
"""

import torch
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAPLPredictor:
    """
    Real-time RAPL power predictor for DECODE-RAPL (v2 and v3)

    Automatically adapts to model's tau value (1, 4, or 8).
    Maintains delay-embedding buffer and creates 100-dim vectors.
    Supports both v2 (autoencoder) and v3 (direct predictor) architectures.
    """

    def __init__(self, checkpoint_path: str):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint (.pt file)
        """
        # Load checkpoint
        logger.info(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'config' not in checkpoint:
            raise ValueError(
                "Checkpoint missing 'config'. This checkpoint was created before "
                "inference support was added. Please retrain the model."
            )

        self.config = checkpoint['config']

        # Extract embedding parameters
        self.tau = self.config['embedding']['tau']
        self.d = self.config['embedding']['d']
        self.n_features = self.config['embedding']['n_features']

        # Calculate buffer size
        # For delay embedding: we need (d-1)*tau + 1 samples
        # Example: tau=1, d=25 → buffer_size = 24*1 + 1 = 25
        # Example: tau=8, d=25 → buffer_size = 24*8 + 1 = 193
        self.buffer_size = (self.d - 1) * self.tau + 1

        # Calculate temporal lookback
        sampling_interval_ms = 16  # 16ms per sample
        lookback_ms = (self.d - 1) * self.tau * sampling_interval_ms

        logger.info(f"Model configuration:")
        logger.info(f"  Tau: {self.tau} samples")
        logger.info(f"  Embedding dimension (d): {self.d}")
        logger.info(f"  Features: {self.n_features}")
        logger.info(f"  Buffer size: {self.buffer_size} samples")
        logger.info(f"  Temporal lookback: {lookback_ms}ms ({lookback_ms/1000:.2f}s)")

        # Initialize model
        from model import create_model

        # Get version from config (v2, v3, or v4)
        version = self.config.get('version', 'v2')
        self.model = create_model(self.config['model'], version=version)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model architecture: {version}")

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        logger.info(f"Model loaded on {self.device}")

        # Load normalization parameters
        if 'normalization' in self.config and self.config['normalization']:
            norm = self.config['normalization']
            self.feature_min = np.array(norm['feature_min'], dtype=np.float32)
            self.feature_range = np.array(norm['feature_range'], dtype=np.float32)
            self.use_normalization = True

            logger.info("="*80)
            logger.info("MINMAX SCALER LOADED FROM CHECKPOINT")
            logger.info("="*80)
            logger.info(f"Normalization type: {norm['type']}")
            logger.info(f"feature_min shape: {self.feature_min.shape}")
            logger.info(f"feature_range shape: {self.feature_range.shape}")
            logger.info(f"\nFeature min (first 10):  {self.feature_min[:10]}")
            logger.info(f"Feature range (first 10): {self.feature_range[:10]}")
            logger.info(f"\nFeature min (last 10):   {self.feature_min[-10:]}")
            logger.info(f"Feature range (last 10):  {self.feature_range[-10:]}")
            logger.info(f"\nMin of all mins: {self.feature_min.min():.4f}")
            logger.info(f"Max of all ranges: {self.feature_range.max():.4f}")
            logger.info(f"Mean of ranges: {self.feature_range.mean():.4f}")
            logger.info("="*80)
        else:
            self.use_normalization = False
            logger.warning("WARNING: No normalization parameters in checkpoint!")
            logger.warning("Model will receive unnormalized inputs - predictions may be poor!")

        # Initialize history buffer
        # Buffer stores raw features: [user%, system%, iowait%, ctx_switches_per_sec]
        self.buffer = deque(maxlen=self.buffer_size)

        # Debug flag for printing normalized values (only once)
        self._debug_printed = False

    def update_metrics(self, user_percent: float, system_percent: float,
                      iowait_percent: float, ctx_switches_per_sec: float):
        """
        Update buffer with new CPU metrics

        Args:
            user_percent: User CPU percentage (0-100)
            system_percent: System CPU percentage (0-100)
            iowait_percent: I/O wait percentage (0-100)
            ctx_switches_per_sec: Context switches per second (raw count)
        """
        # Apply log transform to context switches (matching training)
        log_ctx_switches = np.log1p(ctx_switches_per_sec)

        # Store in buffer
        self.buffer.append([user_percent, system_percent, iowait_percent, log_ctx_switches])

    def predict(self) -> Optional[float]:
        """
        Predict power consumption based on buffered metrics

        Returns:
            Predicted power in Watts, or None if buffer not full
        """
        # Check if buffer is full
        if len(self.buffer) < self.buffer_size:
            logger.debug(f"Buffer not full yet: {len(self.buffer)}/{self.buffer_size}")
            return None

        # Convert buffer to array
        buffer_array = np.array(self.buffer)  # Shape: (buffer_size, 4)

        # Create delay-embedded vector (feature-grouped ordering)
        embedded_vector = self._create_delay_embedding(buffer_array)

        # Apply normalization if available
        if self.use_normalization:
            embedded_vector_normalized = self._apply_normalization(embedded_vector)

            # Debug print (only once)
            if not self._debug_printed:
                logger.info("\n" + "="*80)
                logger.info("NORMALIZATION DEBUG (First Prediction)")
                logger.info("="*80)
                logger.info(f"Raw embedded vector (first 10):  {embedded_vector[:10]}")
                logger.info(f"Normalized vector (first 10):    {embedded_vector_normalized[:10]}")
                logger.info(f"\nRaw embedded vector (last 10):   {embedded_vector[-10:]}")
                logger.info(f"Normalized vector (last 10):     {embedded_vector_normalized[-10:]}")
                logger.info(f"\nRaw range: [{embedded_vector.min():.4f}, {embedded_vector.max():.4f}]")
                logger.info(f"Normalized range: [{embedded_vector_normalized.min():.4f}, {embedded_vector_normalized.max():.4f}]")
                logger.info(f"\nValues in [0, 1]: {np.all((embedded_vector_normalized >= 0) & (embedded_vector_normalized <= 1))}")
                logger.info(f"Values < 0: {np.sum(embedded_vector_normalized < 0)}")
                logger.info(f"Values > 1: {np.sum(embedded_vector_normalized > 1)}")
                logger.info("="*80 + "\n")
                self._debug_printed = True

            embedded_vector = embedded_vector_normalized

        # Convert to tensor
        x = torch.FloatTensor(embedded_vector).unsqueeze(0)  # Shape: (1, 100)
        x = x.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(x)
            power_pred = outputs['power_pred'].cpu().numpy()[0, 0]

        return float(power_pred)

    def _apply_normalization(self, x: np.ndarray) -> np.ndarray:
        """
        Apply MinMaxScaler normalization to input vector

        Args:
            x: Input vector of shape (100,)

        Returns:
            Normalized vector scaled to [0, 1] range
        """
        # Apply: (x - min) / range
        x_normalized = (x - self.feature_min) / self.feature_range

        # Clip to [0, 1] to handle out-of-distribution values
        x_normalized = np.clip(x_normalized, 0.0, 1.0)

        return x_normalized

    def _create_delay_embedding(self, buffer: np.ndarray) -> np.ndarray:
        """
        Create delay-embedded vector from buffer

        Uses feature-grouped ordering to match training:
        [user(t), user(t-τ), ..., user(t-24τ),       positions 0-24
         sys(t), sys(t-τ), ..., sys(t-24τ),          positions 25-49
         iowait(t), iowait(t-τ), ..., iowait(t-24τ), positions 50-74
         log_ctx(t), log_ctx(t-τ), ..., log_ctx(t-24τ)] positions 75-99

        Args:
            buffer: Array of shape (buffer_size, 4)

        Returns:
            Embedded vector of shape (100,) = 4 features × 25 delays
        """
        embedded = np.zeros(self.d * self.n_features)

        # Feature-grouped ordering
        for feat_idx in range(self.n_features):
            for delay_idx in range(self.d):
                # Calculate which buffer position to read from
                # For most recent (delay_idx=0): read from end of buffer
                # For oldest (delay_idx=d-1): read from beginning
                buffer_pos = self.buffer_size - 1 - (delay_idx * self.tau)

                # Store in embedded vector
                embedded_pos = feat_idx * self.d + delay_idx
                embedded[embedded_pos] = buffer[buffer_pos, feat_idx]

        return embedded

    def predict_from_sequence(self, user_pct: np.ndarray, system_pct: np.ndarray,
                             iowait_pct: np.ndarray, ctx_switches: np.ndarray) -> np.ndarray:
        """
        Predict power from complete sequences (batch mode)

        Args:
            user_pct: User CPU percentage array (N,)
            system_pct: System CPU percentage array (N,)
            iowait_pct: I/O wait percentage array (N,)
            ctx_switches: Context switches per second array (N,)

        Returns:
            Predicted power array (N - buffer_size + 1,)
        """
        # Apply log transform to context switches
        log_ctx = np.log1p(ctx_switches)

        # Stack into feature array
        features = np.stack([user_pct, system_pct, iowait_pct, log_ctx], axis=1)  # (N, 4)

        # Create sliding windows
        n_samples = len(features)
        n_predictions = n_samples - self.buffer_size + 1

        if n_predictions <= 0:
            raise ValueError(
                f"Sequence too short. Need at least {self.buffer_size} samples, "
                f"got {n_samples}"
            )

        predictions = []

        for i in range(n_predictions):
            # Extract window
            window = features[i:i + self.buffer_size]

            # Create delay embedding
            embedded = self._create_delay_embedding(window)

            # Apply normalization if available
            if self.use_normalization:
                embedded = self._apply_normalization(embedded)

            # Predict
            x = torch.FloatTensor(embedded).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(x)
                power_pred = outputs['power_pred'].cpu().numpy()[0, 0]

            predictions.append(power_pred)

        return np.array(predictions)

    def reset(self):
        """Clear buffer"""
        self.buffer.clear()

    def get_buffer_info(self) -> Dict:
        """Get buffer status information"""
        return {
            'tau': self.tau,
            'd': self.d,
            'buffer_size': self.buffer_size,
            'current_fill': len(self.buffer),
            'is_ready': len(self.buffer) >= self.buffer_size,
            'lookback_ms': (self.d - 1) * self.tau * 16
        }


def test_inference():
    """Test inference with synthetic data"""
    import sys
    from pathlib import Path

    print("Testing DECODE-RAPL inference...")

    # Check if v3 model exists, otherwise try v2
    checkpoint_path = "checkpoints/v3_tau1/best_model.pt"

    if not Path(checkpoint_path).exists():
        checkpoint_path = "checkpoints/v2_tau1/best_model.pt"
        if not Path(checkpoint_path).exists():
            print(f"Error: No checkpoint found")
            print("Please train a model first using: python src/train.py")
            return 1

    print(f"Using checkpoint: {checkpoint_path}")

    # Test predictor initialization
    print("\n1. Initializing predictor...")
    try:
        predictor = RAPLPredictor(checkpoint_path)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nThis checkpoint was created before inference support.")
        print("Please retrain with: python src/train.py --config config/v2_default.yaml")
        return 1

    print("✓ Predictor initialized")

    # Print buffer info
    info = predictor.get_buffer_info()
    print(f"\nBuffer configuration:")
    print(f"  Tau: {info['tau']}")
    print(f"  Buffer size: {info['buffer_size']} samples")
    print(f"  Temporal lookback: {info['lookback_ms']}ms")

    # Test real-time prediction
    print("\n2. Testing real-time prediction...")

    # Fill buffer with synthetic data
    np.random.seed(42)
    for i in range(info['buffer_size']):
        user_pct = np.random.uniform(10, 50)
        system_pct = np.random.uniform(5, 20)
        iowait_pct = np.random.uniform(0, 5)
        ctx_switches = np.random.uniform(1000, 5000)

        predictor.update_metrics(user_pct, system_pct, iowait_pct, ctx_switches)

        if (i + 1) % 10 == 0:
            print(f"  Buffer: {i+1}/{info['buffer_size']}")

    # Make prediction
    power = predictor.predict()
    print(f"\n✓ Prediction: {power:.2f}W")

    # Test batch prediction
    print("\n3. Testing batch prediction...")
    n_samples = 200
    user_seq = np.random.uniform(10, 50, n_samples)
    system_seq = np.random.uniform(5, 20, n_samples)
    iowait_seq = np.random.uniform(0, 5, n_samples)
    ctx_seq = np.random.uniform(1000, 5000, n_samples)

    predictions = predictor.predict_from_sequence(
        user_seq, system_seq, iowait_seq, ctx_seq
    )

    print(f"✓ Generated {len(predictions)} predictions")
    print(f"  Mean: {predictions.mean():.2f}W")
    print(f"  Range: [{predictions.min():.2f}, {predictions.max():.2f}]W")

    print("\n✓ All tests passed!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(test_inference())

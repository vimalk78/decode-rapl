"""
DECODE-RAPL Inference Module
Real-time power prediction for bare-metal and VM environments
"""

import torch
import numpy as np
from collections import deque
from typing import Optional, Tuple
import time
import logging
import warnings

from src.preprocessing import DelayEmbedding

# Suppress sklearn feature name warnings during inference
warnings.filterwarnings('ignore', message='X does not have valid feature names')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAPLPredictor:
    """
    Real-time RAPL power predictor

    Supports:
    - Bare-metal inference: Direct prediction from CPU usage
    - VM inference: Scaling for virtualized environments
    """

    def __init__(
        self,
        checkpoint_path: str,
        vm_mode: bool = False,
        vm_vcpus: Optional[int] = None,
        host_cores: Optional[int] = None
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            vm_mode: Enable VM scaling mode
            vm_vcpus: Number of VM vCPUs (required if vm_mode=True)
            host_cores: Number of host physical cores (required if vm_mode=True)
        """
        self.vm_mode = vm_mode

        if vm_mode and (vm_vcpus is None or host_cores is None):
            raise ValueError("VM mode requires vm_vcpus and host_cores parameters")

        self.vm_vcpus = vm_vcpus
        self.host_cores = host_cores

        # Load checkpoint
        logger.info(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        self.config = checkpoint['config']
        self.machine_id_map = checkpoint['machine_id_map']
        self.scalers = checkpoint['scalers']

        # Determine if model is multi-feature
        self.feature_columns = self.config['preprocessing'].get('feature_columns', ['cpu_usage'])
        self.n_features = len(self.feature_columns)
        self.is_multifeature = self.n_features > 1

        # Initialize delay embedder
        self.embedder = DelayEmbedding(
            tau=self.config['embedding']['tau'],
            d=self.config['embedding']['d'],
            n_features=self.n_features
        )

        self.buffer_size = self.config['inference']['buffer_size']

        # Initialize model
        from src.model import create_model

        num_machines = len(self.machine_id_map)
        self.model = create_model(self.config, num_machines)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model type: {'Multi-feature' if self.is_multifeature else 'Single-feature'}")
        if self.is_multifeature:
            logger.info(f"Features: {self.feature_columns}")
        logger.info(f"VM mode: {vm_mode}")

        # Initialize history buffer
        self.usage_buffer = deque(maxlen=self.buffer_size + (self.embedder.d - 1) * self.embedder.tau)

    def update_usage(self, cpu_usage):
        """
        Update usage buffer with new measurement

        Args:
            cpu_usage: For single-feature: float (CPU usage 0-100)
                      For multi-feature: dict with keys {'user_percent', 'system_percent', 'context_switches'}
                                        or np.ndarray of shape (3,)
        """
        # Convert dict to array if needed
        if self.is_multifeature:
            if isinstance(cpu_usage, dict):
                # Extract features in correct order
                features = np.array([
                    cpu_usage['user_percent'],
                    cpu_usage['system_percent'],
                    cpu_usage['context_switches']
                ])
            else:
                features = np.array(cpu_usage)  # Assume already in correct format

            # Normalize if needed
            if self.scalers['usage'] is not None:
                features_norm = self.scalers['usage'].transform([features])[0]
            else:
                # Default normalization: user% and system% by 100, context_switches by 100000
                features_norm = features / np.array([100.0, 100.0, 100000.0])

            self.usage_buffer.append(features_norm)
        else:
            # Single-feature mode (backward compatible)
            if self.scalers['usage'] is not None:
                cpu_usage_norm = self.scalers['usage'].transform([[cpu_usage]])[0, 0]
            else:
                cpu_usage_norm = cpu_usage / 100.0

            self.usage_buffer.append(cpu_usage_norm)

    def predict(self) -> Optional[float]:
        """
        Predict power consumption based on buffered usage history

        Returns:
            Predicted power in Watts, or None if buffer not full
        """
        # Check if buffer has enough data
        required_length = self.buffer_size + (self.embedder.d - 1) * self.embedder.tau

        if len(self.usage_buffer) < required_length:
            logger.debug(
                f"Buffer not full yet: {len(self.usage_buffer)}/{required_length}"
            )
            return None

        # Convert buffer to array
        usage_array = np.array(self.usage_buffer)

        # Apply VM scaling if needed
        if self.vm_mode:
            # Scale VM usage to effective host usage
            # effective_host_usage = vm_usage * (vm_vcpus / host_cores)
            usage_array = usage_array * (self.vm_vcpus / self.host_cores)

        # Apply delay embedding
        # For single-feature: usage_array shape is (T,)
        # For multi-feature: usage_array shape is (T, 3)
        embedded = self.embedder.embed(usage_array)  # Shape: (T, d) or (T, d*n_features)

        # Take last window_size timesteps
        window_size = self.config['preprocessing']['window_size']
        if len(embedded) < window_size:
            logger.warning(
                f"Not enough embedded samples: {len(embedded)}/{window_size}"
            )
            return None

        embedded_window = embedded[-window_size:]  # Shape: (window_size, d)

        # Convert to tensor
        x = torch.FloatTensor(embedded_window).unsqueeze(0)  # Shape: (1, window_size, d)
        x = x.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(x)
            power_pred = outputs['power_pred'].cpu().numpy()[0, 0]

        # Denormalize
        if self.scalers['power'] is not None:
            power_pred = self.scalers['power'].inverse_transform([[power_pred]])[0, 0]

        # Apply VM scaling if needed
        if self.vm_mode:
            # Scale back to VM power
            # vm_power = predicted_host_power * (vm_vcpus / host_cores)
            power_pred = power_pred * (self.vm_vcpus / self.host_cores)

        return float(power_pred)

    def predict_from_sequence(self, usage_sequence: np.ndarray) -> float:
        """
        Predict power from a complete usage sequence (batch mode)

        Args:
            usage_sequence: Array of CPU usage values (0-100)

        Returns:
            Predicted power in Watts
        """
        # Normalize
        if self.scalers['usage'] is not None:
            usage_norm = self.scalers['usage'].transform(usage_sequence.reshape(-1, 1)).flatten()
        else:
            usage_norm = usage_sequence / 100.0

        # Apply VM scaling if needed
        if self.vm_mode:
            usage_norm = usage_norm * (self.vm_vcpus / self.host_cores)

        # Apply delay embedding
        embedded = self.embedder.embed(usage_norm)

        # Take last window_size timesteps
        window_size = self.config['preprocessing']['window_size']
        if len(embedded) < window_size:
            raise ValueError(
                f"Sequence too short after embedding: {len(embedded)} < {window_size}"
            )

        embedded_window = embedded[-window_size:]

        # Convert to tensor
        x = torch.FloatTensor(embedded_window).unsqueeze(0)
        x = x.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(x)
            power_pred = outputs['power_pred'].cpu().numpy()[0, 0]

        # Denormalize
        if self.scalers['power'] is not None:
            power_pred = self.scalers['power'].inverse_transform([[power_pred]])[0, 0]

        # Apply VM scaling if needed
        if self.vm_mode:
            power_pred = power_pred * (self.vm_vcpus / self.host_cores)

        return float(power_pred)

    def reset(self):
        """Clear usage buffer"""
        self.usage_buffer.clear()


def run_realtime_inference(
    predictor: RAPLPredictor,
    get_cpu_usage_fn,
    duration_seconds: float = 60,
    sampling_rate_ms: int = 100
):
    """
    Run real-time inference loop

    Args:
        predictor: RAPLPredictor instance
        get_cpu_usage_fn: Function that returns current CPU usage (0-100)
        duration_seconds: Duration to run inference
        sampling_rate_ms: Sampling rate in milliseconds
    """
    logger.info(f"Starting real-time inference for {duration_seconds}s...")

    predictions = []
    timestamps = []

    start_time = time.time()
    iteration = 0

    while (time.time() - start_time) < duration_seconds:
        iter_start = time.time()

        # Get current CPU usage
        cpu_usage = get_cpu_usage_fn()

        # Update buffer
        predictor.update_usage(cpu_usage)

        # Predict
        power = predictor.predict()

        if power is not None:
            predictions.append(power)
            timestamps.append(time.time() - start_time)

            if iteration % 10 == 0:  # Log every 10 iterations
                logger.info(f"Time: {timestamps[-1]:.2f}s, CPU: {cpu_usage:.1f}%, Power: {power:.2f}W")

        iteration += 1

        # Sleep to maintain sampling rate
        elapsed = time.time() - iter_start
        sleep_time = (sampling_rate_ms / 1000.0) - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info(f"Inference completed. {len(predictions)} predictions made.")

    return predictions, timestamps


if __name__ == "__main__":
    # Test inference with synthetic data
    import sys
    sys.path.append('.')
    from pathlib import Path

    print("Testing inference module...")

    # Check if model exists
    checkpoint_path = "checkpoints/best_model.pth"

    if not Path(checkpoint_path).exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train a model first using: python src/train.py")
        sys.exit(1)

    # Test bare-metal inference
    print("\n1. Testing bare-metal inference...")
    predictor_bare = RAPLPredictor(checkpoint_path, vm_mode=False)

    # Simulate usage sequence
    test_usage = np.random.uniform(20, 80, 200)  # 200 samples of CPU usage

    print(f"Test usage shape: {test_usage.shape}")

    power_pred = predictor_bare.predict_from_sequence(test_usage)
    print(f"Predicted power (bare-metal): {power_pred:.2f}W")

    # Test VM inference
    print("\n2. Testing VM inference...")
    predictor_vm = RAPLPredictor(
        checkpoint_path,
        vm_mode=True,
        vm_vcpus=4,
        host_cores=16
    )

    power_pred_vm = predictor_vm.predict_from_sequence(test_usage)
    print(f"Predicted power (VM): {power_pred_vm:.2f}W")

    # Test real-time inference with synthetic CPU usage
    print("\n3. Testing real-time inference (5 seconds)...")

    def get_synthetic_cpu_usage():
        """Simulate CPU usage readings"""
        return np.random.uniform(30, 70)

    predictions, timestamps = run_realtime_inference(
        predictor_bare,
        get_synthetic_cpu_usage,
        duration_seconds=5,
        sampling_rate_ms=100
    )

    print(f"\nMade {len(predictions)} predictions")
    print(f"Average power: {np.mean(predictions):.2f}W")
    print(f"Power range: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]W")

    print("\nInference module test completed successfully!")

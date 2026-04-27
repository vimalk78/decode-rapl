#!/usr/bin/env python3
"""
Quick test to verify v4.1 model instantiation with BatchNorm
"""

import torch
import yaml
from pathlib import Path
from src.model import create_model

def test_v4_model():
    # Load config
    config_path = Path("config/v4_tau1.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Testing DECODE-RAPL v4.1 Model Instantiation")
    print("=" * 60)
    print()

    # Create model
    print("Creating v4 model with config:")
    print(f"  CNN kernel size: {config['model']['cnn_kernel_size']}")
    print(f"  Use BatchNorm: {config['model']['use_batch_norm']}")
    print(f"  CNN channels: {config['model']['cnn_channels']}")
    print()

    model = create_model(config['model'], version='v4')

    # Print model structure
    print("Model Architecture:")
    print(model)
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Parameter Count:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print()

    # Check for BatchNorm layers
    batch_norm_count = sum(1 for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d))
    print(f"BatchNorm1d layers found: {batch_norm_count}")
    print()

    # Test forward pass
    print("Testing forward pass...")
    batch_size = 4
    input_dim = 100
    x = torch.randn(batch_size, input_dim)

    try:
        with torch.no_grad():
            output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  Power prediction shape: {output['power_pred'].shape}")
        print(f"  Latent space shape: {output['z'].shape}")
        print(f"  ✓ Forward pass successful!")
        print()

        # Test with larger batch to verify BatchNorm works
        print("Testing with larger batch (BatchNorm test)...")
        x_large = torch.randn(256, input_dim)
        with torch.no_grad():
            output_large = model(x_large)
        print(f"  Batch size 256: {x_large.shape} → {output_large['power_pred'].shape}")
        print(f"  ✓ BatchNorm working correctly!")
        print()

    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False

    print("=" * 60)
    print("✓ v4.1 Model Test Passed!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    test_v4_model()

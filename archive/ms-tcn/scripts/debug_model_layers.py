#!/usr/bin/env python3
"""
Debug MS-TCN model layer by layer
Tests what each layer produces for idle vs load scenarios
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_model_and_data(model_path, csv_path=None):
    """Load trained model"""
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)

    # Reconstruct model architecture
    from train_model import MSTCN

    config = model_data['config']
    preprocessor = model_data['preprocessor']

    # Get architecture info from preprocessor
    num_features = len(preprocessor['feature_columns'])
    num_targets = len(preprocessor['target_columns'])

    model = MSTCN(
        num_features=num_features,
        num_targets=num_targets,
        dropout=0.3
    )

    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    return model, preprocessor, config

def create_test_sequence(feature_values, feature_columns, sequence_length=64):
    """
    Create a sequence with specific feature values

    Args:
        feature_values: dict of {feature_name: value}
        feature_columns: list of feature column names
        sequence_length: number of samples in sequence
    """
    # Create sequence array
    sequence = np.zeros((sequence_length, len(feature_columns)))

    for i, col in enumerate(feature_columns):
        if col in feature_values:
            sequence[:, i] = feature_values[col]

    return sequence

def normalize_sequence(sequence, preprocessor):
    """Normalize sequence using saved scaler parameters"""
    mean = preprocessor['feature_scaler_mean']
    scale = preprocessor['feature_scaler_scale']

    normalized = (sequence - mean) / scale
    return normalized

def debug_model_inference(model, sequence, layer_name="test"):
    """
    Run inference and print intermediate layer outputs

    Args:
        model: MSTCN model
        sequence: input sequence (seq_len, features)
        layer_name: name for this test
    """
    print(f"\n{'='*80}")
    print(f"Layer-by-Layer Analysis: {layer_name}")
    print(f"{'='*80}\n")

    # Convert to tensor and add batch dimension
    x = torch.FloatTensor(sequence).unsqueeze(0)  # (1, seq_len, features)

    print(f"Input shape: {x.shape}")
    print(f"Input statistics:")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"  Min: {x.min():.4f}, Max: {x.max():.4f}")
    print(f"  Non-zero features: {(x.abs() > 0.01).sum().item()} / {x.numel()}")

    # Transpose for convolution
    x = x.transpose(1, 2)  # (1, features, seq_len)
    print(f"\nAfter transpose: {x.shape}")

    # Multi-scale convolution
    x = model.multiscale_conv(x)
    print(f"\nAfter multiscale_conv: {x.shape}")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"  Min: {x.min():.4f}, Max: {x.max():.4f}")
    print(f"  Dead neurons (all zeros): {(x.abs().sum(dim=(0,2)) == 0).sum().item()}")

    # Dilated temporal blocks
    for i, block in enumerate(model.temporal_blocks):
        x_before = x
        x = block(x)
        change = (x - x_before).abs().mean()
        print(f"\nAfter temporal_block_{i}: {x.shape}")
        print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
        print(f"  Change from previous: {change:.4f}")
        print(f"  Dead neurons: {(x.abs().sum(dim=(0,2)) == 0).sum().item()}")

    # Multi-head attention
    x = model.attention(x)
    print(f"\nAfter attention: {x.shape}")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")

    # Global pooling
    x = model.global_pool(x).squeeze(-1)
    print(f"\nAfter global_pool: {x.shape}")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"  Values: {x[0,:10].detach().numpy()}")

    # FC1
    x = model.fc1(x)
    x_after_fc1 = x.clone()
    x = model.bn_fc1(x)
    x = F.relu(x)
    dead_fc1 = (x == 0).sum().item()
    print(f"\nAfter FC1 (before activation): {x_after_fc1.shape}")
    print(f"  Mean: {x_after_fc1.mean():.4f}, Std: {x_after_fc1.std():.4f}")
    print(f"  Bias contribution: {model.fc1.bias.mean():.4f}")
    print(f"After ReLU:")
    print(f"  Dead neurons (zeroed by ReLU): {dead_fc1} / {x.numel()}")

    x = model.dropout_fc(x)

    # FC2
    x = model.fc2(x)
    x_after_fc2 = x.clone()
    x = model.bn_fc2(x)
    x = F.relu(x)
    dead_fc2 = (x == 0).sum().item()
    print(f"\nAfter FC2 (before activation): {x_after_fc2.shape}")
    print(f"  Mean: {x_after_fc2.mean():.4f}, Std: {x_after_fc2.std():.4f}")
    print(f"  Bias contribution: {model.fc2.bias.mean():.4f}")
    print(f"After ReLU:")
    print(f"  Dead neurons: {dead_fc2} / {x.numel()}")

    # Output heads
    outputs = []
    for i, head in enumerate(model.output_heads):
        out = head(x)
        outputs.append(out)
        print(f"\nOutput head {i}:")
        print(f"  Value: {out.item():.4f}")
        print(f"  Weight norm: {head.weight.norm().item():.4f}")
        print(f"  Bias: {head.bias.item():.4f}")

    final_output = torch.cat(outputs, dim=1)
    print(f"\nFinal output (normalized): {final_output.item():.4f}")

    return final_output.item()

def main():
    # Load model
    model_path = 'models/best_model_diverse_2hr.pth'
    print(f"Loading model from {model_path}...")
    model, preprocessor, config = load_model_and_data(model_path)

    feature_columns = preprocessor['feature_columns']
    sequence_length = preprocessor['sequence_length']

    print(f"\nModel configuration:")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Targets: {preprocessor['target_columns']}")

    # Print normalization parameters
    print(f"\n{'='*80}")
    print("Normalization Parameters")
    print(f"{'='*80}\n")
    print("Feature                        Mean        Std")
    print("-" * 60)
    for i, col in enumerate(feature_columns):
        mean = preprocessor['feature_scaler_mean'][i]
        std = preprocessor['feature_scaler_scale'][i]
        print(f"{col:30s} {mean:10.2f}  {std:10.2f}")

    # Test Case 1: Pure idle (all zeros/idle values)
    idle_values = {
        'cpu_user_percent': 0.0,
        'cpu_system_percent': 0.0,
        'cpu_idle_percent': 100.0,
        'cpu_iowait_percent': 0.0,
        'cpu_irq_percent': 0.0,
        'cpu_softirq_percent': 0.0,
        'context_switches_sec': 1500.0,
        'interrupts_sec': 1500.0,
        'memory_used_mb': 4086.0,
        'memory_cached_mb': 14445.0,
        'memory_buffers_mb': 2.1,
        'memory_free_mb': 77173.0,
        'swap_used_mb': 4.0,
        'page_faults_sec': 0.0,
        'load_1min': 0.59,
        'load_5min': 1.32,
        'load_15min': 3.76,
        'running_processes': 1.0,
        'blocked_processes': 0.0
    }

    sequence_idle = create_test_sequence(idle_values, feature_columns, sequence_length)
    normalized_idle = normalize_sequence(sequence_idle, preprocessor)

    print(f"\n{'='*80}")
    print("IDLE Sequence - Raw Feature Values")
    print(f"{'='*80}\n")
    print("Feature                        Value       Normalized")
    print("-" * 60)
    for i, col in enumerate(feature_columns):
        raw = sequence_idle[0, i]
        norm = normalized_idle[0, i]
        print(f"{col:30s} {raw:10.2f}  {norm:10.4f}")

    idle_output = debug_model_inference(model, normalized_idle, "IDLE (0% CPU)")

    # Denormalize output
    target_mean = preprocessor['target_scaler_mean'][0]
    target_std = preprocessor['target_scaler_scale'][0]
    idle_power = idle_output * target_std + target_mean

    print(f"\n{'='*80}")
    print(f"IDLE Prediction: {idle_power:.2f}W")
    print(f"{'='*80}\n")

    # Test Case 2: Heavy load
    load_values = idle_values.copy()
    load_values.update({
        'cpu_user_percent': 80.0,
        'cpu_idle_percent': 18.0,
        'cpu_system_percent': 2.0,
        'interrupts_sec': 15000.0,
        'context_switches_sec': 8000.0,
        'running_processes': 5.0,
        'page_faults_sec': 5000.0
    })

    sequence_load = create_test_sequence(load_values, feature_columns, sequence_length)
    normalized_load = normalize_sequence(sequence_load, preprocessor)

    print(f"\n{'='*80}")
    print("LOAD Sequence - Raw Feature Values")
    print(f"{'='*80}\n")
    print("Feature                        Value       Normalized")
    print("-" * 60)
    for i, col in enumerate(feature_columns):
        raw = sequence_load[0, i]
        norm = normalized_load[0, i]
        print(f"{col:30s} {raw:10.2f}  {norm:10.4f}")

    load_output = debug_model_inference(model, normalized_load, "LOAD (80% CPU)")

    load_power = load_output * target_std + target_mean

    print(f"\n{'='*80}")
    print(f"LOAD Prediction: {load_power:.2f}W")
    print(f"{'='*80}\n")

    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}\n")
    print(f"Idle prediction:  {idle_power:.2f}W")
    print(f"Load prediction:  {load_power:.2f}W")
    print(f"Delta:            {load_power - idle_power:.2f}W")
    print(f"\n")

    # Check for systematic bias in output layer
    print(f"{'='*80}")
    print("Potential Bias Sources")
    print(f"{'='*80}\n")

    print("Output head bias:", model.output_heads[0].bias.item())
    print("FC1 mean bias:", model.fc1.bias.mean().item())
    print("FC2 mean bias:", model.fc2.bias.mean().item())

    # Check how many features become negative after normalization at idle
    negative_idle = (normalized_idle[0, :] < 0).sum()
    negative_load = (normalized_load[0, :] < 0).sum()
    print(f"\nNegative features after normalization:")
    print(f"  Idle: {negative_idle} / {len(feature_columns)}")
    print(f"  Load: {negative_load} / {len(feature_columns)}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Generate architecture diagrams for DECODE-RAPL documentation
Uses TorchViz to auto-generate from actual model code
"""

import sys
sys.path.append('.')

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.model import create_model
from src.utils import load_config


def create_model_architecture_graph():
    """Create computation graph using TorchViz"""
    try:
        from torchviz import make_dot
    except ImportError:
        print("ERROR: torchviz not installed")
        print("Install with: pip install torchviz")
        return False

    print("Creating model architecture graph...")

    # Load config and create model
    config = load_config()
    model = create_model(config, num_machines=3)
    model.eval()

    # Create dummy input
    batch_size = 1
    seq_len = config['preprocessing']['window_size']
    input_dim = config['embedding']['d']

    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    output = model(x)

    # Generate graph for power prediction path
    dot = make_dot(
        output['power_pred'],
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=False
    )

    # Customize graph
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded')

    # Save
    output_path = 'docs/images/model_computation_graph'
    dot.render(output_path, format='png', cleanup=True)
    print(f"✓ Created {output_path}.png")

    return True


def create_architecture_overview():
    """Create high-level architecture diagram using matplotlib"""
    print("Creating architecture overview...")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'DECODE-RAPL Architecture',
            ha='center', fontsize=18, fontweight='bold')

    # Helper function for boxes
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def draw_box(x, y, w, h, text, color='lightblue', style='round,pad=0.1'):
        box = FancyBboxPatch((x, y), w, h, boxstyle=style,
                             edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=10, fontweight='bold', wrap=True)

    def draw_arrow(x1, y1, x2, y2, label='', style='->', color='black', linestyle='solid'):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle=style, mutation_scale=20,
                               linewidth=2.5, color=color, linestyle=linestyle)
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=9, style='italic',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Input
    draw_box(0.5, 10, 2, 0.8, 'CPU Usage\n(0-100%)', 'lightblue')

    # Preprocessing
    draw_box(0.5, 8.7, 2, 0.8, 'Normalize\n÷ 100', 'lightyellow')
    draw_arrow(1.5, 10, 1.5, 9.5)

    draw_box(0.5, 7.2, 2, 1.2, 'Delay Embed\nτ=1ms, d=25\nh(t) ∈ ℝ²⁵', 'lightyellow')
    draw_arrow(1.5, 8.7, 1.5, 8.4)

    # Main model box
    draw_box(3.5, 3, 6, 5, '', 'white', 'round,pad=0.2')
    ax.text(6.5, 7.7, 'DECODE-RAPL Model', fontsize=12, fontweight='bold', ha='center')

    # Encoder
    draw_box(4, 6.5, 2, 1, 'Encoder\n512→128→64→16', 'lightgreen')
    draw_arrow(2.5, 7.8, 4, 7)

    # Latent space
    draw_box(4, 4.8, 2, 0.8, 'Latent z ∈ ℝ¹⁶', 'orange')
    draw_arrow(5, 6.5, 5, 5.6)

    # Decoder
    draw_box(7, 6.5, 2, 1, 'Decoder\n16→64→128→512', 'lightgreen')
    draw_arrow(5, 5.2, 7, 7, style='<-')

    # Reconstruction
    draw_box(7, 4.8, 2, 0.8, 'Reconstruct\nh′ ∈ ℝ²⁵', 'lightyellow')
    draw_arrow(8, 6.5, 8, 5.6)

    # Reconstruction loss
    draw_arrow(8, 4.8, 1.5, 7.2, style='<->', color='red', linestyle='dashed')
    ax.text(4.5, 5.8, 'MSE Loss', color='red', fontsize=9, fontweight='bold')

    # LSTM path
    draw_box(4, 3.5, 2, 0.8, 'LSTM\nHidden=128', 'lightcoral')
    draw_arrow(5, 4.8, 5, 4.3)

    draw_box(4, 2.2, 2, 0.8, 'FC Layer\n128→1', 'lightcoral')
    draw_arrow(5, 3.5, 5, 3.0)

    # Output
    draw_box(4, 0.8, 2, 0.8, 'Power (W)', 'lightblue')
    draw_arrow(5, 2.2, 5, 1.6)

    # Discriminator (dashed box)
    draw_box(7, 3.2, 2, 1.2, 'Discriminator\n(Training)\nMachine ID', 'pink', 'round,pad=0.1')
    ax.text(8, 3.8, '⚠ Training Only', ha='center', fontsize=8, style='italic')
    draw_arrow(5.5, 5.2, 7, 4, style='->', color='purple', linestyle='dashed')
    ax.text(6.2, 4.5, 'Gradient\nReversal', fontsize=8, color='purple', style='italic')

    # Legend
    legend_y = 0.3
    ax.text(0.5, legend_y + 0.5, 'Legend:', fontsize=10, fontweight='bold')
    ax.plot([0.5, 1], [legend_y, legend_y], 'k-', linewidth=2)
    ax.text(1.2, legend_y, 'Forward', fontsize=8)
    ax.plot([2, 2.5], [legend_y, legend_y], 'r--', linewidth=2)
    ax.text(2.7, legend_y, 'Loss', fontsize=8, color='red')
    ax.plot([3.5, 4], [legend_y, legend_y], color='purple', linestyle='--', linewidth=2)
    ax.text(4.2, legend_y, 'Adversarial', fontsize=8, color='purple')

    plt.tight_layout()
    plt.savefig('docs/images/architecture_overview.png', dpi=300, bbox_inches='tight')
    print("✓ Created docs/images/architecture_overview.png")
    plt.close()


def create_delay_embedding_visualization():
    """Create delay embedding visualization"""
    print("Creating delay embedding visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Generate sample CPU usage time series
    t = np.linspace(0, 100, 1000)
    usage = 50 + 20 * np.sin(2 * np.pi * t / 30) + 10 * np.random.randn(1000)
    usage = np.clip(usage, 0, 100)

    # Plot original time series
    ax = axes[0]
    ax.plot(t[:200], usage[:200], linewidth=2, color='steelblue')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('CPU Usage (%)', fontsize=12)
    ax.set_title('Original Time Series: y(t)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Highlight embedding window
    tau = 1
    d = 25
    window_start = 50
    window_indices = [window_start - i * tau for i in range(d)]

    for idx in window_indices:
        ax.axvline(t[idx], color='red', linestyle='--', alpha=0.3, linewidth=1)

    ax.scatter(t[window_indices], usage[window_indices],
              color='red', s=100, zorder=5, label='Embedding points', edgecolors='black')
    ax.legend(fontsize=10)

    # Annotation
    ax.annotate(f'Delay Embedding:\nτ={tau}ms, d={d}',
               xy=(t[window_start], usage[window_start]),
               xytext=(t[window_start] + 20, 80),
               fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', lw=2))

    # Visualize embedding vector
    ax = axes[1]
    embedding_vector = usage[window_indices]

    x_pos = np.arange(d)
    colors = plt.cm.RdYlGn(embedding_vector / 100)
    bars = ax.bar(x_pos, embedding_vector, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('CPU Usage (%)', fontsize=12)
    ax.set_title(f'Delay-Embedded Vector: h(t) ∈ ℝ²⁵', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos[::5])
    ax.set_xticklabels([f'y(t-{i}τ)' for i in range(0, d, 5)])
    ax.grid(True, alpha=0.3, axis='y')

    # Add text showing vector structure
    vector_text = f"h(t={t[window_start]:.0f}) = [{embedding_vector[0]:.1f}, {embedding_vector[1]:.1f}, ..., {embedding_vector[-1]:.1f}]"
    ax.text(0.5, 0.95, vector_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig('docs/images/delay_embedding.png', dpi=300, bbox_inches='tight')
    print("✓ Created docs/images/delay_embedding.png")
    plt.close()


def create_training_vs_inference():
    """Create training vs inference comparison diagram"""
    print("Creating training vs inference diagram...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for idx, (ax, title, mode) in enumerate(zip(axes,
                                                 ['Training (Bare-Metal)', 'Inference (VM/Bare-Metal)'],
                                                 ['train', 'infer'])):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.text(3, 9.5, title, ha='center', fontsize=14, fontweight='bold')

        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

        def draw_box(x, y, w, h, text, color='lightblue'):
            box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=color, linewidth=2)
            ax.add_patch(box)
            ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                   fontsize=9, fontweight='bold')

        def draw_arrow(x1, y1, x2, y2, label=''):
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=2, color='black')
            ax.add_patch(arrow)
            if label:
                ax.text((x1+x2)/2 + 0.3, (y1+y2)/2, label, fontsize=8, style='italic')

        if mode == 'train':
            # Training flow
            draw_box(1.5, 8, 3, 0.6, '3-5 Machines\nMSR @ 1ms', 'lightyellow')
            draw_box(1.5, 7, 3, 0.6, 'CSV Dataset\nCPU + RAPL', 'lightyellow')
            draw_box(1.5, 6, 3, 0.6, 'Preprocessing\nDelay + Norm', 'lightcyan')
            draw_box(1.5, 5, 3, 0.6, 'Split:\nTrain/Val/Test', 'lightcyan')
            draw_box(1.5, 3.5, 3, 1, 'Train Model\n100 epochs\nEarly Stop', 'lightgreen')
            draw_box(1.5, 2, 3, 0.8, 'Save\nbest_model.pth', 'orange')

            # Arrows
            for y in [8, 7, 6, 5, 4.5]:
                draw_arrow(3, y, 3, y-0.6)
            draw_arrow(3, 3.5, 3, 2.8)

        else:
            # Inference flow
            draw_box(1.5, 8, 3, 0.6, 'Load Model\nbest_model.pth', 'orange')
            draw_box(1.5, 7, 3, 0.6, 'CPU Usage\nStream', 'lightblue')
            draw_box(1.5, 6, 3, 0.6, 'Buffer\n84 samples', 'lightcyan')
            draw_box(1.5, 5, 3, 0.6, 'Delay Embed', 'lightcyan')
            draw_box(1.5, 4, 3, 0.6, 'Model Predict\nEncoder+LSTM', 'lightgreen')
            draw_box(1.5, 2.5, 3, 1, 'VM Scaling?\n(if VM mode)', 'lightyellow')
            draw_box(1.5, 1, 3, 0.8, 'Power (W)', 'lightblue')

            # Arrows
            ax.annotate('', xy=(3, 4), xytext=(3, 8),
                       arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='dashed', color='gray'))
            ax.text(3.3, 6, 'load once', fontsize=8, style='italic', color='gray')

            for y in [7, 6, 5, 4, 3.5]:
                draw_arrow(3, y, 3, y-0.6)
            draw_arrow(3, 2.5, 3, 1.8)

    plt.tight_layout()
    plt.savefig('docs/images/training_vs_inference.png', dpi=300, bbox_inches='tight')
    print("✓ Created docs/images/training_vs_inference.png")
    plt.close()


def main():
    """Generate all diagrams"""
    Path('docs/images').mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating DECODE-RAPL Architecture Diagrams")
    print("=" * 70)
    print()

    # 1. Auto-generated computation graph (requires torchviz)
    success = create_model_architecture_graph()
    if not success:
        print("⚠ Skipping computation graph (torchviz not available)")
    print()

    # 2. High-level architecture overview
    create_architecture_overview()
    print()

    # 3. Delay embedding visualization
    create_delay_embedding_visualization()
    print()

    # 4. Training vs inference
    create_training_vs_inference()
    print()

    print("=" * 70)
    print("All diagrams generated successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - docs/images/model_computation_graph.png (auto-generated from model)")
    print("  - docs/images/architecture_overview.png")
    print("  - docs/images/delay_embedding.png")
    print("  - docs/images/training_vs_inference.png")
    print("\nTo regenerate after model changes: python scripts/generate_diagrams.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate v4.1 Architecture Diagram

Creates a visual representation of the DECODE-RAPL v4.1 model architecture
showing the flow from input through CNN Encoder to Power Head.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects

def create_v4_diagram():
    """Create v4.1 architecture diagram"""

    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # Colors
    color_input = '#E8F4F8'
    color_cnn = '#B8E6F0'
    color_mlp = '#88D4E8'
    color_latent = '#FFF59D'
    color_power = '#FFCCBC'
    color_output = '#C8E6C9'

    # Box parameters
    box_width = 4
    box_height = 0.6
    x_center = 5

    # Starting y position
    y = 19

    # Title
    title = ax.text(x_center, y, 'DECODE-RAPL v4.1 Architecture',
                    ha='center', va='top', fontsize=18, fontweight='bold')
    title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

    y -= 1
    subtitle = ax.text(x_center, y, '1D-CNN Encoder + Power Head (No Decoder)',
                       ha='center', va='top', fontsize=12, style='italic', color='#555')

    y -= 1.2

    def add_box(y, text, shape_text, color, height=box_height):
        """Add a box with text and shape information"""
        box = FancyBboxPatch((x_center - box_width/2, y - height),
                             box_width, height,
                             boxstyle="round,pad=0.1",
                             edgecolor='#333', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x_center, y - height/2, text,
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x_center, y - height - 0.15, shape_text,
                ha='center', va='top', fontsize=8, style='italic', color='#666')
        return y - height

    def add_arrow(y_start, y_end):
        """Add a downward arrow"""
        arrow = FancyArrowPatch((x_center, y_start), (x_center, y_end),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='#333')
        ax.add_patch(arrow)

    def add_section_label(y, text, color='#1976D2'):
        """Add a section label"""
        ax.text(x_center - box_width/2 - 0.5, y, text,
                ha='right', va='center', fontsize=11, fontweight='bold',
                color=color, rotation=90)

    # INPUT
    y_top = add_box(y, 'Input', '(batch, 100)', color_input)
    y -= 0.8
    add_arrow(y_top, y)

    # ENCODER SECTION LABEL
    add_section_label(y - 3.5, 'CNN ENCODER', '#1976D2')

    # Reshape
    y_top = add_box(y, 'Reshape', '(batch, 4, 25)', color_cnn)
    ax.text(x_center + box_width/2 + 0.3, y - box_height/2, '4 features × 25 delays',
            ha='left', va='center', fontsize=8, color='#666')
    y -= 0.8
    add_arrow(y_top, y)

    # Conv1d Block 1
    y_top = add_box(y, 'Conv1d (4 → 32, kernel=5)', '(batch, 32, 25)', color_cnn, height=0.7)
    y -= 0.05
    y_top = add_box(y, 'BatchNorm1d(32)', '', color_cnn, height=0.4)
    y -= 0.05
    y_top = add_box(y, 'ReLU', '', color_cnn, height=0.4)
    y -= 0.8
    add_arrow(y_top, y)

    # Conv1d Block 2
    y_top = add_box(y, 'Conv1d (32 → 64, kernel=5)', '(batch, 64, 25)', color_cnn, height=0.7)
    y -= 0.05
    y_top = add_box(y, 'BatchNorm1d(64)', '', color_cnn, height=0.4)
    y -= 0.05
    y_top = add_box(y, 'ReLU', '', color_cnn, height=0.4)
    y -= 0.8
    add_arrow(y_top, y)

    # MaxPool
    y_top = add_box(y, 'MaxPool1d (kernel=2)', '(batch, 64, 12)', color_cnn)
    ax.text(x_center + box_width/2 + 0.3, y - box_height/2, '25 → 12 timesteps',
            ha='left', va='center', fontsize=8, color='#666')
    y -= 0.8
    add_arrow(y_top, y)

    # Flatten
    y_top = add_box(y, 'Flatten', '(batch, 768)', color_mlp)
    ax.text(x_center + box_width/2 + 0.3, y - box_height/2, '64 × 12 = 768',
            ha='left', va='center', fontsize=8, color='#666')
    y -= 0.8
    add_arrow(y_top, y)

    # MLP Block 1
    y_top = add_box(y, 'Linear (768 → 128)', '(batch, 128)', color_mlp, height=0.7)
    y -= 0.05
    y_top = add_box(y, 'ReLU', '', color_mlp, height=0.4)
    y -= 0.05
    y_top = add_box(y, 'Dropout (0.3)', '', color_mlp, height=0.4)
    y -= 0.8
    add_arrow(y_top, y)

    # Latent Space
    y_top = add_box(y, 'Linear (128 → 64)', '(batch, 64)', color_latent, height=0.7)
    ax.text(x_center + box_width/2 + 0.3, y - 0.35, '⭐ Latent Space',
            ha='left', va='center', fontsize=9, fontweight='bold', color='#F57F17')
    y -= 0.8
    add_arrow(y_top, y)

    # POWER HEAD SECTION LABEL
    add_section_label(y - 1.5, 'POWER HEAD', '#D32F2F')

    # Power Head Block 1
    y_top = add_box(y, 'Linear (64 → 128)', '(batch, 128)', color_power, height=0.7)
    y -= 0.05
    y_top = add_box(y, 'ReLU', '', color_power, height=0.4)
    y -= 0.05
    y_top = add_box(y, 'Dropout (0.3)', '', color_power, height=0.4)
    y -= 0.8
    add_arrow(y_top, y)

    # Power Head Block 2
    y_top = add_box(y, 'Linear (128 → 64)', '(batch, 64)', color_power, height=0.7)
    y -= 0.05
    y_top = add_box(y, 'ReLU', '', color_power, height=0.4)
    y -= 0.8
    add_arrow(y_top, y)

    # Output
    y_top = add_box(y, 'Linear (64 → 1)', '(batch, 1)', color_output, height=0.7)
    y -= 0.8
    add_arrow(y_top, y)

    # Final Output
    y_top = add_box(y, 'Power Prediction', '(batch, 1)', color_output)
    ax.text(x_center + box_width/2 + 0.3, y - box_height/2, 'Watts',
            ha='left', va='center', fontsize=8, fontweight='bold', color='#2E7D32')

    # Add legend/notes
    notes_y = 1.5
    notes = [
        "Key Features:",
        "• 1D Convolutions capture temporal patterns",
        "• BatchNorm after each Conv layer (v4.1)",
        "• Kernel size = 5 (~80ms receptive field)",
        "• No decoder (power-only prediction)",
        "• Dropout in MLP layers only (not CNN)",
        "",
        "Loss: L1Loss (MAE)",
        "Normalization: MinMaxScaler on input features"
    ]

    notes_text = '\n'.join(notes)
    box = FancyBboxPatch((0.5, notes_y - 0.1), 4, 1.8,
                         boxstyle="round,pad=0.1",
                         edgecolor='#666', facecolor='#F5F5F5',
                         linewidth=1.5, linestyle='--')
    ax.add_patch(box)
    ax.text(2.5, notes_y + 1.6, notes_text,
            ha='center', va='top', fontsize=8, family='monospace',
            color='#333')

    # Parameter counts
    param_y = 1.5
    params = [
        "Parameter Counts:",
        "CNN: ~66K params",
        "MLP: ~58K params",
        "Power Head: ~10K params",
        "Total: ~134K params"
    ]

    params_text = '\n'.join(params)
    box2 = FancyBboxPatch((5.5, param_y - 0.1), 3.5, 1.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='#666', facecolor='#F5F5F5',
                          linewidth=1.5, linestyle='--')
    ax.add_patch(box2)
    ax.text(7.25, param_y + 1.6, params_text,
            ha='center', va='top', fontsize=8, family='monospace',
            color='#333', fontweight='bold')

    plt.tight_layout()

    return fig

if __name__ == '__main__':
    import os
    from pathlib import Path

    print("Generating v4.1 architecture diagram...")
    fig = create_v4_diagram()

    # Save to project root
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    output_path = project_dir / 'decode_rapl_v4.1_architecture.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Diagram saved to: {output_path}")
    print(f"Absolute path: {output_path.absolute()}")
    plt.close()

#!/usr/bin/env python3
"""Generate MS-TCN model architecture diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Color scheme
colors = {
    'input': '#E8F4F8',
    'conv': '#B8E6F0',
    'dilated': '#7EC8E3',
    'attention': '#FFE6CC',
    'pool': '#D4EDDA',
    'fc': '#F0D9FF',
    'output': '#FFD9D9',
    'arrow': '#666666'
}

def draw_box(ax, x, y, width, height, text, color, fontsize=10, bold=False):
    """Draw a fancy box with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, weight=weight,
            wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->',
        color=colors['arrow'],
        linewidth=2,
        mutation_scale=20
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label,
                fontsize=8, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Title
ax.text(5, 19.5, 'MS-TCN Power Prediction Architecture',
        ha='center', va='top', fontsize=16, weight='bold')

# Current Y position
y_pos = 18.5

# Input Layer
draw_box(ax, 3, y_pos, 4, 0.6, 'Input Sequence\n64 timesteps × 19 features',
         colors['input'], fontsize=11, bold=True)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.2

# Multi-Scale Convolution
ax.text(5, y_pos + 0.3, 'Multi-Scale Convolution', ha='center', fontsize=12, weight='bold')
y_pos -= 0.5

# Three parallel branches
branch_width = 2.2
branch_x = [1.5, 3.9, 6.3]
branch_labels = ['Conv1D\nKernel=3', 'Conv1D\nKernel=5', 'Conv1D\nKernel=7']

for i, (x, label) in enumerate(zip(branch_x, branch_labels)):
    draw_box(ax, x, y_pos, branch_width, 0.7, label, colors['conv'], fontsize=9)
    # Arrow from input to branch
    draw_arrow(ax, 5, y_pos + 1.2, x + branch_width/2, y_pos + 0.7)
    # Arrow from branch to concat
    draw_arrow(ax, x + branch_width/2, y_pos, x + branch_width/2, y_pos - 0.5)

y_pos -= 0.9

# Concatenation
draw_box(ax, 3.5, y_pos, 3, 0.5, 'Concatenate → 128 channels',
         colors['conv'], fontsize=9)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.1

# Dilated Temporal Blocks
ax.text(5, y_pos + 0.3, 'Dilated Temporal Convolution Blocks', ha='center', fontsize=12, weight='bold')
y_pos -= 0.5

dilations = [1, 2, 4, 8, 16, 32]
for i, dilation in enumerate(dilations):
    receptive_field = 3 ** (i + 1)
    draw_box(ax, 2.5, y_pos, 5, 0.6,
             f'Block {i+1}: Conv1D (dilation={dilation})\n128 channels, residual',
             colors['dilated'], fontsize=8)

    if i < len(dilations) - 1:
        draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
        y_pos -= 0.9
    else:
        draw_arrow(ax, 5, y_pos, 5, y_pos - 0.6)
        y_pos -= 1.0

# Multi-Head Attention
ax.text(5, y_pos + 0.3, 'Multi-Head Attention (8 heads)', ha='center', fontsize=12, weight='bold')
y_pos -= 0.5

# Q, K, V projections
qkv_width = 1.8
qkv_x = [1.8, 4.1, 6.4]
qkv_labels = ['Query\nProjection', 'Key\nProjection', 'Value\nProjection']

for x, label in zip(qkv_x, qkv_labels):
    draw_box(ax, x, y_pos, qkv_width, 0.6, label, colors['attention'], fontsize=8)
    draw_arrow(ax, 5, y_pos + 1.0, x + qkv_width/2, y_pos + 0.6)

y_pos -= 0.9

# Attention computation
draw_box(ax, 2.5, y_pos, 5, 0.6,
         'Attention = softmax((Q × K^T) / √d_k) × V',
         colors['attention'], fontsize=9)
draw_arrow(ax, 5, y_pos + 0.9, 5, y_pos + 0.6)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.0

# Attention output
draw_box(ax, 3, y_pos, 4, 0.5,
         'Attended Features (128 channels)',
         colors['attention'], fontsize=9)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)
y_pos -= 1.0

# Global Average Pooling
draw_box(ax, 3, y_pos, 4, 0.5,
         'Global Average Pooling\n64 timesteps → 1',
         colors['pool'], fontsize=9, bold=True)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5, '64×128')
y_pos -= 1.0

# Fully Connected Layers
draw_box(ax, 3, y_pos, 4, 0.5, 'FC Layer: 128 → 64\nReLU + Dropout(0.3)',
         colors['fc'], fontsize=9)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5, '128')
y_pos -= 0.9

draw_box(ax, 3, y_pos, 4, 0.5, 'FC Layer: 64 → 32\nReLU + Dropout(0.3)',
         colors['fc'], fontsize=9)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5, '64')
y_pos -= 1.0

# Output Heads
ax.text(5, y_pos + 0.2, 'Output Heads', ha='center', fontsize=12, weight='bold')
y_pos -= 0.5

output_x = [2.2, 5.8]
output_labels = ['Package Power\nHead (32→1)', 'Core Power\nHead (32→1)']

for x, label in zip(output_x, output_labels):
    draw_box(ax, x, y_pos, 2, 0.6, label, colors['output'], fontsize=9, bold=True)
    draw_arrow(ax, 5, y_pos + 1.0, x + 1, y_pos + 0.6)

y_pos -= 1.0

# Final outputs
draw_box(ax, 2.2, y_pos, 2, 0.5, 'Package Power\n(Watts)', colors['output'], fontsize=9)
draw_box(ax, 5.8, y_pos, 2, 0.5, 'Core Power\n(Watts)', colors['output'], fontsize=9)
draw_arrow(ax, 3.2, y_pos + 1.1, 3.2, y_pos + 0.5)
draw_arrow(ax, 6.8, y_pos + 1.1, 6.8, y_pos + 0.5)

# Add legend/info box
info_y = 1.2
info_text = (
    'Model Parameters: ~740K\n'
    'Loss Function: Huber Loss\n'
    'Optimizer: AdamW (lr=1e-4)\n'
    'Normalization: Z-score'
)
ax.text(0.5, info_y, info_text,
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='black', linewidth=1.5),
        verticalalignment='top')

# Add receptive field info
rf_text = (
    'Receptive Field Growth:\n'
    'Block 1: 3 timesteps\n'
    'Block 2: 9 timesteps\n'
    'Block 3: 27 timesteps\n'
    'Block 6: 243 timesteps'
)
ax.text(9.5, info_y, rf_text,
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='black', linewidth=1.5),
        verticalalignment='top',
        horizontalalignment='right')

# Add color legend
legend_y = 0.3
legend_elements = [
    mpatches.Patch(color=colors['input'], label='Input'),
    mpatches.Patch(color=colors['conv'], label='Convolution'),
    mpatches.Patch(color=colors['dilated'], label='Dilated Conv'),
    mpatches.Patch(color=colors['attention'], label='Attention'),
    mpatches.Patch(color=colors['pool'], label='Pooling'),
    mpatches.Patch(color=colors['fc'], label='Fully Connected'),
    mpatches.Patch(color=colors['output'], label='Output'),
]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
          ncol=7, fontsize=8, frameon=True, fancybox=True)

plt.tight_layout()
plt.savefig('model_architecture.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved to: model_architecture.png")

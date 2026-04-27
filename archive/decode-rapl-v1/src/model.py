"""
DECODE-RAPL Model Architecture
Autoencoder + LSTM + Adversarial Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Encoder(nn.Module):
    """
    Encoder network: Maps delay-embedded space to compact latent space

    Architecture: Input → Dense layers → Latent space
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        latent_dim: int,
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim: Dimension of delay-embedded input (d * n_vars)
            hidden_dims: List of hidden layer dimensions (e.g., [256, 64, 32])
            latent_dim: Dimension of latent space
            dropout: Dropout probability
        """
        super(Encoder, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            Latent representation of shape (batch, seq_len, latent_dim) or (batch, latent_dim)
        """
        return self.network(x)


class Decoder(nn.Module):
    """
    Decoder network: Reconstructs delay-embedded space from latent space

    Architecture: Latent space → Dense layers → Reconstructed input
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.2
    ):
        """
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions (reverse of encoder)
            output_dim: Dimension of output (same as encoder input)
            dropout: Dropout probability
        """
        super(Decoder, self).__init__()

        layers = []
        prev_dim = latent_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final layer to output
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, z):
        """
        Args:
            z: Latent tensor of shape (batch, seq_len, latent_dim) or (batch, latent_dim)

        Returns:
            Reconstructed tensor of shape (batch, seq_len, output_dim) or (batch, output_dim)
        """
        return self.network(z)


class PowerLSTM(nn.Module):
    """
    LSTM for temporal power prediction from latent sequences
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        """
        Args:
            latent_dim: Dimension of latent input
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout between LSTM layers
        """
        super(PowerLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output layer: LSTM hidden → Power (scalar)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, z_sequence):
        """
        Args:
            z_sequence: Latent sequence of shape (batch, seq_len, latent_dim)

        Returns:
            Power prediction of shape (batch, 1)
        """
        # LSTM processes sequence
        lstm_out, (h_n, c_n) = self.lstm(z_sequence)

        # Use final hidden state for prediction
        # h_n shape: (num_layers, batch, hidden_size)
        final_hidden = h_n[-1]  # Take last layer: (batch, hidden_size)

        # Predict power
        power = self.fc(final_hidden)  # (batch, 1)

        return power


class MachineDiscriminator(nn.Module):
    """
    Adversarial discriminator to predict machine_id from latent space

    Used to encourage encoder to learn machine-invariant representations
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_machines: int
    ):
        """
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden layer dimension
            num_machines: Number of machines to classify
        """
        super(MachineDiscriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_machines)
        )

    def forward(self, z):
        """
        Args:
            z: Latent representation of shape (batch, latent_dim) or (batch, seq_len, latent_dim)

        Returns:
            Machine logits of shape (batch, num_machines)
        """
        # If z is sequential, average over time
        if z.dim() == 3:
            z = z.mean(dim=1)  # Average pooling over sequence

        return self.network(z)


class DECODERAPLModel(nn.Module):
    """
    Complete DECODE-RAPL model combining all components
    """

    def __init__(self, config: dict, num_machines: int):
        """
        Args:
            config: Configuration dictionary
            num_machines: Number of unique machines in dataset
        """
        super(DECODERAPLModel, self).__init__()

        # Extract config parameters
        self.latent_dim = config['model']['latent_dim']
        encoder_layers = config['model']['encoder_layers']
        dropout = config['model']['dropout']

        # Calculate input dimension (delay embedding dimension)
        tau = config['embedding']['tau']
        d = config['embedding']['d']
        # Get number of features from preprocessing config
        n_features = len(config['preprocessing'].get('feature_columns', ['cpu_usage']))
        input_dim = d * n_features  # d delays × n_features variables

        # Build components
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=encoder_layers,
            latent_dim=self.latent_dim,
            dropout=dropout
        )

        # Decoder mirrors encoder
        decoder_layers = list(reversed(encoder_layers))
        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            hidden_dims=decoder_layers,
            output_dim=input_dim,
            dropout=dropout
        )

        # LSTM for power prediction
        self.lstm = PowerLSTM(
            latent_dim=self.latent_dim,
            hidden_size=config['model']['lstm_hidden_size'],
            num_layers=config['model']['lstm_num_layers'],
            dropout=dropout
        )

        # Adversarial discriminator
        self.discriminator = MachineDiscriminator(
            latent_dim=self.latent_dim,
            hidden_dim=config['model']['adversarial_hidden'],
            num_machines=num_machines
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Delay-embedded sequence of shape (batch, seq_len, input_dim)

        Returns:
            Dictionary with:
                - z: Latent representation (batch, seq_len, latent_dim)
                - x_recon: Reconstructed input (batch, seq_len, input_dim)
                - power_pred: Power prediction (batch, 1)
                - machine_logits: Machine classification logits (batch, num_machines)
        """
        # Encode to latent space
        z = self.encoder(x)  # (batch, seq_len, latent_dim)

        # Decode to reconstruct input
        x_recon = self.decoder(z)  # (batch, seq_len, input_dim)

        # Predict power from latent sequence
        power_pred = self.lstm(z)  # (batch, 1)

        # Discriminator predicts machine from latent
        machine_logits = self.discriminator(z)  # (batch, num_machines)

        return {
            'z': z,
            'x_recon': x_recon,
            'power_pred': power_pred,
            'machine_logits': machine_logits
        }

    def encode(self, x):
        """Encode input to latent space (for inference)"""
        return self.encoder(x)

    def predict_power(self, z_sequence):
        """Predict power from latent sequence (for inference)"""
        return self.lstm(z_sequence)


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training

    Forward: Identity
    Backward: Negate gradients (multiply by -lambda)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wrapper for Gradient Reversal"""

    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def create_model(config: dict, num_machines: int) -> DECODERAPLModel:
    """
    Factory function to create DECODE-RAPL model

    Args:
        config: Configuration dictionary
        num_machines: Number of unique machines in dataset

    Returns:
        Initialized model
    """
    model = DECODERAPLModel(config, num_machines)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    return model


if __name__ == "__main__":
    # Test model architecture
    import sys
    sys.path.append('.')
    from src.utils import load_config

    print("Testing model architecture...")

    # Load config
    config = load_config()

    # Create model
    num_machines = 3
    model = create_model(config, num_machines)

    print(f"\nModel architecture:")
    print(model)

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 8
    seq_len = config['preprocessing']['window_size']
    input_dim = config['embedding']['d']

    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"Input shape: {x.shape}")

    output = model(x)

    print(f"\nOutput shapes:")
    print(f"  Latent (z): {output['z'].shape}")
    print(f"  Reconstructed (x_recon): {output['x_recon'].shape}")
    print(f"  Power prediction: {output['power_pred'].shape}")
    print(f"  Machine logits: {output['machine_logits'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nModel architecture test completed successfully!")

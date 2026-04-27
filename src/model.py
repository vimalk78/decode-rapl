"""
DECODE-RAPL v2 Model Architecture

Simplified architecture based on Gemini recommendations:
- No LSTM (delay embedding provides temporal encoding)
- Direct MLP power head (instead of LSTM-based)
- Wider latent space (64 dimensions)
- Processes single delay-embedded vectors (batch, 100)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class Encoder(nn.Module):
    """
    Encoder: Maps 100-dim delay-embedded vectors to 64-dim latent space

    Architecture: 100 → 512 → 128 → 64
    """

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dims: list = [512, 128],
        latent_dim: int = 64,
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim: Dimension of delay-embedded input (default: 100)
            hidden_dims: List of hidden layer dimensions
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Latent representation of shape (batch, 64)
        """
        return self.network(x)


class Decoder(nn.Module):
    """
    Decoder: Reconstructs 100-dim input from 64-dim latent space

    Architecture: 64 → 128 → 512 → 100
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: list = [128, 512],
        output_dim: int = 100,
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape (batch, 64)

        Returns:
            Reconstructed input of shape (batch, 100)
        """
        return self.network(z)


class PowerHead(nn.Module):
    """
    Power Head: MLP that predicts power from latent space

    Architecture: 64 → 128 → 64 → 1

    This is NEW in v2 - replaces the LSTM-based power prediction from v1
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: list = [128, 64],
        dropout: float = 0.2
    ):
        """
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(PowerHead, self).__init__()

        layers = []
        prev_dim = latent_dim

        # Build hidden layers with ReLU
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final layer to power (no activation - regression task)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape (batch, 64)

        Returns:
            Power prediction of shape (batch, 1)
        """
        return self.network(z)


class DECODERAPL_v2(nn.Module):
    """
    DECODE-RAPL v2 Complete Model

    Components:
    1. Encoder: 100 → 64 (compress delay-embedded input to latent space)
    2. Decoder: 64 → 100 (reconstruct input for autoencoder loss)
    3. Power Head: 64 → 1 (predict power from latent space)

    Training uses combined loss: power_mse + reconstruction_mse
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary with keys:
                - input_dim: Input dimension (default: 100)
                - latent_dim: Latent space dimension (default: 64)
                - encoder_layers: List of encoder hidden dims (default: [512, 128])
                - decoder_layers: List of decoder hidden dims (default: [128, 512])
                - power_head_layers: List of power head hidden dims (default: [128, 64])
                - dropout: Dropout probability (default: 0.2)
        """
        super(DECODERAPL_v2, self).__init__()

        # Extract config
        input_dim = config.get('input_dim', 100)
        latent_dim = config.get('latent_dim', 64)
        encoder_layers = config.get('encoder_layers', [512, 128])
        decoder_layers = config.get('decoder_layers', [128, 512])
        power_head_layers = config.get('power_head_layers', [128, 64])
        dropout = config.get('dropout', 0.2)

        # Build components
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=encoder_layers,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_layers,
            output_dim=input_dim,
            dropout=dropout
        )

        self.power_head = PowerHead(
            latent_dim=latent_dim,
            hidden_dims=power_head_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Dictionary with keys:
                - z: Latent representation (batch, 64)
                - x_recon: Reconstructed input (batch, 100)
                - power_pred: Power prediction (batch, 1)
        """
        # Encode to latent space
        z = self.encoder(x)

        # Reconstruct input
        x_recon = self.decoder(z)

        # Predict power
        power_pred = self.power_head(z)

        return {
            'z': z,
            'x_recon': x_recon,
            'power_pred': power_pred
        }

    def predict_power(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict power (inference mode - skip reconstruction)

        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Power prediction of shape (batch, 1)
        """
        z = self.encoder(x)
        return self.power_head(z)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (for visualization)

        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Latent representation of shape (batch, 64)
        """
        return self.encoder(x)


class DECODERAPL_v3(nn.Module):
    """
    DECODE-RAPL v3 Simplified Model (No Decoder)

    Components:
    1. Encoder: 100 → 64 (compress delay-embedded input to latent space)
    2. Power Head: 64 → 1 (predict power from latent space)

    Training uses single loss: power_mse only

    This removes the reconstruction task entirely, focusing only on power prediction.
    The v2 experiments showed reconstruction loss was causing overfitting without
    helping power prediction accuracy.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary with keys:
                - input_dim: Input dimension (default: 100)
                - latent_dim: Latent space dimension (default: 64)
                - encoder_layers: List of encoder hidden dims (default: [512, 128])
                - power_head_layers: List of power head hidden dims (default: [128, 64])
                - dropout: Dropout probability (default: 0.2)
        """
        super(DECODERAPL_v3, self).__init__()

        # Extract config
        input_dim = config.get('input_dim', 100)
        latent_dim = config.get('latent_dim', 64)
        encoder_layers = config.get('encoder_layers', [512, 128])
        power_head_layers = config.get('power_head_layers', [128, 64])
        dropout = config.get('dropout', 0.2)

        # Build components (no decoder!)
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=encoder_layers,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.power_head = PowerHead(
            latent_dim=latent_dim,
            hidden_dims=power_head_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Dictionary with keys:
                - z: Latent representation (batch, 64)
                - power_pred: Power prediction (batch, 1)
        """
        # Encode to latent space
        z = self.encoder(x)

        # Predict power
        power_pred = self.power_head(z)

        return {
            'z': z,
            'power_pred': power_pred
        }

    def predict_power(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict power (same as forward for v3)

        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Power prediction of shape (batch, 1)
        """
        z = self.encoder(x)
        return self.power_head(z)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (for visualization)

        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Latent representation of shape (batch, 64)
        """
        return self.encoder(x)


class CNNEncoder(nn.Module):
    """
    1D-CNN Encoder: Processes temporal structure in delay-embedded features

    Input: (batch, 100) reshaped to (batch, 4, 25)
      - 4 features: user%, system%, iowait%, log(ctx_switches)
      - 25 time delays

    Architecture (v4.1):
      Conv1d: 4 → 32 channels (kernel=5) + BatchNorm + ReLU
      Conv1d: 32 → 64 channels (kernel=5) + BatchNorm + ReLU
      MaxPool1d: reduces 25 → 12 timesteps
      Flatten: 64 × 12 = 768
      MLP: 768 → 128 → 64 (latent space) with dropout
    """

    def __init__(
        self,
        num_features: int = 4,
        sequence_length: int = 25,
        cnn_channels: list = [32, 64],
        kernel_size: int = 5,
        pool_size: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Args:
            num_features: Number of input features (default: 4)
            sequence_length: Length of delay embedding (default: 25)
            cnn_channels: List of CNN output channels (default: [32, 64])
            kernel_size: Convolution kernel size (default: 5)
            pool_size: Max pooling kernel size (default: 2)
            latent_dim: Output latent dimension (default: 64)
            dropout: Dropout probability (applied to MLP only)
            use_batch_norm: Use BatchNorm after Conv layers (default: True)
        """
        super(CNNEncoder, self).__init__()

        self.num_features = num_features
        self.sequence_length = sequence_length

        # CNN layers with BatchNorm (v4.1)
        cnn_layers = []
        in_channels = num_features

        for out_channels in cnn_channels:
            cnn_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # same padding
            ))

            # Add BatchNorm if enabled
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm1d(out_channels))

            cnn_layers.append(nn.ReLU())
            in_channels = out_channels

        # Max pooling
        cnn_layers.append(nn.MaxPool1d(kernel_size=pool_size))

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate flattened dimension after CNN + pooling
        # sequence_length // pool_size * final_channels
        pooled_length = sequence_length // pool_size
        flattened_dim = cnn_channels[-1] * pooled_length  # 64 * 12 = 768

        # MLP layers after CNN
        self.mlp = nn.Sequential(
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Latent representation of shape (batch, 64)
        """
        batch_size = x.size(0)

        # Reshape: (batch, 100) -> (batch, 4, 25)
        x = x.view(batch_size, self.num_features, self.sequence_length)

        # Apply CNN layers
        x = self.cnn(x)  # (batch, 64, 12)

        # Flatten
        x = x.view(batch_size, -1)  # (batch, 768)

        # Apply MLP to get latent representation
        z = self.mlp(x)  # (batch, 64)

        return z


class DECODERAPL_v4(nn.Module):
    """
    DECODE-RAPL v4 Model with 1D-CNN Encoder

    Components:
    1. CNN Encoder: 100 → 64 using 1D convolutions over temporal dimension
    2. Power Head: 64 → 1 (predict power from latent space)

    Key difference from v3: Uses 1D-CNN to extract temporal patterns
    instead of treating delay-embedded input as flat vector.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary with keys:
                - input_dim: Input dimension (default: 100)
                - num_features: Number of features (default: 4)
                - sequence_length: Delay embedding dimension (default: 25)
                - cnn_channels: List of CNN channels (default: [32, 64])
                - cnn_kernel_size: CNN kernel size (default: 5)
                - pool_size: Pooling kernel size (default: 2)
                - use_batch_norm: Use BatchNorm in CNN (default: True)
                - latent_dim: Latent space dimension (default: 64)
                - power_head_layers: List of power head hidden dims (default: [128, 64])
                - dropout: Dropout probability (default: 0.3)
        """
        super(DECODERAPL_v4, self).__init__()

        # Extract config
        num_features = config.get('num_features', 4)
        sequence_length = config.get('sequence_length', 25)
        cnn_channels = config.get('cnn_channels', [32, 64])
        kernel_size = config.get('cnn_kernel_size', 5)
        pool_size = config.get('pool_size', 2)
        use_batch_norm = config.get('use_batch_norm', True)
        latent_dim = config.get('latent_dim', 64)
        power_head_layers = config.get('power_head_layers', [128, 64])
        dropout = config.get('dropout', 0.3)

        # Build CNN encoder
        self.encoder = CNNEncoder(
            num_features=num_features,
            sequence_length=sequence_length,
            cnn_channels=cnn_channels,
            kernel_size=kernel_size,
            pool_size=pool_size,
            latent_dim=latent_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        # Build power head (same as v3)
        self.power_head = PowerHead(
            latent_dim=latent_dim,
            hidden_dims=power_head_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Dictionary with keys:
                - z: Latent representation (batch, 64)
                - power_pred: Power prediction (batch, 1)
        """
        # Encode to latent space using CNN
        z = self.encoder(x)

        # Predict power
        power_pred = self.power_head(z)

        return {
            'z': z,
            'power_pred': power_pred
        }

    def predict_power(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict power (same as forward for v4)

        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Power prediction of shape (batch, 1)
        """
        z = self.encoder(x)
        return self.power_head(z)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (for visualization)

        Args:
            x: Input tensor of shape (batch, 100)

        Returns:
            Latent representation of shape (batch, 64)
        """
        return self.encoder(x)


class CombinedLoss(nn.Module):
    """
    Combined loss for DECODE-RAPL v2:
    - Power MSE: Main objective (prediction accuracy)
    - Reconstruction MSE: Autoencoder quality
    """

    def __init__(self, power_weight: float = 1.0, reconstruction_weight: float = 0.1):
        """
        Args:
            power_weight: Weight for power prediction loss
            reconstruction_weight: Weight for reconstruction loss
        """
        super(CombinedLoss, self).__init__()

        self.power_weight = power_weight
        self.reconstruction_weight = reconstruction_weight
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss

        Args:
            outputs: Model outputs dict with keys:
                - x_recon: Reconstructed input (batch, 100)
                - power_pred: Power prediction (batch, 1)
            x: Original input (batch, 100)
            y: Ground truth power (batch, 1) or (batch,)

        Returns:
            (total_loss, loss_dict)
        """
        # Ensure y has correct shape
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Power prediction loss (main objective)
        power_loss = self.mse_loss(outputs['power_pred'], y)

        # Reconstruction loss (autoencoder quality)
        recon_loss = self.mse_loss(outputs['x_recon'], x)

        # Combined loss
        total_loss = (
            self.power_weight * power_loss +
            self.reconstruction_weight * recon_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'power': power_loss.item(),
            'reconstruction': recon_loss.item()
        }

        return total_loss, loss_dict


def create_model(config: dict, version: str = 'v2'):
    """
    Factory function to create DECODE-RAPL model

    Args:
        config: Configuration dictionary
        version: Model version ('v2', 'v3', or 'v4'), default 'v2'

    Returns:
        Model instance (DECODERAPL_v2, DECODERAPL_v3, or DECODERAPL_v4)
    """
    # Check config for version override
    if 'version' in config:
        version = config['version']

    if version == 'v4':
        model = DECODERAPL_v4(config)
    elif version == 'v3':
        model = DECODERAPL_v3(config)
    else:  # default to v2 for backwards compatibility
        model = DECODERAPL_v2(config)

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    config = {
        'input_dim': 100,
        'latent_dim': 64,
        'encoder_layers': [512, 128],
        'decoder_layers': [128, 512],
        'power_head_layers': [128, 64],
        'dropout': 0.2
    }

    model = create_model(config)
    print("DECODE-RAPL v2 Model")
    print(f"Total parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 100)

    outputs = model(x)
    print(f"\nForward pass test (batch_size={batch_size}):")
    print(f"  Input: {x.shape}")
    print(f"  Latent: {outputs['z'].shape}")
    print(f"  Reconstructed: {outputs['x_recon'].shape}")
    print(f"  Power prediction: {outputs['power_pred'].shape}")

    # Test loss
    y = torch.randn(batch_size, 1)
    criterion = CombinedLoss(power_weight=1.0, reconstruction_weight=0.1)
    loss, loss_dict = criterion(outputs, x, y)
    print(f"\nLoss test:")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Power loss: {loss_dict['power']:.4f}")
    print(f"  Reconstruction loss: {loss_dict['reconstruction']:.4f}")

"""
Neural network models for chess move prediction.
Contains model architectures and a registry for easy instantiation.

Available Models:
- OLP (One Layer Perceptron): Simple linear baseline model

"""

import torch
import torch.nn as nn


class OLP(nn.Module):
    """
    One Layer Perceptron for chess move classification.
    This is a simple linear model that maps board features directly to move 
    predictions. 
    
    Architecture:
        Input (855) → Linear → Output (20,480)
    Args:
        input_dim: Dimension of input features (typically 855 for FEN encoding)
                   - 832: Board position (64 squares × 13 channels)
                   - 1: Turn indicator
                   - 4: Castling rights
                   - 16: En passant possibilities
                   - 2: Move counters
        num_classes: Number of possible moves (typically ~20,480)
                     All possible moves in chess: 8×8×8×8×5 (with promotions)
    
    Example:
        >>> model = OLP(input_dim=855, num_classes=20480)
        >>> x = torch.randn(16, 855)  # Batch of 16 positions
        >>> output = model(x)  # Shape: [16, 20480]
        >>> predictions = torch.argmax(output, dim=1)  # Get predicted moves
    """

    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize the One Layer Perceptron.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes (possible moves)
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
        # Store dimensions for reference
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.linear(x)
    
    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.
        
        Returns:
            Total number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        num_params = self.get_num_parameters()
        return (f"OLP(input_dim={self.input_dim}, num_classes={self.num_classes}, "
                f"parameters={num_params:,})")




class TLP(nn.Module):
    """
    Two Layer Perceptron for chess move classification.
    
    This model adds a hidden layer between input and output, allowing it to learn
    non-linear patterns in chess positions. The hidden layer uses ReLU activation
    to introduce non-linearity.
    
    Architecture:
        Input (855) → Linear → ReLU → Linear → Output (20,480)
    
    The hidden layer dimension can be adjusted to control model capacity.
    A typical choice is between 512 and 2048 neurons.
    
    Args:
        input_dim: Dimension of input features (typically 855 for FEN encoding)
                   - 832: Board position (64 squares × 13 channels)
                   - 1: Turn indicator
                   - 4: Castling rights
                   - 16: En passant possibilities
                   - 2: Move counters
        num_classes: Number of possible moves (typically ~20,480)
                     All possible moves in chess: 8×8×8×8×5 (with promotions)
        hidden_dim: Number of neurons in the hidden layer (default: 1024)
    
    Example:
        >>> model = TLP(input_dim=855, num_classes=20480, hidden_dim=1024)
        >>> x = torch.randn(16, 855)  # Batch of 16 positions
        >>> output = model(x)  # Shape: [16, 20480]
        >>> predictions = torch.argmax(output, dim=1)  # Get predicted moves
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        """
        Initialize the Two Layer Perceptron.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes (possible moves)
            hidden_dim: Number of neurons in hidden layer
        """
        super().__init__()
        
        # First layer: input to hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Second layer: hidden to output
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Store dimensions for reference
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer with activation
        x = self.fc1(x)
        x = self.relu(x)       
        
        # Second layer (output)
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.
        
        Returns:
            Total number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        num_params = self.get_num_parameters()
        return (f"TLP(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
                f"num_classes={self.num_classes}, parameters={num_params:,})")


# Model registry for easy instantiation from configuration
MODELS_DICT = {
    "OLP": OLP,
    "TLP": TLP,
}

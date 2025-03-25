"""
Implement a neural field using a simple MLP.

The MLP maps from `in_features`-dimensional points (e.g., 2D xy positions) 
to `out_features`-dimensional points (e.g., 1D color values). To make your
implementation more general, also let the user specify any activation function.

Some conventions we use that you will need to follow IF you want to use the unit tests:
1. The last layer is always linear.
2. All layers have bias IFF self.bias is True.
3. There is one linear+activation function layer from the input (`in_features`) to `hidden_features` and then `hidden_layers-1` 
    linear+activation function layers from `hidden_features` to `hidden_features`, and then a linear layer from `hidden_features`
    to `out_features`.
"""

import torch
import torch.nn as nn
from typing import Tuple
import jaxtyping

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,  # Number of input features
        out_features: int,  # Number of output features
        hidden_features: int,  # Number of features in the hidden layers
        hidden_layers: int,  # Number of hidden layers in the MLP
        bias: bool = True,  # Whether to include a bias term in the linear layers
        activation: str = "ReLU",  # E.g., "ReLU", "Tanh", "GELU", etc.
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.bias = bias
        self.activation = activation

        self.net = self.initialize_net()

    def initialize_net(self):
        """Build the network according to the provided hyperparameters."""
        layers = []
        
        activation_fn = getattr(nn, self.activation)()
        
        layers.append(nn.Linear(self.in_features, self.hidden_features, bias=self.bias))
        layers.append(activation_fn)
        
        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_features, self.hidden_features, bias=self.bias))
            layers.append(activation_fn)
        
        layers.append(nn.Linear(self.hidden_features, self.out_features, bias=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, coords: jaxtyping.Float[torch.Tensor, "N D"]) -> Tuple[
        jaxtyping.Float[torch.Tensor, "N out_features"],
        jaxtyping.Float[torch.Tensor, "N D"],
    ]:
        """
        Implement a forward pass where the output AND the input require gradients so as to be differentiable.

        Return: tuple of (outputs, gradient-enabled coords). Shape of outputs should be (N, out_features).

        Hint: coords should be (N, d) where N is the number of points (batch size) and d is the dimensionality
        of your input/field. Copy them and enable gradients on the copy. Then, pass them into your network
        recalling that in `utils.py` we use the convention that the input values are in [-1, 1] where
        -1 means "furthest left" or "furthest bottom" (depending on the dimension) and 1 means "furthest right"
        or "furthest top".
        """
        coords_with_grad = coords.clone().detach().requires_grad_(True)
        
        outputs = self.net(coords_with_grad)
        
        return outputs, coords_with_grad

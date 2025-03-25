"""
Implement a neural field using a SIREN.

The SIREN maps from `in_features`-dimensional points (e.g., 2D xy positions)
to `out_features`-dimensional points (e.g., 1D color values). Pay special
attention to the paper linked in the README when implementing this model.

Some conventions we use that you will need to follow IF you want to use the unit tests:
1. If the last layer is linear, then it always has a bias (other terms have a bias IFF self.bias is True)
2. There is one layer from `in_features` to `hidden_features` and then `hidden_layers` layers from 
    `hidden_features` to `hidden_features`; the last layer (which could be linear or not) is from
    `hidden_features` to `out_features`.
"""

import torch
import torch.nn as nn
import jaxtyping
from typing import Tuple


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        is_first_layer: bool,
        omega_0: float,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(is_first_layer, in_features)

    @torch.no_grad()
    def init_weights(self, is_first_layer: bool, in_features: int):
        """
        Initialize the weights of the layer according to the scheme
        described in the SIREN paper.
        """
        # According to the SIREN paper, the weights need special initialization
        # For the first layer, uniform distribution in [-1/in_features, 1/in_features]
        # For subsequent layers, uniform distribution in [-sqrt(6/in_features)/omega_0, sqrt(6/in_features)/omega_0]
        if is_first_layer:
            bound = 1 / in_features
            nn.init.uniform_(self.linear.weight, -bound, bound)
        else:
            bound = torch.sqrt(torch.tensor(6.0) / in_features) / self.omega_0
            nn.init.uniform_(self.linear.weight, -bound, bound)
        
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(
        self,
        in_features: int,  # Number of input features
        out_features: int,  # Number of output features
        hidden_features: int,  # Number of features in the hidden layers
        hidden_layers: int,  # Number of hidden layers
        bias: bool = True,  # Whether to include a bias term in the linear layers
        last_layer_linear: bool = False,  # Whether to use a linear layer for the last layer
        first_omega_0: float = 20.0,  # omega_0 for the first layer
        hidden_omega_0: float = 20.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.bias = bias
        self.last_layer_linear = last_layer_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.net = self.initialize_net()

    def initialize_net(self):
        """
        Build the network according to the provided hyperparameters.
        """
        layers = []
        
        layers.append(
            SineLayer(
                in_features=self.in_features,
                out_features=self.hidden_features,
                bias=self.bias,
                is_first_layer=True,
                omega_0=self.first_omega_0
            )
        )
        
        for _ in range(self.hidden_layers):
            layers.append(
                SineLayer(
                    in_features=self.hidden_features,
                    out_features=self.hidden_features,
                    bias=self.bias,
                    is_first_layer=False,
                    omega_0=self.hidden_omega_0
                )
            )
        
        if self.last_layer_linear:
            layers.append(nn.Linear(self.hidden_features, self.out_features, bias=True))
        else:
            layers.append(
                SineLayer(
                    in_features=self.hidden_features,
                    out_features=self.out_features,
                    bias=self.bias,
                    is_first_layer=False,
                    omega_0=self.hidden_omega_0
                )
            )
        
        return nn.Sequential(*layers)

    def forward(self, coords: jaxtyping.Float[torch.Tensor, "N D"]) -> Tuple[
        jaxtyping.Float[torch.Tensor, "N out_features"],
        jaxtyping.Float[torch.Tensor, "N D"],
    ]:
        """
        Implement a forward pass where the output AND the input require gradients so as to be differentiable.

        Return: tuple of (outputs, gradient-enabled coords). Shape of outputs should be (N, out_features).

        Hint: coords should be (N, d) where N is the number of points (batch size) and d is the dimensionality
        if your input/field. Copy them and enable gradients on the copy. Then, pass them into your network
        recalling that that in `utils.py` we use the convention that the input values are in [-1, 1] where
        -1 means "furthest left" or "furthest bottom" (depending on the dimension) and 1 means "furthest right"
        or "furthest top".
        """
        coords_with_grad = coords.clone().detach().requires_grad_(True)
        
        outputs = self.net(coords_with_grad)
        
        return outputs, coords_with_grad

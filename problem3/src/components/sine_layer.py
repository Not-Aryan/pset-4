import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.d_in = d_in
        
        # Create a linear layer
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize weights according to the SIREN paper
        if self.is_first:
            # First layer initialization
            bound = 1 / self.d_in
            nn.init.uniform_(self.linear.weight, -bound, bound)
        else:
            # Hidden layer initialization
            bound = np.sqrt(6 / self.d_in) / self.omega_0
            nn.init.uniform_(self.linear.weight, -bound, bound)
        
        # Initialize bias if it exists
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, input):
        # Apply linear transformation followed by sine activation with frequency omega_0
        return torch.sin(self.omega_0 * self.linear(input))

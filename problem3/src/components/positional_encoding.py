import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int):
        super().__init__()
        self.num_octaves = num_octaves

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        """Separately encode each channel using a positional encoding. The lowest
        frequency should be 2 * torch.pi, and each frequency thereafter should be
        double the previous frequency. For each frequency, you should encode the input
        signal using both sine and cosine.
        """

        # Create a tensor of frequencies
        # TODO: materize frequency bands
        # Copy the below code (a few lines is enough) to the final submission report.
        # Better to add a small title "Positional Encoding of NeRF" in the report.
        freq_bands = 2.0 * torch.pi * torch.pow(2.0, torch.arange(self.num_octaves, device=samples.device))

        # Create the output tensor
        embedded_dim = list(samples.size())
        embedded_dim[-1] = self.d_out(samples.size()[-1])
        embedded_samples = torch.empty(embedded_dim, device=samples.device)

        # Fill the tensor with sine and cosine embedded values
        for i, freq in enumerate(freq_bands):
            embedded_samples[..., i :: 2 * self.num_octaves] = torch.sin(samples * freq)
            # TODO: materize the PE for cos.
            # Copy the below code (a few lines is enough) to the final submission report.
            # Better to add a small title "Positional Encoding of NeRF" in the report
            embedded_samples[..., i + self.num_octaves :: 2 * self.num_octaves] = torch.cos(samples * freq)

        return embedded_samples

    def d_out(self, dimensionality: int):
        return dimensionality * 2 * self.num_octaves

from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, cat, cumprod, exp, linspace, nn, ones_like, sigmoid, sum

from .field.field import Field


class NeRF(nn.Module):
    cfg: DictConfig
    field: Field

    def __init__(self, cfg: DictConfig, field: Field) -> None:
        super().__init__()
        self.cfg = cfg
        self.field = field

    def forward(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
    ) -> Float[Tensor, "batch 3"]:
        """Render the rays using volumetric rendering. Use the following steps:

        1. Generate sample locations along the rays using self.generate_samples().
        2. Evaluate the neural field at the sample locations. The neural field's output
           has four channels: three for RGB color and one for volumetric density. Don't
           forget to map these channels to valid output ranges.
        3. Compute the alpha values for the evaluated volumetric densities using
           self.compute_alpha_values().
        4. Composite these alpha values together with the evaluated colors from.
        """

        num_samples = self.cfg.num_samples
        # 1. Generate sample locations along the rays
        sample_locations, boundaries = self.generate_samples(
            origins, directions, near, far, num_samples
        )

        # 2. Evaluate the neural field at the sample locations
        field_outputs = self.field(sample_locations.reshape(-1, 3)).reshape(
            origins.size(0), num_samples, -1
        )

        # Ensure channels are in a valid range (e.g., [0, 1])
        colors = sigmoid(field_outputs[..., :])

        colors = field_outputs[..., :3]  # RGB colors
        densities = field_outputs[..., 3]  # Volumetric densities

        # 3. Compute the alpha values
        alphas = self.compute_alpha_values(densities, boundaries)

        # 4. Composite the alpha values and colors
        # TODO: call the final composite function
        # replace the following line with the final composite function
        radiance = self.alpha_composite(alphas, colors)

        return radiance

    def generate_samples(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
        num_samples: int,
    ) -> tuple[
        Float[Tensor, "batch sample 3"],  # xyz sample locations
        Float[Tensor, "batch sample+1"],  # sample boundaries
    ]:
        """For each ray, equally divide the space between the specified near and far
        planes into num_samples segments. Return the segment boundaries (including the
        endpoints at the near and far planes). Also return sample locations, which fall
        at the midpoints of the segments.
        """

        depths = linspace(near, far, num_samples + 1, device=origins.device).expand(
            origins.shape[0], -1
        )  # shape: [batch, num_samples+1]

        # Compute midpoints
        midpoints = 0.5 * (
            depths[:, :-1] + depths[:, 1:]
        )  # shape: [batch, num_samples]

        sample_locations = origins.unsqueeze(1) + midpoints.unsqueeze(
            -1
        ) * directions.unsqueeze(1)
        return sample_locations, depths

    def compute_alpha_values(
        self,
        sigma: Float[Tensor, "batch sample"],
        boundaries: Float[Tensor, "batch sample+1"],
    ) -> Float[Tensor, "batch sample"]:
        """Compute alpha values from volumetric densities (values of sigma) and segment
        boundaries.
        """

        # Calculate the difference in depth between consecutive boundaries
        delta = boundaries[:, 1:] - boundaries[:, :-1]  # shape: [batch, num_samples]

        # Calculate alpha values using the formula
        alpha = 1 - exp(-sigma * delta)

        return alpha

    def alpha_composite(
        self,
        alphas: Float[Tensor, "batch sample"],
        colors: Float[Tensor, "batch sample 3"],
    ) -> Float[Tensor, "batch 3"]:
        """Alpha-composite the supplied alpha values and colors. You may assume that the
        background is black.
        """

        # Step 1: Compute transmittance Ti
        ones = ones_like(alphas[:, :1])  # Create a tensor of ones with shape [batch, 1]
        combined = cat([ones, 1 - alphas], dim=1)
        transmittance = cumprod(combined, dim=1)[:, :-1]

        # Step 2: Compute weights wi
        weights = alphas * transmittance  # shape: [batch, num_samples]

        # Step 3: Compute the expected radiance along the ray c
        radiance = sum(weights.unsqueeze(-1) * colors, dim=1)  # shape: [batch, 3]

        return radiance

"""
Implement a training loop for the MLP and SIREN models.
"""

import torch
from torch.utils.data import DataLoader

from problem_1_gradients import gradient, laplace
from problem_1_mlp import MLP
from problem_1_siren import SIREN
from utils import ImageDataset, plot, psnr
from typing import Dict, Any


def train(
    model,  # "MLP" or "SIREN"
    dataset: ImageDataset,  # Dataset of coordinates and pixels for an image
    lr: float,  # Learning rate
    total_steps: int,  # Number of gradient descent step
    steps_til_summary: int,  # Number of steps between summaries (i.e. print/plot)
    device: torch.device,  # "cuda" or "cpu"
    **kwargs: Dict[str, Any],  # Model-specific arguments
):
    """
    Train the model on the provided dataset.
    
    Given the **kwargs, initialize a neural field model and an optimizer.
    Then, train the model and log the loss and PSNR for each step. Examples
    in the notebook use MSE loss, but feel free to experiment with other
    objective functions. Additionally, in the notebook, we plot the reconstruction
    and various gradients every `steps_til_summary` steps using `utils.plot()`.

    You re allowed to change the arguments as you see fit so long as you can plot
    images of the reconstruction and the gradients/laplacian every `steps_til_summary` steps.
    Look at `should_look_like` for examples of what we would like to see. Make sure to
    also plot (MSE) loss and PSNR every `steps_til_summary` steps.

    You should train for `total_steps` gradient steps on the whole image (look at `ImageDataset` in `utils.py`)
    and visualize the results every `steps_til_summary` steps. The visualization must at least include:
    1. The MSE and PSNR
    2. The reconstructed image
    (Optionally you can also include the laplace or gradient of the image).

    PSNR is defined here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    
    if model == "MLP":
        net = MLP(**kwargs).to(device)
    elif model == "SIREN":
        net = SIREN(**kwargs).to(device)
    else:
        raise ValueError(f"Invalid model: {model}")
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    cords = dataset.coords.to(device)
    pixels = dataset.pixels.to(device)

    losses = []
    psnr_values = []

    for step in range(total_steps):
        optimizer.zero_grad()

        model_output, model_cords = net(cords)
        loss = ((model_output - pixels) ** 2).mean()

        summary_step = step % steps_til_summary == 0 or step == total_steps - 1

        if not summary_step:
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        curr_psnr = psnr(model_output, pixels)
        psnr_values.append(curr_psnr.item())

        if summary_step:
            print(f"Step {step}, Loss: {loss.item():.6f}, PSNR: {curr_psnr:.2f}")
            
            grad = gradient(model_output, model_cords)
            lap = laplace(model_output, model_cords)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            plot(dataset, model_output, grad, lap)
        

    return losses, psnr_values
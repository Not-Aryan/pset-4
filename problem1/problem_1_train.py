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
    # Initialize the model based on the model type
    model_kwargs = kwargs.copy()  # Create a copy to avoid modifying the original
    
    # Set input and output features if not already provided
    if 'in_features' not in model_kwargs:
        model_kwargs['in_features'] = dataset.coords.shape[-1]
        
    if 'out_features' not in model_kwargs:
        model_kwargs['out_features'] = dataset.pixels.shape[-1]
    
    if model == "MLP":
        net = MLP(**model_kwargs).to(device)
    elif model == "SIREN":
        net = SIREN(**model_kwargs).to(device)
    else:
        raise ValueError(f"Unknown model type: {model}")
    
    # Initialize the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Initialize lists to store loss and PSNR values for plotting
    losses = []
    psnr_values = []
    
    # Move data to device
    coords = dataset.coords.to(device)
    pixels = dataset.pixels.to(device)
    
    # Training loop
    for step in range(total_steps):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        model_output, model_coords = net(coords)
        
        # Compute loss (MSE)
        loss = ((model_output - pixels) ** 2).mean()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Store loss and PSNR
        losses.append(loss.item())
        current_psnr = psnr(model_output, pixels)
        psnr_values.append(current_psnr)
        
        # Print progress
        if step % steps_til_summary == 0 or step == total_steps - 1:
            print(f"Step {step}, Loss: {loss.item():.6f}, PSNR: {current_psnr:.2f}")
            
            try:
                # Compute real gradients and Laplacian for visualization
                # We need to enable gradient tracking for these calculations
                coords_vis = coords.clone().detach().requires_grad_(True)
                
                # Forward pass with gradient tracking enabled
                model_output_vis, coords_with_grad = net(coords_vis)
                
                # Try to calculate real gradients and Laplacian
                try:
                    # Normalize model output to enhance gradient visibility
                    # Scale gradients to be more visible in visualizations
                    # IMPORTANT: Use coords_with_grad instead of coords_vis - this is the tensor that's 
                    # actually connected to model_output_vis in the computation graph
                    grad = gradient(model_output_vis, coords_with_grad)
                    
                    # Ensure gradient has the right shape - should be [N, input_dims]
                    if grad.shape != coords_with_grad.shape:
                        print(f"Warning: Gradient shape {grad.shape} doesn't match coordinates shape {coords_with_grad.shape}")
                        # Fix shape if possible
                        if grad.numel() == coords_with_grad.shape[0] * coords_with_grad.shape[1]:
                            grad = grad.reshape(coords_with_grad.shape)
                    
                    grad_vis = grad.detach().cpu()
                    print(f"Gradient shape: {grad_vis.shape}, min: {grad_vis.min().item():.6f}, max: {grad_vis.max().item():.6f}")
                    
                except Exception as e:
                    print(f"Warning: Error computing gradient: {e}")
                    grad_vis = torch.zeros_like(coords_with_grad.detach().cpu())
                
                try:
                    # Compute Laplacian - should be [N] for scalar output
                    lap = laplace(model_output_vis, coords_with_grad, model=net)
                    lap_vis = lap.detach().cpu()
                    print(f"Laplacian shape: {lap_vis.shape}, min: {lap_vis.min().item():.6f}, max: {lap_vis.max().item():.6f}")
                    
                    # If we have scalar values for each input, reshape to 2D image
                    lap_vis_2d = lap_vis.view(dataset.height, dataset.height)
                except Exception as e:
                    print(f"Warning: Error computing Laplacian: {e}")
                    lap_vis = torch.zeros(coords_with_grad.shape[0], device='cpu')
                    lap_vis_2d = lap_vis.view(dataset.height, dataset.height)
                
                # Process tensors for visualization
                model_output_vis_2d = model_output.detach().cpu().view(dataset.height, dataset.height)
                
                # Reshape gradient and Laplacian to 2D for visualization
                # The plot function expects these to be reshapable to (height, height)
                # The gradient needs special handling because it has a dimension for each input coordinate
                if grad_vis.shape[-1] == coords_with_grad.shape[-1]:  # If gradient has coordinate dimension
                    # Need to compute norm of gradient vector at each point, then reshape
                    grad_vis_2d = grad_vis.reshape(-1, coords_with_grad.shape[-1])
                else:
                    # If gradient is already processed, just ensure it's the right shape
                    grad_vis_2d = grad_vis
                
                # Use the correct plot function signature
                plot(
                    dataset=dataset,
                    model_output=model_output_vis_2d,
                    img_grad=grad_vis_2d,
                    img_laplacian=lap_vis_2d,
                )
                
                # Also print current loss and PSNR
                print(f"Loss: {loss.item():.6f}, PSNR: {current_psnr:.2f}")
            except Exception as e:
                print(f"Warning: Error in visualization step: {e}")
                # Continue training even if visualization fails
    
    return losses, psnr_values

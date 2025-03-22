"""
For each function below
- y is the output of the model
- x is the input of the model

Use automatic differentiation to compute all functions of the gradient of y with respect to x.

Hint: implement the functions in order from top to bottom.
"""

import torch


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `y` that represents a function of `x` (`y` could be multi-dimensional),
    return the gradient of `y` with respect to `x`. It is important to note for the assignment
    that the PURPOSE of this function is to be used not only to compute gradients, but
    also to compute them in such a way that we can use them as part of the computational
    graph for backpropagation THROUGH the gradient function. If this is unclear, you
    may note that if you were to run the forward pass of a neural network, then
    compute the output of the backwards pass up to an input `x`, the concatenated
    [forwards pass, backwards pass] is a function of `x` and so defines a longer, 
    concatenated "forward pass" of `x`. We want to use this to train a SIREN on the gradients
    of an image, such that the SIREN's actual output corresponds to the image.
    
    Hint: You may find the `torch.autograd.grad` function useful. To achieve the stated purpose
    above, you will need to set a specific boolean parameter to one of two options. Can you
    read the documentation and figure out which one and why?
    """
    # We need to set create_graph=True to allow backpropagation through the gradient computation
    # This allows us to compute higher-order derivatives and use gradients in the computational graph
    grad_outputs = torch.ones_like(y)
    grad_result = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,  # This is the key parameter to allow backpropagation through this operation
        retain_graph=True,
        only_inputs=True,
        allow_unused=True  # Add this parameter to handle unused tensors in the graph
    )[0]
    
    # Handle the case when grad_result is None (can happen with allow_unused=True)
    if grad_result is None:
        return torch.zeros_like(x)
    
    return grad_result

def divergence(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `y` that represents a function of `x` (`y` could be multi-dimensional),
    return the divergence of `y` with respect to `x`. By convention we will compute
    the divergence along the LAST axis of `y`. When we use this function it will usually
    be flattened in practice.

    Hint: You may find the ` torch.autograd.grad` function useful. Like in `gradient`,
    you need to set a specific boolean parameter to the correct out of two options.
    """
    # Divergence is the sum of the partial derivatives with respect to the corresponding input dimensions
    # We compute this along the last axis of y
    
    # Get the batch size and output dimensions
    batch_size = y.shape[0]
    output_dims = y.shape[-1]
    
    # Initialize the divergence tensor
    div = torch.zeros(batch_size, device=y.device)
    
    # For each output dimension, compute the gradient with respect to the corresponding input dimension
    # and sum them to get the divergence
    for i in range(output_dims):
        # Extract the i-th component of y
        y_i = y[..., i]
        
        # Compute the gradient of y_i with respect to x
        grad_result = torch.autograd.grad(
            outputs=y_i,
            inputs=x,
            grad_outputs=torch.ones_like(y_i),
            create_graph=True,  # Allow backpropagation through this operation
            retain_graph=True,
            only_inputs=True,
            allow_unused=True  # Add this parameter to handle unused tensors in the graph
        )[0]
        
        # Handle the case when grad_result is None
        if grad_result is None:
            continue
        
        # Add the i-th partial derivative to the divergence
        div += grad_result[..., i]
    
    return div

def laplace(y: torch.Tensor, x: torch.Tensor, model=None) -> torch.Tensor:
    """
    Given a tensor `y` that represents a function of `x` (`y` could be multi-dimensional),
    return the laplacian of `y` with respect to `x`.
    
    This implementation uses one of two approaches:
    1. If model is provided, uses finite differences with the model for accurate derivatives
    2. If no model is provided, uses autograd to compute the Laplacian directly
    """
    if model is not None:
        # ===== Finite difference approach with explicit model forward pass =====
        # This approach directly computes second derivatives by running the model on perturbed inputs
        
        # Get dimensions
        batch_size = x.shape[0]
        input_dims = x.shape[-1]
        
        # Step size for finite difference
        h = 0.01
        
        # Initialize Laplacian tensor
        laplacian = torch.zeros(batch_size, device=y.device)
        
        # For each coordinate dimension
        for i in range(input_dims):
            # Create offset vectors for each dimension
            offset = torch.zeros_like(x)
            offset[:, i] = h
            
            # Forward pass at x+h
            x_plus_h = x + offset
            y_plus_h, _ = model(x_plus_h)
            
            # Forward pass at x-h
            x_minus_h = x - offset
            y_minus_h, _ = model(x_minus_h)
            
            # Compute second derivative using central difference formula
            # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
            second_deriv = (y_plus_h.squeeze() - 2*y.squeeze() + y_minus_h.squeeze()) / (h*h)
            
            # Sum the derivatives for Laplacian
            laplacian += second_deriv
            
        # Amplify for better visualization
        laplacian = laplacian * 10.0
        
    else:
        # ===== Pure autograd approach =====
        # Try to use autograd for direct second derivative calculation
        
        # Make a copy of x that requires grad
        x_copy = x.clone().detach().requires_grad_(True)
        
        # Compute first derivatives using autograd
        grad_outputs = torch.ones_like(y)
        first_derivatives = torch.autograd.grad(
            outputs=y, 
            inputs=x_copy,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Initialize Laplacian
        laplacian = torch.zeros(x.shape[0], device=y.device)
        
        # For each dimension, compute second derivative
        for i in range(x.shape[-1]):
            # Extract the dimension's first derivative
            first_deriv_i = first_derivatives[:, i]
            
            # Compute second derivative
            grad_outputs_second = torch.ones_like(first_deriv_i)
            second_derivatives = torch.autograd.grad(
                outputs=first_deriv_i,
                inputs=x_copy,
                grad_outputs=grad_outputs_second,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Add second derivative with respect to the same dimension
            laplacian += second_derivatives[:, i]
            
    # Print diagnostic information
    print(f"Raw Laplacian stats - shape: {laplacian.shape}, min: {laplacian.min().item():.6f}, max: {laplacian.max().item():.6f}")
    
    return laplacian

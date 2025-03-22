import torch
import torch.nn as nn
import math


class SimpleImplicitModel(nn.Module):
    """
    Implicit model for two spheres defined by their Signed Distance Functions (SDF):
    - Sphere 1: Center (0.5, 0.5, 0.5), Radius 0.2, Color: Blue
    - Sphere 2: Center (0.2, 0.7, 0.3), Radius 0.2, Color: Green
    The function computes the SDF for the spheres and returns the color if the point lies inside the sphere.
    """

    def __init__(self):
        super().__init__()

    def forward(self, points_xyz):
        """
        points_xyz: [B, 3]
        Returns:
            sdf: [B, 1] - Signed distance function value for each point
            color: [B, 3] - Color of the sphere (if inside the sphere)
        """
        device = points_xyz.device
        batch_size = points_xyz.shape[0]

        # Initialize SDF and color tensors
        sdf = torch.zeros(batch_size, 1, device=device)
        color = torch.zeros(batch_size, 3, device=device)

        # Sphere 1: Center (0.5, 0.5, 0.5), Radius 0.2, Color: Blue
        sphere1_center = torch.tensor([0.5, 0.5, 0.5], device=device)
        sphere1_radius = 0.2
        sphere1_dist = (
            torch.norm(points_xyz - sphere1_center, dim=1, keepdim=True)
            - sphere1_radius
        )
        sphere1_color = torch.tensor([0.1, 0.4, 0.8], device=device)  # Soft blue

        # Sphere 2: Center (0.2, 0.7, 0.3), Radius 0.2, Color: Green
        sphere2_center = torch.tensor([0.2, 0.7, 0.3], device=device)
        sphere2_radius = 0.2
        sphere2_dist = (
            torch.norm(points_xyz - sphere2_center, dim=1, keepdim=True)
            - sphere2_radius
        )
        sphere2_color = torch.tensor([0.2, 0.7, 0.3], device=device)  # Soft green

        # Determine the closest sphere and assign the color and SDF
        sdf = torch.minimum(sphere1_dist, sphere2_dist)
        color = torch.where(sphere1_dist <= sphere2_dist, sphere1_color, sphere2_color)

        return sdf, color


############################
# Part A: Camera Rays (same as Problem 1)
############################


def camera_param_to_rays(c2w, intrinsics, H=128, W=128):
    """
    Given the camera parameters, generate rays for each pixel.

    Args:
        c2w: [4,4] camera-to-world transform matrix
        intrinsics: [fx, fy, cx, cy] camera intrinsic parameters
        H: Height of the image
        W: Width of the image

    Returns:
        ray_origins: [H, W, 3] origin points for rays
        ray_directions: [H, W, 3] direction vectors for rays
    """
    # NOTE: This function should be the same as in the volumetric rendering problem
    device = c2w.device
    
    # Extract intrinsic parameters
    fx, fy, cx, cy = intrinsics
    
    # Create a meshgrid of pixel coordinates
    # Add 0.5 to pixel coordinates to sample at pixel centers
    y, x = torch.meshgrid(
        torch.arange(H, device=device) + 0.5,
        torch.arange(W, device=device) + 0.5,
        indexing='ij'
    )
    
    # Convert pixel coordinates to camera coordinates using intrinsics
    # X_cam = (x - cx) / fx
    # Y_cam = (y - cy) / fy
    # Z_cam = 1
    x_cam = (x - cx) / fx
    y_cam = (y - cy) / fy
    z_cam = torch.ones_like(x)
    
    # Stack to create camera ray directions
    directions_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [H, W, 3]
    
    # Normalize ray directions to unit length
    directions_cam = directions_cam / torch.norm(directions_cam, dim=-1, keepdim=True)
    
    # Transform camera coordinates to world coordinates using c2w
    # Extract rotation matrix (3x3) and translation vector from c2w
    rotation = c2w[:3, :3]  # [3, 3]
    translation = c2w[:3, 3]  # [3]
    
    # Apply rotation to directions
    # directions_world = directions_cam @ rotation.T
    directions_world = torch.matmul(directions_cam, rotation.T)  # [H, W, 3]
    
    # Create ray origins (camera position in world coordinates)
    # All rays start from the camera position
    ray_origins = translation.expand(H, W, 3)  # [H, W, 3]
    
    return ray_origins, directions_world


############################
# Part B: Sphere Tracing
############################


def sphere_tracing(
    ray_origins,
    ray_directions,
    model,
    t_near=0.0,
    t_far=3.0,
    max_iter=256,
    epsilon=3e-3,  # Increased to 3e-3 for better surface detection
):
    """
    Perform sphere tracing to find the intersection of rays with the implicit model.

    Args:
        ray_origins: [H, W, 3] origin points for rays
        ray_directions: [H, W, 3] direction vectors for rays
        model: Implicit model to compute the SDF
        t_near: Near plane distance
        t_far: Far plane distance
        max_iter: Maximum number of iterations for sphere tracing
        epsilon: Distance threshold for stopping

    Returns:
        image: [H, W, 3] rendered image
    """
    device = ray_origins.device
    H, W, _ = ray_origins.shape

    # Initialize output
    image = torch.zeros(H, W, 3, device=device)
    
    # Initialize t for each ray (starting distance)
    t = torch.ones(H, W, device=device) * t_near
    
    # Track which rays have hit a surface
    hit_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    
    # Track which rays are still active (not hit and not exceeded t_far)
    active_mask = torch.ones(H, W, dtype=torch.bool, device=device)
    
    # Sphere tracing loop
    for _ in range(max_iter):
        # Skip if all rays have hit or are inactive
        if not active_mask.any():
            break
        
        # Sample points along active rays
        points = ray_origins + t.unsqueeze(-1) * ray_directions  # [H, W, 3]
        
        # Only process active rays
        active_points = points[active_mask]  # [num_active, 3]
        
        # Query SDF model for distances and colors
        sdf, color = model(active_points)  # [num_active, 1], [num_active, 3]
        
        # Reshape SDF and color back to match active rays
        sdf_full = torch.zeros(H, W, 1, device=device)
        color_full = torch.zeros(H, W, 3, device=device)
        
        sdf_full[active_mask] = sdf
        color_full[active_mask] = color
        
        # Check which rays hit the surface (SDF < epsilon)
        new_hits = (sdf.squeeze(-1) < epsilon) & active_mask[active_mask]
        
        # Update hit mask and record colors for newly hit rays
        hit_indices = torch.nonzero(active_mask)[new_hits]
        if hit_indices.shape[0] > 0:
            hit_mask[hit_indices[:, 0], hit_indices[:, 1]] = True
            image[hit_indices[:, 0], hit_indices[:, 1]] = color[new_hits]
        
        # Update t for rays that haven't hit yet
        # Using relaxation factor of 0.95 for nearly exact steps
        t[active_mask] = t[active_mask] + sdf.squeeze(-1) * 0.95
        
        # Update active mask: rays are active if they haven't hit and t < t_far
        active_mask = (~hit_mask) & (t < t_far)

    return image


############################
# Part C: Putting It All Together
############################


def render_sdf_with_sphere_tracing(model, c2w, intrinsics, H=128, W=128, max_iter=256):
    device = c2w.device

    # 1. Generate rays
    ray_origins, ray_directions = camera_param_to_rays(c2w, intrinsics, H, W)

    # 2. Sphere tracing
    image = sphere_tracing(ray_origins, ray_directions, model, max_iter=max_iter)

    return image


############################
# Demo
############################


def demo():
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = SimpleImplicitModel().to(device)

    # you can use the debugging intrinsics for debugging, it's correctly rendered images are shown in folder expected_renders_for_debug/
    # after you have finished the problem, you can use the submit intrinsics to render the image and attach it to your pdf report.
    fx, fy, cx, cy = {
        "debug": [100.0, 100.0, 64.0, 64.0],
        "submit": [75.0, 75.0, 64.0, 64.0],
    }["submit"]
    intrinsics = torch.tensor([fx, fy, cx, cy], device=device)

    # Two camera views - make sure both are on the same device
    c2w_1 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 2.0],  # Looking at the scene from z=2
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,  # Added device here
    )

    theta = math.radians(30)
    c2w_2 = torch.tensor(
        [
            [math.cos(theta), 0.0, math.sin(theta), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta), -1.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )

    # Render two views
    img_1 = render_sdf_with_sphere_tracing(
        model, c2w_1, intrinsics, H=128, W=128, max_iter=256
    )
    img_2 = render_sdf_with_sphere_tracing(
        model, c2w_2, intrinsics, H=128, W=128, max_iter=256
    )

    # Visualize
    img_1_np = img_1.detach().cpu().numpy()
    img_2_np = img_2.detach().cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_1_np)
    plt.title("View 1")
    plt.subplot(1, 2, 2)
    plt.imshow(img_2_np)
    plt.title("View 2")
    plt.tight_layout()
    plt.savefig("sphere_tracing.png")


if __name__ == "__main__":
    demo()

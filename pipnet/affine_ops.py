import torch
import torch.nn.functional as F
from typing import Tuple

IMAGE_BATCH_SIZE = Tuple[int, int, int, int]


def get_affine_match_mask(
    t_mtrx: torch.Tensor,
    x_size: IMAGE_BATCH_SIZE,
    device: str,
) -> torch.Tensor:
    """
    Generates match matrix of shape (N, H*W, H*W)
    representing match between
    pre and post-transform image coordinates
    encoded in row-major indexing.
    """
    N, _, _, _ = x_size

    # Get identity transform matrix for image coordinates;
    id_mtrx = get_identity_transform(N)

    # Get coordinates grids;
    id_grid = transform_to_grid(id_mtrx, x_size, device)
    t_grid = transform_to_grid(t_mtrx, x_size, device)

    # Get Out Of Orange coordinates mask and clip invalid coordinates;
    oor_mask = get_oor_mask(t_grid, x_size)
    t_grid = clip_coords(t_grid, x_size)

    # Get match mask for row-major coordinates;
    match_mask = coords_to_match_mask(id_grid, t_grid, oor_mask, x_size, device)
    return match_mask


def get_identity_transform(batch_size: int) -> torch.Tensor:
    """
    Produces a batch of identity transform matrices.
    """
    return torch.eye(2, 3).unsqueeze(0).repeat(batch_size, 1, 1)


def affine(
    x: torch.Tensor,
    t_mtrx: torch.Tensor,
    mode: str = "nearest",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    Differentiable affine transform.
    """
    # Create the affine grid
    grid = F.affine_grid(
        theta=t_mtrx,
        size=x.size(),
        align_corners=False,
    )

    # Apply the grid sample
    x_transformed = F.grid_sample(
        input=x,
        grid=grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )

    return x_transformed


def get_rotation_mtrx(angles: torch.Tensor) -> torch.Tensor:
    """
    Converts angles (in degrees) to rotation matrix batch.
    """
    batch_size = angles.shape[0]

    # Convert to radians;
    angles_rad = angles_to_radians(angles)

    # Calculate affine components;
    sin_x = torch.sin(angles_rad)
    cos_x = torch.cos(angles_rad)
    tx_off = torch.zeros_like(sin_x)
    ty_off = torch.zeros_like(sin_x)

    # Generate matrices;
    t_mtrx = torch.concat(
        tensors=[cos_x, -sin_x, tx_off, sin_x, cos_x, ty_off],
        dim=1,
    )
    return t_mtrx.reshape([batch_size, 2, 3])


def transform_to_grid(
    t_mtrx: torch.Tensor,
    x_size: IMAGE_BATCH_SIZE,
    device: str,
) -> torch.Tensor:
    """
    Converts transformation matrix
    to a grid of transformed coordinates.
    Shape (N, 2, H, W) <- 2 stands for x and y coord.
    """
    # Extract tensor size;
    _, _, H, W = x_size

    # Create coordinates grid;
    # Permute to shape: (N, 2, H, W);
    grid = F.affine_grid(t_mtrx, size=x_size, align_corners=False)
    grid = grid.permute(0, 3, 1, 2)

    # Rescale coords [-1, 1] -> [0, W-1] & [0, H-1]
    grid = (grid + 1) * 0.5
    scale = torch.tensor([W - 1, H - 1]).reshape((1, 2, 1, 1)).to(device)
    grid = (grid * scale).round().long()
    return grid


def get_oor_mask(
    grid: torch.Tensor,
    x_size: IMAGE_BATCH_SIZE,
) -> torch.Tensor:
    """
    Generates Out Of Range coordinates binary mask.
    1 -> x and y coords are in ranges [0, W); [0, H)
    0 -> otherwise
    """
    # Extract tensor size;
    _, _, H, W = x_size

    # Create binary masks for x and y coords (1 -> coord in range);
    mask_x_oor = ((grid[:, 0, ...] >= 0) & (grid[:, 0, ...] <= W - 1)).int()
    mask_y_oor = ((grid[:, 1, ...] >= 0) & (grid[:, 1, ...] <= H - 1)).int()

    # In-range conjunction (1 -> (x in range & y in range));
    mask_oor = mask_x_oor * mask_y_oor
    return mask_oor


def get_zeros_mask(x: torch.Tensor) -> torch.Tensor:
    """
    Computes a binary mask.
    1 -> x != 0
    0 -> x == 0
    """
    return (x != 0).float()


def clip_coords(
    grid: torch.Tensor,
    x_size: IMAGE_BATCH_SIZE,
) -> torch.Tensor:
    """
    Clips grids coords to the max range;
    """
    # Extract tensor size;
    _, _, H, W = x_size

    # Clip coords;
    grid[:, 0, ...] = torch.clamp(grid[:, 0, ...], min=0, max=W - 1)
    grid[:, 1, ...] = torch.clamp(grid[:, 1, ...], min=0, max=H - 1)
    return grid


def coords_to_match_mask(
    id_grid: torch.Tensor,
    t_grid: torch.Tensor,
    oor_mask: torch.Tensor,
    x_size: IMAGE_BATCH_SIZE,
    device: str,
) -> torch.Tensor:
    """
    Generates a binary match mask for row-major image coords.
    1 -> pixel match;
    0 -> no match;
    Resulting shape: (N, H*W, H*W)
    """
    # Extract tensor size;
    N, _, H, W = x_size

    # Convert coords grids to vecs with row-major indexing;
    t_coords_vec = coords_grid_to_vec(t_grid, x_size)
    id_coords_vec = coords_grid_to_vec(id_grid, x_size)

    # Flatten Out Of Range mask;
    oor_mask = oor_mask.reshape((N, -1))

    # Create loss match mask;
    mask = torch.zeros(size=(N, H * W, H * W))
    batch_idxs = torch.arange(N).unsqueeze(1).to(device)
    mask[batch_idxs, id_coords_vec, t_coords_vec] = 1

    # Combine OOR mask with loss match mask;
    mask = mask * oor_mask.unsqueeze(-1)
    return torch.clamp(mask, 0, 1)


def coords_grid_to_vec(
    grid: torch.Tensor,
    x_size: IMAGE_BATCH_SIZE,
) -> torch.Tensor:
    """
    Converts a batch of coordinates meshgrid (N, 2, H, W)
    to a batch of coordinates vector
    using row-major indexing: idx = W*y + x
    """
    # Extract tensor size;
    N, _, _, W = x_size

    coords_vec = (W * grid[:, 1, ...]) + grid[:, 0, ...]
    coords_vec = coords_vec.reshape((N, -1))
    return coords_vec


def draw_angles(
    batch_size: int,
    min_angle: int = -45,
    max_angle: int = 45,
    step: int = 15,
) -> torch.Tensor:
    """
    Randomly samples angles from a range [min_angle, max_angle].
    Performs uniform sampling with the given step.
    """
    angles = torch.arange(min_angle, max_angle + step, step)
    random_indices = torch.randint(0, len(angles), (batch_size,))
    random_angles = angles[random_indices]
    return random_angles.unsqueeze(1)


def angles_to_radians(angles: torch.Tensor) -> torch.Tensor:
    return angles * torch.pi / 180



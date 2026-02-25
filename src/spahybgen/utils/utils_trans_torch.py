import open3d as o3d
import torch
import torch.nn.functional as F


def robust_R6d2R9d(poses: torch.Tensor) -> torch.Tensor:
    """
    Inherent from [GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp)
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector_torch(x_raw)  # batch*3
    y = normalize_vector_torch(y_raw)  # batch*3
    middle = normalize_vector_torch(x + y)
    orthmid = normalize_vector_torch(x - y)
    x = normalize_vector_torch(middle + orthmid)
    y = normalize_vector_torch(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector_torch(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def normalize_vector_torch(v: torch.Tensor) -> torch.Tensor:
    """Normalize a vector in torch"""
    ## for torch operation
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Calculate the cross product of two vectors in torch"""
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out


def orthod6d_to_rot(o6d: torch.Tensor) -> torch.Tensor:
    """Convert orthogonal 6D representation to rotation matrix in torch"""
    x_raw = o6d[:, 0:3]  # batch*3
    y_raw = o6d[:, 3:6]  # batch*3

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = z / torch.norm(z, dim=-1, keepdim=True)  # batch*3
    y = cross_product(z, x)  # batch*3

    matrix = torch.cat([x.view(-1, 3, 1), y.view(-1, 3, 1), z.view(-1, 3, 1)], 2)  # batch*3*3
    return matrix


def quaternion_to_rot(quaternions: torch.Tensor, order: str = "xyzw") -> torch.Tensor:
    """
    Args:
        `quaternions`: B x 4

    Returns:
        B x 3 x 3
    """
    if order == "wxyz":
        w, a, b, c = quaternions.chunk(4, dim=1)
    elif order == "xyzw":
        a, b, c, w = quaternions.chunk(4, dim=1)
    else:
        raise ValueError(f"Unknown quaternion order: {order}")

    return torch.concat(
        [
            torch.concat(
                [1 - 2 * (b.pow(2) + c.pow(2)), 2 * (a * b - w * c), 2 * (a * c + w * b)], dim=-1
            ).unsqueeze(-2),
            torch.concat(
                [2 * (a * b + w * c), 1 - 2 * (a.pow(2) + c.pow(2)), 2 * (b * c - w * a)], dim=-1
            ).unsqueeze(-2),
            torch.concat(
                [2 * (a * c - w * b), 2 * (b * c + w * a), 1 - 2 * (a.pow(2) + b.pow(2))], dim=-1
            ).unsqueeze(-2),
        ],
        dim=-2,
    )

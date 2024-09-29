import open3d as o3d
import torch
import torch.nn.functional as F


def robust_R6d2R9d(poses):
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


def normalize_vector_torch(v):
    ## for torch operation
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out
    

def rot_to_orthod6d(rot):
    return rot.transpose(1, 2)[:, :2].reshape([-1, 6])
    

def orthod6d_to_rot(o6d):
    x_raw = o6d[:, 0:3]  # batch*3
    y_raw = o6d[:, 3:6]  # batch*3

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)     # batch*3
    z = cross_product(x, y_raw)                             # batch*3
    z = z / torch.norm(z, dim=-1, keepdim=True)             # batch*3
    y = cross_product(z, x)                                 # batch*3
        
    matrix = torch.cat([
        x.view(-1, 3, 1),
        y.view(-1, 3, 1),
        z.view(-1, 3, 1)
    ], 2)  # batch*3*3
    return matrix


def random_rot(device='cuda'): # np replace by torch
    rot_angles = torch.randn(3) * torch.pi * 2 
    theta_x, theta_y, theta_z = rot_angles[0], rot_angles[1], rot_angles[2]
    Rx = torch.tensor([[1, 0, 0], [0, torch.cos(theta_x), -torch.sin(theta_x)], [0, torch.sin(theta_x), torch.cos(theta_x)]]).to(device)
    Ry = torch.tensor([[torch.cos(theta_y), 0, torch.sin(theta_y)], [0, 1, 0], [-torch.sin(theta_y), 0, torch.cos(theta_y)]]).to(device)
    Rz = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0], [torch.sin(theta_z), torch.cos(theta_z), 0], [0, 0, 1]]).to(device)
    
    return torch.tensor(Rx @ Ry @ Rz, device=device).float()  # [3, 3]


def hand_param_comp(hand_param, mode='mano_right'):
    if mode == 'mano_right':
        return torch.concat([ torch.tensor([[0., 0., 0., 1., 0., 0., 0., 1., 0.]]).to(hand_param.device).tile((hand_param.shape[0], 1)), hand_param ])
    else:
        raise NotImplementedError()

def btrace(matrix):
    return torch.diagonal(matrix, offset=0, dim1=-1, dim2=-2).sum(dim=-1)

def rot_to_quaternion(rotations):
    """
    Args:
        rotations: B x 3 x 3
    """
    w = torch.sqrt(btrace(rotations) + 1 / 2).unsqueeze(-1)
    return torch.concat([
            w,
            (rotations[:, 1, 2] - rotations[:, 2, 1]).unsqueeze(-1) / w / 4,
            (rotations[:, 2, 0] - rotations[:, 0, 2]).unsqueeze(-1) / w / 4,
            (rotations[:, 0, 1] - rotations[:, 1, 0]).unsqueeze(-1) / w / 4
    ], dim=-1)

def quaternion_to_rot(quaternions, order='xyzw'):
    """
    Args:
        `quaternions`: B x 4
    
    Returns:
        B x 3 x 3
    """
    if order == 'wxyz': w, a, b, c = quaternions.chunk(quaternions.shape[-1], dim=1)
    elif order == 'xyzw': a, b, c, w = quaternions.chunk(quaternions.shape[-1], dim=1)
    
    return torch.concat([
        torch.concat([
            1 - 2 * (b**2 + c**2),
            2 * (a * b - w * c),
            2 * (a * c + w * b)
        ], dim=-1).unsqueeze(-2),
        torch.concat([
            2 * (a * b + w * c),
            1 - 2 * (a**2 + c**2),
            2 * (b * c - w * a)
        ], dim=-1).unsqueeze(-2),
        torch.concat([
            2 * (a * c - w * b),
            2 * (b * c - w * a),
            1 - 2 * (a**2 + b**2)
        ], dim=-1).unsqueeze(-2)
    ], dim=-2)


def do_rotation(points, rots):
    """
    Args:
        `points`: `[batch_shape]` x `num_points` x 3
        `rots`: `[batch_shape]` x 3 x 3
    """
    shape = points.shape
    # Calculate rotation
    points_local = torch.bmm(rots.unsqueeze(-3).expand(list(shape) + [3]).reshape([-1, 3, 3]), 
        points.reshape([-1, 3]).unsqueeze(-1)).reshape(shape)
    return points_local

def do_translation(points, trans):
    """
    Args:
        `points`: `[batch_shape]` x `num_points` x 3
        `trans`: `[batch_shape]`  x 3
    """
    trans = trans.unsqueeze(-2).expand(points.shape)
    return points + trans

def undo_rotation(points, rots, cents=None):
    return do_rotation(points, torch.transpose(rots, -1, -2), cents)

def undo_translation(points, trans):
    return do_translation(points, -trans)

def trans_rot_to_homo_matrix(translation, rotation):
    """
    Args:
        `translation`: `batch_size` x 3 x 3
        `rotation`: `batch_size` x 3
    """
    batch_size = translation.shape[0]
    # `batch_size` x 4 x 3
    rotation = torch.concat([rotation, torch.zeros([batch_size, 1, 3], device=rotation.device)], dim=-2)
    # `batch_size` x 4 x 1
    translation = torch.concat([translation.unsqueeze(-1), torch.ones([batch_size, 1, 1], device=translation.device)], dim=-2)
    # `batch_size` x 4 x 4
    return torch.concat([rotation, translation], dim=-1)

def meshgrid_points(range, ticks, device='cuda', requires_grad=True):
    [[x_lo, x_hi], [y_lo, y_hi], [z_lo, z_hi]] = range
    x_ticks, y_ticks, z_ticks = ticks
    
    xx = torch.linspace(x_lo, x_hi, x_ticks, device=device, requires_grad=requires_grad)
    yy = torch.linspace(y_lo, y_hi, y_ticks, device=device, requires_grad=requires_grad)
    zz = torch.linspace(z_lo, z_hi, z_ticks, device=device, requires_grad=requires_grad)
    
    ii, jj, kk = torch.meshgrid([xx, yy, zz])
    return torch.concat([ii.unsqueeze(-1), jj.unsqueeze(-1), kk.unsqueeze(-1)], dim=-1).reshape([-1, 3])

def mesh_from_points(points, radii=[0.005, 0.01, 0.02, 0.04]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[0])
    pcd = pcd.voxel_down_sample(voxel_size=0.5)
    pcd.estimate_normals()
    
def mesh_to_pts(mesh, padding=False):
    if padding:
        padding_len = torch.max([ m.vertices.shape[0] for m in mesh ]) # np replace by torch
        pts = [
            F.pad(F.pad(torch.Tensor(m.vertices), [0, 1], mode='constant', value=1.), [0, 0, 0, padding_len - m.vertices.shape[0]], mode='constant', value=0.)
            for m in mesh
        ]
        return torch.stack(pts, dim=0)
    
    else:
        pts = [ torch.Tensor(m.verties) for m in mesh ]
        return torch.stack(pts, dim=0)
        
def shuffle_tensor(tensor, dim=-1):
    return tensor[:, torch.randperm(tensor.shape[dim])].view(tensor.size())

def accept_and_tile(tensor, accept_indices, trim=False):
    """Accept parts of a tensor and retile them to recover (may outnumbers by a bit) the batch size.
    """
    old_batch_size = tensor.shape[0]
    tile_factor = [int(tensor.shape[0] / accept_indices.shape[0])] + [1] * (tensor.dim() - 1)
    
    if trim:
        return tensor[accept_indices].tile(tile_factor)[:old_batch_size]
    else:
        return tensor[accept_indices].tile(tile_factor)

def tensor_in_arr(ten, dic):
    for elem in dic:
        if ten.equal(elem):
            return True
    return False

def neighbors_on_mesh(ind, tri=None, neighbor_array=None):
    if tri is None:
        return neighbor_array[ind]
    elif neighbor_array is None:
        neighbors = []
        for t in tri:
            if ind == t[0]:
                if t[1] not in neighbors:
                    neighbors.append(t[1])
                if t[2] not in neighbors:
                    neighbors.append(t[2])
            if ind == t[1]:
                if t[0] not in neighbors:
                    neighbors.append(t[0])
                if t[2] not in neighbors:
                    neighbors.append(t[2])
            if ind == t[2]:
                if t[0] not in neighbors:
                    neighbors.append(t[0])
                if t[1] not in neighbors:
                    neighbors.append(t[1])
    else:
        raise NotImplementedError()
    return neighbors

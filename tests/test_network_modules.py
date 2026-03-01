from spahybgen.networks import get_network
import torch


def test_network_creation():
    """Test netwrok construction with various configurations"""
    voxel_discr = 80
    ntargs_list = [
        {"voxel_discreteness": voxel_discr, "orientation": "quat", "augment": False},
        {"voxel_discreteness": voxel_discr, "orientation": "so3", "augment": False},
        {"voxel_discreteness": voxel_discr, "orientation": "R6d", "augment": False},
        {"voxel_discreteness": voxel_discr, "orientation": "quat", "augment": True},
    ]
    batch_size = 8
    x = torch.randn(batch_size, 1, voxel_discr, voxel_discr, voxel_discr)

    for ntargs in ntargs_list:
        if ntargs["orientation"] == "quat":
            rot_bits = 4
        elif ntargs["orientation"] == "so3":
            rot_bits = 3
        elif ntargs["orientation"] == "R6d":
            rot_bits = 6

        for net_name in ["vgn", "unet"]:
            net = get_network(net_name, ntargs)
            out = net(x)
            assert out[0].shape == (batch_size, 1, voxel_discr, voxel_discr, voxel_discr)  # out_score
            assert out[1].shape == (batch_size, rot_bits, voxel_discr, voxel_discr, voxel_discr)  # out_rot
            assert out[2].shape == (batch_size, 1, voxel_discr, voxel_discr, voxel_discr)  # out_wren

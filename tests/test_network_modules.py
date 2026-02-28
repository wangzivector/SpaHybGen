from spahybgen.networks import get_network
import torch


def test_network_creation():
    voxel_discr = 80

    ntargs_list = [
        {"voxel_discreteness": voxel_discr, "orientation": "quat", "augment": False},
        {"voxel_discreteness": voxel_discr, "orientation": "so3", "augment": False},
        {"voxel_discreteness": voxel_discr, "orientation": "R6d", "augment": False},
        {"voxel_discreteness": voxel_discr, "orientation": "quat", "augment": True},
    ]

    for ntargs in ntargs_list:
        if ntargs["orientation"] == "quat":
            rotation_bits = 4
        elif ntargs["orientation"] == "so3":
            rotation_bits = 3
        elif ntargs["orientation"] == "R6d":
            rotation_bits = 6

        vgn = get_network("vgn", ntargs)
        batch_size = 8
        x = torch.randn(batch_size, 1, voxel_discr, voxel_discr, voxel_discr)
        out = vgn(x)
        assert out[0].shape == (batch_size, 1, voxel_discr, voxel_discr, voxel_discr)  # out_score
        assert out[1].shape == (batch_size, rotation_bits, voxel_discr, voxel_discr, voxel_discr)  # out_rot
        assert out[2].shape == (batch_size, 1, voxel_discr, voxel_discr, voxel_discr)  # out_wren

        unet = get_network("unet", ntargs)
        out = unet(x)
        assert out[0].shape == (batch_size, 1, voxel_discr, voxel_discr, voxel_discr)  # out_score
        assert out[1].shape == (batch_size, rotation_bits, voxel_discr, voxel_discr, voxel_discr)  # out_rot
        assert out[2].shape == (batch_size, 1, voxel_discr, voxel_discr, voxel_discr)  # out_wren

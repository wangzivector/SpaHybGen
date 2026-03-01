import spahybgen.observation as ObsEng
import numpy as np
from spahybgen.observation import CameraIntrinsic
from spahybgen.utils.utils_trans_np import Transform


def test_volume_creation():
    """
    Test volumetric creation process
    """
    TSDF_volume_size = 0.4
    TSDF_discreteness = 80
    d_weight, d_height = 720, 640
    depth = np.random.rand(d_height, d_weight).astype(np.float32)  # m
    depth = ObsEng.depth_inpaint(depth)
    depth_imgs = np.expand_dims(depth, axis=0)
    intrinsics = CameraIntrinsic(d_weight, d_height, d_weight, d_height, d_weight / 2, d_height / 2)
    extrinsics_arrays = np.expand_dims(Transform.identity().to_list(), axis=0)
    voxel = ObsEng.create_voxel(
        TSDF_volume_size, TSDF_discreteness, depth_imgs, intrinsics, extrinsics_arrays
    )
    voxel_grid = voxel.get_grid()
    assert voxel_grid.shape == (1, TSDF_discreteness, TSDF_discreteness, TSDF_discreteness)

    tsdf = ObsEng.create_tsdf(
        TSDF_volume_size, TSDF_discreteness, depth_imgs, intrinsics, extrinsics_arrays, trunc=8
    )
    tsdf_grid = tsdf.get_grid()
    assert tsdf_grid.shape == (1, TSDF_discreteness, TSDF_discreteness, TSDF_discreteness)

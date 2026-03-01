import spahybgen.grasptip as shg_contact
from spahybgen import grasptip as GraspType
import numpy as np
from graspnetAPI import GraspGroup


def test_grasp_decomposition():
    """Test grasp decomposition module"""
    sceneId, annId = -1, -1
    TSDF_volume_size = 0.4
    TSDF_discreteness = 80
    GRASP_ARRAY_LEN = 17
    grasp_size = 64

    voxel_grid = np.random.rand(1, TSDF_discreteness, TSDF_discreteness, TSDF_discreteness)

    grasp_group = GraspGroup(np.random.rand(grasp_size, GRASP_ARRAY_LEN))
    camera_pose = np.identity(4)

    grasps_vis, df_raw_grasps = GraspType.Graspnets2Grasps(grasp_group, camera_pose, sceneId, annId)
    tips_data = GraspType.Grasp2Tips(df_raw_grasps)
    columns_tip = 20
    assert len(grasps_vis) == grasp_size
    assert tips_data.shape == (grasp_size, columns_tip)

    # create voxel-based tips data from raw tips
    df_VoxelTips = GraspType.Tips2TipsDF(
        tips_data,
        TSDF_volume_size,
        TSDF_discreteness,
        interp_ratios=[0.8, 1.0],
        scene_grid=voxel_grid,
        grid_type="voxel",
    )
    columns_voxeltips = 8
    assert df_VoxelTips.shape[1] == columns_voxeltips

    # create voxel-based wrenches data from raw tips
    df_Wrens = GraspType.Tips2WrensDF(
        tips_data,
        TSDF_volume_size,
        TSDF_discreteness,
        interp_ratios=[0.8, 1.0],
    )
    columns_wren = 3
    assert df_Wrens.shape[1] == columns_wren

### Generate dataset from raw pointcloud and grasp, obtaining standard TSDF/Voxel and Grasp/Contacts.
## BEFORE START: Modify graspnet_root and generate_root paths in the script below.

## Steps:
## 1. TSDF from scene depth-camera data, which save as .npz of grid_tsdf
## 2. Create Panda-based voxel-formated Grasp poses from raw grasp data
## 3. Transfer grasp to raw tips data and eventually the voxel-based tips data

# Manage data path
from pathlib import Path
from graspnetAPI import GraspNet
from tqdm import tqdm
import numpy as np

from spahybgen import grasptip as GraspType
from spahybgen.utils import utils_dataio
from spahybgen.utils.utils_trans_np import Transform
import spahybgen.observation as ObsEng

sceneIds_task = [0, 100, 1]
# sceneIds_task = [100, 190, 1] For test data
annIds_task = [0, 256, 4]

TSDF_volume_size, TSDF_discreteness = 0.4, 80

graspnet_root = Path("/path to graspnet/Graspnet")
generate_root = Path("/path to project/spahybgen/dataset/train")
# generate_root = Path("/path to project/spahybgen/dataset/test") For test data

print("Graspnet: {} \nResult: {}".format(graspnet_root.absolute(), generate_root.absolute()))

camera = 'kinect'
TSDF_SHIFT = np.vstack((np.c_[np.eye(3), np.array([0.32, 0.35, 0.10])], [0.0, 0.0, 0.0, 1.0]))

sceneIds = range(sceneIds_task[0], sceneIds_task[1], sceneIds_task[2]) # change to control generation
annIds = range(annIds_task[0], annIds_task[1], annIds_task[2])  # change to control generation

g = GraspNet(graspnet_root, camera=camera, split='all') # all

# TSDF generation
total_jobs = ((sceneIds_task[1] - sceneIds_task[0])/annIds_task[2])*((annIds_task[1] - annIds_task[0])/sceneIds_task[2])
with tqdm(total=total_jobs) as pbar: # task process indication
    pbar.set_description('==> DATASET CONVERSION [{},{}] <== '.format(sceneIds_task[0], sceneIds_task[1])) # task process indication
    for sceneId in sceneIds:
        for annId in annIds:
            pbar.update(1) # task process indication
            
            # Generate and save scene tsdf file
            depth = g.loadDepth(sceneId = sceneId, camera = camera, annId = annId).astype(np.float32) / 1000.0 # m
            depth = ObsEng.depth_inpaint(depth)
            depth_imgs = np.expand_dims(depth, axis = 0)
            intrinsics, camera_poses = utils_dataio.GraspnetCameraInfo.fetch_IntExts(
                graspnet_root, sceneId, camera, depth.shape, align = True, base_shift = TSDF_SHIFT)
            extrinsics_arrays = np.expand_dims(Transform.from_matrix(np.linalg.inv(camera_poses[annId])).to_list(), axis = 0)
            
            tsdf = ObsEng.create_tsdf(TSDF_volume_size, TSDF_discreteness, depth_imgs, intrinsics, extrinsics_arrays, trunc = 8)
            tsdf_grid = tsdf.get_grid()
            utils_dataio.write_tsdf_grid(generate_root, sceneId, annId, tsdf_grid)

            voxel = ObsEng.create_voxel(TSDF_volume_size, TSDF_discreteness, depth_imgs, intrinsics, extrinsics_arrays)
            voxel_grid = voxel.get_grid()
            utils_dataio.write_voxel_grid(generate_root, sceneId, annId, voxel_grid)
            
            # Grasp label information from GraspnetAPI
            grasps_6d = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = camera, fric_coef_thresh = 0.2)
            sample_6d_grasp_group = grasps_6d.random_sample(int(len(grasps_6d)/8))

            # GraspnetAPI-grasps to vgn-grasp to raw tips
            grasps_vis, df_raw_grasps = GraspType.Graspnets2Grasps(sample_6d_grasp_group, camera_poses[annId], sceneId, annId)
            tips_data = GraspType.Grasp2Tips(df_raw_grasps)

            # create voxel-based tips data from raw tips
            df_VoxelTips = GraspType.Tips2TipsDF(tips_data, TSDF_volume_size, TSDF_discreteness, interp_ratios = [0.8, 1.0], scene_grid = voxel_grid, grid_type='voxel')
            utils_dataio.write_df(df_VoxelTips, generate_root, sceneId, annId, name = "VoxelTips")

            # create voxel-based wrenches data from raw tips
            df_Wrens = GraspType.Tips2WrensDF(tips_data, TSDF_volume_size, TSDF_discreteness, interp_ratios = [0.8, 1.0])
            utils_dataio.write_df(df_Wrens, generate_root, sceneId, annId, name = "VoxelWrens")


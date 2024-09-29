# Inherent from [VGN](https://github.com/ethz-asl/vgn) and GraspnetAPI

import numpy as np
import pandas as pd
import os
from spahybgen.observation import CameraIntrinsic

class GraspnetCameraInfo(object):
    """Obtain Graspnet Extrinsic, Intrinsic parameters of a pinhole camera model.

    Attributes:
        data_root (Path): Path of dataset.
        sceneId: Scene ID.
        camera: camera type.
    """ 
    def __init__(self, data_root, sceneId, camera):
        self.data_root = data_root
        self.sceneId = sceneId
        self.camera = camera

    @staticmethod
    def fetch_ori(data_root, sceneId, camera):
        intrinsics = np.load(os.path.join(data_root, 'scenes', 'scene_%04d' % sceneId, camera, 'camK.npy'))
        camera_poses = np.load(os.path.join(data_root, 'scenes', 'scene_%04d' % sceneId, camera, 'camera_poses.npy'))
        align_mat = np.load(os.path.join(data_root, 'scenes', 'scene_%04d' % sceneId, camera, 'cam0_wrt_table.npy'))
        return intrinsics, camera_poses, align_mat

    @staticmethod
    def fetch_IntExts(data_root, sceneId, camera, depth_size, align=True, base_shift=np.eye(4)):
        intrinsics_mat = np.load(os.path.join(data_root, 'scenes', 'scene_%04d' % sceneId, camera, 'camK.npy'))
        camera_poses = np.load(os.path.join(data_root, 'scenes', 'scene_%04d' % sceneId, camera, 'camera_poses.npy'))
        align_mat = np.load(os.path.join(data_root, 'scenes', 'scene_%04d' % sceneId, camera, 'cam0_wrt_table.npy'))
        camera_poses_wrt_table = np.zeros_like(camera_poses)
        if align:
            for i in range(len(camera_poses)):
                camera_poses_wrt_table[i] = base_shift.dot(align_mat.dot(camera_poses[i]))
        
        fx, fy = intrinsics_mat[0][0], intrinsics_mat[1][1]
        cx, cy = intrinsics_mat[0][2], intrinsics_mat[1][2]
        intrinsics = CameraIntrinsic(depth_size[1], depth_size[0], fx, fy, cx, cy)
        return intrinsics, camera_poses_wrt_table


def read_df(root, scene_id, ann_id, name):
    if scene_id is None: return pd.read_csv(root, index_col=0)
    return pd.read_csv(root / ('scene_%04d' % scene_id) / (("ann_%04d" % ann_id) + ("_%s.csv" % name)), index_col=0)


def write_df(df, root, scene_id, ann_id, name):
    df.to_csv(root / ('scene_%04d' % scene_id) / (("ann_%04d" % ann_id) + ("_%s.csv" % name)), index=True)


def write_df_tsdf_grasps(df, root, scene_id, ann_id):
    pass

def write_tsdf_grid(root, scene_id, ann_id, tsdf_grid):
    (root / ('scene_%04d' % scene_id)).mkdir(parents=True, exist_ok=True)
    path = root / ('scene_%04d' % scene_id) / ('ann_%04d.npz' % ann_id)
    np.savez_compressed(path, grid=tsdf_grid)


def read_tsdf_grid(root, scene_id, ann_id):
    if scene_id is None: return np.load(root)["grid"]
    path = root / ('scene_%04d' % scene_id) / ('ann_%04d.npz' % ann_id)
    return np.load(path)["grid"]


def write_voxel_grid(root, scene_id, ann_id, voxel_grid):
    (root / ('scene_%04d' % scene_id)).mkdir(parents=True, exist_ok=True)
    path = root / ('scene_%04d' % scene_id) / ('ann_%04d_voxel.npz' % ann_id)
    np.savez_compressed(path, grid=voxel_grid)


def read_voxel_grid(root, scene_id, ann_id):
    if scene_id is None: return np.load(root)["grid"]
    path = root / ('scene_%04d' % scene_id) / ('ann_%04d_voxel.npz' % ann_id)
    return np.load(path)["grid"]


def read_cam0_to_world(graspnet_root, sceneId, camera):
    align_mat = np.load(os.path.join(graspnet_root, 'scenes', 'scene_%04d' % sceneId, camera, 'cam0_wrt_table.npy'))
    return align_mat


def write_raw_grasp(root, scene_id, ann_id, grasp, erase=False):
    csv_path = root / ('scene_%04d' % scene_id) / ("ann_%04d_rawgrasps.csv" % ann_id)
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "ann_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "depth", "finger_base_depth", "score"],
        )
    if erase:
        erase_csv(csv_path)
        create_csv(
            csv_path,
            ["scene_id", "ann_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "depth", "finger_base_depth", "score"],
        )
        return
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width, depth, finger_base_depth, label = grasp.width, grasp.depth, grasp.finger_base_depth, grasp.score
    append_csv(csv_path, scene_id, ann_id, qx, qy, qz, qw, x, y, z, width, depth, finger_base_depth, label)


def create_csv(path, columns):
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")


def erase_csv(path):
    with path.open("w") as f:
        f.truncate(0)


def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")

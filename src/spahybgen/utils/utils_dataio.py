# Inherent from [VGN](https://github.com/ethz-asl/vgn) and GraspnetAPI

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import os
from pathlib import Path
from spahybgen.observation import CameraIntrinsic
from spahybgen.grasptip import Grasp_neat


class GraspnetCameraInfo(object):
    """
    Obtain Graspnet Extrinsic, Intrinsic parameters of a pinhole camera model.

    Attributes:
        data_root (Path): Path of dataset.
        sceneId: Scene ID.
        camera: camera type.
    """

    def __init__(self, data_root: Path, sceneId: int, camera: str) -> None:
        self.data_root = data_root
        self.sceneId = sceneId
        self.camera = camera

    @staticmethod
    def fetch_ori(data_root: Path, sceneId: int, camera: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetch the transformation information of one scene of the origin

        Args:
            data_root: Path of dataset.
            sceneId: SCene ID.
            camera: Camera type, 'kinetic' or 'realsense'.

        Returns:
            Intrinsics, camera poses, and align matrice of all captures in the scene.
        """
        intrinsics = np.load(os.path.join(data_root, "scenes", "scene_%04d" % sceneId, camera, "camK.npy"))
        camera_poses = np.load(
            os.path.join(data_root, "scenes", "scene_%04d" % sceneId, camera, "camera_poses.npy")
        )
        align_mat = np.load(
            os.path.join(data_root, "scenes", "scene_%04d" % sceneId, camera, "cam0_wrt_table.npy")
        )
        return intrinsics, camera_poses, align_mat

    @staticmethod
    def fetch_IntExts(
        data_root: Path,
        sceneId: int,
        camera: str,
        depth_size: tuple,
        align: bool = True,
        base_shift: np.ndarray = np.eye(4),
    ) -> Tuple[CameraIntrinsic, np.ndarray]:
        """
        Fetch the intrinciss and extrinsics of scene captures

        Args:
            data_root (Path): dataset root path
            sceneId: Scene id
            camera: Camera name
            depth_size: (height, width) of depth images for constructing CameraIntricsic
            align: Whether align to origin of workspace. Defaults to True.
            base_shift: Shift of orgin of the workspace. Defaults to np.eye(4).

        Returns:
            Intrinsics and extrinsics of camera captures
        """
        intrinsics_mat = np.load(
            os.path.join(data_root, "scenes", "scene_%04d" % sceneId, camera, "camK.npy")
        )
        camera_poses = np.load(
            os.path.join(data_root, "scenes", "scene_%04d" % sceneId, camera, "camera_poses.npy")
        )
        align_mat = np.load(
            os.path.join(data_root, "scenes", "scene_%04d" % sceneId, camera, "cam0_wrt_table.npy")
        )
        camera_poses_wrt_table = np.zeros_like(camera_poses)
        if align:
            for i in range(len(camera_poses)):
                camera_poses_wrt_table[i] = base_shift.dot(align_mat.dot(camera_poses[i]))

        fx, fy = intrinsics_mat[0][0], intrinsics_mat[1][1]
        cx, cy = intrinsics_mat[0][2], intrinsics_mat[1][2]
        intrinsics = CameraIntrinsic(depth_size[1], depth_size[0], fx, fy, cx, cy)
        return intrinsics, camera_poses_wrt_table


def read_df(root: Path, scene_id: Optional[int], ann_id: Optional[int], name: Optional[str]) -> pd.DataFrame:
    """
    Read the dataframe of one annotation

    Args:
        root: dataset path
        scene_id: scene id
        ann_id: annotation id
        name: type name of data

    Returns:
        out: dataframe of label for the annotated scene
    """
    if scene_id is None:
        return pd.read_csv(root, index_col=0)
    return pd.read_csv(
        root / ("scene_%04d" % scene_id) / (("ann_%04d" % ann_id) + ("_%s.csv" % name)), index_col=0
    )


def write_df(df: pd.DataFrame, root: Path, scene_id: int, ann_id: int, name: str) -> None:
    """
    Write the dataframe of one annotation

    Args:
        df (Dataframe): dataframe to be saved
        root: dataset path
        scene_id: scene id
        ann_id: annotation id
        name: type name of data
    """
    df.to_csv(root / ("scene_%04d" % scene_id) / (("ann_%04d" % ann_id) + ("_%s.csv" % name)), index=True)


def write_tsdf_grid(root: Path, scene_id: int, ann_id: int, tsdf_grid: np.ndarray) -> None:
    """
    Save the dataframe of tsdf grid volume

    Args:
        root: dataset path
        scene_id: scene id
        ann_id: annotation id
        tsdf_grid: tsdf to be saved
    """
    (root / ("scene_%04d" % scene_id)).mkdir(parents=True, exist_ok=True)
    path = root / ("scene_%04d" % scene_id) / ("ann_%04d.npz" % ann_id)
    np.savez_compressed(path, grid=tsdf_grid)


def read_tsdf_grid(root: Path, scene_id: Optional[int], ann_id: Optional[int]) -> np.ndarray:
    """
    Read the ndarray of tsdf

    Args:
        root: dataset path
        scene_id: scene id. If is None, directly read from the path `root`
        ann_id: annotation id

    Returns:
        out: ndarray of tsdf observation for the annotated scene
    """
    if scene_id is None:
        return np.load(root)["grid"]
    path = root / ("scene_%04d" % scene_id) / ("ann_%04d.npz" % ann_id)
    return np.load(path)["grid"]


def write_voxel_grid(root: Path, scene_id: int, ann_id: int, voxel_grid: np.ndarray) -> None:
    """
    Write the voxel observation of one annotation

    Args:
        root: dataset path
        scene_id: scene id
        ann_id: annotation id
        voxel_grid: voxel volume of scene annotation
    """
    (root / ("scene_%04d" % scene_id)).mkdir(parents=True, exist_ok=True)
    path = root / ("scene_%04d" % scene_id) / ("ann_%04d_voxel.npz" % ann_id)
    np.savez_compressed(path, grid=voxel_grid)


def read_voxel_grid(root: Path, scene_id: Optional[int], ann_id: Optional[int]) -> np.ndarray:
    """
    Read the voxel volume of one annotation

    Args:
        root: dataset path
        scene_id: scene id, If is None, read directly from `root`
        ann_id: annotation id

    Returns:
        out: ndarray of voxel volume for the annotated scene
    """
    if scene_id is None:
        return np.load(root)["grid"]
    path = root / ("scene_%04d" % scene_id) / ("ann_%04d_voxel.npz" % ann_id)
    return np.load(path)["grid"]


def read_cam0_to_world(graspnet_root: str, sceneId: int, camera: str) -> np.ndarray:
    """
    Read the transfermation of the first frame wrt world

    Args:
        graspnet_root: dataset path
        scene_id: scene id
        camera: camera type: `kinect` or `realsense`

    Returns:
        out: ndarray of cam0 wrt table frame
    """
    align_mat = np.load(
        os.path.join(graspnet_root, "scenes", "scene_%04d" % sceneId, camera, "cam0_wrt_table.npy")
    )
    return align_mat


def write_raw_grasp(root: Path, scene_id: int, ann_id: int, grasp: Grasp_neat, erase: bool = False) -> None:
    """
    Write grasp data to csv format

    Args:
        root: dataset path
        scene_id: scene id
        ann_id: annotation id
        grasp (Grasp_neat) : Contact annotation
        erase: Whether delete original label data. Defaults to False.
    """
    csv_path = root / ("scene_%04d" % scene_id) / ("ann_%04d_rawgrasps.csv" % ann_id)
    if not csv_path.exists():
        create_csv(
            csv_path,
            [
                "scene_id",
                "ann_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "depth",
                "finger_base_depth",
                "score",
            ],
        )
    if erase:
        erase_csv(csv_path)
        create_csv(
            csv_path,
            [
                "scene_id",
                "ann_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "depth",
                "finger_base_depth",
                "score",
            ],
        )
        return
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width, depth, finger_base_depth, label = grasp.width, grasp.depth, grasp.finger_base_depth, grasp.score
    append_csv(csv_path, scene_id, ann_id, qx, qy, qz, qw, x, y, z, width, depth, finger_base_depth, label)


def create_csv(path: Path, columns: List[str]) -> None:
    """
    Craete csv strings for saving

    Args:
        path: data path to write
        columns: annotation label to create
    """
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")


def erase_csv(path: Path) -> None:
    """
    erase label information in path

    Args:
        path: data path to write
    """
    with path.open("w") as f:
        f.truncate(0)


def append_csv(path: Path, *args) -> None:
    """
    add one label record to path

    Args:
        path: path to add info
    """
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")

# Inherent from [VGN](https://github.com/ethz-asl/vgn)
"""Render volumes, point clouds, and grasp detections in rviz."""

import matplotlib.colors
import numpy as np
from sensor_msgs.msg import PointCloud2
import rospy
from rospy import Publisher
from visualization_msgs.msg import Marker, MarkerArray

from spahybgen.utils import utils_rosmsg
from spahybgen.utils.utils_trans_np import Transform, Rotation
from spahybgen.grasptip import index_str2nums, Grasp_neat
import pandas as pd
from typing import List, Union, Optional


cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["r", "g"])
DELETE_MARKER_MSG = Marker(action=Marker.DELETEALL)
DELETE_MARKER_ARRAY_MSG = MarkerArray(markers=[DELETE_MARKER_MSG])


def workspace_lines(size: float) -> List[List[float]]:
    """
    Create a box in the origin of Rviz world

    Args:
        size: size of cubic box

    Returns:
        out: list of box corners
    """
    return [
        [0.0, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, size, 0.0],
        [size, size, 0.0],
        [0.0, size, 0.0],
        [0.0, size, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, size],
        [size, 0.0, size],
        [size, 0.0, size],
        [size, size, size],
        [size, size, size],
        [0.0, size, size],
        [0.0, size, size],
        [0.0, 0.0, size],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, size],
        [size, 0.0, 0.0],
        [size, 0.0, size],
        [size, size, 0.0],
        [size, size, size],
        [0.0, size, 0.0],
        [0.0, size, size],
    ]


def visualize_vectors_in_df(df_vectors: pd.DataFrame, voxel_size: float) -> List[dict]:
    """
    Visualize contact features as vectors

    Args:
        df_vectors: dataframe of contact features
        voxel_size: size length of one grid voxel

    Returns:
        out: list of constructed vector for rviz vectors
    """
    tips_vectors = []
    for label in df_vectors.index:
        indexs_uvw = index_str2nums(label)
        # nums = label.split('/')
        # indexs_uvw = np.array([int(nums[0]), int(nums[1]), int(nums[2])])
        location_xyz = indexs_uvw * voxel_size
        tip_vector = {}
        tip_vector["score"] = df_vectors.loc[label]["weighted_score"]
        quat = df_vectors.loc[label][["mean_qx", "mean_qy", "mean_qz", "mean_qw"]].to_numpy()
        pose_tran = Transform.from_list(np.hstack([quat, location_xyz]).tolist())
        pose_vector = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]])  # visualize z-axis
        pose_vector = pose_tran.transform_point(pose_vector)
        tip_vector["points"] = pose_vector
        tips_vectors.append(tip_vector)
    return tips_vectors


def visualize_vectors_in_array(
    array_tips: list, array_scores: Union[list, np.ndarray], voxel_size: float, is_list: bool = False
) -> List[dict]:
    """
    Visualize contact features as vectors

    Args:
        array_tips: ndarray of contact features, poses
        array_scores: scores of contacts for color mapping
        voxel_size: size length of one grid voxel
        is_list: where the pose data is list or Transform

    Returns:
        out: list of constructed vector for rviz vectors
    """
    tips_vectors = []
    pose_vector_ori = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]])  # visualize z-axis
    for ind, (pose, score) in enumerate(zip(array_tips, array_scores)):
        if not is_list:
            location_xyz = pose[1] * voxel_size
            pose_tran = Transform.from_list(np.hstack([pose[0].as_quat(), location_xyz]).tolist())
        else:
            pose[4:] = pose[4:] * voxel_size
            pose_tran = Transform.from_list(pose)

        tip_vector = {}
        tip_vector["score"] = score
        tip_vector["points"] = pose_tran.transform_point(pose_vector_ori)
        tips_vectors.append(tip_vector)
    return tips_vectors


def draw_workspace(
    size: float, frame: str = "task", color: list = [0.5, 0.5, 0.5], pose: Optional[Transform] = None
) -> Marker:
    """
    Draw the cubic workspace in Rviz

    Args:
        size: size of the box
        frame: frame id of box. Defaults to "task".
        color: color of box. Defaults to [0.5, 0.5, 0.5].
        pose: pose of box. Defaults to None.

    Returns:
        Marker of lines to indicate box vertices
    """
    scale = size * 0.01
    if pose == None:
        pose = Transform.identity()
    scale = [scale, 0.0, 0.0]
    color = color
    msg = _create_marker_msg(Marker.LINE_LIST, frame, pose, scale, color)
    msg.points = [utils_rosmsg.to_point_msg(point) for point in workspace_lines(size)]
    pubs["workspace"].publish(msg)
    return msg


def draw_grid(vol: np.ndarray, grid_size: float, threshold: float = 0.01, frame_id: str = "task") -> None:
    """
    Draw volume grid to Rviz

    Args:
        vol: volume data in ndarray
        grid_size: length of the voxel size
        threshold: threshold to filter low quanlity voxels. Defaults to 0.01
        frame_id: frame id of the volume. Defaults to "task"
    """
    msg = _create_vol_msg(vol, grid_size, threshold, frame_id)
    msg.header.frame_id = frame_id
    pubs["grid"].publish(msg)


def draw_points(points: np.ndarray, frame: str = "task") -> None:
    """
    Cast points to Rviz

    Args:
        points: points to cast in ndarray
        frame: frame id of visualization. Defaults to "task".
    """
    msg = utils_rosmsg.to_cloud_msg(points, frame=frame)
    pubs["points"].publish(msg)


def draw_quality(vol: np.ndarray, voxel_size: float, threshold: float = 0.01, frame: str = "task") -> None:
    """
    Draw color-annotated volume in Rviz

    Args:
        vol: volume data
        voxel_size: length of voxel size
        threshold: threshold to filter low quanlity voxels. Defaults to 0.01
        frame_id: frame id of the volume. Defaults to "task"
    """
    msg = _create_vol_msg(vol, voxel_size, threshold, frame)
    pubs["quality"].publish(msg)


def draw_volume(vol: np.ndarray, voxel_size: float, threshold: float = 0.01) -> None:
    """
    Draw volume in Rviz, for debug only

    Args:
        vol: volume data
        voxel_size: length of voxel size
        threshold: threshold to filter low quanlity voxels. Defaults to 0.01
    """
    msg = _create_vol_msg(vol, voxel_size, threshold, frame="task")
    pubs["debug"].publish(msg)


def draw_grasp(grasp: Grasp_neat, score: float, finger_depth: float) -> None:
    """
    Draw grasp pose in Rviz

    Args:
        grasp: grasp data
        score: score of the grasp
        finger_depth: finger length
    """
    radius = 0.1 * finger_depth
    w, d = grasp.width, finger_depth
    color = cmap(float(score))

    markers = []

    # left finger
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, -w / 2, d / 2])
    scale = [radius, radius, d]
    msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    msg.id = 0
    markers.append(msg)

    # right finger
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, w / 2, d / 2])
    scale = [radius, radius, d]
    msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    msg.id = 1
    markers.append(msg)

    # wrist
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, 0.0, -d / 4])
    scale = [radius, radius, d / 2]
    msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    msg.id = 2
    markers.append(msg)

    # palm
    pose = grasp.pose * Transform(Rotation.from_rotvec(np.pi / 2 * np.r_[1.0, 0.0, 0.0]), [0.0, 0.0, 0.0])
    scale = [radius, radius, w]
    msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    msg.id = 3
    markers.append(msg)

    pubs["grasp"].publish(MarkerArray(markers=markers))


def draw_grasps(grasps: List[Grasp_neat]) -> None:
    """
    Draw multiple grasp poses in Rviz

    Args:
        grasps: list of grasps
    """
    markers = []
    for i in range(len(grasps)):
        msg = _create_grasp_marker_msg(grasps[i])
        msg.id = i
        markers.append(msg)
    msg = MarkerArray(markers=markers)
    pubs["grasps"].publish(msg)


def draw_vectors(vectors: list, frame: str = "task", opacity: bool = False) -> None:
    """
    Draw contact poses in Rviz

    Args:
        vectors: contact poses in list
        frame: frame id. Defaults to "task".
        opacity: whether annotate transparency with contact scores. Defaults to False.
    """
    markers = []
    for i in range(len(vectors)):
        msg = _create_vector_marker_msg(vectors[i], frame=frame, opacity=opacity)
        msg.id = i
        markers.append(msg)
    msg = MarkerArray(markers=markers)
    pubs["vectors"].publish(msg)


def clear_grid(frame: str = "task") -> None:
    """
    Clear visualize grid in the Rviz

    Args:
        frame: frame_id. Defaults to "task".
    """
    pubs["grid"].publish(utils_rosmsg.to_cloud_msg(np.array([]), frame=frame))


def clear() -> None:
    """Clear all visualized data in Rviz"""
    pubs["workspace"].publish(DELETE_MARKER_MSG)
    pubs["grid"].publish(utils_rosmsg.to_cloud_msg(np.array([]), frame="task"))
    pubs["points"].publish(utils_rosmsg.to_cloud_msg(np.array([]), frame="task"))
    clear_quality()
    pubs["grasp"].publish(DELETE_MARKER_ARRAY_MSG)
    clear_grasps()
    pubs["debug"].publish(utils_rosmsg.to_cloud_msg(np.array([]), frame="task"))
    clear_vectors()


def clear_quality() -> None:
    """Clear volume"""
    pubs["quality"].publish(utils_rosmsg.to_cloud_msg(np.array([]), frame="task"))


def clear_grasps() -> None:
    """Clear grasp"""
    pubs["grasps"].publish(DELETE_MARKER_ARRAY_MSG)


def clear_vectors() -> None:
    """Clear vector of contact poses"""
    pubs["vectors"].publish(DELETE_MARKER_ARRAY_MSG)


def _create_publishers() -> dict:
    """Initialize visualizing topics in Rviz"""
    pubs = dict()
    pubs["workspace"] = Publisher("/workspace", Marker, queue_size=1, latch=True)
    pubs["grid"] = Publisher("/grid", PointCloud2, queue_size=1, latch=True)
    pubs["points"] = Publisher("/points", PointCloud2, queue_size=1, latch=True)
    pubs["quality"] = Publisher("/quality", PointCloud2, queue_size=1, latch=True)
    pubs["grasp"] = Publisher("/grasp", MarkerArray, queue_size=1, latch=True)
    pubs["grasps"] = Publisher("/grasps", MarkerArray, queue_size=1, latch=True)
    pubs["vectors"] = Publisher("/vectors", MarkerArray, queue_size=1, latch=True)
    pubs["debug"] = Publisher("/debug", PointCloud2, queue_size=1, latch=True)
    return pubs


def _create_marker_msg(
    marker_type: int, frame: str, pose: Transform, scale: list, color: Union[tuple, list, np.ndarray]
) -> Marker:
    """
    Create a general Marker type msg

    Args:
        marker_type: Marker type
        frame: frame_id
        pose: pose of feature in Transform
        scale: scale parameter of Marker
        color: color info

    Returns:
        Marker msg
    """
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.type = marker_type
    msg.action = Marker.ADD
    msg.pose = utils_rosmsg.to_pose_msg(pose)
    msg.scale = utils_rosmsg.to_vector3_msg(scale)
    msg.color = utils_rosmsg.to_color_msg(color)
    return msg


def _create_vol_msg(vol: np.ndarray, voxel_size: float, threshold: float, frame: str) -> PointCloud2:
    """
    Create volume msg using PointCloud2

    Args:
        vol: volume data to visualize
        voxel_size: length of voxel
        threshold: threshold to filter low-quality data
        frame: frame_id

    Returns:
        PointCloud2 msg
    """
    vol = vol.squeeze()
    points = np.argwhere(vol > threshold) * voxel_size
    rospy.logdebug("Grid visual points number with threshold {}: {}".format(threshold, points.shape[0]))
    values = np.expand_dims(vol[vol > threshold], 1)
    return utils_rosmsg.to_cloud_msg(points, values, frame)


def _create_grasp_marker_msg(grasp: Grasp_neat) -> Marker:
    """
    Instantialize grasp data to Marker for visualization

    Args:
        grasp: grasp data in list of Grasp_neat

    Returns:
        Marker msg
    """
    finger_depth = grasp.depth + grasp.finger_base_depth
    radius = 0.1 * finger_depth
    w, d = grasp.width, finger_depth
    scale = [radius, 0.0, 0.0]
    color = cmap(float(grasp.score))
    msg = _create_marker_msg(Marker.LINE_LIST, "task", grasp.pose, scale, color)
    msg.points = [utils_rosmsg.to_point_msg(point) for point in _gripper_lines(w, d)]
    return msg


def _create_vector_marker_msg(vector: dict, frame: str, opacity: bool = False) -> Marker:
    """
    Instantialize vector (contacts) to Marker

    Args:
        vector: vector data (contact)
        frame: frame_id in Rviz
        opacity: Whether map quality of vectors to set opacity. Defaults to False.

    Returns:
        list of Markers
    """
    length, width, height = 0.002, 0.004, 0.002
    # scale.x: shaft diameter; scale.y: head diameter; If scale.z: head length.
    scale = [length, width, height]
    color = cmap(vector["score"])
    if opacity:
        color = (color[0], color[1], color[2], vector["score"])
    pose = Transform.identity()
    msg = _create_marker_msg(Marker.ARROW, frame, pose, scale, color)
    msg.points = [utils_rosmsg.to_point_msg(point) for point in vector["points"]]
    return msg


def _gripper_lines(width: float, depth: float) -> list:
    """
    Lines for drawing a gripper frame

    Args:
        width: width of grasp
        depth: depth of grasp

    Returns:
        list of lines for drawing grasps
    """
    return [
        [0.0, 0.0, -depth / 2.0],
        [0.0, 0.0, 0.0],
        [0.0, -width / 2.0, 0.0],
        [0.0, -width / 2.0, depth],
        [0.0, width / 2.0, 0.0],
        [0.0, width / 2.0, depth],
        [0.0, -width / 2.0, 0.0],
        [0.0, width / 2.0, 0.0],
    ]


pubs = _create_publishers()

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
from spahybgen.grasptip import index_str2nums


cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["r", "g"])
DELETE_MARKER_MSG = Marker(action=Marker.DELETEALL)
DELETE_MARKER_ARRAY_MSG = MarkerArray(markers=[DELETE_MARKER_MSG])


def workspace_lines(size):
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


def visualize_vectors_in_df(df_vectors, voxel_size):
    tips_vectors = []
    for label in df_vectors.index:
        indexs_uvw = index_str2nums(label)
        # nums = label.split('/')
        # indexs_uvw = np.array([int(nums[0]), int(nums[1]), int(nums[2])])
        location_xyz = indexs_uvw * voxel_size
        tip_vector = {}
        tip_vector['score'] = df_vectors.loc[label]['weighted_score']
        quat = df_vectors.loc[label][['mean_qx', 'mean_qy', 'mean_qz', 'mean_qw']].to_numpy()
        pose_tran = Transform.from_list(np.hstack([quat, location_xyz]))
        pose_vector = np.array([[0., 0., 0.], [0., 0., 0.01]]) # visualize z-axis
        pose_vector = pose_tran.transform_point(pose_vector)
        tip_vector['points'] = pose_vector
        tips_vectors.append(tip_vector)
    return tips_vectors


def visualize_vectors_in_array(array_tips, array_scores, voxel_size, is_list=False):
    tips_vectors = []
    pose_vector_ori = np.array([[0., 0., 0.], [0., 0., 0.01]]) # visualize z-axis
    for ind, (pose, score) in enumerate(zip(array_tips, array_scores)):
        if not is_list:
            location_xyz = pose[1] * voxel_size
            pose_tran = Transform.from_list(np.hstack([pose[0].as_quat(), location_xyz]))
        else: 
            pose[4:] = pose[4:] * voxel_size
            pose_tran = Transform.from_list(pose)

        tip_vector = {}
        tip_vector['score'] = score
        tip_vector['points'] = pose_tran.transform_point(pose_vector_ori)
        tips_vectors.append(tip_vector)
    return tips_vectors


def draw_workspace(size, frame="task", color = [0.5, 0.5, 0.5], pose = None):
    scale = size * 0.01
    if pose == None: pose = Transform.identity()
    scale = [scale, 0.0, 0.0]
    color = color
    msg = _create_marker_msg(Marker.LINE_LIST, frame, pose, scale, color)
    msg.points = [utils_rosmsg.to_point_msg(point) for point in workspace_lines(size)]
    pubs["workspace"].publish(msg)
    return msg


def draw_grid(vol, grid_size, threshold=0.01, frame_id = 'task'):
    msg = _create_vol_msg(vol, grid_size, threshold, frame_id)
    msg.header.frame_id = frame_id
    pubs["grid"].publish(msg)


def draw_points(points, frame="task"):
    msg = utils_rosmsg.to_cloud_msg(points, frame=frame)
    pubs["points"].publish(msg)


def draw_quality(vol, voxel_size, threshold=0.01, frame="task"):
    msg = _create_vol_msg(vol, voxel_size, threshold, frame)
    pubs["quality"].publish(msg)


def draw_volume(vol, voxel_size, threshold=0.01):
    msg = _create_vol_msg(vol, voxel_size, threshold)
    pubs["debug"].publish(msg)


def draw_grasp(grasp, score, finger_depth):
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
    pose = grasp.pose * Transform(
        Rotation.from_rotvec(np.pi / 2 * np.r_[1.0, 0.0, 0.0]), [0.0, 0.0, 0.0]
    )
    scale = [radius, radius, w]
    msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    msg.id = 3
    markers.append(msg)

    pubs["grasp"].publish(MarkerArray(markers=markers))


def draw_grasps(grasps):
    markers = []
    for i in range(len(grasps)):
        msg = _create_grasp_marker_msg(grasps[i])
        msg.id = i
        markers.append(msg)
    msg = MarkerArray(markers=markers)
    pubs["grasps"].publish(msg)


def draw_vectors(vectors, frame="task", opacity=False):
    markers = []
    for i in range(len(vectors)):
        msg = _create_vector_marker_msg(vectors[i], frame=frame, opacity=opacity)
        msg.id = i
        markers.append(msg)
    msg = MarkerArray(markers=markers)
    pubs["vectors"].publish(msg)


def clear_grid(frame="task"):
    pubs["grid"].publish(utils_rosmsg.to_cloud_msg(np.array([]), frame=frame))


def clear():
    pubs["workspace"].publish(DELETE_MARKER_MSG)
    pubs["grid"].publish(utils_rosmsg.to_cloud_msg(np.array([]), frame="task"))
    pubs["points"].publish(utils_rosmsg.to_cloud_msg(np.array([]), frame="task"))
    clear_quality()
    pubs["grasp"].publish(DELETE_MARKER_ARRAY_MSG)
    clear_grasps()
    pubs["debug"].publish(utils_rosmsg.to_cloud_msg(np.array([]), frame="task"))
    clear_vectors()

def clear_quality():
    pubs["quality"].publish(utils_rosmsg.to_cloud_msg(np.array([]), frame="task"))


def clear_grasps():
    pubs["grasps"].publish(DELETE_MARKER_ARRAY_MSG)

def clear_vectors():
    pubs["vectors"].publish(DELETE_MARKER_ARRAY_MSG)


def _create_publishers():
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


def _create_marker_msg(marker_type, frame, pose, scale, color):
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.type = marker_type
    msg.action = Marker.ADD
    msg.pose = utils_rosmsg.to_pose_msg(pose)
    msg.scale = utils_rosmsg.to_vector3_msg(scale)
    msg.color = utils_rosmsg.to_color_msg(color)
    return msg


def _create_vol_msg(vol, voxel_size, threshold, frame):
    vol = vol.squeeze()
    points = np.argwhere(vol > threshold) * voxel_size
    rospy.logdebug("Grid visual points number with threshold {}: {}".format(threshold, points.shape[0]))
    values = np.expand_dims(vol[vol > threshold], 1)
    return utils_rosmsg.to_cloud_msg(points, values, frame)


def _create_grasp_marker_msg(grasp):
    finger_depth = grasp.depth + grasp.finger_base_depth
    radius = 0.1 * finger_depth
    w, d = grasp.width, finger_depth
    scale = [radius, 0.0, 0.0]
    color = cmap(float(grasp.score))
    msg = _create_marker_msg(Marker.LINE_LIST, "task", grasp.pose, scale, color)
    msg.points = [utils_rosmsg.to_point_msg(point) for point in _gripper_lines(w, d)]
    return msg


def _create_vector_marker_msg(vector, frame, opacity=False):
    length, width, height = 0.002, 0.004, 0.002
    # scale.x: shaft diameter; scale.y: head diameter; If scale.z: head length.
    scale = [length, width, height]
    color = cmap(vector['score'])
    if opacity: color = (color[0], color[1], color[2], vector['score']) 
    pose = Transform.identity()
    msg = _create_marker_msg(Marker.ARROW, frame, pose, scale, color)
    msg.points = [utils_rosmsg.to_point_msg(point) for point in vector['points']]
    return msg


def _gripper_lines(width, depth):
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

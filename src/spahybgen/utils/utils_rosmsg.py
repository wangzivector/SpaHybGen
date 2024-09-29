# Inherent from [VGN](https://github.com/ethz-asl/vgn)

import math
import geometry_msgs.msg
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg

from spahybgen.utils.utils_trans_np import Rotation, Transform

from geometry_msgs.msg import TransformStamped, PoseStamped


def converToTransMsg(father_frame, child_frame, translation, rotation):
    trans_msg = TransformStamped() 
    trans_msg.header.frame_id = father_frame
    trans_msg.child_frame_id = child_frame
    trans_msg.header.stamp = rospy.Time.now()

    trans_msg.transform.translation.x = translation[0]
    trans_msg.transform.translation.y = translation[1]
    trans_msg.transform.translation.z = translation[2]
    if len(rotation) == 3:
        q = Rotation.from_rotvec(rotation).as_quat()
    elif len(rotation) == 4:
        q = rotation
    trans_msg.transform.rotation.x = q[0]
    trans_msg.transform.rotation.y = q[1]
    trans_msg.transform.rotation.z = q[2]
    trans_msg.transform.rotation.w = q[3]
    return trans_msg

def to_point_msg(position):
    """Convert numpy array to a Point message."""
    msg = geometry_msgs.msg.Point()
    msg.x = position[0]
    msg.y = position[1]
    msg.z = position[2]
    return msg


def from_point_msg(msg):
    """Convert a Point message to a numpy array."""
    return np.r_[msg.x, msg.y, msg.z]


def to_vector3_msg(vector3):
    """Convert numpy array to a Vector3 message."""
    msg = geometry_msgs.msg.Vector3()
    msg.x = vector3[0]
    msg.y = vector3[1]
    msg.z = vector3[2]
    return msg


def from_vector3_msg(msg):
    """Convert a Vector3 message to a numpy array."""
    return np.r_[msg.x, msg.y, msg.z]


def to_quat_msg(orientation):
    """Convert a `Rotation` object to a Quaternion message."""
    quat = orientation.as_quat()
    msg = geometry_msgs.msg.Quaternion()
    msg.x = quat[0]
    msg.y = quat[1]
    msg.z = quat[2]
    msg.w = quat[3]
    return msg


def from_quat_msg(msg):
    """Convert a Quaternion message to a Rotation object."""
    return Rotation.from_quat([msg.x, msg.y, msg.z, msg.w])


def to_pose_msg(transform):
    """Convert a `Transform` object to a Pose message."""
    msg = geometry_msgs.msg.Pose()
    msg.position = to_point_msg(transform.translation)
    msg.orientation = to_quat_msg(transform.rotation)
    return msg


def to_transform_msg(transform):
    """Convert a `Transform` object to a Transform message."""
    msg = geometry_msgs.msg.Transform()
    msg.translation = to_vector3_msg(transform.translation)
    msg.rotation = to_quat_msg(transform.rotation)
    return msg


def from_transform_msg(msg):
    """Convert a Transform message to a Transform object."""
    translation = from_vector3_msg(msg.translation)
    rotation = from_quat_msg(msg.rotation)
    return Transform(rotation, translation)


def to_color_msg(color):
    """Convert a numpy array to a ColorRGBA message."""
    msg = std_msgs.msg.ColorRGBA()
    msg.r = color[0]
    msg.g = color[1]
    msg.b = color[2]
    msg.a = color[3] if len(color) == 4 else 1.0
    return msg


def to_cloud_msg(points, intensities=None, frame=None, stamp=None):
    """Convert list of unstructured points to a PointCloud2 message.

    Args:
        points: Point coordinates as array of shape (N,3).
        colors: Colors as array of shape (N,3).
        frame
        stamp
    """
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.header.stamp = stamp or rospy.Time.now()

    msg.height = 1
    msg.width = points.shape[0]
    msg.is_bigendian = False
    msg.is_dense = False

    msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    msg.point_step = 12
    data = points

    if intensities is not None:
        msg.fields.append(PointField("intensity", 12, PointField.FLOAT32, 1))
        msg.point_step += 4
        data = np.hstack([points, intensities])

    msg.row_step = msg.point_step * points.shape[0]
    msg.data = data.astype(np.float32).tostring()

    return msg

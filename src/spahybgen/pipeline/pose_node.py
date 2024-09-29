### TF tree and robot execution management, including UR5e

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as SR


"""
URClient Node
"""
class PoseNode:
    def __init__(self, robot_prefix, camera_prefix):
        self.robot_prefix, self.camera_prefix = robot_prefix, camera_prefix
        ### Init static pose sequence
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.static_br = tf2_ros.StaticTransformBroadcaster()
        self.dynamic_br = tf2_ros.TransformBroadcaster()
        rospy.sleep(0.1) # [pipeline]: Awaiting all TCP connection done.

        self.ur_pose_pub = rospy.Publisher(self.robot_prefix + "pose_servo_cmd", PoseStamped, queue_size=1)
        # self.sub_servo = rospy.Subscriber(self.robot_prefix + "ur_pose_state", PoseStamped, self.ur_pose_state_CB)
        rospy.sleep(0.1) # [pipeline]: Awaiting all TCP connection done.
        

    def characterize_grasp_transform(self, armend2gripper, gripper2grip, gripper_tfname, gripper_tfname_vis, upper_offset_height = - 0.10):
        trans_end2gri = self.PosRotToTransMsg(self.robot_prefix + "tool", self.robot_prefix + "gripper", [0.0, 0.0, armend2gripper], [0.0, 0.0, 0.0])
        trans_gri2grispe = self.PosRotToTransMsg(self.robot_prefix + "gripper", "hand_" + self.robot_prefix + gripper_tfname, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        trans_obj2gri = self.PosRotToTransMsg(self.robot_prefix + "object", self.robot_prefix +  "object_grip", [0.0, 0.0, - gripper2grip], [0.0, 0.0, 0.0]) # grip excution pose
        trans_gri2grispe_vis = self.PosRotToTransMsg(self.robot_prefix + "object_grip", gripper_tfname_vis, [0.0, 0.0, 0], [0.0, 0.0, 0.0])
        trans_end2gri_rev4vis = self.PosRotToTransMsg(self.robot_prefix + "object_grip", self.robot_prefix + "object_grip_endpos", [0.0, 0.0, - armend2gripper], [0.0, 0.0, 0.0])
        trans_obj2upp = self.PosRotToTransMsg(self.robot_prefix + "object_grip_endpos", self.robot_prefix + "endpos_upper", [0.0, 0.0, upper_offset_height], [0.0, 0.0, 0.0]) # motion prepare pose

        self.static_br.sendTransform([trans_obj2upp, trans_end2gri, trans_obj2gri, trans_gri2grispe, 
                                     trans_gri2grispe_vis, trans_end2gri_rev4vis])
        rospy.sleep(0.2)

    def publish_ur_pose(self, pose_type, pose, repeat=False):
        pose_to_send = PoseStamped()
        pose_to_send.header.frame_id = "COM_" + pose_type
        if pose_type in ['APOSE', 'POSE']:
            pose_to_send.pose.position.x = pose[0]
            pose_to_send.pose.position.y = pose[1]
            pose_to_send.pose.position.z = pose[2]
            # direct axis-angle rotation 
            pose_to_send.pose.orientation.x = pose[3]
            pose_to_send.pose.orientation.y = pose[4]
            pose_to_send.pose.orientation.z = pose[5]
            pose_to_send.pose.orientation.w = 1
            # print(pose_type, pose_to_send)
        elif pose_type in ['JOINT', 'AJOINT']:
            pose_to_send.pose.position.x = pose[0]
            pose_to_send.pose.position.y = pose[1]
            pose_to_send.pose.position.z = pose[2]
            # direct axis-angle rotation 
            pose_to_send.pose.orientation.x = pose[3]
            pose_to_send.pose.orientation.y = pose[4]
            pose_to_send.pose.orientation.z = pose[5]
            pose_to_send.pose.orientation.w = 1
            
        self.ur_pose_pub.publish(pose_to_send)
        if repeat: 
            rospy.sleep(0.1)
            self.ur_pose_pub.publish(pose_to_send)
        # print(pose_type, pose_to_send)


    def publish_dynamic_tf(self, father_frame, child_frame, PosRotVec, add_prefix=(False, False)):
        father_frame = self.robot_prefix + father_frame if add_prefix[0] else father_frame
        child_frame = self.robot_prefix + child_frame if add_prefix[1] else child_frame

        translation, rotvec = PosRotVec[:3], PosRotVec[3:]
        trans_dynamic = self.PosRotToTransMsg(father_frame, child_frame, translation, rotvec)
        self.dynamic_br.sendTransform(trans_dynamic)

    
    def fetch_tf_posquat(self, target_frame, source_frame, wait_block=True):
        """
        Transformation from source_frame to target_frame
        """
        loop_rate_rece = rospy.Rate(2)
        while(not rospy.is_shutdown()):
            try:
                trans_listener = self.tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time())
                return trans_listener
            except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Fail to fetch transformation... from {} to {}".format(target_frame, source_frame))
                if wait_block is False: return None

                loop_rate_rece.sleep()
                continue


    def fetch_tf_posrot(self, target_frame, source_frame, wait_block=True, add_prefix=(False, False)):
        """
        Transformation from source_frame to target_frame
        Frame: "world", "base", "armend", "camera", "gripper", "object", "endpos_upper", "object_grip"
        """
        target_frame = self.robot_prefix + target_frame if add_prefix[0] else target_frame
        source_frame = self.robot_prefix + source_frame if add_prefix[1] else source_frame

        Trans_msg = self.fetch_tf_posquat(target_frame, source_frame, wait_block)
        Pos_rota = self.TransMsgToPosRot(Trans_msg) if Trans_msg is not None else None
        return Pos_rota


    @staticmethod
    def PosRotToTransMsg(father_frame, child_frame, translation, rot_vecter):
        trans_msg = TransformStamped()
        trans_msg.header.frame_id = father_frame
        trans_msg.child_frame_id = child_frame
        trans_msg.header.stamp = rospy.Time.now()

        trans_msg.transform.translation.x = translation[0]
        trans_msg.transform.translation.y = translation[1]
        trans_msg.transform.translation.z = translation[2]

        q = SR.from_rotvec(rot_vecter).as_quat()

        trans_msg.transform.rotation.x = q[0]
        trans_msg.transform.rotation.y = q[1]
        trans_msg.transform.rotation.z = q[2]
        trans_msg.transform.rotation.w = q[3]
        return trans_msg
    

    @staticmethod
    def TransMsgToPosRot(trans_msg):
        translation = [
            trans_msg.transform.translation.x, 
            trans_msg.transform.translation.y, 
            trans_msg.transform.translation.z
        ]
        quat_num = [
            trans_msg.transform.rotation.x,
            trans_msg.transform.rotation.y,
            trans_msg.transform.rotation.z,
            trans_msg.transform.rotation.w
        ]
        quat = SR.from_quat(quat_num)
        rot_vecter = quat.as_rotvec().tolist()
        pos_rot = translation + rot_vecter
        return pos_rot
    

    @staticmethod
    def PosRotVec2TransMat(pose_posrotvec):
        pose_rotation_SR = SR.from_rotvec(pose_posrotvec[3:])
        position_vec = np.array([pose_posrotvec[:3]]).T
        homon_vec = np.array([[0.0, 0.0, 0.0, 1.0]])
        tran_mat  = np.concatenate((np.concatenate((pose_rotation_SR.as_dcm(), position_vec), axis=1), homon_vec), axis=0)
        return tran_mat


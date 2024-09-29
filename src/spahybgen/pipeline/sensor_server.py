#!/usr/bin/env python

import numpy as np
import rospy
import ros_numpy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray
import spahybgen.observation as ObsEng
from spahybgen.observation import CameraIntrinsic
from spahybgen.utils.utils_trans_np import Transform
import spahybgen.visualization as Visualization


"""
Azure RGBD ROS Node for grid
"""
class SensorServer:
    def __init__(self, voxel_disc, grid_length, depth_topic, caminfo_topic, grid_topic, grid_request_topic, visualize_grid):
        self.voxel_disc = voxel_disc
        self.grid_length = grid_length
        self.show_grid_with_issue = visualize_grid
        self.current_cam2world = [0,0,0, 0,0,0]
        
        self.depth_buff = None
        rospy.Subscriber(depth_topic, Image, self.depth_callback)
        rospy.loginfo('[Sensor]: ROS subcriping to {}'.format(depth_topic))

        self.caminfo_buff = None
        rospy.Subscriber(caminfo_topic, CameraInfo, self.caminfo_callback)
        rospy.loginfo('[Sensor]: ROS subcriping to {}'.format(caminfo_topic))

        self.grid_request = rospy.Subscriber(grid_request_topic, Float32MultiArray, self.grid_request_callback)
        rospy.loginfo('[Sensor]: ROS subcriping to {}'.format(grid_request_topic))

        self.grid_pub = rospy.Publisher(grid_topic, Float32MultiArray, queue_size=1, latch=True)
        self.reset_depthpose_buff()


    def reset_depthpose_buff(self):
        self.g3d_extrinsics_buff = None
        self.g3d_depth_buff = None


    def append_depthpose_buff(self, cam2world_k):
        extrinsics = np.expand_dims(self.convert_extrinsics(cam2world_k), axis=0)
        depth_image = np.expand_dims(self.fetch_depth_image(), axis=0)

        if self.g3d_extrinsics_buff is None: self.g3d_extrinsics_buff = extrinsics
        else: self.g3d_extrinsics_buff = np.vstack((self.g3d_extrinsics_buff, extrinsics))

        if self.g3d_depth_buff is None: self.g3d_depth_buff = depth_image
        else: self.g3d_depth_buff = np.vstack((self.g3d_depth_buff, depth_image))


    def grid_request_callback(self, msg):
        data = np.array(msg.data)
        if len(data) == 1:
            if data.mean() == 0: 
                rospy.loginfo('[Sensor]: clear buff by msg: {}'.format(data))
                self.reset_depthpose_buff()
            elif data.mean() == 1:
                rospy.loginfo('[Sensor]: add tsdf buff by msg: {}'.format(data))
                self.issue_grid_using_buff(grid_type='tsdf')
            elif data.mean() == 2:
                rospy.loginfo('[Sensor]: add voxel buff by msg: {}'.format(data))
                self.issue_grid_using_buff(grid_type='voxel')
            elif data.mean() == 3:
                rospy.loginfo('[Sensor]: issue single tsdf by msg: {}'.format(data))
                self.issue_grid_using_buff(grid_type='tsdf')
            elif data.mean() == 4:
                rospy.loginfo('[Sensor]: issue single voxel by msg: {}'.format(data))
                self.issue_grid_using_buff(grid_type='voxel')
        else:
            rospy.loginfo("Buffing depth with pose data: \n{}".format(data))
            self.append_depthpose_buff(cam2world_k=data)
            self.current_cam2world = data
        

    def depth_callback(self, msg):
        self.depth_buff = msg

    def caminfo_callback(self, msg):
        self.caminfo_buff = msg

    def fetch_depth_image(self):
        ## Depth Image
        while self.depth_buff is None:
            rospy.loginfo("[Sensor]: depth_image buff is None, retrying...")
            rospy.sleep(1.0)
        
        depth_image = ros_numpy.image.image_to_numpy(self.depth_buff).astype(np.float32)
        depth_image = np.nan_to_num(depth_image)
        # rospy.loginfo('Depth_image size: {}'.format(depth_image.shape))
        depth_image = self.depth_inpaint(depth_image, missing_value = 0)
        if depth_image.mean() > 1: depth_image = depth_image / 1000.0
        return depth_image


    @staticmethod
    def depth_inpaint(image, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in the depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (image == missing_value).astype(np.uint8)
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        imax, imin = np.abs(image).max(), np.abs(image).min()
        irange = imax - imin
        image = ((image - imin) / irange).astype(np.float32) # Has be float32, 64 not supported. get -1:1
        image = cv2.inpaint(image, mask, 2, cv2.INPAINT_NS) # repair with fluid alg. radius 1
        # Back to original size and value range.
        image = image[1:-1, 1:-1] # cut the 1 pixel boarder
        image = image.astype(np.float32) * irange + imin
        return image


    def fetch_intrinsic(self):
        ## Camera Info
        while self.caminfo_buff is None:
            rospy.loginfo("[Sensor]: depth_image buff is None, retrying...")
            rospy.sleep(1.0)
        else: cam_info = self.caminfo_buff
        fx, fy, cx, cy = cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5]
        width, height = cam_info.width, cam_info.height
        intrinsics = CameraIntrinsic(width, height, fx, fy, cx, cy)
        return intrinsics


    def convert_extrinsics(self, cam2world_k):
        cam2world_k_tranf = Transform.from_list_transrotvet(cam2world_k)
        extrinsics = cam2world_k_tranf.inverse().to_list()
        return extrinsics


    def fetch_single_grid(self, grid_type):
        intrinsic = self.fetch_intrinsic()
        depth_imgs = np.expand_dims(self.fetch_depth_image(), axis=0)
        extrinsics = np.expand_dims(self.convert_extrinsics(self.current_cam2world), axis=0)
        return self.gen_grid(depth_imgs, intrinsic, extrinsics, grid_type)


    def gen_grid(self, depth_imgs, intrinsic, extrinsics, grid_type):
        if grid_type == 'tsdf':
            tsdf = ObsEng.create_tsdf(self.grid_length, self.voxel_disc, depth_imgs, intrinsic, 
                                    extrinsics, trunc = 8)
            grid_data = tsdf.get_grid()
        elif grid_type == 'voxel':
            voxel = ObsEng.create_voxel(self.grid_length, self.voxel_disc, depth_imgs, intrinsic, 
                                    extrinsics)
            grid_data = voxel.get_grid()
        return grid_data


    def issue_grid(self, grid_data):
        msg = Float32MultiArray(data=grid_data.astype(np.float32).reshape(-1))
        self.grid_pub.publish(msg)


    def issue_grid_using_buff(self, grid_type):
        if self.g3d_extrinsics_buff is None:
            self.append_depthpose_buff(self.current_cam2world)
            rospy.logwarn("extrinsics_buff is None, appending ext.: {}".format(self.current_cam2world))
        intrinsic = self.fetch_intrinsic()
        depth_imgs = self.g3d_depth_buff
        extrinsics = self.g3d_extrinsics_buff
        frame_size = extrinsics.shape[0]
        self.reset_depthpose_buff()
        grid_data = self.gen_grid(depth_imgs, intrinsic, extrinsics, grid_type)
        rospy.loginfo("[Sensor]: issuing {} with buff size: {}".format(grid_type, frame_size))
        self.issue_grid(grid_data)

        if self.show_grid_with_issue:
            Visualization.visualize_grid(grid_data, 'grid_ws', self.grid_length, self.voxel_disc)
            rospy.loginfo("[Sensor]: visualize buffed {} data".format(grid_type))
        return grid_data


class SensorClient:
    def __init__(self, voxel_disc, grid_topic, grid_request_topic):
        self.grid = None
        self.voxel_disc = voxel_disc
        self.request_grid_pub = rospy.Publisher(grid_request_topic, Float32MultiArray, queue_size=1)
        rospy.Subscriber(grid_topic, Float32MultiArray, self.grid_cb)


    def grid_cb(self, msg):
        self.grid = np.array(msg.data).astype(np.float32).reshape(self.voxel_disc, self.voxel_disc, 
                                                                  self.voxel_disc)
        # rospy.loginfo("[Inference]: Received grid msg: {}".format(self.grid.shape))


    def request_grid_cmd(self, cmd, pose = None):
        self.grid = None
        if cmd == "clear":
            issue_buff_signal = [0]
        elif cmd == "tsdf":
            issue_buff_signal = [1]
        elif cmd == "voxel":
            issue_buff_signal = [2]
        elif cmd == "pose": 
            issue_buff_signal = pose
        elif cmd == "single_tsdf":
            issue_buff_signal = [3]
        elif cmd == "single_voxel":
            issue_buff_signal = [4]
        else:
            raise KeyError("wrong grid request key: {}".format(cmd))
        self.request_grid_pub.publish(Float32MultiArray(data=issue_buff_signal))
    
    def await_grid(self):
        while self.grid is None: rospy.sleep(0.1)
        return self.grid


###
### Sensor Server Instance
###
if __name__ == '__main__':
    from spahybgen.pipeline.param_server import GraspParameter
    GP = GraspParameter("./config/grasp_generation.yaml")

    rospy.init_node("sensor_server")
    rospy.loginfo("[Sensor]: Started sensor_server node.")

    azure_node = SensorServer(
        GP.sensor.voxel_disc, 
        GP.sensor.grid_length, 
        GP.sensor.depth_topic, 
        GP.sensor.caminfo_topic, 
        GP.sensor.grid_topic, 
        GP.sensor.grid_request_topic, 
        GP.sensor.visualize_grid
    )
    
    rospy.loginfo("[Sensor]: SensorNode Created and Spinning.")
    rospy.spin()
    rospy.loginfo("[Sensor]: SensorNode Finished.")


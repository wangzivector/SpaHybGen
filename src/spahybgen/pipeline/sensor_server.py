#!/usr/bin/env python
from typing import Union, List
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


class SensorServer:
    """
    Azure RGBD ROS Node for grid
    Args:
        voxel_disc: discretization of the output grid
        grid_length: length of the output grid in meters
        depth_topic: ROS topic for depth image
        caminfo_topic: ROS topic for camera info
        grid_topic: ROS topic for publishing the generated grid
        grid_request_topic: ROS topic for receiving grid generation requests
        visualize_grid: whether to visualize the generated grid in rviz
    """

    def __init__(
        self,
        voxel_disc: int,
        grid_length: float,
        depth_topic: str,
        caminfo_topic: str,
        grid_topic: str,
        grid_request_topic: str,
        visualize_grid: bool,
    ) -> None:
        self.voxel_disc = voxel_disc
        self.grid_length = grid_length
        self.show_grid_with_issue = visualize_grid
        self.current_cam2world = [0, 0, 0, 0, 0, 0]

        self.depth_buff = None
        rospy.Subscriber(depth_topic, Image, self.depth_callback)
        rospy.loginfo("[Sensor]: ROS subcriping to {}".format(depth_topic))

        self.caminfo_buff = None
        rospy.Subscriber(caminfo_topic, CameraInfo, self.caminfo_callback)
        rospy.loginfo("[Sensor]: ROS subcriping to {}".format(caminfo_topic))

        self.grid_request = rospy.Subscriber(
            grid_request_topic, Float32MultiArray, self.grid_request_callback
        )
        rospy.loginfo("[Sensor]: ROS subcriping to {}".format(grid_request_topic))

        self.grid_pub = rospy.Publisher(grid_topic, Float32MultiArray, queue_size=1, latch=True)
        self.reset_depthpose_buff()

    def reset_depthpose_buff(self) -> None:
        """reset the buffer for depth images and extrinsics used for grid generation"""
        self.g3d_extrinsics_buff = None
        self.g3d_depth_buff = None

    def append_depthpose_buff(self, cam2world_k: Union[list, np.ndarray]) -> None:
        """append depth image and extrinsics to the buffer for grid generation
        Args:
            cam2world_k: the extrinsics of the current depth image, in the form of [x, y, z, roll, pitch, yaw]
        """
        extrinsics = np.expand_dims(self.convert_extrinsics(cam2world_k), axis=0)
        depth_image = np.expand_dims(self.fetch_depth_image(), axis=0)

        if self.g3d_extrinsics_buff is None:
            self.g3d_extrinsics_buff = extrinsics
        else:
            self.g3d_extrinsics_buff = np.vstack((self.g3d_extrinsics_buff, extrinsics))

        if self.g3d_depth_buff is None:
            self.g3d_depth_buff = depth_image
        else:
            self.g3d_depth_buff = np.vstack((self.g3d_depth_buff, depth_image))

    def grid_request_callback(self, msg):
        """callback function for receiving grid generation requests"""
        data = np.array(msg.data)
        if len(data) == 1:
            if data.mean() == 0:
                rospy.loginfo("[Sensor]: clear buff by msg: {}".format(data))
                self.reset_depthpose_buff()
            elif data.mean() == 1:
                rospy.loginfo("[Sensor]: add tsdf buff by msg: {}".format(data))
                self.issue_grid_using_buff(grid_type="tsdf")
            elif data.mean() == 2:
                rospy.loginfo("[Sensor]: add voxel buff by msg: {}".format(data))
                self.issue_grid_using_buff(grid_type="voxel")
            elif data.mean() == 3:
                rospy.loginfo("[Sensor]: issue single tsdf by msg: {}".format(data))
                self.issue_grid_using_buff(grid_type="tsdf")
            elif data.mean() == 4:
                rospy.loginfo("[Sensor]: issue single voxel by msg: {}".format(data))
                self.issue_grid_using_buff(grid_type="voxel")
        else:
            rospy.loginfo("Buffing depth with pose data: \n{}".format(data))
            self.append_depthpose_buff(cam2world_k=data)
            self.current_cam2world = data

    def depth_callback(self, msg):
        """callback function for receiving depth images"""
        self.depth_buff = msg

    def caminfo_callback(self, msg):
        """callback function for receiving camera info"""
        self.caminfo_buff = msg

    def fetch_depth_image(self) -> np.ndarray:
        """obtain depth image from the buffer, inpaint missing values and convert to meters if necessary

        Returns:
           depth_image: the processed depth image in numpy array format
        """
        ## Depth Image
        while self.depth_buff is None:
            rospy.loginfo("[Sensor]: depth_image buff is None, retrying...")
            rospy.sleep(1.0)

        depth_image = ros_numpy.image.image_to_numpy(self.depth_buff).astype(np.float32)
        depth_image = np.nan_to_num(depth_image)
        # rospy.loginfo('Depth_image size: {}'.format(depth_image.shape))
        depth_image = self.depth_inpaint(depth_image, missing_value=0)
        if depth_image.mean() > 1:
            depth_image = depth_image / 1000.0
        return depth_image

    @staticmethod
    def depth_inpaint(image, missing_value: int = 0) -> np.ndarray:
        """
        Inpaint missing values in depth image

        Args:
            image: the input depth image in numpy array format
            missing_value: value to fill in the depth image

        Returns:
            the inpainted depth image in numpy array format
        As noted in reference, cv2 inpainting doesn't handle the border properly
        https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        """

        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (image == missing_value).astype(np.uint8)
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        imax, imin = np.abs(image).max(), np.abs(image).min()
        irange = imax - imin
        image = ((image - imin) / irange).astype(np.float32)  # Has be float32, 64 not supported. get -1:1
        image = cv2.inpaint(image, mask, 2, cv2.INPAINT_NS)  # repair with fluid alg. radius 1
        # Back to original size and value range.
        image = image[1:-1, 1:-1]  # cut the 1 pixel boarder
        image = image.astype(np.float32) * irange + imin
        return image

    def fetch_intrinsic(self) -> CameraIntrinsic:
        """obtain camera intrinsic from the buffer and convert to CameraIntrinsic

        Returns:
            intrinsics: the camera intrinsic parameters in CameraIntrinsic dataclass format
        """
        ## Camera Info
        while self.caminfo_buff is None:
            rospy.loginfo("[Sensor]: depth_image buff is None, retrying...")
            rospy.sleep(1.0)
        else:
            cam_info = self.caminfo_buff
        fx, fy, cx, cy = cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5]
        width, height = cam_info.width, cam_info.height
        intrinsics = CameraIntrinsic(width, height, fx, fy, cx, cy)
        return intrinsics

    def convert_extrinsics(self, cam2world_k: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
        """convert extrinsics from [x, y, z, roll, pitch, yaw] to 4x4 transformation matrix

        Args:
            cam2world_k: the extrinsics of the current depth image, in the form of [x, y, z, roll, pitch, yaw]

        Returns:
            extrinsics: the extrinsics in 4x4 transformation matrix format
        """
        cam2world_k_tranf = Transform.from_list_transrotvet(cam2world_k)
        extrinsics = cam2world_k_tranf.inverse().to_list()
        return extrinsics

    def fetch_single_grid(self, grid_type: str) -> np.ndarray:
        """fetch a single grid without using the buffer, which will directly use the current depth image
        and extrinsics for grid generation. This is used for the case when we only want to generate one grid without
        buffering multiple frames.
        Args:
            grid_type: type of the grid to be generated, can be "tsdf" or "voxel"
        Returns:
            grid_data: the generated grid data in numpy array format
        """
        intrinsic = self.fetch_intrinsic()
        depth_imgs = np.expand_dims(self.fetch_depth_image(), axis=0)
        extrinsics = np.expand_dims(self.convert_extrinsics(self.current_cam2world), axis=0)
        return self.gen_grid(depth_imgs, intrinsic, extrinsics, grid_type)

    def gen_grid(
        self, depth_imgs: np.ndarray, intrinsic: CameraIntrinsic, extrinsics: np.ndarray, grid_type: str
    ) -> np.ndarray:
        """generate grid data from depth images, camera intrinsic and extrinsics
        Args:
            depth_imgs: the input depth images in numpy array format, with shape (N, H, W)
            intrinsic: the camera intrinsic parameters in CameraIntrinsic dataclass format
            extrinsics: the extrinsics of the depth images, in 4x4 transformation matrix format, with shape (N, 4, 4)
            grid_type: type of the grid to be generated, can be "tsdf" or "voxel"
        Returns:
            grid_data: the generated grid data in numpy array format, with shape (voxel_disc, voxel_disc, voxel_disc)
        """
        if grid_type == "tsdf":
            tsdf = ObsEng.create_tsdf(
                self.grid_length, self.voxel_disc, depth_imgs, intrinsic, extrinsics, trunc=8
            )
            grid_data = tsdf.get_grid()
        elif grid_type == "voxel":
            voxel = ObsEng.create_voxel(self.grid_length, self.voxel_disc, depth_imgs, intrinsic, extrinsics)
            grid_data = voxel.get_grid()
        return grid_data

    def issue_grid(self, grid_data: np.ndarray) -> None:
        """publish the generated grid data to the ROS topic

        Args:
            grid_data: the generated grid data in numpy array format, with shape (voxel_disc, voxel_disc, voxel_disc)
        """
        msg = Float32MultiArray(data=grid_data.astype(np.float32).reshape(-1))
        self.grid_pub.publish(msg)

    def issue_grid_using_buff(self, grid_type: str):
        """generate and publish grid data using the buffered depth images and extrinsics,
        which allows us to use multiple frames of data for grid generation, potentially improving the quality of the generated grid.

        Args:
            grid_type: type of the grid to be generated, can be "tsdf" or "voxel"

        Returns:
            grid_data: the generated grid data in numpy array format, with shape (voxel_disc, voxel_disc, voxel_disc)
        """
        if self.g3d_extrinsics_buff is None:
            self.append_depthpose_buff(self.current_cam2world)
            rospy.logwarn("extrinsics_buff is None, appending ext.: {}".format(self.current_cam2world))
        intrinsic = self.fetch_intrinsic()
        depth_imgs = self.g3d_depth_buff
        extrinsics = self.g3d_extrinsics_buff
        assert extrinsics is not None, "extrinsics_buff is None, cannot issue grid using buff"
        frame_size = extrinsics.shape[0]
        self.reset_depthpose_buff()
        assert depth_imgs is not None, "depth_buff is None, cannot issue grid using buff"
        grid_data = self.gen_grid(depth_imgs, intrinsic, extrinsics, grid_type)
        rospy.loginfo("[Sensor]: issuing {} with buff size: {}".format(grid_type, frame_size))
        self.issue_grid(grid_data)

        if self.show_grid_with_issue:
            Visualization.visualize_grid(grid_data, "grid_ws", self.grid_length, self.voxel_disc)
            rospy.loginfo("[Sensor]: visualize buffed {} data".format(grid_type))
        return grid_data


class SensorClient:
    def __init__(self, voxel_disc: int, grid_topic: str, grid_request_topic: str) -> None:
        """Client for receiving grid data from SensorServer and requesting grid generation

        Args:
            voxel_disc: discretization of the output grid
            grid_topic: ROS topic for receiving the generated grid
            grid_request_topic: ROS topic for requesting grid generation
        """
        self.grid = None
        self.voxel_disc = voxel_disc
        self.request_grid_pub = rospy.Publisher(grid_request_topic, Float32MultiArray, queue_size=1)
        rospy.Subscriber(grid_topic, Float32MultiArray, self.grid_cb)

    def grid_cb(self, msg):
        """Callback function for receiving grid data from SensorServer"""
        self.grid = (
            np.array(msg.data).astype(np.float32).reshape(self.voxel_disc, self.voxel_disc, self.voxel_disc)
        )
        # rospy.loginfo("[Inference]: Received grid msg: {}".format(self.grid.shape))

    def request_grid_cmd(self, cmd: str, pose: Union[list, None] = None) -> None:
        """Request grid generation from SensorServer by publishing a command to the grid_request_topic
        Args:
            cmd: command for grid generation, can be "clear", "tsdf", "voxel", "pose", "single_tsdf" or "single_voxel"
            pose: the extrinsics of the current depth image, only used when cmd is "pose"
        """
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
        """wait for the grid data to be received from SensorServer and return it"""
        while self.grid is None:
            rospy.sleep(0.1)
        return self.grid


###
### Sensor Server Instance
###
if __name__ == "__main__":
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
        GP.sensor.visualize_grid,
    )

    rospy.loginfo("[Sensor]: SensorNode Created and Spinning.")
    rospy.spin()
    rospy.loginfo("[Sensor]: SensorNode Finished.")

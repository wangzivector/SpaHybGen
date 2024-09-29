# Inherent from [VGN](https://github.com/ethz-asl/vgn)

from math import cos, sin
import time

import numpy as np
import open3d as o3d

from spahybgen.utils.utils_trans_np import Transform


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution, origin=np.array([0.0, 0.0, 0.0]), trunc=4):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = trunc * self.voxel_size
        self.origin = origin

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
            origin = self.origin
        )

    def integrate(self, depth_img, intrinsic, extrinsic):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        extrinsic = extrinsic.as_matrix()

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        cloud = self._volume.extract_voxel_point_cloud()
        points = np.asarray(cloud.points)
        distances = np.asarray(cloud.colors)[:, [0]]
        grid = np.zeros((1, self.resolution, self.resolution, self.resolution), dtype=np.float32)
        for idx, point in enumerate(points):
            i, j, k = np.floor(point / self.voxel_size).astype(int)
            grid[0, i, j, k] = distances[idx]
        return grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()


def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics, trunc=4):
    tsdf = TSDFVolume(size, resolution, trunc=trunc)
    for i in range(depth_imgs.shape[0]):
        extrinsic = Transform.from_list(extrinsics[i])
        tsdf.integrate(depth_imgs[i], intrinsic, extrinsic)
    return tsdf


class VoxelVolume(object):
    """Integration of multiple depth images for Voxel."""

    def __init__(self, size, resolution, origin=np.array([0.0, 0.0, 0.0])):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.origin = origin

        # setup dense voxel grid
        self.voxel_carving = o3d.geometry.VoxelGrid.create_dense(
            width=self.size,
            height=self.size,
            depth=self.size,
            voxel_size=self.size / self.resolution,
            origin=self.origin,
            color=[1.0, 1.0, 1.0])
        

    def integrate(self, depth_img, intrinsic, extrinsic):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )
        extrinsic = extrinsic.as_matrix()
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = intrinsic
        param.extrinsic = extrinsic

        self.voxel_carving.carve_depth_map(o3d.geometry.Image(depth_img), param)

    def get_grid(self):
        voxels = self.voxel_carving.get_voxels()
        grid = np.zeros((1, self.resolution, self.resolution, self.resolution), dtype=np.float32)
        for voxel in voxels:
            i, j, k = voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2]
            grid[0, i, j, k] = voxel.color[0]
        return grid
    

def create_voxel(size, resolution, depth_imgs, intrinsic, extrinsics):
    voxel = VoxelVolume(size, resolution)
    for i in range(depth_imgs.shape[0]):
        extrinsic = Transform.from_list(extrinsics[i])
        voxel.integrate(depth_imgs[i], intrinsic, extrinsic)
    return voxel


class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic


def depth_inpaint(image, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in the depth image.
    """
    # cv2 inpainting doesn't handle the border properly
    # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
    import cv2
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
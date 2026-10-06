from math import cos, sin

import numpy as np
import open3d as o3d

from grasp_env.simulator.transform import Transform


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
            "fx": self.K.flatten()[0],
            "fy": self.K.flatten()[4],
            "cx": self.K.flatten()[2],
            "cy": self.K.flatten()[5],
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


class TSDFVolume(object):

    def integrate(self, depth_img, intrinsic, extrinsic, rgb_img=None):
        """
        Args:
            depth_img: The depth image (as numpy array).
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsic: The transform from TSDF/world to camera coordinates.
        """
        color = (
            np.empty((*depth_img.shape, 3), dtype=np.uint8)
            if rgb_img is None
            else np.ascontiguousarray(rgb_img)
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
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
        self._povs.append(extrinsic.inverse().as_matrix())
        extrinsic = extrinsic.as_matrix()
        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_cloud(self) -> o3d.geometry.PointCloud:
        return self._volume.extract_point_cloud()

    def get_mesh(self):
        return self._volume.extract_triangle_mesh()


class UniformTSDFVolume(TSDFVolume):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size=0.3, resolution=120, color_type=None):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=color_type or o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )
        self._povs = list()

    def index_of(self, x, y, z):
        return x * self._volume.resolution * self._volume.resolution + y * self._volume.resolution + z


class ScalableTSDFVolume(TSDFVolume):
    """Integration of multiple depth images using a Scalable TSDF."""

    def __init__(self, voxel_size=0.0025):
        self.voxel_size = voxel_size
        self.sdf_trunc = 4 * self.voxel_size  # Same idea: truncation distance

        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )
        self._povs = list()



def camera_on_sphere(origin, radius, theta, phi):
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()


def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics, rgb_imgs=None):
    color_type = (
        o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        if rgb_imgs is None
        else o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    tsdf = UniformTSDFVolume(size, resolution, color_type=color_type)
    for i in range(len(depth_imgs)):
        extrinsic = Transform.from_list(extrinsics[i])
        rgb_img = None if rgb_imgs is None else rgb_imgs[i]
        tsdf.integrate(depth_imgs[i], intrinsic, extrinsic, rgb_img=rgb_img)
    return tsdf

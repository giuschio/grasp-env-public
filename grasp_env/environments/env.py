from math import cos, sin
from time import sleep

import numpy as np
import tkinter as tk
import open3d as o3d


from grasp_env.simulator.grasp import Grasp, Label
from grasp_env.simulator.perception import camera_on_sphere
from grasp_env.simulator.simulation_clutter_bandit import ClutterRemovalSim
from grasp_env.simulator.transform import Rotation, Transform
from grasp_env.simulator.perception import TSDFVolume, ScalableTSDFVolume, UniformTSDFVolume

from typing import Optional


def get_color(name: str, alpha: int = 255):
    """
    Convert a Tk color name or "random" into a PyBullet-compatible (R, G, B, A)
    tuple with float components in [0.0, 1.0].

    :param name: any Tk color string or "random"
    :param alpha: alpha channel in 0–255, default 255
    :return: (r, g, b, a) all floats in [0.0, 1.0]
    """
    import random
    if name == "random":
        r, g, b = (random.randint(0, 255) for _ in range(3))
    else:
        root = tk.Tk()
        root.withdraw()
        r16, g16, b16 = root.winfo_rgb(name)
        root.destroy()
        r, g, b = (v // 256 for v in (r16, g16, b16))
    
    # Normalize to [0.0, 1.0]
    return (r / 255.0, g / 255.0, b / 255.0, alpha / 255.0)


class GraspEnv:
    def __init__(
        self,
        seed: int,
        scene_type: str = "packed",
        object_set: str = "vgn-packed-test",
        randomize_object_scale: bool = True,
        show_gui: bool = False,
        output_video: str = None,
        asset_catalog_path: str = None,
        assets_root: str = None,
    ) -> None:

        # Internal simulator from VGN
        self.sim = ClutterRemovalSim(
            seed=seed,
            scene_type=scene_type,
            object_set=object_set,
            randomize_object_scale=randomize_object_scale,
            gui=show_gui,
            rand=True,  # use random object orientations
            sort_file_list=True,  # more reproducible
            output_video=output_video,
            asset_catalog_path=asset_catalog_path,
            assets_root=assets_root,
        )
        self.show_gui = show_gui
        self._visualization_markers = []
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

    def close(self):
        if getattr(self, "sim", None) is not None:
            self.sim.close()
            self.sim = None

    def saveBullet(self, fpath):
        self.sim.world.p.saveBullet(fpath)

    def loadBullet(self, fpath):
        self.sim.world.p.loadBullet(fpath)
        self.sim.world.step()

    def reset(self, object_count, index=None, seed=None):
        self.sim.reset(object_count=object_count, index=index, seed=seed)
        return True

    def get_object_poses(self):
        return self.sim.get_object_poses()

    def step(
        self,
        pose: np.ndarray,
        noise: Optional[dict] = None,
        width_pre: Optional[np.ndarray] = None,
        remove_grasped_object=True,
        keep_gripper=True,
    ) -> dict:
        """Evaluate grasp robustness by executing the grasp multiple times with noise.

        Args:
            pose (np.ndarray): 4x4 homogeneous transformation matrix of the grasp.
            noise (dict): Std dev of Gaussian noise for ['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']. Rotations are in degrees
            width_pre (np.ndarray, optional): Predefined grasp width.
            remove_grasped_object (bool): Whether to remove object on successful grasp.
            keep_gripper (bool): Whether to keep the hand constrained at its final pose.

        Returns:
            dict: {
                'success_rate': float,
                'width_average': float,
                'trials': List[Dict]
            }
        """
        base_transform = Transform.from_matrix(pose)
        # safety factor
        width = min(width_pre + 0.03, self.sim.gripper.max_opening_width) if width_pre is not None else self.sim.gripper.max_opening_width
        
        if noise is not None:
            # Sample noise
            trans_noise = self._rng.normal(
                loc=0.0,
                scale=[noise.get(k, 0) for k in ["t_x", "t_y", "t_z"]]
            )
            rot_noise = self._rng.normal(
                loc=0.0,
                scale=[noise.get(k, 0) for k in ["r_x", "r_y", "r_z"]]
            )

            # Apply noise in the grasp frame
            noise_transform = Transform.from_6dofs(np.r_[trans_noise, rot_noise], degrees=True)
            noisy_transform = base_transform * noise_transform
            candidate = Grasp(noisy_transform, width=width)
        else:
            candidate = Grasp(base_transform, width=width)

        # # if the grasp is slightly in collision with the table, fix it by raising it
        # todo: THERE SHOULD BE AN EXTERNAL STEP POF GRASP REFINEMENT
        # table_height = self.sim.lower[2] + 0.015
        # p1_margin = table_height - candidate.point1[2]
        # p2_margin = table_height - candidate.point2[2]
        # margin = min(p1_margin, p2_margin)
        # if margin > 0.: candidate.pose[3, 2] += margin

        outcome, result_width, description, collision = self.sim.execute_grasp_quick(
            candidate,
            allow_contact=True,
            remove_grasped_object=remove_grasped_object,
            keep_gripper=keep_gripper,
        )

        success = 1 if outcome == Label.SUCCESS else 0

        return {
            "success": success,
            "width": result_width,
        }
    
    def step_robust(self, pose: np.ndarray, noise: dict, num_trials : int = 10,  width_pre: Optional[np.ndarray] = None, remove_grasped_object=True) -> dict:
        """Evaluate grasp robustness by executing the grasp multiple times with noise.

        Args:
            pose (np.ndarray): 4x4 homogeneous transformation matrix of the grasp.
            noise (dict): Std dev of Gaussian noise for ['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']. Rotations are in degrees
            width_pre (np.ndarray, optional): Predefined grasp width.
            remove_grasped_object (bool): Whether to remove object on successful grasp.

        Returns:
            dict: {
                'success': float, # average success rate
                'width': float, # average width
            }
        """
                
        success = np.empty(num_trials, dtype=np.float32)
        width = np.empty(num_trials, dtype=np.float32)
        for i in range(num_trials):
            self.sim.save_state('robust_step_backup')
            res = self.step(
                pose,
                noise,
                width_pre,
                remove_grasped_object=False,
                keep_gripper=False,
            )
            success[i] = res["success"]
            width[i] = res["width"]
            self.sim.restore_state('robust_step_backup')
        return {
            "success": float(np.mean(success)),
            "width": float(np.mean(width)),
        }

    def render_images_fixed_viewpoints(self):
        height, width = (
            self.sim.camera.intrinsic.height,
            self.sim.camera.intrinsic.width,
        )
        origin = Transform(
            Rotation.identity(),
            np.r_[self.sim.workspace_size / 2, self.sim.workspace_size / 2, 0.0 + 0.25],
        )
        n_views = 5
        eyes = np.empty((n_views, 3), np.float32)
        extrinsics = np.empty((n_views, 7), np.float32)
        depth_imgs = np.empty((n_views, height, width), np.float32)
        rgb_imgs = np.empty((n_views, height, width, 3), np.uint8)
        indexes = [0, 1, 2, 3, 4]
        r_s = [0.60, 0.60, 0.60, 0.60, 0.60]
        theta_s = [-0.01, np.pi/6, np.pi/6, np.pi/6, np.pi/6]
        phi_s = [0., 0., np.pi/2, np.pi, 3*np.pi/2]
        for i, r, theta, phi in zip(indexes, r_s, theta_s, phi_s):
            extrinsic = camera_on_sphere(origin, r, theta, phi)

            imgs = self.sim.camera.render(extrinsic)
            extrinsics[i] = extrinsic.to_list()
            depth_imgs[i] = imgs['depth']
            rgb_imgs[i] = imgs['rgb']
            eye = np.r_[
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta),
            ]
            cam_pos = eye + origin.translation

            eyes[i] = cam_pos

        res = {
            "depth_imgs": depth_imgs,
            "rgb_imgs": rgb_imgs,
            "intrinsics": self.sim.camera.intrinsic,
            "extrinsics": extrinsics,
            "eye": eyes,
        }
        return res

    def render_images_top_down(self):
        height, width = (
            self.sim.camera.intrinsic.height,
            self.sim.camera.intrinsic.width,
        )
        origin = Transform(
            Rotation.identity(),
            np.r_[self.sim.workspace_size / 2, self.sim.workspace_size / 2, 0.0 + 0.25],
        )
        n_views = 1
        extrinsics = np.empty((n_views, 7), np.float32)
        depth_imgs = np.empty((n_views, height, width), np.float32)
        rgb_imgs = np.empty((n_views, height, width, 3), np.uint8)
        eyes = np.empty((n_views, 3), np.float32)

        r, theta, phi = 0.60, -0.01, 0.
        extrinsic = camera_on_sphere(origin, r, theta, phi)

        imgs = self.sim.camera.render(extrinsic)
        extrinsics[0] = extrinsic.to_list()
        depth_imgs[0] = imgs['depth']
        rgb_imgs[0] = imgs['rgb']
        eye = np.r_[
            r * sin(theta) * cos(phi),
            r * sin(theta) * sin(phi),
            r * cos(theta),
        ]
        cam_pos = eye + origin.translation
        eyes[0] = cam_pos
        res = {
            "depth_imgs": depth_imgs,
            "rgb_imgs": rgb_imgs,
            "intrinsics": self.sim.camera.intrinsic,
            "extrinsics": extrinsics,
            "eye": eyes,
        }
        return res
    
    def render_images_random_viewpoint(self):
        height, width = (
            self.sim.camera.intrinsic.height,
            self.sim.camera.intrinsic.width,
        )
        origin = Transform(
            Rotation.identity(),
            np.r_[self.sim.workspace_size / 2, self.sim.workspace_size / 2, 0.0 + 0.25],
        )
        n_views = 1
        extrinsics = np.empty((n_views, 7), np.float32)
        depth_imgs = np.empty((n_views, height, width), np.float32)
        rgb_imgs = np.empty((n_views, height, width, 3), np.uint8)

        r = self._rng.uniform(2, 2.5) * self.sim.workspace_size
        theta = self._rng.uniform(np.pi / 4, np.pi / 3)
        phi = self._rng.uniform(0.0, np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)

        imgs = self.sim.camera.render(extrinsic)
        extrinsics[0] = extrinsic.to_list()
        depth_imgs[0] = imgs['depth']
        rgb_imgs[0] = imgs['rgb']
        eye = np.r_[
            r * sin(theta) * cos(phi),
            r * sin(theta) * sin(phi),
            r * cos(theta),
        ]
        cam_pos = eye + origin.translation
        eyes = np.empty((n_views, 3), np.float32)
        eyes[0] = cam_pos
        res = {
            "depth_imgs": depth_imgs,
            "rgb_imgs": rgb_imgs,
            "intrinsics": self.sim.camera.intrinsic,
            "extrinsics": extrinsics,
            "eye": eyes,
        }
        return res

    def get_observation_images_fixed_viewpoints(self):
        return self.render_images_fixed_viewpoints()
    
    def get_observation_random_viewpoint(self):
        return self.render_images_random_viewpoint()

    def get_observation_top_down(self):
        return self.render_images_top_down()

    def get_observation_tsdf(self) -> TSDFVolume:
        return self.sim.acquire_tsdf(n_viewpoints=5)

    def get_observation_cloud(self) -> o3d.geometry.PointCloud:
        tsdf = self.sim.acquire_tsdf(n_viewpoints=5)
        cloud = tsdf.get_cloud()
        return cloud

    def get_observation_mesh(self, resolution=120, n_viewpoints=5):
        return self.get_observation_tsdf().get_mesh()

    @property
    def n_objects(self):
        return self.sim.num_objects

    def _create_cylinder_between(self, p_start, p_end, radius, color):
        """Create a visual-only cylinder in PyBullet between two points."""
        p_start = np.array(p_start)
        p_end = np.array(p_end)
        center = (p_start + p_end) / 2
        direction = p_end - p_start
        length = np.linalg.norm(direction)

        if length == 0:
            orientation = [0, 0, 0, 1]
        else:
            z_axis = np.array([0, 0, 1])
            axis = np.cross(z_axis, direction)
            angle = np.arccos(np.clip(np.dot(z_axis, direction) / length, -1.0, 1.0))
            if np.linalg.norm(axis) < 1e-6:
                orientation = [0, 0, 0, 1] if np.dot(z_axis, direction) > 0 else [1, 0, 0, 0]
            else:
                axis = axis / np.linalg.norm(axis)
                orientation = self.sim.world.p.getQuaternionFromAxisAngle(axis.tolist(), angle)

        visual_shape_id = self.sim.world.p.createVisualShape(
            shapeType=self.sim.world.p.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=color
        )

        marker_id = self.sim.world.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,     # No collisions ✅
            baseVisualShapeIndex=visual_shape_id,
            basePosition=center.tolist(),
            baseOrientation=orientation
        )

        return marker_id

    def _create_sphere_marker(self, position, radius, color):
        visual_shape_id = self.sim.world.p.createVisualShape(
            shapeType=self.sim.world.p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        marker_id = self.sim.world.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )
        return marker_id

    def create_grasp_markers(self, grasps, color="random", radius=0.005):
        """Create a U-shaped grasp marker made of 3 cylinders."""
        if self.show_gui is False: return
        for grasp in grasps:
            pose = grasp["pose"]
            p1 = np.array(grasp["point1"])
            p2 = np.array(grasp["point2"])
            z_dir = pose[:3, 2]  # z-direction from the pose

            p1_end = p1 - z_dir * 0.05
            p2_end = p2 - z_dir * 0.05

            color_values = get_color(color)
            side1 = self._create_cylinder_between(p1, p1_end, radius, color_values)
            side2 = self._create_cylinder_between(p2, p2_end, radius, color_values)
            bar = self._create_cylinder_between(p1_end, p2_end, radius, color_values)

            sphere1 = self._create_sphere_marker(p1.tolist(), radius=radius*1.5, color=color_values)
            sphere2 = self._create_sphere_marker(p2.tolist(), radius=radius*1.5, color=color_values)

            self._visualization_markers += [side1, side2, bar, sphere1, sphere2]

    def remove_grasp_markers(self):
        """Remove all visual markers from the scene."""
        for marker_id in self._visualization_markers:
            self.sim.world.p.removeBody(marker_id)
        self._visualization_markers = []

    def start_visualization_loop(self):
        while True:
            sleep(0.01)  # let GUI stay responsive

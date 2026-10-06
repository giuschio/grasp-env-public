import argparse

import numpy as np

from grasp_env.environments.env import GraspEnv
from grasp_env.environments.offline_visualization import visualize_scene_pcd
from grasp_env.environments.perception_utils import cloud_from_image_data


class TopDownGraspSampler:
    """Minimal top-down sampler without collision checking."""

    def __init__(self, seed=None, top_layer=0.02, max_width=0.08):
        self.rng = np.random.default_rng(seed)
        self.top_layer = top_layer
        self.max_width = max_width

    def __call__(self, cloud_objects, n_grasps=1):
        points = np.asarray(cloud_objects.points, dtype=np.float32)
        if len(points) < 2:
            return []

        top_points = points[points[:, 2] >= points[:, 2].max() - self.top_layer]
        grasps = []
        for _ in range(100 * n_grasps):
            if len(grasps) >= n_grasps:
                break
            grasp = self._sample_single_grasp(top_points)
            if grasp is not None:
                grasps.append(grasp)
        return grasps

    def _sample_single_grasp(self, points):
        if len(points) < 2:
            return None

        point1 = points[self.rng.integers(len(points))]
        offsets = points[:, :2] - point1[:2]
        distances = np.linalg.norm(offsets, axis=1)
        candidates = np.flatnonzero(
            (distances >= 0.01) & (distances <= self.max_width)
        )
        if len(candidates) == 0:
            return None

        point2 = points[self.rng.choice(candidates)]
        closing_direction = point2[:2] - point1[:2]
        width = float(np.linalg.norm(closing_direction))
        y_axis = np.array(
            [closing_direction[0] / width, closing_direction[1] / width, 0.0],
            dtype=np.float32,
        )
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        x_axis = np.cross(y_axis, z_axis)

        center = 0.5 * (point1 + point2)
        center += 0.01 * z_axis

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        pose[:3, 3] = center
        return {
            "pose": pose,
            "width": width,
            "point1": point1,
            "point2": point2,
            "sampler": "topdown",
        }


def main():
    parser = argparse.ArgumentParser(description="Minimal grasp-env observation example.")
    parser.add_argument("--object_set", default="vgn-pile-train")
    parser.add_argument("--scene_type", default="packed")
    parser.add_argument("--max_object_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = GraspEnv(
        seed=args.seed,
        scene_type=args.scene_type,
        object_set=args.object_set,
        randomize_object_scale=False,
        show_gui=True,
        output_video=None,
    )
    try:
        env.reset(object_count=args.max_object_count)
        sampler = TopDownGraspSampler(seed=args.seed)
        attempt = 0
        while env.sim.num_objects > 0:
            attempt += 1
            image_data = env.get_observation_random_viewpoint()
            pointcloud = cloud_from_image_data(
                image_data,
                lower_bounds=env.sim.lower - [0, 0, 0.04],
                upper_bounds=env.sim.upper,
                noise_level=0.0008,
            )
            if pointcloud is None:
                print(f"attempt={attempt}: observation produced no usable point cloud; retrying")
                continue

            print(
                f"attempt={attempt}, remaining_objects={env.sim.num_objects}, "
                f"pointcloud_points={len(pointcloud.points)}"
            )
            grasps = sampler(pointcloud, n_grasps=1)
            if not grasps:
                print(f"attempt={attempt}: could not sample a grasp; retrying")
                continue

            grasp = grasps[0]
            visualize_scene_pcd(
                pointcloud,
                [grasp],
                visualize_grasp_rotation=True,
            )

            result = env.step(grasp["pose"], width_pre=grasp["width"])
            print(
                f"attempt={attempt}, grasp_width={grasp['width']:.3f} m, "
                f"success={bool(result['success'])}, remaining_objects={env.sim.num_objects}"
            )

            if env.sim.num_objects > 0:
                env.sim.gripper.park()

        print(f"Scene cleared in {attempt} grasp attempts")
    finally:
        env.close()


if __name__ == "__main__":
    main()

import numpy as np
import os
import pybullet

from typing import List, Tuple
from scipy.spatial.transform import Slerp
try:
    from importlib.resources import files as pkg_resources
except ImportError:
    from importlib_resources import files as pkg_resources

from grasp_env.simulator import btsim
from grasp_env.simulator.grasp import Label
from grasp_env.simulator.perception import CameraIntrinsic, TSDFVolume, UniformTSDFVolume, ScalableTSDFVolume, camera_on_sphere
from grasp_env.simulator.transform import Rotation, Transform
from grasp_env.simulator.workspace_lines import workspace_lines
from grasp_env.simulator.object_models import ModelDatabase, get_preferred_rotation


def find_information(dictionary, key, debug_name):
    if key in dictionary:
        return dictionary[key]
    else:
        raise KeyError(f"Could not find {debug_name} information")

class ClutterRemovalSim(object):
    def __init__(self, seed, scene_type, object_set, randomize_object_scale, gui=True, rand=False, sort_file_list=False, output_video=None, asset_catalog_path=None, assets_root=None):
        assert scene_type in ["pile", "packed", "singulated"]
        self.env_urdfs = pkg_resources("grasp_env") / "assets/environment"
        # self.object_folder = pkg_resources("grasp_env") / "assets" / object_set
        self.scene_type = scene_type
        self.sort_file_list = sort_file_list
        self.object_db = ModelDatabase(
            object_set,
            catalog_path=asset_catalog_path,
            assets_root=assets_root,
        )
        self.rand = rand
        self.object_scales = 1.0
        self.gui = gui
        self.output_video = output_video
        self._rng = np.random.RandomState(seed)
        self.world = btsim.BtWorld(self.gui, output_video)
        self.gripper = Gripper(self.world)
        self.table_height = 0.05
        self.workspace_size = 0.3
        self.scale_bounds = (0.7, 1.0) if randomize_object_scale else (0.9, 0.9)
        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)
        self.turn_off_rendering_during_scene_generation = False
        self._saved_states = {}
        lx, ux = 0.02, self.workspace_size - 0.02
        ly, uy = 0.02, self.workspace_size - 0.02
        lz, uz = 0.055, self.workspace_size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]
        self.scene_object = None
        self.table_body = None

    def close(self):
        if getattr(self, "world", None) is not None:
            self.world.close()
            self.world = None

    @staticmethod
    def _set_body_asset_metadata(body, object_file):
        body.asset_path = str(object_file)
        body.asset_name = os.path.basename(object_file)

    @property
    def num_objects(self):
        cnt = 0
        for b in self.world.bodies.values():
            if b.body_type == "object": cnt += 1
        return cnt

    def save_state(self, state_key='default'):
        self._saved_states[state_key] = self.world.save_state()

    def restore_state(self, state_key='default'):
        self.world.restore_state(self._saved_states.pop(state_key))

    def rendering_on(self):
        self.world.p.configureDebugVisualizer(self.world.p.COV_ENABLE_RENDERING, 1)

    def rendering_off(self):
        self.world.p.configureDebugVisualizer(self.world.p.COV_ENABLE_RENDERING, 0)

    def reset(self, object_count, index=None, seed=None):
        if self.turn_off_rendering_during_scene_generation: self.rendering_off()
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        # self.draw_workspace()
        # self.world.p.configureDebugVisualizer(self.world.p.COV_ENABLE_GUI, 0)
        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        if seed is not None: self._rng = np.random.RandomState(seed)

        self.place_table()
        self.gripper.spawn()
        if self.scene_type == "pile":
            self.generate_pile(object_count)
        elif self.scene_type == "packed":
            self.generate_packed(object_count)
        elif self.scene_type == "singulated":
            self.generate_singulated()
        else:
            raise ValueError("Invalid scene argument")
        if self.turn_off_rendering_during_scene_generation: self.rendering_on()
        return self.num_objects

    def place_table(self):
        urdf = self.env_urdfs / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, self.table_height])
        self.table_body = self.world.load_urdf(urdf, pose, scale=0.6, body_type="table")

    def place_box(self):
        urdf = self.env_urdfs / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, self.table_height])
        self.box = self.world.load_urdf(urdf, pose, scale=1.3, body_type="table")

    def remove_box(self):
        self.world.remove_body(self.box)

    def generate_pile(self, object_count):
        # place box
        self.place_box()
        object_indexes = self._rng.choice(len(self.object_db), size=object_count)
        
        for idx in object_indexes:
            object_file = self.object_db.get_file(idx)
            preferred_orientation = self.object_db.get_canonical_axis(idx)
            relative_scale = self.object_db.get_scale(idx)

            if self.rand:
                rotation = Rotation.random(random_state=self._rng)
            else:
                rotation = Rotation.from_matrix(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
            # xy = self._rng.uniform(1.0 / 3.0 * self.table_size, 2.0 / 3.0 * self.table_size, 2)
            xy = self._rng.uniform(
                self.workspace_size/2.0 - 1.0/6.0*self.workspace_size,
                self.workspace_size/2.0 + 1.0/6.0*self.workspace_size,
                2
            )
            initial_pose = Transform(rotation, np.r_[xy, self.table_height + 0.2])
            alignment_rot = get_preferred_rotation(preferred_orientation)
            initial_pose.rotation = initial_pose.rotation * alignment_rot

            relative_scale = None if relative_scale is None else relative_scale * self._rng.uniform(*self.scale_bounds)

            if os.path.splitext(object_file)[1] == ".urdf":
                body = self.world.load_urdf(object_file, initial_pose, scale=relative_scale, body_type="object")
            else:
                # assume it's an obj file
                body = self.world.load_obj(
                    obj_path=object_file, 
                    relative_scale=relative_scale,
                    initial_pose=initial_pose,
                    body_type="object")
            self._set_body_asset_metadata(body, object_file)
            self.wait_for_objects_to_rest(timeout=1.0)
        # remove box
        self.remove_box()
        self.remove_and_wait()

    def generate_singulated(self):
        self.save_state()
        idx = self._rng.choice(len(self.object_db))

        x = self._rng.uniform(0.08, 0.22)
        y = self._rng.uniform(0.08, 0.22)
        z = 1.0
        angle = self._rng.uniform(0.0, 2.0 * np.pi)
        rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
        initial_pose = Transform(rotation, np.r_[x, y, z])

        object_file = self.object_db.get_file(idx)
        preferred_orientation = self.object_db.get_canonical_axis(idx)
        relative_scale = self.object_db.get_scale(idx)

        alignment_rot = get_preferred_rotation(preferred_orientation)
        initial_pose.rotation = initial_pose.rotation * alignment_rot

        relative_scale = None if relative_scale is None else relative_scale * self._rng.uniform(*self.scale_bounds)

        if os.path.splitext(object_file)[1] == ".urdf":
            body = self.world.load_urdf(object_file, initial_pose, scale=relative_scale, body_type="object")
        else:
            # assume it's an obj file
            body = self.world.load_obj(
                obj_path=object_file, 
                relative_scale=relative_scale,
                initial_pose=initial_pose,
                body_type="object")
        self._set_body_asset_metadata(body, object_file)
            
        lower, upper = self.world.p.getAABB(body.uid)
        z = self.table_height + 0.5 * (upper[2] - lower[2]) + 0.002

        body.set_pose(pose=Transform(initial_pose.rotation, np.r_[x, y, z]))
        self.scene_object = object_file
        self.remove_and_wait()


    def generate_packed(self, object_count):
        attempts = 0
        max_attempts = 100
        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            idx = self._rng.choice(len(self.object_db))

            x = self._rng.uniform(0.08, 0.22)
            y = self._rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self._rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            initial_pose = Transform(rotation, np.r_[x, y, z])

            object_file = self.object_db.get_file(idx)
            preferred_orientation = self.object_db.get_canonical_axis(idx)
            relative_scale = self.object_db.get_scale(idx)

            alignment_rot = get_preferred_rotation(preferred_orientation)
            initial_pose.rotation = initial_pose.rotation * alignment_rot

            relative_scale = None if relative_scale is None else relative_scale * self._rng.uniform(*self.scale_bounds)

            if os.path.splitext(object_file)[1] == ".urdf":
                body = self.world.load_urdf(object_file, initial_pose, scale=relative_scale, body_type="object")
            else:
                # assume it's an obj file
                body = self.world.load_obj(
                    obj_path=object_file, 
                    relative_scale=relative_scale,
                    initial_pose=initial_pose,
                    body_type="object")
            self._set_body_asset_metadata(body, object_file)
            lower, upper = self.world.p.getAABB(body.uid)
            z = self.table_height + 0.5 * (upper[2] - lower[2]) + 0.002

            body.set_pose(pose=Transform(initial_pose.rotation, np.r_[x, y, z]))
            self.world.step()
            # self.wait_for_objects_to_rest()
            contacts = self.world.get_contacts(body)
            # exclude contact with the table
            contacts = [c for c in contacts if not "plane" in [c.bodyA.name, c.bodyB.name]]
            if contacts:
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1

    def get_object_poses(self) -> List[Tuple[str, np.ndarray]]:
        object_poses = []
        for body in self.world.bodies.values():
            if body.body_type != "object":
                continue
            asset_name = getattr(body, "asset_name", body.name)
            object_poses.append((asset_name, body.get_pose().as_matrix()))
        return object_poses

    def acquire_tsdf(self, n_viewpoints=5) -> TSDFVolume:
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.
        If N is None, the n viewpoints are equally distributed on circular trajectory.
        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        # open3d TSDFs are positioned with a bounding box defined by origin + size
        # tsdf = UniformTSDFVolume(size=self.workspace_size, resolution=resolution)
        tsdf = UniformTSDFVolume()

        origin = Transform(Rotation.identity(), np.r_[self.workspace_size / 2, self.workspace_size / 2, 0])
        r = 1.5 * self.workspace_size
        theta = np.pi / 6.0

        phi_list = 2.0 * np.pi * np.arange(n_viewpoints) / n_viewpoints
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]
        for extrinsic in extrinsics:
            depth_img = self.camera.render(extrinsic)['depth']
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)

        return tsdf

    def rotate_gripper(self, theta, eef_step=0.05, vel=0.80, axis="z"):
        # eef_step=0.05, vel=0.40,
        T_world_grasp = self.gripper.get_T_world_grasp()
        # pre_position = T_world_tcp.translation
        diff = theta
        n_step = int(abs(theta) / eef_step)
        if n_step == 0:
            n_step = 1  # avoid divide by zero
        dist_step = diff / n_step
        dur_step = abs(dist_step) / vel
        for _ in range(n_step):
            T_world_grasp = T_world_grasp * Transform(Rotation.from_euler(axis, -dist_step), [0.0, 0.0, 0.0])
            self.gripper.update_graspframe_constraint(T_world_grasp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()

    def advance_sim(self, frames):
        for _ in range(frames):
            self.world.step()

    def gripper_dance(self, n_rotations=9):
        yaw = np.pi / np.float(n_rotations)
        for _ in range(n_rotations):
            self.rotate_gripper(yaw, axis="y")

    def get_closest_body(self, point):
        p = self.world.p
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=1e-3)
        test_body = p.createMultiBody(baseCollisionShapeIndex=collision_shape, basePosition=point)

        closest = None
        min_dist = float("inf")

        for body_id in range(p.getNumBodies()):
            if body_id == test_body:
                continue
            pts = p.getClosestPoints(test_body, body_id, distance=1000.0)
            if pts:
                d = pts[0][8]  # contactDistance
                if d < min_dist:
                    min_dist = d
                    closest = body_id

        # Remove the test sphere
        p.removeBody(test_body)
        # print("Closest body:", closest, "distance:", min_dist)

    def execute_grasp_quick(
        self,
        grasp,
        remove_grasped_object=True,
        allow_contact=True,
        keep_gripper=True,
    ):
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        # self.get_closest_body(grasp.pose.translation)

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.clip(np.dot(approach, np.r_[0.0, 0.0, -1.0]), -1.0, 1.0))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset_graspframe_pose(T_world_pregrasp, grasp.width)

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width, "pregrasp", None
        else:
            contacts = self.gripper.move_graspframe_xyz(T_world_grasp, abort_on_contact=False)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width, "grasp", contacts
            else:
                self.gripper.move_fingers(0.0)
                # check if we grasp the object
                if self.gripper.grasped_object is not None:
                    # Remove collision with the grasped object
                    bignore = [contact.bodyB for contact in self.gripper.grasped_object]
                    bignore += [contact.bodyA for contact in self.gripper.grasped_object]
                    # we have an attached object.
                    contacts = [c for c in contacts if not (c.bodyA in bignore and c.bodyB in bignore)]
                    contacts = [c for c in contacts if not "plane" in [c.bodyA.name, c.bodyB.name]]
                contacts += self.gripper.move_graspframe_xyz(T_world_retreat, abort_on_contact=False)
                contacts = list(set(contacts))

                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read(), "success", contacts
                    if remove_grasped_object:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width, "after", contacts

        # CLEANUP
        if not keep_gripper:
            self.gripper.remove()
        if remove_grasped_object: self.remove_and_wait()
        # RETURN
        return result

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60): self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if body.body_type != "object": continue
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            if body.body_type != "object": continue
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz[:2] > self.workspace_size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        contacts = [c for c in contacts if c.bodyA.name != "plane" and c.bodyB.name != "plane"]
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res


class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world):
        self.world: btsim.BtWorld = world
        self.urdf_path = pkg_resources("grasp_env") / "assets/robots/panda/hand.urdf"
        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_grasp = Transform(Rotation.identity(), [0.0, 0.0, 0.06])
        self.T_grasp_body = self.T_body_grasp.inverse()

        self.grasped_object = None

    def spawn(self):
        # ---- NEW: load the hand **once** ----
        # spawn it at some out‐of‐the‐way pose; we'll teleport it later
        offscreen = Transform(Rotation.identity(), [0, 0, -0.5])
        self.body = self.world.load_urdf(self.urdf_path, offscreen, scale=1.0, body_type="robot")
        # placeholder constraints
        self.origin_constraint = None
        self.finger_constraint = None

    def remove(self):
        # we no longer destroy the URDF body here—just tear down its constraints
        if self.origin_constraint is not None:
            self.world.remove_constraint(self.origin_constraint)
            self.origin_constraint = None
        if self.finger_constraint is not None:
            self.world.remove_constraint(self.finger_constraint)
            self.finger_constraint = None
        self.grasped_object = None

    def park(self):
        """Move the unconstrained hand outside the workspace."""
        self.remove()
        offscreen = Transform(Rotation.identity(), [0.0, 0.0, -0.5])
        self.body.set_pose(offscreen)
        self.world.p.resetBaseVelocity(
            self.body.uid,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )

    def reset_graspframe_pose(self, T_world_grasp, opening_width=0.08):
        # teleport the hand
        T_world_body = T_world_grasp * self.T_grasp_body
        self.body.set_pose(T_world_body)

        pybullet.changeDynamics(self.body.uid, 0, lateralFriction=0.75, spinningFriction=0.05)
        pybullet.changeDynamics(self.body.uid, 1, lateralFriction=0.75, spinningFriction=0.05)

        # remove old constraints
        if self.origin_constraint:
            self.world.remove_constraint(self.origin_constraint)
        if self.finger_constraint:
            self.world.remove_constraint(self.finger_constraint)

        # sets up a new fixed‐joint that pins the hand at T_world_grasp
        self.origin_constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_graspframe_constraint(T_world_grasp)
        # constraint to keep fingers centered
        self.finger_constraint = self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        )
        self.finger_constraint.change(gearRatio=-1, erp=0.1, maxForce=30)

        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * opening_width, kinematics=True)
        self.grasped_object = None

    def update_graspframe_constraint(self, T_world_grasp):
        T_world_body = T_world_grasp * self.T_grasp_body
        self.origin_constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def get_T_world_body(self):
        return self.body.get_pose()

    def get_T_world_grasp(self):
        return self.get_T_world_body() * self.T_body_grasp

    def move_graspframe_xyz(self, T_world_grasp_target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        gripper_contacts = []
        contacts = []
        T_world_grasp = self.get_T_world_grasp()

        diff = T_world_grasp_target.translation - T_world_grasp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_grasp.translation += dist_step
            self.update_graspframe_constraint(T_world_grasp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()

            contacts += self.world.get_contacts(self.body)

            # Check if attached object collides
            if self.grasped_object is not None:
                bignore = [contact.bodyB for contact in self.grasped_object]
                bignore += [contact.bodyA for contact in self.grasped_object]
                for body in bignore:
                    contacts += self.world.get_contacts(body)

                # we have an attached object.
                contacts = [c for c in contacts if not (c.bodyA in bignore and c.bodyB in bignore)]
                contacts = [c for c in contacts if not "plane" in [c.bodyA.name, c.bodyB.name]]

            gripper_contacts = self.world.get_contacts(self.body)
            if abort_on_contact and len(gripper_contacts) > 0:
                return contacts

        return contacts
    
    def move_graspframe_pose(self, T_world_grasp_target, eef_step1=0.002, vel1=0.10, abs=False):
        T_world_body = self.get_T_world_body()
        T_world_grasp = self.get_T_world_grasp()
        
        diff = T_world_grasp_target.translation - T_world_grasp.translation
        n_steps = max(int(np.linalg.norm(diff) / eef_step1), 10)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel1
        key_rots = np.stack((T_world_body.rotation.as_quat(), T_world_grasp_target.rotation.as_quat()), axis=0)
        key_rots = Rotation.from_quat(key_rots)
        slerp = Slerp([0.0, 1.0], key_rots)
        times = np.linspace(0, 1, n_steps)
        orientations = slerp(times).as_quat()
        for ii in range(n_steps):
            T_world_grasp.translation += dist_step
            T_world_grasp.rotation = Rotation.from_quat(orientations[ii])
            if abs is True:
                # todo by haojie add the relation transformation later
                self.origin_constraint.change(
                    jointChildPivot=T_world_grasp.translation,
                    jointChildFrameOrientation=T_world_grasp.rotation.as_quat(),
                    maxForce=300,
                )
            else:
                self.update_graspframe_constraint(T_world_grasp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()

    def detect_contact(self, threshold=5):
        # make sure we are stepping the simulator at least once, otherwise this is not reliable
        # on the other hand, it kills the performance of ICG-net, so I'll leave it off for now
        # TODO: deal with this
        # self.world.step()
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def grasp_object_id(self):
        contacts = self.world.get_contacts(self.body)
        for contact in contacts:
            # contact = contacts[0]
            # get rid body
            grased_id = contact.bodyB
            if grased_id.uid != self.body.uid:
                return grased_id.uid

    def move_fingers(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)

        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()
            if width == 0.0 and self.grasped_object is None:
                contacts = self.world.get_contacts(self.body)
                if len(contacts) > 0:
                    self.grasped_object = contacts

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width

    def move_gripper_top_down(self):
        current_pose = self.body.get_pose()
        pos = current_pose.translation + 0.1
        flip = Rotation.from_euler("y", np.pi)
        target_ori = Rotation.identity() * flip
        self.move_graspframe_pose(Transform(rotation=target_ori, translation=pos), abs=True)

    def get_distance_from_hand(
        self,
    ):
        object_id = self.grasp_object_id()
        pos, _ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        return dist_from_hand

    def is_dropped(self, object_id, prev_dist):
        pos, _ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        if np.isclose(prev_dist, dist_from_hand, atol=0.1):
            return False
        else:
            return True

    def shake_hand(self, pre_dist):
        grasp_id = self.grasp_object_id()
        current_pose = self.body.get_pose()
        x, y, z = current_pose.translation[0], current_pose.translation[1], current_pose.translation[2]
        default_position = [x, y, z]
        shake_position = [x, y, z + 0.05]
        hand_orientation2 = pybullet.getQuaternionFromEuler([np.pi, 0, -np.pi / 2])
        shake_orientation1 = pybullet.getQuaternionFromEuler([np.pi, -np.pi / 12, -np.pi / 2])
        shake_orientation2 = pybullet.getQuaternionFromEuler([np.pi, np.pi / 12, -np.pi / 2])
        new_trans = current_pose.translation + np.array([0.0, 0.0, 0.05])
        self.move_graspframe_pose(T_world_grasp_target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=new_trans))
        # check drop
        if self.is_dropped(grasp_id, pre_dist):
            return False
        self.move_graspframe_pose(
            T_world_grasp_target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=default_position)
        )
        # check drop
        if self.is_dropped(grasp_id, pre_dist):
            return False
        self.move_graspframe_pose(T_world_grasp_target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=shake_position))
        # check drop
        if self.is_dropped(grasp_id, pre_dist):
            return False
        self.move_graspframe_pose(
            T_world_grasp_target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=default_position)
        )
        # check drop
        if self.is_dropped(grasp_id, pre_dist):
            return False
        self.move_graspframe_pose(
            T_world_grasp_target=Transform(rotation=Rotation.from_quat(shake_orientation1), translation=default_position)
        )
        # check drop
        if self.is_dropped(grasp_id, pre_dist):
            return False
        self.move_graspframe_pose(
            T_world_grasp_target=Transform(rotation=Rotation.from_quat(shake_orientation2), translation=default_position)
        )
        # check drop
        if self.is_dropped(grasp_id, pre_dist):
            return False
        else:
            return True
        # self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(shake_orientation1), translation=default_position))
        # # check drop
        # if self.is_dropped(grasp_id,pre_dist):
        #     return False
        # self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(shake_orientation2), translation=default_position))
        # # check drop
        # if self.is_dropped(grasp_id,pre_dist):
        #     return False
        # else:
        #     return True

import time
import os
import open3d as o3d
import ctypes
import sys
import numpy as np
import pybullet
from pybullet_utils import bullet_client

from grasp_env.simulator.transform import Rotation, Transform
from grasp_env.simulator.object_models import get_canonical_rotation

assert pybullet.isNumpyEnabled(), "Pybullet needs to be built with NumPy"


class RedirectStream(object):
    @staticmethod
    def _flush_c_stream(stream):
        streamname = stream.name[1:-1]
        libc = ctypes.CDLL(None)
        libc.fflush(ctypes.c_void_p.in_dll(libc, streamname))

    def __init__(self, stream=sys.stdout, file=os.devnull):
        self.stream = stream
        self.file = file

    def __enter__(self):
        self.stream.flush()  # ensures python stream unaffected 
        self.fd = open(self.file, "w+")
        self.dup_stream = os.dup(self.stream.fileno())
        os.dup2(self.fd.fileno(), self.stream.fileno()) # replaces stream

    def __exit__(self, type, value, traceback):
        RedirectStream._flush_c_stream(self.stream)  # ensures C stream buffer empty
        os.dup2(self.dup_stream, self.stream.fileno()) # restores stream
        os.close(self.dup_stream)
        self.fd.close()


class BtWorld(object):
    """Interface to a PyBullet physics server.
    Attributes:
        dt: Time step of the physics simulation.
        rtf: Real time factor. If negative, the simulation is run as fast as possible.
        sim_time: Virtual time elpased since the last simulation reset.
    """

    def __init__(self, gui=True, output_video=None):
        options = ""
        if output_video is not None: options = f"--mp4={output_video} --mp4fps=30 --width=1920 --height=1080"
        with RedirectStream():
            connection_mode = pybullet.GUI if gui else pybullet.DIRECT
            try:
                self.p = bullet_client.BulletClient(connection_mode, options=options)
            except TypeError as e:
                print(e)
                self.p = bullet_client.BulletClient(connection_mode)
        self.gui = gui
        # self.dt = 1.0 / 240.0
        self.dt = 1.0 / 80.0
        self.solver_iterations = 100
        self.reset()
        self.bodies = {}

    def set_gravity(self, gravity):
        self.p.setGravity(*gravity)

    def load_urdf(self, urdf_path, pose, scale, body_type):
        # the plane don't have mass
        body = Body.from_urdf(self.p, urdf_path, pose, scale, body_type=body_type)
        self.bodies[body.uid] = body
        return body

    def load_obj(
        self,
        obj_path,
        relative_scale,
        initial_pose,
        body_type
    ):
        # the plane don't have mass
        body = Body.from_obj(
            self.p, 
            obj_path, 
            relative_scale, 
            initial_pose, 
            body_type=body_type)
        self.bodies[body.uid] = body
        return body

    def remove_body(self, body):
        self.p.removeBody(body.uid)
        self.bodies.pop(body.uid)

    def remove_constraint(self, constraint):
        self.p.removeConstraint(constraint.uid)

    def add_constraint(self, *argv, **kwargs):
        """See `Constraint` below."""
        constraint = Constraint(self.p, *argv, **kwargs)
        return constraint

    def add_camera(self, intrinsic, near, far):
        camera = Camera(self.p, intrinsic, near, far)
        return camera

    def get_contacts(self, bodyA):
        points = self.p.getContactPoints(bodyA.uid)
        contacts = []
        for point in points:
            contact = Contact(
                bodyA=self.bodies[point[1]],
                bodyB=self.bodies[point[2]],
                point=point[5],
                normal=point[7],
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        return contacts

    def reset(self):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSolverIterations=self.solver_iterations)
        self.bodies = {}
        self.sim_time = 0.0

    def step(self):
        self.p.stepSimulation()
        self.sim_time += self.dt
        if self.gui:
            time.sleep(self.dt)
            time.sleep(0.0)
            pass

    def save_state(self):
        return self.p.saveState()

    def restore_state(self, state_uid):
        self.p.restoreState(stateId=state_uid)
        self.p.removeState(state_uid)

    def close(self):
        self.p.disconnect()


def get_middle_length(obj_path):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    aabb = mesh.get_axis_aligned_bounding_box()
    size = aabb.get_extent()
    representative_length = np.partition(size, -2)[-2]
    return representative_length


class Body(object):
    """Interface to a multibody simulated in PyBullet.
    Attributes:
        uid: The unique id of the body within the physics server.
        name: The name of the body.
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
    """
    default_mass = 0.5
    default_lateral_friction = 1.0
    default_spinning_friction = 0.001
    default_rolling_friction = 0.001
    default_linear_damping = 0.

    def __init__(self, physics_client, body_uid):
        self.p = physics_client
        self.uid = body_uid
        self.name = self.p.getBodyInfo(self.uid)[1].decode("utf-8")
        self.joints, self.links = {}, {}
        for i in range(self.p.getNumJoints(self.uid)):
            joint_info = self.p.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            self.joints[joint_name] = Joint(self.p, self.uid, i)
            link_name = joint_info[12].decode("utf8")
            self.links[link_name] = Link(self.p, self.uid, i)
        self.body_type = "object"

    @classmethod
    def from_urdf(cls, physics_client, urdf_path, pose, scale, body_type="object"):
        # Load the URDF with the given pose and scale
        body_uid = physics_client.loadURDF(
            str(urdf_path),
            pose.translation,
            pose.rotation.as_quat(),
            globalScaling=scale,
        )

        # Randomize only scene objects. Applying this override to robots or
        # environment bodies discards the materials defined by their URDFs.
        if body_type == "object":
            color = np.random.uniform(0.6, 1, (4,))
            color[-1] = 1  # Ensure full opacity

            # Apply the same color to the base and all child links.
            num_joints = physics_client.getNumJoints(body_uid)
            physics_client.changeVisualShape(body_uid, -1, rgbaColor=color)
            for link_index in range(num_joints):
                physics_client.changeVisualShape(body_uid, link_index, rgbaColor=color)

        # If it's not a table, modify dynamics
        if body_type == "object":
            physics_client.changeDynamics(body_uid, -1, mass=cls.default_mass, lateralFriction=cls.default_lateral_friction)
        instance = cls(physics_client, body_uid)
        instance.body_type = body_type
        return instance
    
    

    @classmethod
    def from_obj(cls, 
                 physics_client, 
                 obj_filepath, 
                 relative_scale,
                 initial_pose,
                 body_type="object"):
        """
        Create a PyBullet object from an OBJ file with either fixed or dynamic scaling.

        Args:
            physics_client: The PyBullet physics client.
            obj_filepath (str): Path to the .obj file.
            pose: An object with `translation` and `rotation.as_quat()` methods.
            scale (float or str): A number for fixed scale, or 'dynamic' to auto-scale.
            low_bound (float): Minimum target size (used only if scale='dynamic').
            high_bound (float): Maximum target size (used only if scale='dynamic').

        Returns:
            Instance of the class.
        """
        if relative_scale is None:
            # Dynamically rescale to deal with objects that are too big or too little
            current_size = get_middle_length(obj_filepath)
            low_bound = 0.02
            high_bound = 0.08

            target_size = np.random.uniform(low_bound, high_bound)
            obj_scale = target_size / current_size
        else:
            obj_scale = relative_scale

        # Generate a random RGBA color
        color = np.random.uniform(0.6, 1, (4,))
        color[-1] = 1  # Fully opaque

        mesh_scale = [obj_scale] * 3  # Uniform scaling

        # Create visual and collision shapes
        obj_visual = physics_client.createVisualShape(
            physics_client.GEOM_MESH,
            fileName=str(obj_filepath),
            rgbaColor=color,
            meshScale=mesh_scale
        )

        obj_collision = physics_client.createCollisionShape(
            physics_client.GEOM_MESH,
            fileName=str(obj_filepath),
            meshScale=mesh_scale
        )

        # Create the object in the simulation
        object_id = physics_client.createMultiBody(
            baseMass=cls.default_mass,
            baseCollisionShapeIndex=obj_collision,
            baseVisualShapeIndex=obj_visual,
            basePosition=initial_pose.translation,
            baseOrientation=initial_pose.rotation.as_quat()
        )

        # Set dynamics
        # If it's not a table, modify dynamics
        if body_type == "object":
            physics_client.changeDynamics(
                object_id, -1,
                lateralFriction=cls.default_lateral_friction,
                spinningFriction=cls.default_spinning_friction,
                rollingFriction=cls.default_rolling_friction,
                linearDamping=cls.default_linear_damping
            )

        instance = cls(physics_client, object_id)
        instance.body_type = body_type
        return instance


    def get_pose(self):
        pos, ori = self.p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), np.asarray(pos))

    def set_pose(self, pose):
        self.p.resetBasePositionAndOrientation(self.uid, pose.translation, pose.rotation.as_quat())

    def get_velocity(self):
        linear, angular = self.p.getBaseVelocity(self.uid)
        return linear, angular


class Link(object):
    """Interface to a link simulated in Pybullet.
    Attributes:
        link_index: The index of the joint.
    """

    def __init__(self, physics_client, body_uid, link_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.link_index = link_index

    def get_pose(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos, ori = link_state[0], link_state[1]
        return Transform(Rotation.from_quat(ori), pos)

    def get_position(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos = link_state[0]
        return pos


class Joint(object):
    """Interface to a joint simulated in PyBullet.
    Attributes:
        joint_index: The index of the joint.
        lower_limit: Lower position limit of the joint.
        upper_limit: Upper position limit of the joint.
        effort: The maximum joint effort.
    """

    def __init__(self, physics_client, body_uid, joint_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.joint_index = joint_index

        joint_info = self.p.getJointInfo(body_uid, joint_index)
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.effort = joint_info[10]

    def get_position(self):
        joint_state = self.p.getJointState(self.body_uid, self.joint_index)
        return joint_state[0]

    def set_position(self, position, kinematics=False):
        if kinematics:
            self.p.resetJointState(self.body_uid, self.joint_index, position)
        self.p.setJointMotorControl2(
            self.body_uid,
            self.joint_index,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.effort,
        )


class Constraint(object):
    """Interface to a constraint in PyBullet.
    Attributes:
        uid: The unique id of the constraint within the physics server.
    """

    def __init__(
        self,
        physics_client,
        parent,
        parent_link,
        child,
        child_link,
        joint_type,
        joint_axis,
        parent_frame,
        child_frame,
    ):
        """
        Create a new constraint between links of bodies.
        Args:
            parent:
            parent_link: None for the base.
            child: None for a fixed frame in world coordinates.
        """
        self.p = physics_client
        parent_body_uid = parent.uid
        parent_link_index = parent_link.link_index if parent_link else -1
        child_body_uid = child.uid if child else -1
        child_link_index = child_link.link_index if child_link else -1

        self.uid = self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_frame.translation,
            parentFrameOrientation=parent_frame.rotation.as_quat(),
            childFramePosition=child_frame.translation,
            childFrameOrientation=child_frame.rotation.as_quat(),
        )

    def change(self, **kwargs):
        self.p.changeConstraint(self.uid, **kwargs)


class Contact(object):
    """Contact point between two multibodies.
    Attributes:
        point: Contact point.
        normal: Normal vector from ... to ...
        depth: Penetration depth
        force: Contact force acting on body ...
    """

    def __init__(self, bodyA, bodyB, point, normal, depth, force):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.point = point
        self.normal = normal
        self.depth = depth
        self.force = force


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.
    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, physics_client, intrinsic, near, far):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.p = physics_client

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.
        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )

        rgb, z_buffer = result[2][:, :, :3], result[3]
        depth = 1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)

        res = {
            'rgb': rgb,
            'depth': depth
        }
        return res


def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag([2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0])
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho

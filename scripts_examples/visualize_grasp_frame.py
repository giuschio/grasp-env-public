import pybullet as p
import pybullet_data
import time

try:
    from importlib.resources import files as pkg_resources
except ImportError:
    from importlib_resources import files as pkg_resources

# Start PyBullet in GUI mode (no simulation needed)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)

# Load the Panda hand URDF at the world origin
panda_hand_urdf_path = pkg_resources("grasp_env") / "assets/robots/panda/hand.urdf"
panda_hand_id = p.loadURDF(
    str(panda_hand_urdf_path),
    basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True
)

# Open the fingers
# Find joint indices
num_joints = p.getNumJoints(panda_hand_id)
joint_name_to_index = {p.getJointInfo(panda_hand_id, i)[1].decode('utf-8'): i for i in range(num_joints)}

finger_joint1_index = joint_name_to_index['panda_finger_joint1']
finger_joint2_index = joint_name_to_index['panda_finger_joint2']

# Set joint positions
# Maximum opening: 0.04 meters (from URDF limits)
open_distance = 0.04

# Move joints (for prismatic joints, position is displacement along axis)
p.resetJointState(panda_hand_id, finger_joint1_index, open_distance)
p.resetJointState(panda_hand_id, finger_joint2_index, open_distance)

# Define the transform T_gripper_finger (e.g., 6cm in Z direction of gripper frame)
T_gripper_finger_translation = [0, 0, 0.06]  # adjust as needed

# Get the hand base pose
gripper_pos, gripper_orn = p.getBasePositionAndOrientation(panda_hand_id)
gripper_mat = p.getMatrixFromQuaternion(gripper_orn)

# Convert to 3x3 rotation matrix
gripper_rot = [
    [gripper_mat[0], gripper_mat[1], gripper_mat[2]],
    [gripper_mat[3], gripper_mat[4], gripper_mat[5]],
    [gripper_mat[6], gripper_mat[7], gripper_mat[8]],
]

# Compute world position of T_gripper_finger
finger_mid_world_pos = [
    gripper_pos[0] + sum(gripper_rot[0][i] * T_gripper_finger_translation[i] for i in range(3)),
    gripper_pos[1] + sum(gripper_rot[1][i] * T_gripper_finger_translation[i] for i in range(3)),
    gripper_pos[2] + sum(gripper_rot[2][i] * T_gripper_finger_translation[i] for i in range(3)),
]

# Visualize a reference frame at T_gripper_finger (three cylinders for axes)
axis_length = 0.05
axis_radius = 0.001

axis_defs = [
    ("x", [1, 0, 0], [1, 0, 0, 1], p.getQuaternionFromEuler([0, -1.57079632679, 0])),
    ("y", [0, 1, 0], [0, 1, 0, 1], p.getQuaternionFromEuler([1.57079632679, 0, 0])),
    ("z", [0, 0, 1], [0, 0, 1, 1], p.getQuaternionFromEuler([0, 0, 0])),
]

for _, axis_local, color, local_axis_quat in axis_defs:
    axis_world = [
        sum(gripper_rot[0][i] * axis_local[i] for i in range(3)),
        sum(gripper_rot[1][i] * axis_local[i] for i in range(3)),
        sum(gripper_rot[2][i] * axis_local[i] for i in range(3)),
    ]
    axis_center = [
        finger_mid_world_pos[0] + axis_world[0] * (axis_length / 2.0),
        finger_mid_world_pos[1] + axis_world[1] * (axis_length / 2.0),
        finger_mid_world_pos[2] + axis_world[2] * (axis_length / 2.0),
    ]
    _, axis_world_quat = p.multiplyTransforms(
        [0, 0, 0], gripper_orn, [0, 0, 0], local_axis_quat
    )

    axis_visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=axis_radius,
        length=axis_length,
        rgbaColor=color
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=axis_visual_shape_id,
        basePosition=axis_center,
        baseOrientation=axis_world_quat
    )

# Step simulation to update visualization (optional, but helps)
for _ in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

# Pause indefinitely to allow manual visualization
input("Press Enter to exit visualization...")
p.disconnect()

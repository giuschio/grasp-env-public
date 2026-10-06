import enum


class Label(enum.IntEnum):
    FAILURE = 0  # grasp execution failed due to collision or slippage
    SUCCESS = 1  # object was successfully removed


class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.

    The grasp pose is defined as the pose of the midpoint between the two fingers
    z is the approach direction
    """

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width

    @property
    def point1(self):
        center = self.pose[:3, 3]
        y_dir = self.pose[:3, 1]
        p1 = center + 0.5*self.width*y_dir
        p2 = center - 0.5*self.width*y_dir
        return p1

    @property
    def point2(self):
        center = self.pose[:3, 3]
        y_dir = self.pose[:3, 1]
        p1 = center + 0.5*self.width*y_dir
        p2 = center - 0.5*self.width*y_dir
        return p1


def to_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation /= voxel_size
    width = grasp.width / voxel_size
    return Grasp(pose, width)


def from_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation *= voxel_size
    width = grasp.width * voxel_size
    return Grasp(pose, width)

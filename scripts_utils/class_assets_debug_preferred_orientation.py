import argparse
import csv
import os
import shutil
import tempfile
from copy import deepcopy

import numpy as np
import open3d as o3d

from grasp_env.simulator.object_models import get_assets_root, get_category_directory
from grasp_env.simulator.transform import Rotation


AXIS_CANDIDATES = ("x", "y", "z", "-x", "-y", "-z")


def create_wireframe_aabb(mesh, color):
    aabb = mesh.get_axis_aligned_bounding_box()
    return o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb).paint_uniform_color(color)


def create_wireframe_obb(mesh, color):
    obb = mesh.get_oriented_bounding_box(robust=True)
    return o3d.geometry.LineSet.create_from_oriented_bounding_box(obb).paint_uniform_color(color)


def create_world_plane(half_extent, color=(0.6, 0.6, 0.6), grid_steps=10):
    points = []
    lines = []
    step = (2.0 * half_extent) / grid_steps

    for idx in range(grid_steps + 1):
        coord = -half_extent + idx * step

        points.append([coord, -half_extent, 0.0])
        points.append([coord, half_extent, 0.0])
        lines.append([len(points) - 2, len(points) - 1])

        points.append([-half_extent, coord, 0.0])
        points.append([half_extent, coord, 0.0])
        lines.append([len(points) - 2, len(points) - 1])

    plane = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points, dtype=float)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=int)),
    )
    plane.paint_uniform_color(color)
    return plane


def axis_rotation_map():
    return {
        "x": Rotation.from_rotvec(-0.5 * np.pi * np.array([0.0, 1.0, 0.0])),
        "y": Rotation.from_rotvec(0.5 * np.pi * np.array([1.0, 0.0, 0.0])),
        "z": Rotation.identity(),
        "-x": Rotation.from_rotvec(0.5 * np.pi * np.array([0.0, 1.0, 0.0])),
        "-y": Rotation.from_rotvec(-0.5 * np.pi * np.array([1.0, 0.0, 0.0])),
        "-z": Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])),
    }


def detect_axis_key(rotation, atol_deg=1.0):
    for axis_key, candidate in axis_rotation_map().items():
        relative = candidate.inv() * rotation
        if np.rad2deg(relative.magnitude()) <= atol_deg:
            return axis_key
    return None


class ManualOrientationAnnotator:
    def __init__(self, mesh, step_degrees):
        self.step_degrees = step_degrees
        self.step_radians = np.deg2rad(step_degrees)
        self.world_extent = self._world_extent(mesh)

        self.base_mesh = deepcopy(mesh)
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self._frame_size(mesh) * 1.15)
        self.world_plane = create_world_plane(self.world_extent)
        self.current_rotation = Rotation.identity()
        self.current_translation = np.zeros(3, dtype=float)
        self.direction_sign = 1.0
        self.result = "quit"

        self.mesh = deepcopy(self.base_mesh)
        self.aabb = create_wireframe_aabb(self.mesh, color=(0.15, 0.55, 0.15))
        self.obb = create_wireframe_obb(self.mesh, color=(0.85, 0.45, 0.15))
        self._dynamic_geometries = []

    def _frame_size(self, mesh):
        extent = mesh.get_axis_aligned_bounding_box().get_extent()
        return max(float(np.max(extent)) * 0.35, 0.02)

    def _world_extent(self, mesh):
        extent = mesh.get_axis_aligned_bounding_box().get_extent()
        return max(float(np.max(extent)) * 0.9, 0.05)

    def _refresh_geometries(self, vis):
        self.mesh = deepcopy(self.base_mesh)
        self.mesh.rotate(self.current_rotation.as_matrix(), center=(0.0, 0.0, 0.0))
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()

        aabb = self.mesh.get_axis_aligned_bounding_box()
        self.current_translation = np.array([0.0, 0.0, -aabb.get_min_bound()[2]], dtype=float)
        self.mesh.translate(self.current_translation)
        self.aabb = create_wireframe_aabb(self.mesh, color=(0.15, 0.55, 0.15))
        self.obb = create_wireframe_obb(self.mesh, color=(0.85, 0.45, 0.15))

        for geometry in self._dynamic_geometries:
            vis.remove_geometry(geometry, reset_bounding_box=False)

        vis.add_geometry(self.mesh, reset_bounding_box=False)
        vis.add_geometry(self.aabb, reset_bounding_box=False)
        vis.add_geometry(self.obb, reset_bounding_box=False)
        self._dynamic_geometries = [self.mesh, self.aabb, self.obb]
        vis.poll_events()
        vis.update_renderer()

    def _set_side_view(self, vis):
        controller = vis.get_view_control()
        controller.set_lookat([0.0, 0.0, 0.25 * self.world_extent])
        controller.set_front([0.0, -1.0, 0.35])
        controller.set_up([0.0, 0.0, 1.0])
        controller.set_zoom(0.6)

    def _apply_increment(self, axis):
        delta = Rotation.from_rotvec(self.direction_sign * self.step_radians * np.asarray(axis, dtype=float))
        self.current_rotation = self.current_rotation * delta

    def _rotate_x(self, vis):
        self._apply_increment([1.0, 0.0, 0.0])
        self._refresh_geometries(vis)
        return False

    def _rotate_y(self, vis):
        self._apply_increment([0.0, 1.0, 0.0])
        self._refresh_geometries(vis)
        return False

    def _rotate_z(self, vis):
        self._apply_increment([0.0, 0.0, 1.0])
        self._refresh_geometries(vis)
        return False

    def _reset(self, vis):
        self.current_rotation = Rotation.identity()
        self.direction_sign = 1.0
        self._refresh_geometries(vis)
        return False

    def _toggle_direction(self, vis=None):
        self.direction_sign *= -1.0
        print(f"  Rotation direction: {'positive' if self.direction_sign > 0 else 'negative'}")
        return False

    def _accept(self, vis):
        self.result = "accept"
        vis.close()
        return False

    def _undetermined(self, vis):
        self.result = "undetermined"
        vis.close()
        return False

    def _quit(self, vis):
        self.result = "quit"
        vis.close()
        return False

    def run(self, window_name):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=window_name, width=1280, height=960)

        vis.register_key_callback(ord("J"), self._rotate_x)
        vis.register_key_callback(ord("I"), self._rotate_y)
        vis.register_key_callback(ord("L"), self._rotate_z)
        vis.register_key_callback(ord("R"), self._reset)
        vis.register_key_callback(257, self._accept)
        vis.register_key_callback(ord("U"), self._undetermined)
        vis.register_key_callback(ord("X"), self._quit)
        vis.register_key_callback(340, self._toggle_direction)

        vis.add_geometry(self.world_plane, reset_bounding_box=True)
        vis.add_geometry(self.world_frame, reset_bounding_box=False)
        self._refresh_geometries(vis)

        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.line_width = 2.0
        render_option.background_color = np.array([0.08, 0.08, 0.08])
        self._set_side_view(vis)

        vis.run()
        vis.destroy_window()
        return self.result, self.current_rotation


def center_mesh(mesh):
    centered = deepcopy(mesh)
    center = centered.get_center()
    centered.translate(-center)
    return centered, center


def ensure_orientation_fields(fieldnames):
    updated = list(fieldnames)
    for key in ("preferred_orientation", "preferred_qx", "preferred_qy", "preferred_qz", "preferred_qw"):
        if key not in updated:
            updated.append(key)
    if "canonical_axis" not in updated:
        updated.append("canonical_axis")
    return updated


def set_row_orientation(row, result, rotation):
    for key in ("preferred_orientation", "preferred_qx", "preferred_qy", "preferred_qz", "preferred_qw"):
        row[key] = ""

    if result == "undetermined":
        row["canonical_axis"] = "u"
        row["preferred_orientation"] = "u"
        return "u"

    axis_key = detect_axis_key(rotation)
    if axis_key is not None:
        row["canonical_axis"] = axis_key
        row["preferred_orientation"] = axis_key
        return axis_key

    quat = rotation.as_quat()
    row["canonical_axis"] = ""
    row["preferred_orientation"] = "quat"
    row["preferred_qx"] = f"{quat[0]:.10f}"
    row["preferred_qy"] = f"{quat[1]:.10f}"
    row["preferred_qz"] = f"{quat[2]:.10f}"
    row["preferred_qw"] = f"{quat[3]:.10f}"
    return "quat"


def annotate_and_save(csv_path, start_index=0, step_degrees=5.0):
    root_path = get_assets_root()
    temp_fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(temp_fd)

    with open(csv_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        fieldnames = ensure_orientation_fields(reader.fieldnames)

    stop_index = len(rows)
    try:
        with open(temp_path, "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row_idx, row in enumerate(rows):
                if row_idx < start_index:
                    writer.writerow(row)
                    continue

                object_path = row["path"]
                mesh_path = os.path.join(root_path, object_path)
                mesh = o3d.io.read_triangle_mesh(mesh_path)

                if mesh.is_empty():
                    print(f"\n[{row_idx}] Skipping empty mesh: {object_path}")
                    writer.writerow(row)
                    continue
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()

                centered_mesh, original_center = center_mesh(mesh)
                print(f"\n[{row_idx}] Object: {object_path}")
                print(f"Centered preview around mesh center: [{original_center[0]:.6f}, {original_center[1]:.6f}, {original_center[2]:.6f}]")
                print("Controls: J rotate X, I rotate Y, L rotate Z, Shift toggles direction, R reset, Enter accept, U undetermined, X quit")
                print("Rotation is applied in the object's local frame, matching grasp_learning_cl.grasp_datagen/annotation.py.")
                print("Viewer: world frame and world XY plane are fixed; the green box is the live world-aligned AABB and the orange box is the live OBB.")
                print("After each rotation, the object is translated so the AABB rests on top of the XY plane.")
                print(f"Step size: {step_degrees:.1f} degrees")

                annotator = ManualOrientationAnnotator(centered_mesh, step_degrees=step_degrees)
                result, rotation = annotator.run(window_name=f"Manual preferred orientation: {object_path}")

                if result == "quit":
                    stop_index = row_idx
                    writer.writerow(row)
                    break

                saved_kind = set_row_orientation(row, result, rotation)
                writer.writerow(row)

                if result == "undetermined":
                    print(f"  Saved: undetermined for {object_path}")
                else:
                    quat = rotation.as_quat()
                    matrix = rotation.as_matrix()
                    translation = annotator.current_translation
                    print("  Saved manual orientation:")
                    print(f"  mode   = {saved_kind}")
                    print(f"  quat   = [{quat[0]: .8f}, {quat[1]: .8f}, {quat[2]: .8f}, {quat[3]: .8f}]")
                    print(f"  trans  = [{translation[0]: .8f}, {translation[1]: .8f}, {translation[2]: .8f}]")
                    print("  matrix =")
                    print(f"    [{matrix[0,0]: .8f}, {matrix[0,1]: .8f}, {matrix[0,2]: .8f}]")
                    print(f"    [{matrix[1,0]: .8f}, {matrix[1,1]: .8f}, {matrix[1,2]: .8f}]")
                    print(f"    [{matrix[2,0]: .8f}, {matrix[2,1]: .8f}, {matrix[2,2]: .8f}]")

            for row in rows[stop_index + 1:]:
                writer.writerow(row)
    except Exception:
        os.unlink(temp_path)
        raise
    overwrite = None
    while overwrite not in ["y", "n"]:
        overwrite = input(f"\nOverwrite original file {csv_path}? (y/n): ").strip().lower()
        if overwrite == "y":
            shutil.move(temp_path, csv_path)
            print(f"{csv_path} has been overwritten.")
        elif overwrite == "n":
            print(f"Original file unchanged. New file at {temp_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate preferred object orientations by manually rotating each mesh and save them back to the CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=get_category_directory() / "animal.csv",
        help="CSV file containing asset rows with a 'path' column.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Row index to start from in the CSV.",
    )
    parser.add_argument(
        "--step-degrees",
        type=float,
        default=5.0,
        help="Rotation step in degrees for each key press.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    annotate_and_save(args.csv_path, start_index=args.start_index, step_degrees=args.step_degrees)

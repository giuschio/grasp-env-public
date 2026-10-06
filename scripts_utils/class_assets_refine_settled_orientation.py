import argparse
import csv
import os
import shutil
import tempfile

import numpy as np

from grasp_env.simulator.object_models import (
    get_assets_root,
    get_category_directory,
    get_preferred_rotation,
    get_row_preferred_orientation,
)
from grasp_env.simulator.simulation_clutter_bandit import ClutterRemovalSim
from grasp_env.simulator.transform import Transform, Rotation


AXIS_CANDIDATES = ("x", "y", "z", "-x", "-y", "-z")


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


def ensure_orientation_fields(fieldnames):
    updated = list(fieldnames)
    for key in ("preferred_orientation", "preferred_qx", "preferred_qy", "preferred_qz", "preferred_qw"):
        if key not in updated:
            updated.append(key)
    if "canonical_axis" not in updated:
        updated.append("canonical_axis")
    return updated


def set_row_orientation(row, rotation):
    for key in ("preferred_orientation", "preferred_qx", "preferred_qy", "preferred_qz", "preferred_qw"):
        row[key] = ""

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


def spawn_single_object(sim, row):
    assets_root = get_assets_root()
    object_file = os.path.join(assets_root, row["path"])
    preferred_orientation = get_row_preferred_orientation(row)
    preferred_orientation = "z" if preferred_orientation is None else preferred_orientation
    relative_scale = float(row.get("relative_size", 1.0))

    rotation = get_preferred_rotation(preferred_orientation)
    # Spawn close to the table, then reseat using the actual PyBullet AABB.
    initial_pose = Transform(rotation, np.array([0.15, 0.15, sim.table_height + 0.1]))

    if os.path.splitext(object_file)[1] == ".urdf":
        body = sim.world.load_urdf(object_file, initial_pose, scale=relative_scale, body_type="object")
    else:
        body = sim.world.load_obj(
            obj_path=object_file,
            relative_scale=relative_scale,
            initial_pose=initial_pose,
            body_type="object",
        )
    sim._set_body_asset_metadata(body, object_file)

    lower, upper = sim.world.p.getAABB(body.uid)
    z = sim.table_height + 0.5 * (upper[2] - lower[2]) + 0.002
    body.set_pose(Transform(rotation, np.array([0.15, 0.15, z])))
    return body


def settle_object_orientation(row, show_gui=False, seed=0, timeout=4.0):
    sim = ClutterRemovalSim(
        seed=seed,
        scene_type="singulated",
        object_set="category-mug",
        randomize_object_scale=False,
        gui=show_gui,
        rand=False,
        sort_file_list=True,
    )

    try:
        sim.world.reset()
        sim.world.set_gravity([0.0, 0.0, -9.81])
        if show_gui:
            sim.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )
            sim.rendering_off()
        sim.place_table()
        body = spawn_single_object(sim, row)
        if show_gui:
            sim.rendering_on()
        sim.wait_for_objects_to_rest(timeout=timeout)
        return body.get_pose().rotation
    finally:
        sim.close()


def refine_and_save(csv_path, start_index=0, show_gui=False, seed=0, timeout=4.0):
    with open(csv_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        fieldnames = ensure_orientation_fields(reader.fieldnames)

    temp_fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(temp_fd)

    try:
        with open(temp_path, "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row_idx, row in enumerate(rows):
                if row_idx < start_index:
                    writer.writerow(row)
                    continue

                object_path = row["path"]
                print(f"\n[{row_idx}] Settling: {object_path}")
                rotation = settle_object_orientation(
                    row=row,
                    show_gui=show_gui,
                    seed=seed,
                    timeout=timeout,
                )
                saved_kind = set_row_orientation(row, rotation)
                quat = rotation.as_quat()
                print(f"  saved mode = {saved_kind}")
                print(f"  quat       = [{quat[0]: .8f}, {quat[1]: .8f}, {quat[2]: .8f}, {quat[3]: .8f}]")
                writer.writerow(row)
    except Exception:
        os.unlink(temp_path)
        raise

    overwrite = input(f"\nOverwrite original file {csv_path}? (y/n): ").strip().lower()
    if overwrite == "y":
        shutil.move(temp_path, csv_path)
        print(f"{csv_path} has been overwritten.")
    else:
        print(f"Original file unchanged. New file at {temp_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine preferred orientations by spawning each object alone in grasp_env, letting it settle, and saving the settled orientation."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=get_category_directory() / "fasteners.csv",
        help="CSV file containing asset rows with a 'path' column.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Row index to start from in the CSV.",
    )
    parser.add_argument(
        "--show-gui",
        default=True,
        help="Show the PyBullet GUI while objects settle.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to initialize the simulation.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=4.0,
        help="Maximum settle time in seconds of simulated time.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    refine_and_save(
        csv_path=args.csv_path,
        start_index=args.start_index,
        show_gui=args.show_gui,
        seed=args.seed,
        timeout=args.timeout,
    )

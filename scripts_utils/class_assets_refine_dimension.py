import open3d as o3d
import os
import csv
import tempfile
import shutil

from grasp_env.simulator.object_models import get_assets_root, get_category_directory

# def scale_mesh_to_dimension(mesh, target_dimension):
#     aabb = mesh.get_axis_aligned_bounding_box()
#     extent = aabb.get_extent()
#     max_extent = max(extent)
#     scale_factor = target_dimension / max_extent
#     mesh.scale(scale_factor, center=aabb.get_center())
#     return mesh

def get_max_length(obj_path):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    aabb = mesh.get_axis_aligned_bounding_box()
    size = max(aabb.get_extent())
    return size

def scale_objects(csv_path):
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
    os.close(temp_fd)

    root_path = get_assets_root()

    with open(csv_path, 'r', newline='') as infile, open(temp_path, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            object_path = row['path']
            # dimension_str = str(row['dimension']).strip()
            # dimension = float(dimension_str)

            meshpath = os.path.join(root_path, object_path)
            mesh = o3d.io.read_triangle_mesh(meshpath)
            mesh.compute_vertex_normals()

            aabb = mesh.get_axis_aligned_bounding_box()
            size = max(aabb.get_extent())
            print(size)

            # mesh = scale_mesh_to_dimension(mesh, dimension)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            o3d.visualization.draw_geometries([mesh, frame], window_name=f"Object: {object_path}")

            print("Ref system is 20cm")
            print("This should be considered the max dimension")
            target_size = input(f"Enter dimension in cm: ").strip()
            # from random import choice
            # target_size = choice(["20", "18", "17", "15"])
            if target_size:
                target_size = float(target_size) / 100.0
                relative_size = target_size / size
                row['relative_size'] = str(relative_size)

            writer.writerow(row)

    print(f"\nScaled CSV written to: {temp_path}")
    overwrite = input(f"Overwrite original file {csv_path}? (y/n): ").strip().lower()
    if overwrite == 'y':
        shutil.move(temp_path, csv_path)
        print(f"{csv_path} has been overwritten.")
    else:
        print(f"Original file unchanged. New file at {temp_path}")

if __name__ == "__main__":
    fname = get_category_directory() / "bowl.csv"
    scale_objects(fname)

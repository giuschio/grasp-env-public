import open3d as o3d
import os
import csv
import tempfile
import shutil

from grasp_env.simulator.object_models import get_assets_root, get_category_directory

def set_orientation(csv_path):
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
    os.close(temp_fd)

    root_path = get_assets_root()

    with open(csv_path, 'r', newline='') as infile, open(temp_path, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames.copy()
        if 'canonical_axis' not in fieldnames:
            fieldnames.append('canonical_axis')

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            object_path = row['path']
            meshpath = os.path.join(root_path, object_path)
            mesh = o3d.io.read_triangle_mesh(meshpath)
            mesh.compute_vertex_normals()

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            o3d.visualization.draw_geometries([mesh, frame], window_name=f"Object: {object_path}")

            valid_directions = ['x', 'y', 'z', '-x', '-y', '-z', 'u']
            canonical_axis = ''
            while canonical_axis not in valid_directions:
                canonical_axis = input(f"Which axis is up? x, y, z, -x, -y, -z, u (undetermined): ").strip().lower()
                if canonical_axis not in valid_directions:
                    print("Invalid input. Valid options: x, y, z, -x, -y, -z, u (undetermined)")

            row['canonical_axis'] = canonical_axis
            writer.writerow(row)

    print(f"\nOriented CSV written to: {temp_path}")
    overwrite = input(f"Overwrite original file {csv_path}? (y/n): ").strip().lower()
    if overwrite == 'y':
        shutil.move(temp_path, csv_path)
        print(f"{csv_path} has been overwritten.")
    else:
        print(f"Original file unchanged. New file at {temp_path}")

if __name__ == "__main__":
    fname = get_category_directory() / "pliers.csv"
    set_orientation(fname)

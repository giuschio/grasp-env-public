import open3d as o3d
import os
import csv
import tempfile
import shutil

from grasp_env.simulator.object_models import get_assets_root, get_category_directory

def filter_objects(csv_path):
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
    os.close(temp_fd)

    root_path = get_assets_root()

    with open(csv_path, 'r', newline='') as infile, open(temp_path, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        rows = list(reader)
        num_rows = len(rows)
        index = [0]  # mutable index reference
        decision = {'keep': None}

        print("Press y/n to keep or discard the displayed mesh")

        def show_mesh():
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            row = rows[index[0]]
            object_path = row['path']
            meshpath = os.path.join(root_path, object_path)
            mesh = o3d.io.read_triangle_mesh(meshpath)
            mesh.compute_vertex_normals()
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name=f"Object {index[0]+1}/{num_rows}: {object_path}")
            vis.add_geometry(mesh)
            vis.add_geometry(frame)

            def keep_callback(vis):
                decision['keep'] = True
                vis.close()

            def discard_callback(vis):
                decision['keep'] = False
                vis.close()

            vis.register_key_callback(ord('Y'), keep_callback)
            vis.register_key_callback(ord('N'), discard_callback)
            vis.run()
            vis.destroy_window()

        while index[0] < num_rows:
            show_mesh()
            row = rows[index[0]]
            if decision['keep']:
                writer.writerow(row)
            else:
                print(f"Object {index[0]+1} marked for deletion.")
            index[0] += 1

    print(f"\nFiltered CSV written to: {temp_path}")
    overwrite = input(f"Overwrite original file {csv_path}? (y/n): ").strip().lower()
    if overwrite == 'y':
        shutil.move(temp_path, csv_path)
        print(f"{csv_path} has been overwritten.")
    else:
        print(f"Original file unchanged. New file at {temp_path}")

if __name__ == "__main__":
    fname = get_category_directory() / "bowl.csv"
    filter_objects(fname)

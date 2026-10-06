import open3d as o3d
import numpy as np
import math

from grasp_env.simulator.object_models import ModelDatabase, get_preferred_rotation

# Dataset configuration
dataset = "category-animal"

def main(dataset):
    db = ModelDatabase(dataset)
    n = len(db)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    spacing = db.spacing

    meshes = []
    for idx in range(len(db)):
        mesh = o3d.io.read_triangle_mesh(db.get_file(idx))
        scale = db.get_scale(idx)
        scale = 1.0 if scale is None else scale
        preferred_orientation = db.get_canonical_axis(idx)
        preferred_orientation = "z" if preferred_orientation is None else preferred_orientation
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        rotation = get_preferred_rotation(preferred_orientation)
        mesh = mesh.scale(scale, center=(0, 0, 0))
        mesh.rotate(rotation.as_matrix(), center=(0, 0, 0))  # Apply rotation here
        row = idx // cols
        col = idx % cols
        translation = np.array([col * spacing, -row * spacing, 0])
        mesh.translate(translation)
        meshes.append(mesh)

    # Use Visualizer to set window title
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{dataset} dataset ({n} objects)", width=1024, height=768)
    for mesh in meshes:
        vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main(dataset)

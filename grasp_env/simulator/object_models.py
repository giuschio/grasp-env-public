import os
import csv
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from grasp_env.simulator.transform import Rotation, Transform

from natsort import natsorted


GRASP_ENV_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_CATALOG = GRASP_ENV_REPO_ROOT / "asset_directories" / "datasets.yaml"


def load_asset_catalog(catalog_path=None):
    catalog_path = Path(
        catalog_path
        or os.environ.get("GRASP_ENV_ASSET_CATALOG", DEFAULT_ASSET_CATALOG)
    ).expanduser().resolve()
    if not catalog_path.exists():
        raise FileNotFoundError(f"Asset catalog not found: {catalog_path}")
    with catalog_path.open("r", encoding="utf-8") as catalog_file:
        catalog = yaml.safe_load(catalog_file) or {}
    return catalog_path, catalog


def get_assets_root(catalog_path=None, assets_root=None):
    catalog_path, catalog = load_asset_catalog(catalog_path)
    configured_root = assets_root or os.environ.get("GRASP_ENV_ASSET_ROOT") or catalog.get("asset_root", "../assets")
    root = Path(os.path.expandvars(str(configured_root))).expanduser()
    if not root.is_absolute():
        root = catalog_path.parent / root
    return root.resolve()


def get_category_directory(catalog_path=None):
    catalog_path, catalog = load_asset_catalog(catalog_path)
    category_directory = Path(catalog.get("category_directory", "categories"))
    if not category_directory.is_absolute():
        category_directory = catalog_path.parent / category_directory
    return category_directory.resolve()


def get_dataset_root(dataset_type, catalog_path=None, assets_root=None):
    catalog_path, catalog = load_asset_catalog(catalog_path)
    try:
        dataset_directory = catalog["datasets"][dataset_type]["directory"]
    except KeyError as error:
        available = ", ".join(sorted(catalog.get("datasets", {})))
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. Available datasets: {available}"
        ) from error
    return get_assets_root(catalog_path, assets_root) / dataset_directory

def _get_axis_alignment_rotation(canonical_axis):
    # Mapping canonical axis strings to vectors
    axis_map = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1]),
        '-x': np.array([-1, 0, 0]),
        '-y': np.array([0, -1, 0]),
        '-z': np.array([0, 0, -1])
    }

    # Determine alignment rotation if canonical_axis is given
    if canonical_axis in axis_map:
        source_vec = axis_map[canonical_axis.lower()]
        target_vec = np.array([0, 0, 1])  # world z-axis

        # Compute rotation between source_vec and target_vec
        cross = np.cross(source_vec, target_vec)
        dot = np.dot(source_vec, target_vec)
        if np.allclose(cross, 0):
            if dot > 0:
                alignment_quat = [0, 0, 0, 1]  # Identity quaternion
            else:
                # 180-degree rotation about any orthogonal axis
                ortho = np.array([1, 0, 0]) if not np.allclose(source_vec, [1, 0, 0]) else np.array([0, 1, 0])
                axis = np.cross(source_vec, ortho)
                axis = axis / np.linalg.norm(axis)
                alignment_quat = Rotation.from_rotvec(np.pi * axis).as_quat()
        else:
            axis = cross / np.linalg.norm(cross)
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            alignment_quat = Rotation.from_rotvec(angle * axis).as_quat()

        # Apply the alignment rotation before pose rotation
        alignment_rot = Rotation.from_quat(alignment_quat)
    elif canonical_axis == "u":
        alignment_rot = Rotation.random()
    else:
        alignment_rot = Rotation.identity()

    return alignment_rot


def get_canonical_rotation(canonical_axis):
    return _get_axis_alignment_rotation(canonical_axis)


def get_preferred_rotation(preferred_orientation):
    if preferred_orientation is None:
        return Rotation.identity()

    if isinstance(preferred_orientation, Rotation):
        return preferred_orientation

    if isinstance(preferred_orientation, (list, tuple, np.ndarray)):
        values = np.asarray(preferred_orientation, dtype=float)
        if values.shape == (4,):
            return Rotation.from_quat(values)
        raise ValueError(f"Unsupported preferred orientation shape: {values.shape}")

    if isinstance(preferred_orientation, str):
        value = preferred_orientation.strip()
        if not value:
            return Rotation.identity()

        lowered = value.lower()
        axis_keys = {"x", "y", "z", "-x", "-y", "-z", "u"}
        if lowered in axis_keys:
            return _get_axis_alignment_rotation(lowered)

        normalized = value.replace("[", "").replace("]", "").replace(";", " ")
        parts = [part for chunk in normalized.split(",") for part in chunk.split()]
        if len(parts) == 4:
            quat = np.asarray([float(part) for part in parts], dtype=float)
            return Rotation.from_quat(quat)

    raise ValueError(f"Unsupported preferred orientation value: {preferred_orientation!r}")


def get_row_preferred_orientation(row):
    quat_keys = ("preferred_qx", "preferred_qy", "preferred_qz", "preferred_qw")
    quat_values = [row.get(key, "").strip() for key in quat_keys]
    if all(quat_values):
        return np.asarray([float(value) for value in quat_values], dtype=float)

    preferred_orientation = row.get("preferred_orientation", "").strip()
    if preferred_orientation:
        return preferred_orientation

    return row.get("canonical_axis", "z")


class ModelDatabase:
    def __init__(self, dataset, catalog_path=None, assets_root=None):
        self.dataset = dataset
        self.dataset_type = dataset.split("-")[0]
        self.catalog_path, self.catalog = load_asset_catalog(catalog_path)
        self.assets_root = get_assets_root(self.catalog_path, assets_root)
        self.category_directory = get_category_directory(self.catalog_path)
        self.spacing = None
        self.object_files = []
        self.relative_scales = []
        self.preferred_orientations = []
        self._load_files()

    def __len__(self):
        return len(self.object_files)
    
    def get_file(self, idx):
        return self.object_files[idx]
    
    def get_scale(self, idx):
        return self.relative_scales[idx]
    
    def get_canonical_axis(self, idx):
        return self.preferred_orientations[idx]
        
    def _load_files(self):
        if self.dataset_type == "category":
            self._load_category(self.dataset.removeprefix("category-"))
        elif self.dataset_type == "object":
            self._load_object(self.dataset.removeprefix("object-"))
        else:
            self._load_dataset()

        if not self.object_files:
            raise ValueError(f"Object set '{self.dataset}' did not resolve to any object files")

        missing_files = [path for path in self.object_files if not Path(path).is_file()]
        if missing_files:
            preview = ", ".join(missing_files[:3])
            raise FileNotFoundError(
                f"Object set '{self.dataset}' references {len(missing_files)} missing files under "
                f"{self.assets_root}. First missing paths: {preview}"
            )

        combined = natsorted(
            zip(self.object_files, self.relative_scales, self.preferred_orientations),
            key=lambda item: item[0],
        )
        self.object_files, self.relative_scales, self.preferred_orientations = map(list, zip(*combined))

    def _load_dataset(self):
        datasets = self.catalog.get("datasets", {})
        if self.dataset_type not in datasets:
            raise ValueError(
                f"Unknown dataset type '{self.dataset_type}'. Available datasets: {', '.join(sorted(datasets))}"
            )

        spec = datasets[self.dataset_type]
        dataset_root = get_dataset_root(
            self.dataset_type,
            catalog_path=self.catalog_path,
            assets_root=self.assets_root,
        )
        if not dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset '{self.dataset_type}' is not installed at {dataset_root}. "
                "See the Assets section of the grasp-env README for download instructions."
            )

        selector = self.dataset[len(self.dataset_type):].lstrip("-")
        loader_name = spec["loader"]
        loaders = {
            "vgn_collision": self._load_vgn,
            "recursive_obj": self._load_recursive_obj,
            "dexgraspnet_coacd": self._load_dexgraspnet,
        }
        if loader_name not in loaders:
            raise ValueError(f"Unknown asset loader '{loader_name}' for dataset '{self.dataset_type}'")

        self.object_files = loaders[loader_name](dataset_root, selector, spec)
        self.relative_scales = [spec.get("default_scale")] * len(self.object_files)
        self.preferred_orientations = [spec.get("preferred_orientation", "z")] * len(self.object_files)
        self.spacing = float(spec.get("spacing", 0.3))

    def _load_category(self, category):
        csv_file = self.category_directory / f"{category}.csv"
        if not csv_file.exists():
            available = ", ".join(path.stem for path in sorted(self.category_directory.glob("*.csv")))
            raise FileNotFoundError(
                f"Category '{category}' not found at {csv_file}. Available categories: {available}"
            )

        self.spacing = float(self.catalog.get("category_defaults", {}).get("spacing", 0.3))
        with csv_file.open(newline="", encoding="utf-8") as category_file:
            for row in csv.DictReader(category_file):
                self.object_files.append(str(self.assets_root / row["path"]))
                self.relative_scales.append(float(row.get("relative_size") or 1.0))
                self.preferred_orientations.append(get_row_preferred_orientation(row))

    def _load_object(self, chosen_object):
        self.spacing = float(self.catalog.get("category_defaults", {}).get("spacing", 0.3))
        for csv_file in sorted(self.category_directory.glob("*.csv")):
            with csv_file.open(newline="", encoding="utf-8") as category_file:
                for row in csv.DictReader(category_file):
                    if row["path"] == chosen_object:
                        self.object_files.append(str(self.assets_root / row["path"]))
                        self.relative_scales.append(float(row.get("relative_size") or 1.0))
                        self.preferred_orientations.append(get_row_preferred_orientation(row))

    @staticmethod
    def _load_vgn(root_folder, selector, spec):
        subfolder = selector.replace("-", "/") if selector else spec.get("subdirectory", "")
        root_folder = root_folder / subfolder
        if not root_folder.is_dir():
            raise FileNotFoundError(f"VGN object split not found: {root_folder}")
        return [
            str(path)
            for path in root_folder.iterdir()
            if path.name.endswith("collision.obj") and path.is_file()
        ]

    @staticmethod
    def _load_recursive_obj(root_folder, selector, spec):
        subfolder = selector or spec.get("subdirectory", "")
        search_root = root_folder / subfolder
        if not search_root.is_dir():
            raise FileNotFoundError(f"Object directory not found: {search_root}")
        return [str(path) for path in search_root.rglob("*.obj")]

    @staticmethod
    def _load_dexgraspnet(root_folder, selector, spec):
        objects_path = root_folder / spec.get("subdirectory", "objects")
        if not objects_path.is_dir():
            raise FileNotFoundError(f"DexGraspNet object directory not found: {objects_path}")

        tag = selector or "all"
        object_directories = [path for path in objects_path.iterdir() if path.is_dir()]
        if tag != "all":
            tag = tag.lower()
            object_directories = [path for path in object_directories if tag in path.name.lower()]
        return [str(path / "coacd" / "decomposed.obj") for path in object_directories]


def get_max_length(obj_path):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    aabb = mesh.get_axis_aligned_bounding_box()
    size = max(aabb.get_extent())
    return size

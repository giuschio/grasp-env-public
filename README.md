# Grasp env

This repository contains a lightweight simulation environment for grasping in
clutter. It is designed to be simple to install and use, is self-contained, and
includes examples of basic grasping heuristics. The environment builds on the
work of [Breyer et al.](https://github.com/ethz-asl/vgn) and incorporates
modifications from [Huang et al.](https://github.com/HaojHuang/Edge-Grasp-Network)
and [Zurbrügg et al.](https://github.com/renezurbruegg/icg_benchmark/tree/master).

<p align="center">
  <img src="figures/screenshot_0.png" alt="Cluttered grasping simulation with RGB, depth, and segmentation observations" width="49%" />
  <img src="figures/screenshot_1.png" alt="A second generated grasping scene with RGB, depth, and segmentation observations" width="49%" />
</p>

## Associated paper and code

This repository is part of the code release for our CoRL 2026 paper:

> **Continual Learning for 6-DoF Grasp Synthesis via Experience and
> Demonstrations** \
> Giulio Schiavi, Andrei Cramariuc, Michael Pantic, and Roland Siegwart
>
> [Paper](https://arxiv.org/abs/2610.01301) |
> [Project page](https://giuschio.github.io/cl_grasping/)

To keep the individual components reusable, the code supporting the paper is
being released across separate repositories:

- **Simulation environment (this repository):** scene generation, data
  collection, and simulated grasp evaluation. This is the environment used for
  the paper's data collection and simulation experiments.
- **[Grasp detection model]():** model training and grasp inference (coming
  soon).
- **[Robot control and deployment]():** real-robot execution code (coming
  soon).
- **[Paper benchmark]():** the top-level reproduction repository, which brings
  the three components above together as Git submodules and provides the
  benchmarking scripts used for the paper (coming soon).

If your goal is to reproduce the paper's results end to end, use the benchmark
repository once it is available. This repository can be used independently for
simulation and data collection.

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{schiavi2026continuallearning6dofgrasp,
  title         = {Continual Learning for 6-DoF Grasp Synthesis via Experience and Demonstrations},
  author        = {Giulio Schiavi and Andrei Cramariuc and Michael Pantic and Roland Siegwart},
  year          = {2026},
  eprint        = {2610.01301},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.01301}
}
```

## System requirements

Python 3.10 or newer is required.

## Installation

Install in editable mode because the simulator resolves its object and robot
assets from the source checkout:

```bash
python -m pip install -e .
```

## Assets

Robot and environment files used internally by the simulator are tracked under
`grasp_env/assets/`. Object datasets are not part of this repository due to licensing constraints. You can download them as follows:
- **VGN:** [download the VGN data](https://github.com/ethz-asl/vgn#installation)
  and extract it so that `packed/` and `pile/` are directly under
  `assets/vgn_objects/`.
- **EGAD:** [download the EGAD evaluation set](https://dougsm.github.io/egad/)
  and place its OBJ files under `assets/egad_objects/test/`.
- **GraspNet-1Billion:** [download the object models](https://graspnet.net/datasets.html).
  This project expects each selected object's `nontextured_simplified.ply` under
  `assets/graspnet_1B_objects/test/<object-id>/`.
- **DexGraspNet:** [download DexGraspNet](https://pku-epic.github.io/DexGraspNet/).
  This project uses CoACD-processed meshes at
  `assets/dexgraspnet_objects/objects/<object-id>/coacd/decomposed.obj`.
- **Custom objects:** [download the custom object archive](https://drive.google.com/file/d/1LTIz7IBYkk5Ka1KP6Q3Qrp6kMgn_n2SD/view?usp=sharing)
  and extract it so that the object files are under `assets/custom_objects/`.

Put them in the repository-level `assets/` directory. Below is the directory stucture assumed by the package:

```text
grasp-env/
|-- asset_directories/
|   |-- datasets.yaml
|   `-- categories/*.csv
|-- assets/
|   |-- custom_objects/
|   |-- dexgraspnet_objects/
|   |   `-- objects/
|   |       |-- core-bottle-1a7ba1f4c892e2da30711cdbdbc73924/
|   |       `-- .../
|   |-- egad_objects/
|   |-- graspnet_1B_objects/
|   |   `-- test/
|   |       |-- 000/
|   |       `-- .../
|   `-- vgn_objects/
|      |-- packed/
|      `-- pile/
`-- grasp_env/assets/             # tracked simulator environment and robot files
```

The default asset catalog is `asset_directories/datasets.yaml`. An asset catalog
is a small YAML configuration file that tells `ModelDatabase` which directory
and loader correspond to names such as `vgn` and `dexgraspnet`, together with
their default scale, orientation, and spacing. It does not contain object data.
Most users should use the default catalog.

The CSV files under `asset_directories/categories/` are tracked. Each row maps a
category member to a path relative to `assets/`, with its relative scale and
preferred orientation. For example:

```python
from grasp_env.simulator.object_models import ModelDatabase

objects = ModelDatabase("category-mug")
vgn_test = ModelDatabase("vgn-packed-test")
```

By default, object data is loaded from this checkout's `assets/` directory. To
store the downloaded object directories elsewhere, set:

```bash
export GRASP_ENV_ASSET_ROOT=/data/grasp-env-assets
```

That directory should directly contain `vgn_objects/`, `egad_objects/`, and any
other installed dataset directories shown in the layout above. The equivalent
Python argument is `assets_root` on `ModelDatabase`, `ClutterRemovalSim`, and
`GraspEnv`.

To use a modified or independently maintained catalog, set:

```bash
export GRASP_ENV_ASSET_CATALOG=/path/to/my-datasets.yaml
```

The equivalent Python argument is `catalog_path` on `ModelDatabase` and
`asset_catalog_path` on `ClutterRemovalSim` and `GraspEnv`. Relative
`asset_root` and `category_directory` values inside a custom catalog are
resolved relative to that catalog file.

## Reference frames
All grasps are expressed in the grasp frame (center of two fingers, closing direction in the y axis).

<img src="figures/hand_w_grasp_frame.png" alt="Grasp frame" width="300"/>

The visualized frames are:
- origin of the URDF (I think)
- origin of the grasp frame (no rotation wrt urdf)

See ```scripts/visualize_grasp_frame.py```.

In pybullet, the center of the body is not (as one might expect) the origin of the URDF. Instead, it is further inside the hand. In the simulation code, we define the gripper frame relative to this body frame. The Gripper class defines the transform.
```python
class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world):
        # ...
        self.T_body_graspframe = Transform(Rotation.identity(), [0.0, 0.0, 0.06])
```
We can therefore call:
```python
Gripper.reset(DesiredGraspFrame)
Gripper.move(DesiredGraspFrame)
```
and the Gripper class will take care of the offsets.

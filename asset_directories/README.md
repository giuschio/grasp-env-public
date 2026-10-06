# Asset directory metadata

This directory is tracked by Git; the object files in the repository-level
`assets/` directory are not.

- `datasets.yaml` maps dataset names to local directories and `ModelDatabase`
  loaders, with their default scale, orientation, and spacing.
- `categories/*.csv` defines the project-specific object categories. Paths in
  these files are relative to the configured object asset root.

To add another dataset, add a named entry to `datasets.yaml` and register its
loader in `grasp_env.simulator.object_models.ModelDatabase`. To add or edit a
category, create or update its CSV; category aliases and semantic groups are not
expanded implicitly.

Download and directory-layout instructions belong in the repository README,
not in the machine-readable catalog.

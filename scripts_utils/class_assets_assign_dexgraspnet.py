import csv
from pathlib import Path

from grasp_env.simulator.object_models import (
    ModelDatabase,
    get_assets_root,
    get_category_directory,
)

# Dataset configuration
category = "bowl"
dataset = f"dexgraspnet-{category}"

def main():
    # Load models and spacing using the ModelDatabase class
    db = ModelDatabase(dataset)
    files = db.object_files

    assets_base_path = get_assets_root()

    # Create CSV file
    csv_filename = get_category_directory() / f"{category}.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["path", "relative_size"])  # Write headers
        for file_path in files:
            file_path = Path(file_path).resolve()
            try:
                relative_path = file_path.relative_to(assets_base_path)
            except ValueError:
                print(f"Warning: {file_path} is not under {assets_base_path}. Skipping.")
                continue
            writer.writerow([str(relative_path), 1.0])  # Relative path and fixed dimension

    print(f"CSV file '{csv_filename}' created with {len(files)} entries.")

if __name__ == '__main__':
    main()

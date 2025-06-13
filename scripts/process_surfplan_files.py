from pathlib import Path
from SurfplanAdapter import generate_geometry_csv_files, generate_wing_yaml
from SurfplanAdapter.utils import PROJECT_DIR


def main(kite_name="V9_60J-Inertia"):
    """It is crucial that the kite_name matches the name of the surfplan file"""

    data_dir = Path(PROJECT_DIR) / "data" / f"{kite_name}"
    path_surfplan_file = Path(data_dir) / f"{kite_name}.txt"
    save_dir = Path(PROJECT_DIR) / "processed_data" / f"{kite_name}"

    if not path_surfplan_file.exists():
        raise FileNotFoundError(
            f"\nSurfplan file {path_surfplan_file} does not exist. "
            "Please check the .txt file name and ensure it matches the data_dir name."
            "It is essential that the kite_name matches the name of the surfplan file."
        )

    generate_geometry_csv_files.main(
        path_surfplan_file=path_surfplan_file,
        save_dir=save_dir,
        profile_load_dir=Path(data_dir) / "profiles",
        profile_save_dir=Path(save_dir) / "profiles",
    )

    generate_wing_yaml.main(
        path_surfplan_file=path_surfplan_file,
        save_dir=save_dir,
        profile_load_dir=Path(data_dir) / "profiles",
        profile_save_dir=Path(save_dir) / "profiles",
        airfoil_type="neuralfoil",  # Default: masure_regression with .dat files and parameters
    )


if __name__ == "__main__":
    main()

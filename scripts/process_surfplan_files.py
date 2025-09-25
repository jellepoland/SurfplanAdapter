from pathlib import Path
from SurfplanAdapter.plotting import (
    plot_airfoils_3d_from_yaml,
    plot_struc_geometry_yaml,
)
from SurfplanAdapter.generate_yaml import main_generate_yaml
from SurfplanAdapter.process_wing import main_process_wing
from SurfplanAdapter.process_bridle_lines import main_process_bridle_lines

PROJECT_DIR = Path(__file__).resolve().parent.parent


def main(kite_name="TUDELFT_V3_KITE", airfoil_type="masure_regression"):
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

    profile_load_dir = Path(data_dir) / "profiles"
    profile_save_dir = Path(save_dir) / "profiles"

    # process wing
    ribs_data = main_process_wing.main(
        surfplan_txt_file_path=path_surfplan_file,
        profile_load_dir=profile_load_dir,
        profile_save_dir=profile_save_dir,
    )

    # process bridle lines
    bridle_lines = main_process_bridle_lines.main(path_surfplan_file)

    # generate yaml file
    yaml_file_path = save_dir / "config_kite.yaml"

    main_generate_yaml.main(
        ribs_data=ribs_data,
        bridle_lines=bridle_lines,
        yaml_file_path=yaml_file_path,
        airfoil_type=airfoil_type,
    )

    # Generate 3D plot of airfoils from the created YAML file
    plot_struc_geometry_yaml(Path(save_dir) / "struc_geometry.yaml")
    plot_airfoils_3d_from_yaml(
        yaml_file_path=yaml_file_path,
        profile_base_dir=Path(yaml_file_path.parent / "profiles"),
        save_path=save_dir / "3d_airfoil_plot.png",  # if given it will also save
        show_plot=True,  # Set to False to avoid blocking in automated runs
    )


if __name__ == "__main__":
    main()

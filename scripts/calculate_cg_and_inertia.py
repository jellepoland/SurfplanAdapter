from pathlib import Path
from SurfplanAdapter import calculate_cg_and_inertia
from SurfplanAdapter.generate_yaml import main_generate_yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]


def main_generate_yaml(
    kite_name="TUDELFT_V3_KITE",
    total_wing_mass=10.0,
    canopy_kg_p_sqm=0.05,
    le_to_strut_mass_ratio=0.7,
    sensor_mass=0.5,
    is_show_plot=True,
    desired_point=[0, 0, 0],
):
    """Calculate CG and inertia for a given kite_name. Requires config_kite.yaml to exist or will generate it."""
    data_dir = Path(PROJECT_DIR) / "data" / f"{kite_name}"
    path_surfplan_file = Path(data_dir) / f"{kite_name}.txt"
    save_dir = Path(PROJECT_DIR) / "processed_data" / f"{kite_name}"
    yaml_file_path = Path(save_dir) / "config_kite.yaml"

    # Generate YAML config if it does not exist
    if not yaml_file_path.exists():
        main_generate_yaml.main(
            path_surfplan_file=path_surfplan_file,
            save_dir=save_dir,
            profile_load_dir=Path(data_dir) / "profiles",
            profile_save_dir=Path(save_dir) / "profiles",
            airfoil_type="masure_regression",
        )

    calculate_cg_and_inertia.main(
        yaml_file_path,
        total_wing_mass=total_wing_mass,
        canopy_kg_p_sqm=canopy_kg_p_sqm,
        le_to_strut_mass_ratio=le_to_strut_mass_ratio,
        sensor_mass=sensor_mass,
        desired_point=desired_point,
        is_show_plot=is_show_plot,
    )


if __name__ == "__main__":
    main_generate_yaml()

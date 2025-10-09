from pathlib import Path
from SurfplanAdapter import calculate_cg_and_inertia


def main(
    kite_name="TUDELFT_V3_KITE",
    total_wing_mass=10.0,
    canopy_kg_p_sqm=0.05,
    le_to_strut_mass_ratio=0.7,
    sensor_mass=0.5,
    is_show_plot=True,
    desired_point=[0, 0, 0],
):
    """Calculate CG and inertia for a given kite_name. Requires config_kite.yaml to exist or will generate it."""
    PROJECT_DIR = Path(__file__).resolve().parents[1]

    save_dir = Path(PROJECT_DIR) / "processed_data" / f"{kite_name}"
    yaml_file_path = Path(save_dir) / "aero_geometry.yaml"

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
    main()

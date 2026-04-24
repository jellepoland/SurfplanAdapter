from pathlib import Path
from SurfplanAdapter import calculate_cg_and_inertia


def main(
    kite_name="TUDELFT_V3_KITE",
    yaml_filename="aero_geometry.yaml",
    yaml_file_path=None,
    total_wing_mass=10.0,
    canopy_kg_p_sqm=0.1,  # 100g/m2
    le_to_strut_mass_ratio=None,  # if None it is auto-derived
    sensor_mass=0.0,
    mid_span_valve_weight=0.0,
    strut_tube_weight=0.0,
    include_bridle_mass=True,
    is_show_plot=True,
    desired_point=[0, 0, 0],
):
    """Calculate CG and inertia for aero_geometry.yaml or struc_geometry.yaml."""
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    save_dir = Path(PROJECT_DIR) / "processed_data" / f"{kite_name}"
    if yaml_file_path is None:
        yaml_file_path = Path(save_dir) / yaml_filename
    else:
        yaml_file_path = Path(yaml_file_path)

    calculate_cg_and_inertia.main(
        yaml_file_path,
        total_wing_mass=total_wing_mass,
        canopy_kg_p_sqm=canopy_kg_p_sqm,
        le_to_strut_mass_ratio=le_to_strut_mass_ratio,
        sensor_mass=sensor_mass,
        mid_span_valve_weight=mid_span_valve_weight,
        strut_tube_weight=strut_tube_weight,
        include_bridle_mass=include_bridle_mass,
        desired_point=desired_point,
        is_show_plot=is_show_plot,
    )


if __name__ == "__main__":
    main(yaml_filename="struc_geometry.yaml")

from pathlib import Path
from SurfplanAdapter import calculate_cg_and_inertia
from SurfplanAdapter.utils import PROJECT_DIR
from SurfplanAdapter.surfplan_to_vsm.generate_vsm_input import generate_VSM_input

if __name__ == "__main__":

    ### User inputs
    data_folder_name = "TUDELFT_V3_LEI_KITE"
    kite_file_name = "TUDELFT_V3_LEI_KITE_3d"
    kite_name = "TUDELFT_V3_LEI_KITE"

    ## Creating Paths
    geometry_file_path = (
        Path(PROJECT_DIR) / "processed_data" / f"{kite_name}" / "geometry.csv"
    )
    path_surfplan_file = (
        Path(PROJECT_DIR) / "data" / f"{data_folder_name}" / f"{kite_file_name}.txt"
    )
    path_to_save_geometry = (
        Path(PROJECT_DIR) / "processed_data" / f"{data_folder_name}" / "geometry.csv"
    )
    ## Create geometry file once
    if not geometry_file_path.exists():
        wing_aero_breukels = generate_VSM_input(
            path_surfplan_file=path_surfplan_file,
            n_panels=30,
            spanwise_panel_distribution="unchanged",
            airfoil_input_type="lei_airfoil_breukels",
            is_save_geometry=True,
            path_to_save_geometry=path_to_save_geometry,
        )
        print(f'Generated geometry file at "{path_to_save_geometry}"')

    file_path = Path(PROJECT_DIR) / "processed_data" / f"{kite_name}" / "geometry.csv"
    total_wing_mass = 10.0
    canopy_kg_p_sqm = 0.05
    le_to_strut_mass_ratio = 0.7
    sensor_mass = 0.5
    is_show_plot = True
    desired_point = [0, 0, 0]

    calculate_cg_and_inertia.main(
        file_path,
        total_wing_mass,
        canopy_kg_p_sqm,
        le_to_strut_mass_ratio,
        sensor_mass,
        desired_point=desired_point,
        is_show_plot=is_show_plot,
    )

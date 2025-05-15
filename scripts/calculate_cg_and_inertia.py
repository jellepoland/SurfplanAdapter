from pathlib import Path
from SurfplanAdapter import calculate_cg_and_inertia
from SurfplanAdapter.utils import PROJECT_DIR
from SurfplanAdapter import generate_geometry_csv_files

if __name__ == "__main__":

    ### User inputs
    data_folder_name = "TUDELFT_V3_KITE"
    kite_file_name = "TUDELFT_V3_KITE_3d"
    kite_name = "TUDELFT_V3_KITE"

    ## Creating Paths
    path_surfplan_file = (
        Path(PROJECT_DIR) / "data" / f"{data_folder_name}" / f"{kite_file_name}.txt"
    )
    dir_to_save_in = Path(PROJECT_DIR) / "processed_data" / f"{data_folder_name}"

    ## Create geometry file once
    geometry_file_path = Path(dir_to_save_in) / "wing_geometry.csv"
    if not geometry_file_path.exists():
        data_dir = Path(PROJECT_DIR) / "data" / f"{data_folder_name}"
        save_dir = Path(PROJECT_DIR) / "processed_data" / f"{data_folder_name}"

        generate_geometry_csv_files.main(
            path_surfplan_file=path_surfplan_file,
            save_dir=save_dir,
            profile_load_dir=Path(data_dir) / "profiles",
            profile_save_dir=Path(save_dir) / "profiles",
        )

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

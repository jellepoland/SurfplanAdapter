from pathlib import Path
from SurfplanAdapter import generate_geometry_csv_files
from SurfplanAdapter.utils import PROJECT_DIR

if __name__ == "__main__":

    data_folder_name = "TUDELFT_V3_KITE"
    kite_file_name = "TUDELFT_V3_KITE_3d"

    data_dir = Path(PROJECT_DIR) / "data" / f"{data_folder_name}"
    path_surfplan_file = Path(data_dir) / f"{kite_file_name}.txt"
    save_dir = Path(PROJECT_DIR) / "processed_data" / f"{data_folder_name}"

    generate_geometry_csv_files.main(
        path_surfplan_file=path_surfplan_file,
        save_dir=save_dir,
        profile_load_dir=Path(data_dir) / "profiles",
        profile_save_dir=Path(save_dir) / "profiles",
    )

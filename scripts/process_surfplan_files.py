from pathlib import Path
import matplotlib.pyplot as plt
from SurfplanAdapter.plotting import (
    plot_airfoils_3d_from_yaml,
    plot_struc_geometry_yaml,
    plot_struct_geometry_all_in_surfplan_yaml,
)
from SurfplanAdapter.generate_yaml import main_generate_yaml
from SurfplanAdapter.process_wing import main_process_wing
from SurfplanAdapter.process_bridle_lines import main_process_bridle_lines
from SurfplanAdapter.find_airfoil_parameters.plot_airfoils_comparison import (
    plot_all_airfoils,
)


def main(kite_name="TUDELFT_V3_KITE", airfoil_type="masure_regression"):
    """It is crucial that the kite_name matches the name of the surfplan file"""

    PROJECT_DIR = Path(__file__).resolve().parents[1]

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
    config_kite_yaml_path = save_dir / "config_kite.yaml"

    main_generate_yaml.main(
        ribs_data=ribs_data,
        bridle_lines=bridle_lines,
        yaml_file_path=config_kite_yaml_path,
        airfoil_type=airfoil_type,
    )

    # Generate 3D plot of airfoils from the created YAML file
    plot_struc_geometry_yaml(Path(save_dir) / "struc_geometry.yaml")
    plot_struct_geometry_all_in_surfplan_yaml(
        Path(save_dir) / "struc_geometry_all_in_surfplan.yaml"
    )
    plot_airfoils_3d_from_yaml(
        yaml_file_path=config_kite_yaml_path,
        profile_base_dir=Path(config_kite_yaml_path.parent / "profiles"),
        save_path=save_dir / "3d_airfoil_plot.png",  # if given it will also save
        show_plot=True,  # Set to False to avoid blocking in automated runs
    )

    # Generate comparison plot of parametric vs CAD-sliced airfoils
    print("\n" + "=" * 60)
    print("Generating airfoil comparison plot")
    print("=" * 60)
    aero_geometry_path = save_dir / "aero_geometry.yaml"
    comparison_plot_path = save_dir / f"airfoils_in_{aero_geometry_path.stem}.pdf"
    plot_all_airfoils(
        yaml_path=aero_geometry_path,
        output_path=comparison_plot_path,
        surfplan_airfoils_dir=profile_load_dir,
    )

    # plot the distribution of chambers along the span
    print("\n" + "=" * 60)
    print("Processing completed successfully.")
    print("=" * 60)

    # try:
    #     # Extract spanwise positions and chamber (max camber) values
    #     # ribs_data is expected to be a sequence of dicts containing 'LE' and 'y_max_camber' keys
    #     spans = [rib["LE"][1] for rib in ribs_data]
    #     chambers = [
    #         rib.get("y_max_camber", rib.get("kappa_val", None)) for rib in ribs_data
    #     ]

    #     # Filter out ribs with missing chamber data
    #     span_filtered = []
    #     chamber_filtered = []
    #     for s, c in zip(spans, chambers):
    #         if c is None:
    #             continue
    #         span_filtered.append(s)
    #         chamber_filtered.append(c)

    #     # Sort by span coordinate so the plot is along the span
    #     paired = sorted(zip(span_filtered, chamber_filtered), key=lambda x: x[0])
    #     if paired:
    #         span_sorted, chamber_sorted = map(list, zip(*paired))
    #     else:
    #         span_sorted, chamber_sorted = [], []

    #     plt.figure(figsize=(8, 4))
    #     plt.plot(span_sorted, chamber_sorted, "-o")
    #     plt.grid(True, linestyle="--", alpha=0.5)
    #     plt.xlabel("Span coordinate (m)")
    #     plt.ylabel("Max camber (m)")
    #     plt.title("Distribution of Chambers Along the Span")

    #     # Ensure save directory exists and save plot
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     outpath = save_dir / "chamber_distribution.png"
    #     plt.tight_layout()
    #     plt.savefig(outpath)
    #     try:
    #         plt.show()
    #     except Exception:
    #         # In non-interactive environments show() may fail or block — ignore
    #         pass
    #     print(f"Saved chamber distribution plot to: {outpath}")
    # except Exception as exc:
    #     print(f"Could not create chamber distribution plot: {exc}")


if __name__ == "__main__":
    main()

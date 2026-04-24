from pathlib import Path
import matplotlib.pyplot as plt
import inspect
import sys
from SurfplanAdapter.plotting import (
    plot_airfoils_3d_from_yaml,
    plot_struc_geometry_yaml,
    plot_struct_geometry_all_in_surfplan_yaml,
)
from SurfplanAdapter.generate_yaml import main_generate_yaml
from SurfplanAdapter.generate_yaml.utils import yaml_reader
from SurfplanAdapter.process_wing import main_process_wing
from SurfplanAdapter.process_bridle_lines import main_process_bridle_lines
from SurfplanAdapter.find_airfoil_parameters.plot_airfoils_comparison import (
    plot_all_airfoils,
)
from SurfplanAdapter import calculate_cg_and_inertia


def _parse_cli_kite_name(argv):
    """
    Parse CLI kite-name overrides.

    Supported forms:
    - positional: TUDELFT_V3_KITE
    - named option: --kite-name TUDELFT_V3_KITE
    - inline option: --kite-name=TUDELFT_V3_KITE
    - flag-as-name: --TUDELFT_V3_KITE
    """
    if any(arg in ("-h", "--help") for arg in argv):
        print("Usage:")
        print("  python scripts/process_surfplan_files.py [KITE_NAME]")
        print("  python scripts/process_surfplan_files.py --kite-name KITE_NAME")
        print("  python -m scripts.process_surfplan_files --KITE_NAME")
        print("")
        print("Examples:")
        print("  python -m scripts.process_surfplan_files --TUDELFT_V3_KITE")
        print("  python scripts/process_surfplan_files.py TUDELFT_V3_KITE")
        sys.exit(0)

    kite_name = None
    unknown_args = []
    i = 0
    while i < len(argv):
        token = str(argv[i])

        if token in ("--kite-name", "--kite_name"):
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value after {token}")
            kite_name = str(argv[i + 1])
            i += 2
            continue

        if token.startswith("--kite-name=") or token.startswith("--kite_name="):
            kite_name = token.split("=", 1)[1]
            i += 1
            continue

        # Support style: --TUDELFT_V3_KITE
        if token.startswith("--") and len(token) > 2 and "=" not in token:
            if kite_name is None:
                kite_name = token[2:]
            else:
                unknown_args.append(token)
            i += 1
            continue

        if token.startswith("-"):
            unknown_args.append(token)
            i += 1
            continue

        # Positional fallback
        if kite_name is None:
            kite_name = token
        else:
            unknown_args.append(token)
        i += 1

    return kite_name, unknown_args


def main(
    kite_name="TUDELFT_V3_KITE",
    ### Plotting
    is_with_struc_geometry_plot=False,
    is_with_struc_geometry_all_in_surfplan_yaml=False,
    is_with_airfoil_3d_plot=False,
    is_with_cg_and_inertia_plot=False,
    is_with_spanwise_chamber_plot=False,
    ### Aerodynamic specific input
    airfoil_type="masure_regression",
    ### For calculating CG and inertia
    total_wing_mass=10.0,
    canopy_kg_p_sqm=0.1,  # g/m2
    tube_kg_p_sqm=0.4,  # if None it is auto-derived to match total_wing_mass
    le_to_strut_mass_ratio=None,  # if None it is auto-derived
    sensor_mass=0.0,
    mid_span_valve_weight=0.1,  # kg, added at mid-span LE node(s)
    strut_tube_weight=0.1,  # kg per non-tip LE strut-attachment node
    include_bridle_mass=False,
    desired_point=[0, 0, 0],
):
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
        total_wing_mass=total_wing_mass,
        canopy_kg_p_sqm=canopy_kg_p_sqm,
        le_to_strut_mass_ratio=le_to_strut_mass_ratio,
        tube_kg_p_sqm=tube_kg_p_sqm,
        sensor_mass=sensor_mass,
        mid_span_valve_weight=mid_span_valve_weight,
        strut_tube_weight=strut_tube_weight,
    )

    # Generate 3D plot of airfoils from the created YAML file
    if is_with_struc_geometry_plot:
        plot_struc_geometry_yaml(Path(save_dir) / "struc_geometry.yaml")
    if is_with_struc_geometry_all_in_surfplan_yaml:
        plot_struct_geometry_all_in_surfplan_yaml(
            Path(save_dir) / "struc_geometry_all_in_surfplan.yaml"
        )
    if is_with_airfoil_3d_plot:
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

    calculate_cg_and_inertia.main(
        Path(save_dir) / "struc_geometry.yaml",
        total_wing_mass=total_wing_mass,
        canopy_kg_p_sqm=canopy_kg_p_sqm,
        le_to_strut_mass_ratio=le_to_strut_mass_ratio,
        tube_kg_p_sqm=tube_kg_p_sqm,
        sensor_mass=sensor_mass,
        mid_span_valve_weight=mid_span_valve_weight,
        strut_tube_weight=strut_tube_weight,
        include_bridle_mass=include_bridle_mass,
        desired_point=desired_point,
        is_show_plot=is_with_cg_and_inertia_plot,
    )

    if is_with_spanwise_chamber_plot:
        try:
            # Extract spanwise positions and chamber (max camber) values
            # ribs_data is expected to be a sequence of dicts containing 'LE' and 'y_max_camber' keys
            spans = [rib["LE"][1] for rib in ribs_data]
            chambers = [
                rib.get("y_max_camber", rib.get("kappa_val", None)) for rib in ribs_data
            ]

            # Filter out ribs with missing chamber data
            span_filtered = []
            chamber_filtered = []
            for s, c in zip(spans, chambers):
                if c is None:
                    continue
                span_filtered.append(s)
                chamber_filtered.append(c)

            # Sort by span coordinate so the plot is along the span
            paired = sorted(zip(span_filtered, chamber_filtered), key=lambda x: x[0])
            if paired:
                span_sorted, chamber_sorted = map(list, zip(*paired))
            else:
                span_sorted, chamber_sorted = [], []

            plt.figure(figsize=(8, 4))
            plt.plot(span_sorted, chamber_sorted, "-o")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xlabel("Span coordinate (m)")
            plt.ylabel("Max camber (m)")
            plt.title("Distribution of Chambers Along the Span")

            # Ensure save directory exists and save plot
            save_dir.mkdir(parents=True, exist_ok=True)
            outpath = save_dir / "chamber_distribution.png"
            plt.tight_layout()
            plt.savefig(outpath)
            try:
                plt.show()
            except Exception:
                # In non-interactive environments show() may fail or block — ignore
                pass
            print(f"Saved chamber distribution plot to: {outpath}")
        except Exception as exc:
            print(f"Could not create chamber distribution plot: {exc}")


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    default_config_path = PROJECT_DIR / "data" / "default_kite" / "config.yaml"

    # 1) Start from shared defaults.
    runtime_config = yaml_reader(default_config_path, required=True)

    # 2) Optional CLI override for kite_name.
    #    Examples:
    #      python scripts/process_surfplan_files.py TUDELFT_V3_KITE
    #      python scripts/process_surfplan_files.py --kite-name TUDELFT_V3_KITE
    #      python -m scripts.process_surfplan_files --TUDELFT_V3_KITE
    kite_name_cli, unknown_cli_args = _parse_cli_kite_name(sys.argv[1:])
    if kite_name_cli:
        runtime_config["kite_name"] = kite_name_cli
    if unknown_cli_args:
        print(
            "Ignoring unknown CLI argument(s): "
            + ", ".join(str(arg) for arg in unknown_cli_args)
        )

    # 3) Merge per-kite overrides if that config exists.
    kite_name = str(runtime_config.get("kite_name", "default_kite"))
    kite_config_path = PROJECT_DIR / "data" / kite_name / "config.yaml"
    kite_config = yaml_reader(kite_config_path, required=False)
    runtime_config = {**runtime_config, **kite_config}

    # 4) Guard against unknown keys in YAML.
    allowed_keys = set(inspect.signature(main).parameters.keys())
    filtered_config = {k: v for k, v in runtime_config.items() if k in allowed_keys}
    unknown_keys = sorted(set(runtime_config.keys()) - allowed_keys)
    if unknown_keys:
        print(
            "Ignoring unknown config key(s): "
            + ", ".join(str(key) for key in unknown_keys)
        )

    main(**filtered_config)

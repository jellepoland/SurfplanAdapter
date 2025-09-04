# %% importing necessary modules
from pathlib import Path
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.plotting import plot_polars
from VSM.plot_geometry_matplotlib import plot_geometry
from VSM.plot_geometry_plotly import interactive_plot
import numpy as np
from SurfplanAdapter.utils import PROJECT_DIR


def main(kite_name="TUDELFT_V3_KITE"):
    """
    Example: 3D Aerodynamic Analysis of TUDELFT_V3_KITE using VSM

    This script demonstrates the workflow for performing a 3D aerodynamic analysis of the TUDELFT_V3_KITE
    using the Vortex Step Method (VSM) library. The workflow is structured as follows:

    Step 1: Instantiate BodyAerodynamics objects from different YAML configuration files.
        - Each YAML config defines the geometry and airfoil/polar data for a specific modeling approach.
        - Supported approaches include:
            - Masure regression (empirical model with comprehensive parameters)
            - Breukels regression (simplified empirical model)
            - NeuralFoil-based polars
            - CFD-based polars
            - Inviscid theory

    Step 2: Set inflow conditions for each aerodynamic object.
        - Specify wind speed (Umag), angle of attack, side slip, and yaw rate.
        - Initialize the apparent wind for each BodyAerodynamics object.

    Step 3: Plot the kite geometry using Matplotlib.
        - Visualize the panel mesh, control points, and aerodynamic centers.

    Step 4: Create an interactive 3D plot using Plotly.
        - Allows for interactive exploration of the geometry and panel arrangement.

    Step 5: Plot and save polar curves for different angles of attack and side slip angles.
        - Compare the results of different aerodynamic models.
        - Optionally include literature/CFD data for validation.

    Step 5a: Plot alpha sweep (angle of attack variation).
    Step 5b: Plot beta sweep (side slip variation).

    Returns:
        None
    """
    # Step 1: Define paths and settings
    config_file_path = (
        Path(PROJECT_DIR) / "processed_data" / f"{kite_name}" / "config_kite.yaml"
    )
    # Aerodynamic analysis settings
    n_panels = 36
    spanwise_panel_distribution = "uniform"
    solver_base_version = Solver()

    # Flight conditions
    Umag = 10.0  # Wind speed [m/s]
    angle_of_attack = 6.8  # Angle of attack [degrees]
    side_slip = 0.0  # Side slip angle [degrees]
    yaw_rate = 0.0  # Yaw rate [rad/s]

    print(f"Loading configuration from: {config_file_path}")
    print("Analysis settings:")
    print(f"  - Number of panels: {n_panels}")
    print(f"  - Panel distribution: {spanwise_panel_distribution}")
    print(f"  - Wind speed: {Umag} m/s")
    print(f"  - Angle of attack: {angle_of_attack}°")
    print(f"  - Side slip: {side_slip}°")
    print(f"  - Yaw rate: {yaw_rate} rad/s")

    # Step 2: Instantiate BodyAerodynamics objects
    print("\nStep 1: Instantiating BodyAerodynamics objects...")

    # Main configuration with masure regression (default)
    body_aero = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=config_file_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        # is_with_bridles=False,
    )
    print("  ✓ BodyAerodynamics Instantiated")

    # Same configuration but with bridles
    # body_aero_with_bridles = BodyAerodynamics.instantiate(
    #     n_panels=n_panels,
    #     file_path=config_file_path,
    #     spanwise_panel_distribution=spanwise_panel_distribution,
    #     # is_with_bridles=True,
    # )
    # print("  ✓ BodyAerodynamics Instantiated with Bridles")

    # Step 3: Set inflow conditions for each aerodynamic object
    print("\nStep 2: Setting inflow conditions...")
    body_aero.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
    # body_aero_with_bridles.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)

    print("  ✓ Inflow conditions initialized for all models")

    # Step 4: Plot the kite geometry using Matplotlib
    print("\nStep 3: Plotting kite geometry...")
    plot_geometry(
        body_aero,
        title=f"{kite_name}",
        data_type=".pdf",
        save_path=".",
        is_save=False,
        is_show=True,
    )
    print("  ✓ Geometry plot generated")

    # Step 5: Create an interactive plot using Plotly
    print("\nStep 4: Creating interactive 3D plot...")
    interactive_plot(
        body_aero,
        vel=Umag,
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        is_with_aerodynamic_details=True,
        title=f"{kite_name}",
        # is_with_bridles=True,
    )
    print("  ✓ Interactive plot generated")

    # Step 6: Define results folder
    save_folder = Path(PROJECT_DIR) / "results" / f"{kite_name}"
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {save_folder}")

    # Step 7a: Plot alpha sweep (angle of attack)
    print("\nStep 5a: Generating alpha sweep polar plots...")

    # Define angle ranges for sweeps
    alpha_range = np.linspace(3, 18, 8)  # 3° to 18° in 8 steps
    beta_range = np.linspace(-6, 6, 7)  # -6° to 6° in 7 steps

    plot_polars(
        solver_list=[
            solver_base_version,
        ],
        body_aero_list=[
            body_aero,
        ],
        label_list=[
            "VSM",
            "VSM with Bridles",
        ],
        literature_path_list=[],
        angle_range=alpha_range,
        angle_type="angle_of_attack",
        angle_of_attack=0,
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        Umag=Umag,
        title=f"{kite_name}_alpha_sweep",
        data_type=".pdf",
        save_path=save_folder,
        is_save=True,
        is_show=True,
    )
    print("  ✓ Alpha sweep completed")

    # Step 7b: Plot beta sweep (side slip)
    print("\nStep 5b: Generating beta sweep polar plots...")
    plot_polars(
        solver_list=[
            solver_base_version,
        ],
        body_aero_list=[
            body_aero,
        ],
        label_list=[
            "VSM",
        ],
        literature_path_list=[],
        angle_range=beta_range,
        angle_type="side_slip",
        angle_of_attack=angle_of_attack,
        side_slip=0,
        yaw_rate=yaw_rate,
        Umag=Umag,
        title=f"{kite_name}_beta_sweep",
        data_type=".pdf",
        save_path=save_folder,
        is_save=True,
        is_show=True,
    )
    print("  ✓ Beta sweep completed")

    # Step 8: Print summary
    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Configuration file: {config_file_path.name}")
    print(f"Number of panels: {n_panels}")
    print(f"Results saved to: {save_folder}")
    print("=" * 60)

    # Step 9: Single point analysis at current conditions
    print(f"\nSingle point analysis at α={angle_of_attack}°, β={side_slip}°:")

    # Solve for current conditions
    results_masure = solver_base_version.solve(body_aero)
    # results_with_bridles = solver_base_version.solve(body_aero_with_bridles)

    print(f"Coefficients:")
    print(f"  CL = {results_masure['cl']:.4f}")
    print(f"  CD = {results_masure['cd']:.4f}")
    print(f"  CS = {results_masure['cs']:.4f}")
    print(f"  L/D = {results_masure['cl']/results_masure['cd']:.2f}")

    # print(f"Coefficients with Bridles:")
    # print(f"  CL = {results_with_bridles['cl']:.4f}")
    # print(f"  CD = {results_with_bridles['cd']:.4f}")
    # print(f"  CS = {results_with_bridles['cs']:.4f}")
    # print(f"  L/D = {results_with_bridles['cl']/results_with_bridles['cd']:.2f}")

    # # Calculate bridle drag penalty
    # delta_cd = results_with_bridles["cd"] - results_masure["cd"]
    # delta_ld = (results_with_bridles["cl"] / results_with_bridles["cd"]) - (
    #     results_masure["cl"] / results_masure["cd"]
    # )
    # print(f"Bridle drag penalty:")
    # print(f"  ΔCD = {delta_cd:.4f}")
    # print(f"  Δ(L/D) = {delta_ld:.2f}")


if __name__ == "__main__":
    main()

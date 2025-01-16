# %% importing necessary modules
import numpy as np
from pathlib import Path
from SurfplanAdapter.surfplan_to_vsm.generate_vsm_input import (
    generate_VSM_input,
)
from SurfplanAdapter.utils import project_dir
from SurfplanAdapter.logging_config import *
from VSM.Solver import Solver
import VSM.plotting as plotting
from VSM.interactive import interactive_plot

# %% Defining kite name
# should be equal to folder name, e.g "default_kite/"
# should also be "{kite_name}_3d".txt, e.g "default_kite_3d.txt"
kite_name = "TUDELFT_V3_LEI_KITE"

# %% Running VSM

## Using breukels model without saving the geometry
csv_save_file = Path(project_dir) / "processed_data" / f"{kite_name}" / "geometry.csv"
wing_aero_breukels = generate_VSM_input(
    kite_name,
    n_panels=30,
    spanwise_panel_distribution="linear",
    is_save_geometry=True,
    csv_file_path=csv_save_file,
)

## Using polar_data input, requires one to change the "airfoil_input_type" entry
# A polar data folder should be created and placed inside the same kite_name folder in data/
# Polar data must be provided for all profiles (prof_1.dat,prof_2.dat, ..)
# The polar dat should be named prof_1_polar.csv, prof_2_polar.csv etc.
wing_aero_polar = generate_VSM_input(
    kite_name,
    n_panels=30,
    spanwise_panel_distribution="unchanged",
    airfoil_input_type="polar_data",
    is_save_geometry=False,
    # csv_file_path=Path(project_dir) / "processed_data" / f"{kite_name}" / "geometry.csv",
)

# interactive plot
interactive_plot(
    wing_aero_polar,
    vel=3.15,
    angle_of_attack=6.75,
    side_slip=0,
    yaw_rate=0,
    is_with_aerodynamic_details=True,
)

# 2. Set the flow conditions
aoa = np.deg2rad(10)
sideslip = 0
Umag = 20
wing_aero_polar.va = (
    np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)]) * Umag,
    0,
)

# ### Plotting the wing
save_path = Path(project_dir) / "examples" / f"{kite_name}" / "results"
# plotting.plot_geometry(
#     wing_aero_polar,
#     title="geometry",
#     data_type=".pdf",
#     save_path=save_path,
#     is_save=True,
#     is_show=True,
# )

### Solve the aerodynamics
# cl,cd,cs coefficients are flipped to "normal ref frame"
# x (+) downstream, y(+) left and z-up reference frame
solver = Solver(aerodynamic_model_type="VSM")


### plotting distributions
result = solver.solve(wing_aero_polar)
plotting.plot_distribution(
    y_coordinates_list=[
        [panel.aerodynamic_center[1] for panel in wing_aero_polar.panels]
    ],
    results_list=[result],
    label_list=["V3"],
    title=f"spanwise_distributions for aoa:{np.rad2deg(aoa):.1f} [deg]",
    data_type=".pdf",
    save_path=save_path,
    is_save=True,
    is_show=True,
)

### plotting polar
plotting.plot_polars(
    solver_list=[solver, solver],
    wing_aero_list=[wing_aero_breukels, wing_aero_polar],
    label_list=["Breukels", "Polar"],
    literature_path_list=[],
    angle_range=np.linspace(0, 20, 20),
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=10,
    title="default_kite_polars",
    data_type=".pdf",
    save_path=save_path,
    is_save=True,
    is_show=True,
)

# %% importing necessary modules
import numpy as np
from pathlib import Path
from SurfplanAdapter.surfplan_to_vsm.generate_vsm_input import (
    generate_VSM_input,
)
from SurfplanAdapter.utils import PROJECT_DIR
from SurfplanAdapter.logging_config import *
from VSM.Solver import Solver
import VSM.plotting as plotting
from VSM.interactive import interactive_plot

## User Inputs
data_folder_name = "v9"
kite_file_name = "V9.60J-Inertia"

## Creating Paths
path_surfplan_file = (
    Path(PROJECT_DIR) / "data" / f"{data_folder_name}" / f"{kite_file_name}.txt"
)
path_to_save_geometry = (
    Path(PROJECT_DIR) / "processed_data" / f"{data_folder_name}" / "geometry.csv"
)

## Using polar_data input, requires one to change the "airfoil_input_type" entry to "polar_data"
# A polar data folder should be created and placed inside the same kite_name folder in data/
# Polar data must be provided for all profiles (prof_1.dat,prof_2.dat, ..)
# The polar dat should be named prof_1_polar.csv, prof_2_polar.csv etc.
wing_aero_breukels = generate_VSM_input(
    path_surfplan_file=path_surfplan_file,
    n_panels=30,
    spanwise_panel_distribution="unchanged",
    airfoil_input_type="lei_airfoil_breukels",
    is_save_geometry=True,
    path_to_save_geometry=path_to_save_geometry,
)

# interactive plot
interactive_plot(
    wing_aero_breukels,
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
wing_aero_breukels.va = (
    np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)]) * Umag,
    0,
)

### Solve the aerodynamics
# cl,cd,cs coefficients are flipped to "normal ref frame"
# x (+) downstream, y(+) left and z(+) upwards reference frame
solver = Solver(aerodynamic_model_type="VSM")


### plotting distributions
result = solver.solve(wing_aero_breukels)
plotting.plot_distribution(
    y_coordinates_list=[
        [panel.aerodynamic_center[1] for panel in wing_aero_breukels.panels]
    ],
    results_list=[result],
    label_list=["V3"],
    title=f"spanwise_distributions for aoa:{np.rad2deg(aoa):.1f} [deg]",
    data_type=".pdf",
    save_path=None,
    is_save=False,
    is_show=True,
)

### plotting polar
plotting.plot_polars(
    solver_list=[solver],
    wing_aero_list=[wing_aero_breukels],
    label_list=["Polar"],
    literature_path_list=[],
    angle_range=np.linspace(0, 20, 20),
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=10,
    title="default_kite_polars",
    data_type=".pdf",
    save_path=None,
    is_save=False,
    is_show=True,
)

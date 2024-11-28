### The main script that will
# 1. Load the data
# 2. Process the data and store in processed_data
# 3. Run the solver, iteratively for multiple angles of attack
# 4. Store the results in results folder
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from SurfplanAdapter.surfplan_to_vsm.generate_vsm_input import (
    generate_VSM_input,
)
from SurfplanAdapter.logging_config import *
from VSM.Solver import Solver
import VSM.plotting as plotting
from VSM.interactive import interactive_plot


# Find the root directory of the repository
root_dir = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
    root_dir = os.path.abspath(os.path.join(root_dir, ".."))
    if root_dir == "/":
        raise FileNotFoundError("Could not find the root directory of the repository.")

# 1. Defining paths
# filepath = Path(root_dir) / "data" / "TUDELFT_V3_LEI_KITE" / "V3D_3d.txt"
filepath = Path(root_dir) / "data" / "default_kite" / "default_kite_3d.txt"

# 2. Transforming the data into VSM input format
wing_aero_breukels = generate_VSM_input(
    filepath,
    n_panels=30,
    spanwise_panel_distribution="linear",
    is_save_geometry=True,
    csv_file_path=Path(root_dir) / "processed_data" / "default_kite" / "geometry.csv",
)

wing_aero_polar = generate_VSM_input(
    filepath,
    n_panels=30,
    spanwise_panel_distribution="linear",
    airfoil_input_type="polar_data",
    is_save_geometry=False,
    # csv_file_path=Path(root_dir) / "processed_data" / "default_kite" / "geometry.csv",
)

# # interactive plot
# interactive_plot(
#     wing_aero,
#     vel=3.15,
#     angle_of_attack=6.75,
#     side_slip=0,
#     yaw_rate=0,
#     is_with_aerodynamic_details=True,
# )

# # 2. Set the flow conditions
# aoa = np.deg2rad(10)
# sideslip = 0
# Umag = 20
# wing_aero.va = (
#     np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)]) * Umag,
#     0,
# )

# ### Plotting the wing
save_path = Path(root_dir) / "examples" / "default_kite" / "results"
# plotting.plot_geometry(
#     wing_aero,
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


# ### plotting distributions
# plotting.plot_distribution(
# result = solver.solve(wing_aero)
#     y_coordinates_list=[[panel.aerodynamic_center[1] for panel in wing_aero.panels]],
#     results_list=[result],
#     label_list=["V3"],
#     title=f"spanwise_distributions for aoa:{np.rad2deg(aoa):.1f} [deg]",
#     data_type=".pdf",
#     save_path=save_path,
#     is_save=True,
#     is_show=True,
# )

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

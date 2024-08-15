### The main script that will
# 1. Load the data
# 2. Process the data and store in processed_data
# 3. Run the solver, iteratively for multiple angles of attack
# 4. Store the results in results folder

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
from VSM.color_palette import set_plot_style


# Find the root directory of the repository
root_dir = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
    root_dir = os.path.abspath(os.path.join(root_dir, ".."))
    if root_dir == "/":
        raise FileNotFoundError("Could not find the root directory of the repository.")

# defining paths
filepath = Path(root_dir) / "data" / "V3" / "V3D_3d.txt"
# filepath = Path(root_dir) / "data" / "default_kite" / "default_kite_3d.txt"

# 1. Load the data
wing_aero = generate_VSM_input(
    filepath, n_panels=40, spanwise_panel_distribution="linear"
)

# 2. Set the flow conditions
aoa = np.deg2rad(10)
sideslip = 0
Umag = 20

wing_aero.va = (
    np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)]) * Umag,
    0,
)
# 3 plotting the wing
wing_aero.plot()

# 4. Run the solver for a single value
results, wing_aero = Solver(
    aerodynamic_model_type="VSM",
).solve(wing_aero)
logging.info(f"CL: {results['cl']}, CD: {results['cd']}, CS: {results['cs']}")
logging.info(f"Gamma distribution: {results['gamma_distribution']}")

# 5. Plotting
set_plot_style()
# plotting(
#     wing_aero,
#     variable="angle_of_attack",
#     range=[0, 20],
#     angle_of_attack=0,
#     side_slip=0,
#     yaw_rate=0,
# )


# aoas = np.arange(0, 21, 1)
# aoas = np.arange(10, 12, 1)
# cl_straight = np.zeros(len(aoas))
# cd_straight = np.zeros(len(aoas))
# cs_straight = np.zeros(len(aoas))
# gamma_straight = np.zeros((len(aoas), len(wing_aero.panels)))
# cl_turn = np.zeros(len(aoas))
# cd_turn = np.zeros(len(aoas))
# cs_turn = np.zeros(len(aoas))
# gamma_turn = np.zeros((len(aoas), len(wing_aero.panels)))
# yaw_rate = 1.5
# for i, aoa in enumerate(aoas):
#     aoa = np.deg2rad(aoa)
#     sideslip = 0
#     Umag = 20

#     wing_aero.va = (
#         np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)])
#         * Umag,
#         0,
#     )
#     results, wing_aero = VSM.solve(wing_aero)
#     cl_straight[i] = results["cl"]
#     cd_straight[i] = results["cd"]
#     cs_straight[i] = results["cs"]
#     gamma_straight[i] = results["gamma_distribution"]
#     print(
#         f"Straight: aoa: {aoa}, CL: {cl_straight[i]}, CD: {cd_straight[i]}, CS: {cs_straight[i]}"
#     )

#     wing_aero.va = (
#         np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)])
#         * Umag,
#         yaw_rate,
#     )
#     results, wing_aero = VSM.solve(wing_aero)
#     cl_turn[i] = results["cl"]
#     cd_turn[i] = results["cd"]
#     cs_turn[i] = results["cs"]
#     gamma_turn[i] = results["gamma_distribution"]
#     print(f"Turn: aoa: {aoa}, CL: {cl_turn[i]}, CD: {cd_turn[i]}, CS: {cs_turn[i]}")

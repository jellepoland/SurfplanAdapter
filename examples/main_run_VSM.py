### The main script that will
# 1. Load the data
# 2. Process the data and store in processed_data
# 3. Run the solver, iteratively for multiple angles of attack
# 4. Store the results in results folder

from SurfplanAdapter.main_generating_vsm_input import generate_VSM_input
from VSM import Solver
import numpy as np

# INPUT
filepath = "data/V3/V3D_3d.txt"  # path of the file to load the data from

# 1. Load the data
wing_aero = generate_VSM_input(filepath)

# 3. Run the solver, iteratively for multiple angles of attack
VSM = Solver(
    aerodynamic_model_type="VSM",
)

aoas = np.arange(0, 21, 1)
cl_straight = np.zeros(len(aoas))
cd_straight = np.zeros(len(aoas))
cs_straight = np.zeros(len(aoas))
gamma_straight = np.zeros((len(aoas), len(wing_aero.panels)))
cl_turn = np.zeros(len(aoas))
cd_turn = np.zeros(len(aoas))
cs_turn = np.zeros(len(aoas))
gamma_turn = np.zeros((len(aoas), len(wing_aero.panels)))
yaw_rate = 1.5
for i, aoa in enumerate(aoas):
    aoa = np.deg2rad(aoa)
    sideslip = 0
    Umag = 20

    wing_aero.va = (
        np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)])
        * Umag,
        0,
    )
    results, wing_aero = VSM.solve(wing_aero)
    cl_straight[i] = results["cl"]
    cd_straight[i] = results["cd"]
    cs_straight[i] = results["cs"]
    gamma_straight[i] = results["gamma_distribution"]
    print(
        f"Straight: aoa: {aoa}, CL: {cl_straight[i]}, CD: {cd_straight[i]}, CS: {cs_straight[i]}"
    )

    wing_aero.va = (
        np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)])
        * Umag,
        yaw_rate,
    )
    results, wing_aero = VSM.solve(wing_aero)
    cl_turn[i] = results["cl"]
    cd_turn[i] = results["cd"]
    cs_turn[i] = results["cs"]
    gamma_turn[i] = results["gamma_distribution"]
    print(f"Turn: aoa: {aoa}, CL: {cl_turn[i]}, CD: {cd_turn[i]}, CS: {cs_turn[i]}")

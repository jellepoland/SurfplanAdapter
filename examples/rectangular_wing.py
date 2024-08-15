import numpy as np
import matplotlib.pyplot as plt
import logging
from copy import deepcopy

from VSM.color_palette import set_plot_style
from VSM.WingAerodynamics import WingAerodynamics
from VSM.WingGeometry import Wing
from VSM.Solver import Solver

set_plot_style()

logging.basicConfig(level=logging.INFO)


# Body EastNorthUp (ENU) Reference Frame (aligned with Earth direction)
# x: along the chord / parallel to flow direction
# y: left
# z: upwards

## Create a wing object
# optional arguments are:
#   spanwise_panel_distribution: str = "linear"
#   - "linear"
#   - "cosine"
#   - "cosine_van_Garrel" (http://dx.doi.org/10.13140/RG.2.1.2773.8000)
# spanwise_direction: np.array = np.array([0, 1, 0])
wing = Wing(n_panels=20, spanwise_panel_distribution="split_provided")

## Add sections to the wing
# MUST be done in order from left-to-right
# Sections MUST be defined perpendicular to the quarter-chord line
# arguments are: (leading edge position [x,y,z], trailing edge position [x,y,z], airfoil data)
# airfoil data can be:
# ['inviscid']
# ['lei_airfoil_breukels', [tube_diameter, chamber_height]]
# ['polar_data', [[alpha[rad], cl, cd, cm]]]

## Rectangular wing
span = 20
# wing.add_section([0, -span / 2, 0], [1, -span / 2, 0], ["inviscid"])
wing.add_section([0, span / 2, 0], [1, span / 2, 0], ["inviscid"])
# wing.add_section([0, span / 4, 0], [1, span / 4, 0], ["inviscid"])
# wing.add_section([0, 0, 0], [1, 0, 0], ["inviscid"])
# wing.add_section([0, -span / 4, 0], [1, -span / 4, 0], ["inviscid"])
wing.add_section([0, -span / 2, 0], [1, -span / 2, 0], ["inviscid"])

# Initialize wing aerodynamics
# Default parameters are used (elliptic circulation distribution, 5 filaments per ring)
wing_aero = WingAerodynamics([wing])

# Initialize solver
# Default parameters are used (VSM, no artificial damping)
LLT = Solver(aerodynamic_model_type="LLT")
VSM = Solver(aerodynamic_model_type="VSM")

Umag = 20
aoa = 30
aoa = np.deg2rad(aoa)
Uinf = np.array([np.cos(aoa), 0, np.sin(aoa)]) * Umag

# Define inflow conditions
wing_aero.va = Uinf
wing_aero_LLT = deepcopy(wing_aero)
# Plotting the wing
wing_aero.plot()

## Solve the aerodynamics
# cl,cd,cs coefficients are flipped to "normal ref frame"
# x (+) downstream, y(+) left and z-up reference frame
results_VSM, wing_aero_VSM = VSM.solve(wing_aero)
results_LLT, wing_aero_LLT = LLT.solve(wing_aero_LLT)

###############
# Plotting
###############


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Spanwise distributions", fontsize=16)

# CL plot
axs[0, 0].plot(results_VSM["cl_distribution"], label="VSM")
axs[0, 0].plot(results_LLT["cl_distribution"], label="LLT")
axs[0, 0].set_title(r"$C_L$ Distribution")
axs[0, 0].set_xlabel(r"Spanwise Position $y/b$")
axs[0, 0].set_ylabel(r"Lift Coefficient $C_L$")
axs[0, 0].legend()

# CD plot
axs[0, 1].plot(results_VSM["cd_distribution"], label="VSM")
axs[0, 1].plot(results_LLT["cd_distribution"], label="LLT")
axs[0, 1].set_title(r"$C_D$ Distribution")
axs[0, 1].set_xlabel(r"Spanwise Position $y/b$")
axs[0, 1].set_ylabel(r"Drag Coefficient $C_D$")
axs[0, 1].legend()

# Gamma plot
axs[1, 0].plot(results_VSM["gamma_distribution"], label="VSM")
axs[1, 0].plot(results_LLT["gamma_distribution"], label="LLT")
axs[1, 0].set_title(r"$\Gamma$ Distribution")
axs[1, 0].set_xlabel(r"Spanwise Position $y/b$")
axs[1, 0].set_ylabel(r"Circulation $\Gamma$")
axs[1, 0].legend()

# Alpha plot
axs[1, 1].plot(
    results_VSM["alpha_uncorrected"], label="Uncorrected (alpha at 3/4c, i.e. c.p.)"
)
axs[1, 1].plot(results_VSM["alpha_at_ac"], label="Corrected (alpha at 1/4c, i.e. a.c.)")
axs[1, 1].plot(results_LLT["alpha_geometric"], label="Geometric")
axs[1, 1].set_title(r"$\alpha$ Comparison (from VSM)")
axs[1, 1].set_xlabel(r"Spanwise Position $y/b$")
axs[1, 1].set_ylabel(r"Angle of Attack $\alpha$ (rad)")
axs[1, 1].legend()

plt.tight_layout()
plt.show()

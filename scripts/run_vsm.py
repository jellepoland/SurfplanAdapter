# %% importing necessary modules
import numpy as np
from pathlib import Path
from SurfplanAdapter import generate_geometry_csv_files
from SurfplanAdapter.utils import PROJECT_DIR
from SurfplanAdapter.logging_config import *
from VSM.Solver import Solver
import VSM.plotting as plotting
from VSM.interactive import interactive_plot
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.WingGeometry import Wing


## User Inputs
data_folder_name = "TUDELFT_V3_KITE"
reference_point_for_moment_calculation = [0.910001, -4.099458, 0.052295]
n_panels = 100
spanwise_panel_distribution = "uniform"
va = np.array([10, 3, 0])
alpha = 6.75
beta_s = 0
yaw_rate = 0


load_dir = Path(PROJECT_DIR) / "processed_data" / f"{data_folder_name}"
wing_instance = Wing(n_panels, spanwise_panel_distribution)
body_aero_polar_with_bridles = BodyAerodynamics.from_file(
    wing_instance,
    file_path=Path(load_dir) / "wing_geometry.csv",
    is_with_corrected_polar=False,
    path_polar_data_dir=Path(load_dir) / "polar_data",
    is_with_bridles=True,
    path_bridle_data=Path(load_dir) / "bridle_lines.csv",
)

## interactive plot
interactive_plot(
    body_aero_polar_with_bridles,
    vel=va,
    angle_of_attack=alpha,
    side_slip=beta_s,
    yaw_rate=0,
    is_with_aerodynamic_details=True,
)

# # 2. Set the flow conditions
# aoa = np.deg2rad(6.75)
# sideslip = 0
# Umag = 10
# wing_aero_breukels.va = (
#     np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)]) * Umag,
#     0,
# )

# ### Solve the aerodynamics
# # cl,cd,cs coefficients are flipped to "normal ref frame"
# # x (+) downstream, y(+) left and z(+) upwards reference frame
# solver = Solver()

# ### plotting distributions
# result = solver.solve(wing_aero_breukels)
# plotting.plot_distribution(
#     y_coordinates_list=[
#         [panel.aerodynamic_center[1] for panel in wing_aero_breukels.panels]
#     ],
#     results_list=[result],
#     label_list=["V3"],
#     title=f"spanwise_distributions for aoa:{np.rad2deg(aoa):.1f} [deg]",
#     data_type=".pdf",
#     save_path=None,
#     is_save=False,
#     is_show=True,
# )

# ### plotting polar
# plotting.plot_polars(
#     solver_list=[solver],
#     body_aero_list=[wing_aero_breukels],
#     label_list=["Polar"],
#     literature_path_list=[],
#     angle_range=np.linspace(0, 20, 10),
#     angle_type="angle_of_attack",
#     angle_of_attack=0,
#     side_slip=0,
#     yaw_rate=0,
#     Umag=10,
#     title="default_kite_polars",
#     data_type=".pdf",
#     save_path=None,
#     is_save=False,
#     is_show=True,
# )
# plotting.plot_polars(
#     solver_list=[solver],
#     body_aero_list=[wing_aero_breukels],
#     label_list=["Polar"],
#     literature_path_list=[],
#     angle_range=np.linspace(-10, 10, 10),
#     angle_type="side_slip",
#     angle_of_attack=6.8,
#     side_slip=0,
#     yaw_rate=0,
#     Umag=10,
#     title="default_kite_polars",
#     data_type=".pdf",
#     save_path=None,
#     is_save=False,
#     is_show=True,
# )

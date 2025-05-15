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
n_panels = 40
va_norm = 10
alpha = 6.75
beta_s = 0
yaw_rate = 0


load_dir = Path(PROJECT_DIR) / "processed_data" / f"{data_folder_name}"

# body_aero_breukels = BodyAerodynamics.from_file(
#     file_path,
#     n_panels,
#     spanwise_panel_distribution,
#     is_with_corrected_polar=False,
# )

body_aero = BodyAerodynamics.from_file(
    file_path=Path(load_dir) / "wing_geometry.csv",
    n_panels=n_panels,
    spanwise_panel_distribution="uniform",
    is_with_corrected_polar=False,
    polar_data_dir=Path(load_dir) / "polar_data",
    is_with_bridles=False,
)
body_aero_polar_with_bridles = BodyAerodynamics.from_file(
    file_path=Path(load_dir) / "wing_geometry.csv",
    n_panels=n_panels,
    spanwise_panel_distribution="uniform",
    is_with_corrected_polar=False,
    polar_data_dir=Path(load_dir) / "polar_data",
    is_with_bridles=True,
    bridle_data_path=Path(load_dir) / "bridle_lines.csv",
)
body_aero.va_initialize(va_norm, alpha, beta_s, yaw_rate)
body_aero_polar_with_bridles.va_initialize(va_norm, alpha, beta_s, yaw_rate)

# # MATPLOTLIB Plot the wing geometry
# from VSM import plotting

# plotting.plot_geometry(
#     body_aero_polar_with_bridles,
#     title="V3",
#     data_type=".pdf",
#     save_path=".",
#     is_save=False,
#     is_show=True,
# )

## interactive plot
interactive_plot(
    body_aero_polar_with_bridles,
    vel=va_norm,
    angle_of_attack=alpha,
    side_slip=beta_s,
    yaw_rate=0,
    is_with_aerodynamic_details=True,
)

### Solve the aerodynamics
# cl,cd,cs coefficients are flipped to "normal ref frame"
# x (+) downstream, y(+) left and z(+) upwards reference frame
solver = Solver()


### plotting polar
plotting.plot_polars(
    solver_list=[solver, solver],
    body_aero_list=[body_aero, body_aero_polar_with_bridles],
    label_list=["Wing", "Wing with Bridles"],
    literature_path_list=[],
    angle_range=np.linspace(0, 20, 10),
    angle_type="angle_of_attack",
    angle_of_attack=0,
    side_slip=0,
    yaw_rate=0,
    Umag=10,
    title=f"alpha_sweep_{data_folder_name}",
    data_type=".pdf",
    save_path=None,
    is_save=False,
    is_show=True,
)
plotting.plot_polars(
    solver_list=[solver, solver],
    body_aero_list=[body_aero, body_aero_polar_with_bridles],
    label_list=["Wing", "Wing with Bridles"],
    literature_path_list=[],
    angle_range=np.linspace(-10, 10, 10),
    angle_type="side_slip",
    angle_of_attack=6.8,
    side_slip=0,
    yaw_rate=0,
    Umag=10,
    title=f"beta_sweep_{data_folder_name}",
    data_type=".pdf",
    save_path=None,
    is_save=False,
    is_show=True,
)
### plotting distributions
plotting.plot_distribution(
    alpha_list=[5, 10],
    Umag=va_norm,
    side_slip=beta_s,
    yaw_rate=yaw_rate,
    solver_list=[solver, solver],
    body_aero_list=[body_aero, body_aero_polar_with_bridles],
    label_list=["Wing", "Wing with Bridles"],
    title="spanwise_distribution",
    data_type=".pdf",
    save_path=None,
    is_save=False,
    is_show=True,
    run_time_list=None,
)

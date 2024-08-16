import os
from pathlib import Path
import numpy as np

from SurfplanAdapter.surfplan_to_vsm.read_surfplan_txt import read_surfplan_txt
from SurfplanAdapter.surfplan_to_vsm.transform_coordinate_system_surfplan_to_VSM import (
    transform_coordinate_system_surfplan_to_VSM,
)
from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics


def generate_VSM_input(filepath, n_panels, spanwise_panel_distribution):
    """
    Generate Input for the Vortex Step Method

    Args:
        filepath (string) : path of the file to extract the wing data from
                        it should be a txt file exported from Surfplan
        n_panels (int) : number of panels to divide the wing into
        spanwise_panel_distribution (string) : type of spanwise panel distribution
                        can be: 'split_provided', 'unchanged', 'linear', 'cosine'

    Returns:
        None: This function return an instance of WingAerodynamics which represent the wing described by the txt file
    """
    ribs_data = read_surfplan_txt(filepath)

    # Sorting TRANSFORMED ribs by y-coordinate (should be from positive to negative)
    ribs_data = sorted(
        ribs_data,
        key=lambda rib: transform_coordinate_system_surfplan_to_VSM(rib["LE"])[1],
        reverse=True,
    )
    # Create wing geometry
    wing = Wing(n_panels, spanwise_panel_distribution)
    for rib in ribs_data:
        wing.add_section(
            transform_coordinate_system_surfplan_to_VSM(rib["LE"]),
            transform_coordinate_system_surfplan_to_VSM(rib["TE"]),
            ["lei_airfoil_breukels", [rib["d_tube"], rib["camber"]]],
        )

    return WingAerodynamics([wing])


if __name__ == "__main__":
    # Find the root directory of the repository
    root_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
        root_dir = os.path.abspath(os.path.join(root_dir, ".."))
        if root_dir == "/":
            raise FileNotFoundError(
                "Could not find the root directory of the repository."
            )
    # defining paths
    filepath = Path(root_dir) / "data" / "V3" / "V3D_3d.txt"
    wing_aero = generate_VSM_input(
        filepath, n_panels=50, spanwise_panel_distribution="linear"
    )

    # setting arbitrary flow conditions
    # 3. Set the flow conditions
    aoa = np.deg2rad(10)
    sideslip = 0
    Umag = 20

    wing_aero.va = (
        np.array([np.cos(aoa) * np.cos(sideslip), np.sin(sideslip), np.sin(aoa)])
        * Umag,
        0,
    )

    # plotting
    wing_aero.plot()

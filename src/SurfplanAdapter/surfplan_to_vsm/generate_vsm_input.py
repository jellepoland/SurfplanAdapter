import os
from pathlib import Path
import numpy as np
import csv

from SurfplanAdapter.surfplan_to_vsm.read_surfplan_txt import read_surfplan_txt
from SurfplanAdapter.surfplan_to_vsm.transform_coordinate_system_surfplan_to_VSM import (
    transform_coordinate_system_surfplan_to_VSM,
)
from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.plotting import plot_geometry


def generate_VSM_input(
    filepath,
    n_panels,
    spanwise_panel_distribution,
    is_save_geometry=False,
    csv_file_path=None,
):
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
    row_input_list = []
    for rib in ribs_data:
        LE = transform_coordinate_system_surfplan_to_VSM(rib["LE"])
        TE = transform_coordinate_system_surfplan_to_VSM(rib["TE"])
        airfoil_input = ["lei_airfoil_breukels", [rib["d_tube"], rib["camber"]]]
        wing.add_section(LE, TE, airfoil_input)
        row_input_list.append([LE, TE, airfoil_input])
        # wing.add_section(
        #     transform_coordinate_system_surfplan_to_VSM(rib["LE"]),
        #     transform_coordinate_system_surfplan_to_VSM(rib["TE"]),
        #     ["lei_airfoil_breukels", [rib["d_tube"], rib["camber"]]],
        # )

    # Save wing geometry in a csv file
    if is_save_geometry:
        if csv_file_path is None:
            raise ValueError("You must provide a csv_file_path.")
        if not os.path.exists(csv_file_path):
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        with open(csv_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(
                ["LE_x", "LE_y", "LE_z", "TE_x", "TE_y", "TE_z", "d_tube", "camber"]
            )
            # Write the data for each rib
            for row in row_input_list:
                print(f"row:{row}")
                writer.writerow(
                    [
                        row[0][0],
                        row[0][1],
                        row[0][2],
                        row[1][0],
                        row[1][1],
                        row[1][2],
                        row[2][1][0],
                        row[2][1][1],
                    ]
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
    filepath = Path(root_dir) / "data" / "TUDELFT_V3_LEI_KITE" / "V3D_3d.txt"
    wing_aero = generate_VSM_input(
        filepath,
        n_panels=50,
        spanwise_panel_distribution="linear",
        is_save_geometry=True,
        csv_file_path=Path(root_dir)
        / "processed_data"
        / "TUDELFT_V3_LEI_KITE"
        / "geometry.csv",
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
    plot_geometry(
        wing_aero,
        title="geometry",
        data_type=".pdf",
        is_save=False,
        is_show=True,
        save_path=None,
    )

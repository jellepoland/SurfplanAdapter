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


def sort_ribs_by_proximity(ribs_data):
    # Helper function to calculate radial distance
    def radial_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # Transform the leading-edge points
    transformed_ribs = [
        {"rib": rib, "LE_point": transform_coordinate_system_surfplan_to_VSM(rib["LE"])}
        for rib in ribs_data
    ]

    # Find the rib with the farthest point with positive y-coordinate
    farthest_rib = None
    max_distance = -1
    for rib_data in transformed_ribs:
        LE_point = rib_data["LE_point"]
        if LE_point[1] > 0:  # Ensure the y-coordinate is positive
            total_distance = sum(
                radial_distance(LE_point, other["LE_point"])
                for other in transformed_ribs
            )
            if total_distance > max_distance:
                max_distance = total_distance
                farthest_rib = rib_data

    if not farthest_rib:
        raise ValueError(
            "No rib has a positive y-coordinate in its leading-edge point."
        )

    # Remove the farthest rib and use it as the starting point
    sorted_ribs = [farthest_rib]
    remaining_ribs = [
        rib_data
        for rib_data in transformed_ribs
        if not np.allclose(rib_data["LE_point"], farthest_rib["LE_point"])
    ]

    # Iteratively sort the remaining ribs based on proximity
    while remaining_ribs:
        last_point = sorted_ribs[-1]["LE_point"]
        closest_index = min(
            range(len(remaining_ribs)),
            key=lambda i: radial_distance(last_point, remaining_ribs[i]["LE_point"]),
        )
        closest_rib = remaining_ribs.pop(closest_index)
        sorted_ribs.append(closest_rib)

    # Extract the sorted ribs
    sorted_ribs_data = [rib_data["rib"] for rib_data in sorted_ribs]

    return sorted_ribs_data


def generate_VSM_input(
    filepath,
    n_panels,
    spanwise_panel_distribution,
    airfoil_input_type: str = "lei_airfoil_breukels",
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
    ribs_data = read_surfplan_txt(filepath, airfoil_input_type)
    # Sorting ribs data
    ribs_data = sort_ribs_by_proximity(ribs_data)

    # Create wing geometry
    wing = Wing(n_panels, spanwise_panel_distribution)
    row_input_list = []
    for rib in ribs_data:
        LE = transform_coordinate_system_surfplan_to_VSM(rib["LE"])
        TE = transform_coordinate_system_surfplan_to_VSM(rib["TE"])
        if airfoil_input_type == "lei_airfoil_breukels":
            polar_data = ["lei_airfoil_breukels", [rib["d_tube"], rib["camber"]]]
        elif airfoil_input_type == "polar_data":
            polar_data = ["polar_data", rib["polar_data"]]
        else:
            raise ValueError(
                f"current airfoil_input_type: {airfoil_input_type} is not recognized, change string value"
            )
        wing.add_section(LE, TE, polar_data)
        row_input_list.append([LE, TE, polar_data])
        # wing.add_section(
        #     transform_coordinate_system_surfplan_to_VSM(rib["LE"]),
        #     transform_coordinate_system_surfplan_to_VSM(rib["TE"]),
        #     ["lei_airfoil_breukels", [rib["d_tube"], rib["camber"]]],
        # )

    # Save wing geometry in a csv file
    if is_save_geometry:
        if airfoil_input_type != "lei_airfoil_breukels":
            raise ValueError(
                f"Current airfoil_input_type: {airfoil_input_type} can't be saved yet."
            )
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

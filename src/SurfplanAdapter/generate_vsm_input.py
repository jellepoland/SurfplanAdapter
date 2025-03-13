import os
from pathlib import Path
import numpy as np
import csv

from SurfplanAdapter.utils import PROJECT_DIR
from SurfplanAdapter.surfplan_to_vsm.read_surfplan_txt import (
    read_surfplan_txt,
)
from SurfplanAdapter.surfplan_to_vsm.transform_coordinate_system_surfplan_to_VSM import (
    transform_coordinate_system_surfplan_to_VSM,
)
from SurfplanAdapter.plotting import plot_and_save_all_profiles
from VSM.WingGeometry import Wing
from VSM.plotting import plot_geometry
from VSM.WingAerodynamics import WingAerodynamics


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


def saving_wing_geometry(
    dir_to_save_in,
    row_input_list,
    is_strut_list,
    airfoil_input_type,
):
    if airfoil_input_type != "lei_airfoil_breukels":
        raise ValueError(
            f"Current airfoil_input_type: {airfoil_input_type} can't be saved yet."
        )
    path_to_save_geometry = Path(dir_to_save_in) / "wing_geometry.csv"
    if path_to_save_geometry is None:
        raise ValueError("You must provide a csv_file_path.")
    if not os.path.exists(path_to_save_geometry):
        os.makedirs(os.path.dirname(path_to_save_geometry), exist_ok=True)

    with open(path_to_save_geometry, "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header
        writer.writerow(
            [
                "LE_x",
                "LE_y",
                "LE_z",
                "TE_x",
                "TE_y",
                "TE_z",
                "d_tube",
                "camber",
                "is_strut",
            ]
        )
        # Write the data for each rib
        for row, is_strut in zip(row_input_list, is_strut_list):
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
                    is_strut,
                ]
            )


def saving_bridle_lines(bridle_lines, dir_to_save_in):
    file_path = Path(dir_to_save_in) / "bridle_lines.csv"
    # Now, write the data to a CSV file
    with open(file_path, "w", newline="") as csvfile:
        # Define the column names
        fieldnames = ["p1_x", "p1_y", "p1_z", "p2_x", "p2_y", "p2_z", "diameter"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write each row, unpacking the points and diameter
        for bridle_line in bridle_lines:
            # Ensure the bridle_line has the expected structure
            if bridle_line is None or len(bridle_line) != 3:
                continue
            p1, p2, diameter = bridle_line
            # Create the row dictionary; use conditional checks if needed
            row = {
                "p1_x": p1[0] if p1 is not None else None,
                "p1_y": p1[1] if p1 is not None else None,
                "p1_z": p1[2] if p1 is not None else None,
                "p2_x": p2[0] if p2 is not None else None,
                "p2_y": p2[1] if p2 is not None else None,
                "p2_z": p2[2] if p2 is not None else None,
                "diameter": diameter,
            }
            writer.writerow(row)


def generate_VSM_input(
    path_surfplan_file: str,
    n_panels: int,
    spanwise_panel_distribution: str = "linear",
    airfoil_input_type: str = "lei_airfoil_breukels",
    is_save=False,
    dir_to_save_in=None,
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
    ribs_data, bridle_lines = read_surfplan_txt(path_surfplan_file, airfoil_input_type)
    if len(bridle_lines) > 0:
        is_with_bridle_lines = True
        bridle_lines = [
            [
                transform_coordinate_system_surfplan_to_VSM(bridle_line[0]),
                transform_coordinate_system_surfplan_to_VSM(bridle_line[1]),
                bridle_line[2] / 1e3,  # Convert from mm to m
            ]
            for bridle_line in bridle_lines
        ]
    else:
        is_with_bridle_lines = False
    # Saving all the airfoil plots
    data_folder = Path(path_surfplan_file).parent
    profile_folder = data_folder.joinpath("profiles")
    if data_folder.joinpath("profiles").exists():
        plot_and_save_all_profiles(profile_folder)
    # Sorting ribs data
    ribs_data = sort_ribs_by_proximity(ribs_data)

    # Create wing geometry
    wing = Wing(n_panels, spanwise_panel_distribution)
    row_input_list = []
    is_strut_list = []
    for rib in ribs_data:
        LE = transform_coordinate_system_surfplan_to_VSM(rib["LE"])
        TE = transform_coordinate_system_surfplan_to_VSM(rib["TE"])
        if airfoil_input_type == "lei_airfoil_breukels":
            print("d_tube:", rib["d_tube"], "camber:", rib["camber"])
            polar_data = [
                "lei_airfoil_breukels",
                [rib["d_tube"], rib["camber"]],
            ]
            is_strut_list.append(rib["is_strut"])
        elif airfoil_input_type == "polar_data":
            polar_data = ["polar_data", rib["polar_data"]]
        else:
            raise ValueError(
                f"current airfoil_input_type: {airfoil_input_type} is not recognized, change string value"
            )
        wing.add_section(LE, TE, polar_data)
        row_input_list.append(
            [
                LE,
                TE,
                polar_data,
            ]
        )

    # Save wing geometry in a csv file
    if is_save:
        saving_wing_geometry(
            dir_to_save_in, row_input_list, is_strut_list, airfoil_input_type
        )

        if is_with_bridle_lines:
            saving_bridle_lines(bridle_lines, dir_to_save_in)

    return wing, bridle_lines


if __name__ == "__main__":

    data_folder_name = "TUDELFT_V3_LEI_KITE"
    kite_file_name = "TUDELFT_V3_LEI_KITE_3d"
    path_surfplan_file = (
        Path(PROJECT_DIR) / "data" / f"{data_folder_name}" / f"{kite_file_name}.txt"
    )
    dir_to_save_in = Path(PROJECT_DIR) / "processed_data" / f"{data_folder_name}"

    wing, bridle_lines = generate_VSM_input(
        path_surfplan_file=path_surfplan_file,
        n_panels=30,
        spanwise_panel_distribution="unchanged",
        airfoil_input_type="lei_airfoil_breukels",
        is_save=True,
        dir_to_save_in=dir_to_save_in,
    )
    wing_aero = WingAerodynamics([wing])

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

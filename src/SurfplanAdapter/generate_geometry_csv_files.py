import os
from pathlib import Path
import numpy as np
import csv
import pandas as pd

from SurfplanAdapter.utils import PROJECT_DIR
from SurfplanAdapter.process_surfplan import main_process_surfplan
from SurfplanAdapter.process_surfplan.transform_coordinate_system_surfplan_to_VSM import (
    transform_coordinate_system_surfplan_to_VSM,
)


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


def saving_bridle_lines(bridle_lines, save_dir):
    file_path = Path(save_dir) / "bridle_lines.csv"
    # Now, write the data to a CSV file
    with open(file_path, "w", newline="") as csvfile:
        # Define the column names
        fieldnames = [
            "p1_x",
            "p1_y",
            "p1_z",
            "p2_x",
            "p2_y",
            "p2_z",
            "name",
            "length",
            "diameter",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write each row, unpacking the points, name, length and diameter
        for bridle_line in bridle_lines:
            # Ensure the bridle_line has the expected structure
            if bridle_line is None or len(bridle_line) != 5:
                continue
            p1, p2, name, length, diameter = bridle_line
            # Create the row dictionary; use conditional checks if needed
            row = {
                "p1_x": p1[0] if p1 is not None else None,
                "p1_y": p1[1] if p1 is not None else None,
                "p1_z": p1[2] if p1 is not None else None,
                "p2_x": p2[0] if p2 is not None else None,
                "p2_y": p2[1] if p2 is not None else None,
                "p2_z": p2[2] if p2 is not None else None,
                "name": name,
                "length": length,
                "diameter": diameter,
            }
            writer.writerow(row)


def main(
    path_surfplan_file: str,
    save_dir: Path,
    profile_load_dir: Path,
    profile_save_dir: Path,
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
        None: This function return an instance of BodyAerodynamics which represent the wing described by the txt file
    """
    ribs_data, bridle_lines = main_process_surfplan.main(
        surfplan_txt_file_path=path_surfplan_file,
        profile_load_dir=profile_load_dir,
        profile_save_dir=profile_save_dir,
        is_make_plots=True,
    )
    if len(bridle_lines) > 0:
        is_with_bridle_lines = True
        bridle_lines = [
            [
                transform_coordinate_system_surfplan_to_VSM(bridle_line[0]),  # point1
                transform_coordinate_system_surfplan_to_VSM(bridle_line[1]),  # point2
                bridle_line[2],  # name (string)
                bridle_line[3],  # length (float)
                bridle_line[4] / 1e3,  # Convert diameter from mm to m
            ]
            for bridle_line in bridle_lines
        ]
    else:
        is_with_bridle_lines = False

    # Sorting ribs data
    ribs_data = sort_ribs_by_proximity(ribs_data)

    # Loading profile_parameters
    df_profiles = pd.read_csv(
        profile_save_dir / "profile_parameters.csv", index_col="profile_number"
    )

    # Save wing geometry in a csv file
    path_to_save_geometry = Path(save_dir) / "wing_geometry.csv"
    if not os.path.exists(path_to_save_geometry):
        os.makedirs(os.path.dirname(path_to_save_geometry), exist_ok=True)

    n_ribs = len(ribs_data)
    n_profiles = len(df_profiles)
    print(f"n_ribs: {n_ribs}, n_profiles: {n_profiles}")
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
                "is_strut",
                "profile_number",
                "d_tube",
                "x_camber",
                "y_camber",
                "delta_te_angle",
                "chord",
            ]
        )
        for i, rib in enumerate(ribs_data):
            LE = transform_coordinate_system_surfplan_to_VSM(rib["LE"])
            TE = transform_coordinate_system_surfplan_to_VSM(rib["TE"])

            # read row of df_profiles
            if i < (n_profiles):
                idx = i
            elif i == n_profiles:
                idx = n_profiles - 1
            else:
                idx = n_ribs - (i + 1)
            profile_i = df_profiles.iloc[idx]

            writer.writerow(
                [
                    LE[0],
                    LE[1],
                    LE[2],
                    TE[0],
                    TE[1],
                    TE[2],
                    rib["is_strut"],
                    profile_i["profile_name"],
                    profile_i["t"],
                    profile_i["eta"],
                    profile_i["kappa"],
                    profile_i["delta"],
                    profile_i["c"],
                ]
            )
    print(f'Generated geometry file, and saved at "{path_to_save_geometry}"')

    if is_with_bridle_lines:
        saving_bridle_lines(bridle_lines, save_dir)
        print(f'Generated bridle lines file, and save at "{save_dir}/bridle_lines.csv"')

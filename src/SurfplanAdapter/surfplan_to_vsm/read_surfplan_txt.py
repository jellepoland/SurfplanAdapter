import numpy as np
import pandas as pd
import os
from pathlib import Path
from SurfplanAdapter.surfplan_to_vsm.read_profile_from_airfoil_dat_files import (
    reading_profile_from_airfoil_dat_files,
)


def read_airfoil_polar_data(
    airfoil_input_type: str, kite_dir_path: Path, profile_name: str
):
    if airfoil_input_type == "lei_airfoil_breukels":
        polar_data_i = None
        return polar_data_i
    elif airfoil_input_type == "polar_data":
        polar_data_i_path = (
            Path(kite_dir_path) / "polar_data" / f"{profile_name}_polar.csv"
        )
        df_polar_data = pd.read_csv(polar_data_i_path, sep=";")
        # print(f"profile_name: {profile_name} \npolar_data:{df_polar_data.head()}")
        # print(f' df_polar_data["aoa"].values: { df_polar_data["aoa"].values}')
        polar_data_i = np.array(
            [
                [np.deg2rad(alpha_i) for alpha_i in df_polar_data["aoa"].values],
                df_polar_data["cl"].values,
                df_polar_data["cd"].values,
                df_polar_data["cm"].values,
            ]
        )
        return polar_data_i
    else:
        raise ValueError(f"airfoil_input_type {airfoil_input_type} not recognized")


def read_surfplan_txt(filepath, airfoil_input_type):
    """
    Read the main characteristics of kite ribs and LE (Leading Edge) tube sections from the .txt file from Surfplan.

    Parameters:
    filepath (str): The name of the file containing the 3D rib and LE tube data.

    Returns:
    list of dict: A list of dictionaries, each containing the leading edge (LE) position, trailing edge (TE) position,
                  and airfoil characteristics (tube diameter and camber height) for each rib.
    """
    kite_dir_path = os.path.dirname(filepath)
    with open(filepath, "r") as file:
        lines = file.readlines()

    ribs_data = []  # Output list to store the ribs' data
    ribs = (
        []
    )  # List to store rib data: [LE position [x,y,z], TE position [x,y,z], Up vector VUP [x,y,z]]
    wingtip = []  # List to store wingtip  data
    le_tube = []  # List to store diameters of the LE tube sections
    n_ribs = 0  # Number of ribs
    n_wingtip_segments = 0  # Number of wingtip segments
    n_le_sections = 0  # Number of LE sections
    txt_section = None  # Current section being read ('ribs' or 'le_tube')

    strut_id = None  # Current strut id being read
    strut_id_list = []

    for line in lines:
        line = line.strip()
        if line.startswith("3d rib positions"):
            txt_section = "ribs"
            continue
        elif line.startswith("3d curved wingtip positions"):
            txt_section = "wingtip"
            continue
        elif line.startswith("LE tube"):
            txt_section = "le_tube"
            continue
        elif line.startswith("Strut"):
            txt_section = "strut"
            strut_id = line.split()[
                1
            ]  # Extract the strut number (e.g., "2" from "Strut 2")
            strut_id_list.append(int(strut_id) - 1)
            continue

        # Read kite ribs and store data in ribs
        if txt_section == "ribs":
            if not line:  # Empty line indicates the end of the section
                txt_section = None
                continue
            if not any(char.isdigit() for char in line):
                continue  # Skip comment lines
            if line.isdigit():
                n_ribs = int(line)
                continue
            values = list(map(float, line.replace(",", ".").split(";")))
            if len(values) == 9:
                le = np.array(values[0:3])  # Leading edge position
                te = np.array(values[3:6])  # Trailing edge position
                vup = np.array(values[6:9])  # Up vector
                ribs.append([le, te, vup])

        # Read kite wingtips ribs
        if txt_section == "wingtip":
            if not line:  # Empty line indicates the end of the section
                txt_section = None
                continue
            if not any(char.isdigit() for char in line):
                continue  # Skip comment lines
            if line.isdigit():
                n_wingtip_segments = int(line)
                continue
            values = list(map(float, line.replace(",", ".").split(";")))
            if len(values) == 9:
                le = np.array(values[0:3])  # Leading edge position
                te = np.array(values[3:6])  # Trailing edge position
                vup = np.array(values[6:9])  # Up vector
                wingtip.append([le, te, vup])

        # Read Kite LE tube and store data in le_tube
        elif txt_section == "le_tube":
            if not line:  # Empty line indicates the end of the section
                txt_section = None
                continue
            if not any(char.isdigit() for char in line):
                continue  # Skip comment lines
            if line.isdigit():
                n_le_sections = int(line)
                continue
            values = list(map(float, line.replace(",", ".").split(";")))
            if len(values) == 4:
                # centre = np.array(values[0:3])  #centre position [x,y,y] of the LE tube section
                diameter = values[3]  # Diameter of the LE tube section
                # le_tube.append([centre, diam])
                le_tube.append(diameter)

        # elif txt_section == "strut":
        #     if not line:  # Empty line indicates the end of the section
        #         txt_section = None
        #         continue
        #     if not any(char.isdigit() for char in line):
        #         continue  # Skip comment lines
        #     if line.isdigit():
        #         n_struts = int(line)  # Number of strut entries
        #         continue
        #     values = list(map(float, line.replace(",", ".").split(";")))
        #     if len(values) == 4:
        #         center = np.array(values[0:3])  # Strut center position (X, Y, Z)
        #         diameter = values[3]  # Strut diameter
        #         struts.append({"center": center, "diameter": diameter})

    # We remove wingtips sections from LE tube sections list to make LE and rib lists the same size
    le_tube_without_wingtips = np.concatenate(
        (
            [le_tube[0]],
            le_tube[n_wingtip_segments + 1 : -n_wingtip_segments - 1],
            [le_tube[-1]],
        )
    )

    is_strut = False
    for i in range(n_ribs):
        # Rib position
        rib_le = ribs[i][0]
        rib_te = ribs[i][1]
        # Tube diameter
        # normalize tube diameter with local chord
        tube_diameter = le_tube_without_wingtips[i] / np.linalg.norm(rib_te - rib_le)
        # Associate each rib with its airfoil .dat file name
        k = n_ribs // 2
        # First case, kite has one central rib
        if n_ribs % 2 == 1:
            # print(1 +abs(k-i))
            profile_name = f"prof_{1 +abs(-k+i)}"
        # Second case, kite has two central ribs
        else:
            if i < k:
                # print(k-i)
                profile_name = f"prof_{k-i}"
            else:
                # print(i-k+1)
                profile_name = f"prof_{i-k+1}"
        # Read camber height from .dat airfoil file
        airfoil = reading_profile_from_airfoil_dat_files(
            Path(kite_dir_path) / "profiles" / f"{profile_name}.dat"
        )
        camber = airfoil["depth"]
        # It's possible to add here more airfoil parameters to read in the dat file for more complete airfoil data
        # x_camber = airfoil["x_depth"]
        # TE_angle = airfoil["TE_angle"]
        polar_data_i = read_airfoil_polar_data(
            airfoil_input_type, kite_dir_path, profile_name
        )

        ## checking if at this rib location there is a strut
        print(f"i: {i}, strut_id_list: {strut_id_list}")
        if i in strut_id_list:
            is_strut = True
        else:
            is_strut = False

        ribs_data.append(
            {
                "LE": rib_le,
                "TE": rib_te,
                "d_tube": tube_diameter,
                "camber": camber,
                "polar_data": polar_data_i,
                "is_strut": is_strut,
                # "x_camber" : x_camber,
                # "TE_angle" : TE_angle
            }
        )

    ## ADDING WINGTIPS
    if (
        len(wingtip) > 0
    ):  # if wingtips segments are described in the export file, insert them in ribs data
        # Wingtip airfoil is the read from the last profile
        profile_name = f"prof_{(n_ribs + 1) // 2}"
        wingtip_airfoil = reading_profile_from_airfoil_dat_files(
            Path(kite_dir_path) / "profiles" / f"{profile_name}.dat"
        )

        # Insert wingtips segments ribs
        previous_wt_te = None
        for i in range(n_wingtip_segments - 1, -1, -1):
            wt_te = wingtip[i][1]
            # Condition to not read wingtips segments that have the same TE (resulting in triangular panels)
            if np.array_equal(wt_te, previous_wt_te):
                continue
            previous_wt_te = wt_te
            wt_le = wingtip[i][0]
            tube_diameter = le_tube[i + 1] / np.linalg.norm(rib_te - rib_le)
            camber = wingtip_airfoil["depth"]
            # x_camber = airfoil["x_depth"]
            # TE_angle = airfoil["TE_angle"]

            polar_data_i = read_airfoil_polar_data(
                airfoil_input_type, kite_dir_path, profile_name
            )

            # Insert right wingtips segments ribs
            ribs_data.insert(
                1,
                {
                    "LE": wt_le,
                    "TE": wt_te,
                    "d_tube": tube_diameter,
                    "camber": camber,
                    "polar_data": polar_data_i,
                    # "x_camber" : x_camber,
                    # "TE_angle" : TE_angle
                },
            )
            # Insert left wingtips segments ribs
            ribs_data.insert(
                -1,
                {
                    "LE": np.array([-wt_le[0], wt_le[1], wt_le[2]]),
                    "TE": np.array([-wt_te[0], wt_te[1], wt_te[2]]),
                    "d_tube": tube_diameter,
                    "camber": camber,
                    "polar_data": polar_data_i,
                    # "x_camber" : x_camber,
                    # "TE_angle" : TE_angle
                },
            )
        # Delete wingtip rib that have been replaced by wingtips segment ribs
        ribs_data = ribs_data[1:-1]
    return ribs_data


if __name__ == "__main__":
    from SurfplanAdapter.utils import project_dir

    filepath = Path(project_dir) / "data" / "default_kite" / "default_kite_3d.txt"
    ribs_data = read_surfplan_txt(filepath, "lei_airfoil_breukels")
    for rib in ribs_data:
        print(rib)
        # print("\n")
        # print(rib["d_tube"])

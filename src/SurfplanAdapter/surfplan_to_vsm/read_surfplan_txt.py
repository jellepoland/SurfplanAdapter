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


def line_parser(line):
    """
    Parse a line from the .txt file from Surfplan.

    Parameters:
        line (str): The line to parse.

    Returns:
        list: A list of floats containing the parsed values.
    """
    if ";" in line:
        return list(map(float, line.replace(",", ".").split(";")))
    else:
        return list(map(float, line.split(",")))


def read_bridle_lines(filepath):
    """
    Read the bridle line data from a Surfplan .txt file.

    This function locates the "3d Bridle" section in the file, skips its header,
    and then parses each subsequent line to extract:
      - point1: [TopX, Y, Z]
      - point2: [BottomX, Y, Z]
      - diameter: the value in the 'Diameter' column

    Each bridle line is stored as:
        bridle_line = [point1, point2, diameter]
    and all such lines are collected in a list which is returned.

    Parameters:
        filepath (str): Path to the Surfplan .txt file.

    Returns:
        list: A list of bridle lines, each as [point1, point2, diameter].
              If a diameter field is empty, it is set to None.
    """
    bridle_lines = []
    in_bridle_section = False
    header_skipped = False

    with open(filepath, "r") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for the start of the bridle section.
        if "3d Bridle" in line:
            in_bridle_section = True
            header_skipped = False  # Reset header skip for the new section.
            continue

        if in_bridle_section:
            # Skip the header line (which contains column names)
            if not header_skipped:
                header_skipped = True
                continue

            # If the line does not contain a semicolon, assume the bridle section has ended.
            if ";" not in line:
                break

            # Split the line by semicolons and strip extra whitespace.
            parts = [part.strip() for part in line.split(";")]

            # Expecting at least 10 columns based on the header:
            # TopX, Y, Z, BottomX, Y, Z, Name, Length, Material, Diameter
            if len(parts) < 10:
                continue

            try:
                # Extract point1 from the first three columns.
                point1 = [float(parts[i].replace(",", ".")) for i in range(3)]
                # Extract point2 from columns 4-6.
                point2 = [float(parts[i].replace(",", ".")) for i in range(3, 6)]
            except ValueError:
                # Skip this line if conversion fails.
                continue

            # The diameter is expected to be in the 10th column (index 9).
            diam_str = parts[9].replace(",", ".")
            try:
                diameter = float(diam_str) if diam_str else None
            except ValueError:
                diameter = None

            bridle_line = [point1, point2, diameter]
            bridle_lines.append(bridle_line)

    return bridle_lines


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
            values = line_parser(line)
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
            values = line_parser(line)
            if len(values) == 9:
                le = np.array(values[0:3])  # Leading edge position
                te = np.array(values[3:6])  # Trailing edge position
                vup = np.array(values[6:9])  # Up vector
                wingtip.append([le, te, vup])

        # Read Kite LE tube and store data in le_tube
        if txt_section == "le_tube":
            if not line:  # Empty line indicates the end of the section
                txt_section = None
                continue
            if not any(char.isdigit() for char in line):
                continue  # Skip comment lines
            if line.isdigit():
                n_le_sections = int(line)
                continue
            values = line_parser(line)
            if len(values) == 4:
                diameter = values[3]  # Diameter of the LE tube section
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
            profile_name = f"prof_{1 +abs(-k+i)}"
        # Second case, kite has two central ribs
        else:
            if i < k:
                profile_name = f"prof_{k-i}"
            else:
                profile_name = f"prof_{i-k+1}"
        # Read camber height from .dat airfoil file
        airfoil = reading_profile_from_airfoil_dat_files(
            Path(kite_dir_path) / "profiles" / f"{profile_name}.dat"
        )
        camber = airfoil["depth"]
        # It's possible to add here more airfoil parameters to read in the dat file for more complete airfoil data
        polar_data_i = read_airfoil_polar_data(
            airfoil_input_type, kite_dir_path, profile_name
        )

        ## checking if at this rib location there is a strut
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

    ## ADDING WINGTIPS, if described in surfplan txt export file
    if len(wingtip) > 0:
        ## Preparing LE, TE from the wingtip list
        wingtip = wingtip[::-1]
        le_list = [wingtip[i][0] for i in range(n_wingtip_segments)]
        te_list = [wingtip[i][1] for i in range(n_wingtip_segments)]
        chord_len_list = [
            np.linalg.norm(te_i - le_i) for le_i, te_i in zip(le_list, te_list)
        ]

        ## profile of most outer rib thats not the wingtip
        n_profiles = int(n_ribs // 2)
        profile_outer_rib = reading_profile_from_airfoil_dat_files(
            Path(kite_dir_path) / "profiles" / f"prof_{int(n_profiles-1)}.dat"
        )
        profile_tip = reading_profile_from_airfoil_dat_files(
            Path(kite_dir_path) / "profiles" / f"prof_{n_profiles}.dat"
        )
        ## Assuming camber & tube_diam decrease linearly
        camber_outer_rib = profile_outer_rib["depth"]
        camber_tip = profile_tip["depth"]
        camber_list = np.linspace(camber_outer_rib, camber_tip, n_wingtip_segments)
        tube_diam_outer_rib = le_tube[-2]
        tube_diam_tip = le_tube[-1]
        tube_diam_list = np.linspace(
            tube_diam_outer_rib, tube_diam_tip, n_wingtip_segments
        ) / np.array(chord_len_list)
        ## Assuming polar_data decreases linearly
        if airfoil_input_type == "polar_data":
            polar_data_outer_rib = read_airfoil_polar_data(
                airfoil_input_type, kite_dir_path, f"prof_{int(n_profiles-1)}"
            )
            polar_data_tip = read_airfoil_polar_data(
                airfoil_input_type, kite_dir_path, f"prof_{n_profiles}"
            )
            polar_data_list = []
            # Perform linear interpolation for each segment
            for alpha in np.linspace(0, 1, n_wingtip_segments):
                interpolated_data = (
                    1 - alpha
                ) * polar_data_outer_rib + alpha * polar_data_tip
                polar_data_list.append(interpolated_data)
        elif airfoil_input_type == "lei_airfoil_breukels":
            polar_data_list = [None for i in range(n_wingtip_segments)]
        else:
            raise ValueError(f"airfoil_input_type {airfoil_input_type} not recognized")

        ## Make lists for all ribs from left wing tip to right wing tip
        left_wing_tip_additions = []
        for i, (le_i, te_i, d_tube_i, camber_i, polar_data_i) in enumerate(
            zip(
                le_list[::-1],
                te_list[::-1],
                tube_diam_list[::-1],
                camber_list[::-1],
                polar_data_list[::-1],
            )
        ):

            left_wing_tip_additions.append(
                {
                    "LE": le_i,
                    "TE": te_i,
                    "d_tube": d_tube_i,
                    "camber": camber_i,
                    "polar_data": polar_data_i,
                    "is_strut": False,
                }
            )
        # excluding the tips as the LEs are wrong
        middle_data_non_wing_tip = ribs_data[1:-1]
        right_wing_tip_additions = []
        for i, (le_i, te_i, d_tube_i, camber_i, polar_data_i) in enumerate(
            zip(le_list, te_list, tube_diam_list, camber_list, polar_data_list)
        ):

            right_wing_tip_additions.append(
                {
                    "LE": np.array([-le_i[0], le_i[1], le_i[2]]),
                    "TE": np.array([-te_i[0], te_i[1], te_i[2]]),
                    "d_tube": d_tube_i,
                    "camber": camber_i,
                    "polar_data": polar_data_i,
                    "is_strut": False,
                }
            )
        ## concetanating the ribs data
        ribs_data = np.concatenate(
            [
                left_wing_tip_additions,
                middle_data_non_wing_tip,
                right_wing_tip_additions,
            ]
        )

    bridle_lines = read_bridle_lines(filepath)
    return ribs_data, bridle_lines


if __name__ == "__main__":
    from SurfplanAdapter.utils import PROJECT_DIR

    filepath = Path(PROJECT_DIR) / "data" / "default_kite" / "default_kite_3d.txt"
    ribs_data = read_surfplan_txt(filepath, "lei_airfoil_breukels")
    for rib in ribs_data:
        print(rib)
    bridle_lines = read_bridle_lines(filepath)
    for bridle_line in bridle_lines:
        print(bridle_line)

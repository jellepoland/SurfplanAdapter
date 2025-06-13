import numpy as np
import pandas as pd
import os
from pathlib import Path
from SurfplanAdapter.process_surfplan.read_profile_from_airfoil_dat_files import (
    reading_profile_from_airfoil_dat_files,
)
from SurfplanAdapter.plotting import plot_and_save_all_profiles
import shutil


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
      - name: the line name from the 'Name' column
      - length: the line length from the 'Length' column
      - diameter: the value in the 'Diameter' column

    Each bridle line is stored as:
        bridle_line = [point1, point2, name, length, diameter]
    and all such lines are collected in a list which is returned.

    Parameters:
        filepath (str): Path to the Surfplan .txt file.

    Returns:
        list: A list of bridle lines, each as [point1, point2, name, length, diameter].
              If a field is empty, default values are used.
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

            # Extract name from column 7 (index 6)
            name = parts[6].strip() if len(parts) > 6 else f"line_{len(bridle_lines)+1}"

            # Extract length from column 8 (index 7)
            length_str = parts[7].replace(",", ".") if len(parts) > 7 else "0"
            try:
                length = float(length_str) if length_str else 0.0
            except ValueError:
                length = 0.0

            # The diameter is expected to be in the 10th column (index 9).
            diam_str = parts[9].replace(",", ".") if len(parts) > 9 else "0"
            try:
                diameter = (
                    float(diam_str) if diam_str else 0.002
                )  # Default 2mm diameter
            except ValueError:
                diameter = 0.002

            bridle_line = [point1, point2, name, length, diameter]
            bridle_lines.append(bridle_line)

    return bridle_lines


def generate_wingtip_point_lists(
    point_list_outer_rib, point_list_tip, n_wingtip_segments
):
    """
    Generate interpolated point lists for wingtip segments, handling profiles with different numbers of points.

    Parameters:
    point_list_outer_rib (list): Point list of the outermost rib before the wingtip
    point_list_tip (list): Point list of the wingtip profile
    n_wingtip_segments (int): Number of wingtip segments to generate

    Returns:
    list: A list of point lists for each wingtip segment
    """
    # Determine a common number of points - use the maximum of both profiles
    n_points_common = max(len(point_list_outer_rib), len(point_list_tip))

    # Function to resample a point list to a specified number of points
    def resample_profile(points, n_target):
        # Convert to numpy arrays
        points_array = np.array(points)

        # Calculate cumulative distance along the profile
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            distances[i] = distances[i - 1] + np.linalg.norm(
                points_array[i] - points_array[i - 1]
            )

        # Normalize distances to [0, 1]
        if distances[-1] > 0:
            distances /= distances[-1]

        # Create new parameter values for resampling
        new_params = np.linspace(0, 1, n_target)

        # Interpolate x and y coordinates separately
        x_interp = np.interp(new_params, distances, points_array[:, 0])
        y_interp = np.interp(new_params, distances, points_array[:, 1])

        # Combine into new point list
        return np.column_stack((x_interp, y_interp))

    # Resample both profiles to the common number of points
    outer_rib_resampled = resample_profile(point_list_outer_rib, n_points_common)
    tip_resampled = resample_profile(point_list_tip, n_points_common)

    # Generate interpolation factors
    factors = np.linspace(0, 1, n_wingtip_segments)

    # Generate each intermediate point list
    wingtip_point_lists = []
    for factor in factors:
        # Linear interpolation: (1-factor) * outer_rib + factor * tip
        intermediate_points = (
            1 - factor
        ) * outer_rib_resampled + factor * tip_resampled

        # Convert back to list format
        intermediate_point_list = intermediate_points.tolist()
        wingtip_point_lists.append(intermediate_point_list)

    return wingtip_point_lists


def read_lines(surfplan_txt_file_path: Path):
    """
    Read the data from a Surfplan .txt file and return the ribs' data and bridle lines.

    Parameters:
        surfplan_txt_file_path (Path): Path to the Surfplan .txt file.

    Returns:
        tuple: A tuple containing the ribs' data and bridle lines.
    """
    with open(surfplan_txt_file_path, "r") as file:
        lines = file.readlines()

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
    return ribs, wingtip, le_tube, n_ribs, n_wingtip_segments, strut_id_list


def correcting_wingtip_by_adding_ribs(
    wingtip,
    n_wingtip_segments,
    profile_save_dir,
    le_tube,
    kite_dir_path,
    n_profiles,
    ribs_data,
    profile_load_dir=None,  # new argument for fallback
):
    ## Preparing LE, TE from the wingtip list
    wingtip = wingtip[::-1]
    le_list = [wingtip[i][0] for i in range(n_wingtip_segments)]
    te_list = [wingtip[i][1] for i in range(n_wingtip_segments)]
    chord_len_list = [
        np.linalg.norm(te_i - le_i) for le_i, te_i in zip(le_list, te_list)
    ]

    ## profile of most outer rib thats not the wingtip

    # Ensure the processed profiles directory exists
    os.makedirs(profile_save_dir, exist_ok=True)

    # Helper to get profile path: prefer processed, fallback to data
    def get_profile_path(profile_name, profile_save_dir, profile_load_dir):
        processed_path = Path(profile_save_dir) / profile_name
        if processed_path.exists():
            return processed_path
        elif profile_load_dir is not None:
            fallback = Path(profile_load_dir) / profile_name
            if fallback.exists():
                return fallback
        raise FileNotFoundError(
            f"Profile {profile_name} not found in processed or data directory."
        )

    profile_name_outer = f"prof_{int(n_profiles-1)}.dat"
    profile_outer_rib, point_list_outer_rib = reading_profile_from_airfoil_dat_files(
        get_profile_path(profile_name_outer, profile_save_dir, profile_load_dir),
        is_return_point_list=True,
    )
    profile_name_tip = f"prof_{n_profiles}.dat"
    profile_tip, point_list_tip = reading_profile_from_airfoil_dat_files(
        get_profile_path(profile_name_tip, profile_save_dir, profile_load_dir),
        is_return_point_list=True,
    )

    ## Generating wingtip point lists
    wingtip_point_lists = generate_wingtip_point_lists(
        point_list_outer_rib, point_list_tip, n_wingtip_segments - 1
    )
    # saving these again as .dat files
    for i, wingtip_point_list in enumerate(wingtip_point_lists):
        n_profile = n_profiles + i
        profile_name = f"prof_{n_profile}.dat"
        profile_path = Path(profile_save_dir) / profile_name
        os.makedirs(profile_path.parent, exist_ok=True)
        with open(profile_path, "w") as file:
            file.write(f"{profile_name}\n")
            for point in wingtip_point_list:
                file.write(f"{point[0]} {point[1]}\n")

    ## Assuming things decrease linearly
    def linear_decrease(parameter):
        return np.linspace(
            profile_outer_rib[parameter], profile_tip[parameter], n_wingtip_segments
        )

    x_max_camber_list = linear_decrease("x_max_camber")
    y_max_camber_list = linear_decrease("y_max_camber")
    TE_angle_list = linear_decrease("TE_angle")

    tube_diam_outer_rib = le_tube[-2]
    tube_diam_tip = le_tube[-1]
    tube_diam_list = np.linspace(
        tube_diam_outer_rib, tube_diam_tip, n_wingtip_segments
    ) / np.array(chord_len_list)
    chord_list = chord_len_list

    ## Make lists for all ribs from left wing tip to right wing tip
    left_wing_tip_additions = []
    for i, (
        le_i,
        te_i,
        d_tube_i,
        x_max_camber_i,
        y_max_camber_i,
        TE_angle_i,
        chord_i,
    ) in enumerate(
        zip(
            le_list[::-1],
            te_list[::-1],
            tube_diam_list[::-1],
            x_max_camber_list[::-1],
            y_max_camber_list[::-1],
            TE_angle_list[::-1],
            chord_list[::-1],
        )
    ):

        left_wing_tip_additions.append(
            {
                "LE": le_i,
                "TE": te_i,
                "d_tube": d_tube_i,
                "x_max_camber": x_max_camber_i,
                "y_max_camber": y_max_camber_i,
                "is_strut": False,
                "TE_angle": TE_angle_i,
                "chord": chord_i,
            }
        )
    # excluding the tips as the LEs are wrong
    middle_data_non_wing_tip = ribs_data[1:-1]
    right_wing_tip_additions = []
    for i, (
        le_i,
        te_i,
        d_tube_i,
        x_max_camber_i,
        y_max_camber_i,
        TE_angle_i,
        chord_i,
    ) in enumerate(
        zip(
            le_list,
            te_list,
            tube_diam_list,
            x_max_camber_list,
            y_max_camber_list,
            TE_angle_list,
            chord_list,
        )
    ):

        right_wing_tip_additions.append(
            {
                "LE": np.array([-le_i[0], le_i[1], le_i[2]]),
                "TE": np.array([-te_i[0], te_i[1], te_i[2]]),
                "d_tube": d_tube_i,
                "x_max_camber": x_max_camber_i,
                "y_max_camber": y_max_camber_i,
                "is_strut": False,
                "TE_angle": TE_angle_i,
                "chord": chord_i,
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
    return ribs_data


def main(
    surfplan_txt_file_path: Path,
    profile_load_dir: Path,
    profile_save_dir: Path,
    is_make_plots: bool = False,
):
    """Read the data from a Surfplan .txt file and return the ribs' data and bridle lines.

    Parameters:
        surfplan_txt_file_path (Path): Path to the Surfplan .txt file.
        profile_save_dir (Path): Path to the directory where the adjusted airfoil_dat files will be saved.
        airfoil_input_dir (str): The airfoil input directory. Default is "lei_airfoil_breukels".

    Returns:
        tuple: A tuple containing the ribs' data and bridle lines.
    """

    kite_dir_path = os.path.dirname(surfplan_txt_file_path)
    ribs, wingtip, le_tube, n_ribs, n_wingtip_segments, strut_id_list = read_lines(
        surfplan_txt_file_path
    )

    # We remove wingtips sections from LE tube sections list to make LE and rib lists the same size
    le_tube_without_wingtips = np.concatenate(
        (
            [le_tube[0]],
            le_tube[n_wingtip_segments + 1 : -n_wingtip_segments - 1],
            [le_tube[-1]],
        )
    )

    is_strut = False
    point_list_list = []
    ribs_data = []
    for i in range(n_ribs):
        # Rib position
        rib_le = ribs[i][0]
        rib_te = ribs[i][1]
        # Tube diameter
        tube_diameter_i = le_tube_without_wingtips[i]
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
        airfoil, point_list = reading_profile_from_airfoil_dat_files(
            Path(kite_dir_path) / "profiles" / f"{profile_name}.dat",
            is_return_point_list=True,
        )
        point_list_list.append(point_list)
        x_max_camber = airfoil["x_max_camber"]
        y_max_camber = airfoil["y_max_camber"]
        TE_angle = airfoil["TE_angle"]

        # Non-dimensionalize by chord
        chord_i = np.linalg.norm(rib_te - rib_le)
        tube_diameter_i /= chord_i
        x_max_camber /= chord_i
        y_max_camber /= chord_i

        ## checking if at this rib location there is a strut
        if i in strut_id_list:
            is_strut = True
        else:
            is_strut = False

        ribs_data.append(
            {
                "LE": rib_le,
                "TE": rib_te,
                "d_tube": tube_diameter_i,
                "x_max_camber": x_max_camber,
                "y_max_camber": y_max_camber,
                "is_strut": is_strut,
                "TE_angle": TE_angle,
                "chord": chord_i,
            }
        )

    n_profiles = int(n_ribs // 2)
    ## ADDING WINGTIPS, if described in surfplan txt export file
    if len(wingtip) > 0:
        ribs_data = correcting_wingtip_by_adding_ribs(
            wingtip,
            n_wingtip_segments,
            profile_save_dir,
            le_tube,
            kite_dir_path,
            n_profiles,
            ribs_data,
            profile_load_dir=profile_load_dir,  # pass for fallback
        )

        ## Loading and saving profiles from load to save
        for i in range(n_profiles + n_wingtip_segments - 2):
            i = i + 1
            # print(f"i: {i}, profile_name = prof_{i}")
            try:
                profile_name = f"prof_{i}"
                profile_path = Path(profile_load_dir) / f"{profile_name}.dat"
                # make a copy of this file into the save dir
                with open(profile_path, "r") as file:
                    lines = file.readlines()
                with open(Path(profile_save_dir) / f"{profile_name}.dat", "w") as file:
                    for line in lines:
                        file.write(line)
            except:
                pass

        # ## Do the process seperately for the last profile
        profile_name_initial_tip = f"prof_{n_profiles}"
        profile_name_last = f"prof_{n_profiles+n_wingtip_segments-1}"
        print(f"    profile_name_initial_tip: {profile_name_initial_tip}")
        print(f"    profile_name_last: {profile_name_last}")
        profile_path = Path(profile_load_dir) / f"{profile_name_initial_tip}.dat"
        # make a copy of this file into the save dir
        with open(profile_path, "r") as file:
            lines = file.readlines()
        with open(Path(profile_save_dir) / f"{profile_name_last}.dat", "w") as file:
            for line in lines:
                file.write(line)

        if is_make_plots:
            plot_and_save_all_profiles(
                profile_save_dir,
                ribs_data,
            )

    bridle_lines = read_bridle_lines(surfplan_txt_file_path)

    # print("ribs_data: ", ribs_data)
    print(f"len(ribs_data): {len(ribs_data)}")

    return ribs_data, bridle_lines


if __name__ == "__main__":
    from SurfplanAdapter.utils import PROJECT_DIR

    # filepath = Path(PROJECT_DIR) / "data" / "default_kite" / "default_kite_3d.txt"
    # ribs_data = read_surfplan_txt(filepath, "lei_airfoil_breukels")

    surfplan_txt_file_path = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "TUDELFT_V3_KITE_3d.txt"
    )
    profile_load_dir = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "profiles"
    profile_save_dir = (
        Path(PROJECT_DIR) / "processed_data" / "TUDELFT_V3_KITE" / "profiles"
    )
    ribs_data, bridle_lines = main(
        surfplan_txt_file_path=surfplan_txt_file_path,
        profile_load_dir=profile_load_dir,
        profile_save_dir=profile_save_dir,
        is_make_plots=True,
    )

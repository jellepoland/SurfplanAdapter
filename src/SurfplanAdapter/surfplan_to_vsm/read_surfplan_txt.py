import numpy as np
import os
from pathlib import Path
from SurfplanAdapter.surfplan_to_vsm.read_profile_from_airfoil_dat_files import (
    reading_profile_from_airfoil_dat_files,
)


def read_surfplan_txt(filepath):
    """
    Read the main characteristics of kite ribs and LE (Leading Edge) tube sections from the .txt file from Surfplan.

    Parameters:
    filepath (str): The name of the file containing the 3D rib and LE tube data.

    Returns:
    list of dict: A list of dictionaries, each containing the leading edge (LE) position, trailing edge (TE) position,
                  and airfoil characteristics (tube diameter and camber height) for each rib.
    """
    with open(filepath, "r") as file:
        lines = file.readlines()

    ribs_data = []  # Output list to store the ribs' data
    ribs = (
        []
    )  # List to store rib data: [LE position [x,y,z], TE position [x,y,z], Up vector VUP [x,y,z]]
    le_tube = []  # List to store diameters of the LE tube sections
    n_ribs = 0  # Number of ribs
    n_le_sections = 0  # Number of LE sections
    txt_section = None  # Current section being read ('ribs' or 'le_tube')

    for line in lines:
        line = line.strip()
        if line.startswith("3d rib positions"):
            txt_section = "ribs"
            continue
        elif line.startswith("LE tube"):
            txt_section = "le_tube"
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
            values = list(map(float, line.split(",")))
            if len(values) == 9:
                le = np.array(values[0:3])  # Leading edge position
                te = np.array(values[3:6])  # Trailing edge position
                vup = np.array(values[6:9])  # Up vector
                ribs.append([le, te, vup])
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
            values = list(map(float, line.split(",")))
            if len(values) == 4:
                # centre = np.array(values[0:3])  #centre position [x,y,y] of the LE tube section
                diameter = values[3]  # Diameter of the LE tube section
                # le_tube.append([centre, diam])
                le_tube.append(diameter)

    # LE tube sections list is bigger than ribs list because the wingtip are decomposed of more LE section than ribs
    # We remove wingtips sections from LE tube sections list to make LE and rib lists the same size
    wingtip_size = int((n_le_sections - n_ribs) / 2)
    le_tube = le_tube[wingtip_size:-wingtip_size]

    for i in range(n_ribs):
        # Rib position
        rib_le = ribs[i][0]
        rib_te = ribs[i][1]
        # Tube diameter
        # normalize tube diameter with local chord
        tube_diameter = le_tube[i] / np.linalg.norm(rib_te - rib_le)
        # Associate each rib with its airfoil .dat file name
        k = n_ribs // 2
        # First case, kite has one central rib
        if n_ribs % 2 == 1:
            # print(1 +abs(k-i))
            profile_name = f"prof_{1 +abs(-k+i)}.dat"
        # Second case, kite has two central ribs
        else:
            if i < k:
                # print(k-i)
                profile_name = f"prof_{k-i}.dat"
            else:
                # print(i-k+1)
                profile_name = f"prof_{i-k+1}.dat"
        # Extract the directory path
        profile_directory_path = os.path.dirname(filepath) + "/profiles/"
        # Read camber height from .dat airfoil file
        airfoil = reading_profile_from_airfoil_dat_files(
            profile_directory_path + profile_name
        )
        camber = airfoil["depth"]
        # It's possible to add here more airfoil parameters to read in the dat file for more complete airfoil data
        # x_camber = airfoil["x_depth"]
        # TE_angle = airfoil["TE_angle"]
        ribs_data.append(
            {
                "LE": rib_le,
                "TE": rib_te,
                "d_tube": tube_diameter,
                "camber": camber,
                # "x_camber" : x_camber,
                # "TE_angle" : TE_angle
            }
        )
    return ribs_data


if __name__ == "__main__":
    # Find the root directory of the repository
    root_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
        root_dir = os.path.abspath(os.path.join(root_dir, ".."))
        if root_dir == "/":
            raise FileNotFoundError(
                "Could not find the root directory of the repository."
            )
    # Example usage:
    filepath = Path(root_dir) / "data" / "TUDELFT_V3_LEI_KITE" / "V3D_3d.txt"

    # filepath = 'data/Seakite50_VH/SK50-VH_3d.txt'
    ribs_data = read_surfplan_txt(filepath)
    for rib in ribs_data:
        print(rib)
        # print("\n")
        # print(rib["d_tube"])

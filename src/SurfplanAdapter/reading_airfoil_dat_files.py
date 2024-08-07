import matplotlib.pyplot as plt
import math

def read_profile(filepath):
    """
    Read main characteristics of an ILE profile from a .dat file.

    Parameters:
    filepath (str): The name of the file containing the profile data. Should be a .dat file

    The .dat file should follow the XFoil norm: points start at the trailing edge (TE),
    go to the leading edge (LE) through the extrado, and come back to the TE through the intrado.

    Returns:
    dict: A dictionary containing the profile name, tube diameter, depth, x_depth, and TE angle.
          The keys are "name", "tube_diameter", "depth", "x_depth", and "TE_angle".
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Initialize variables
    profile_name = lines[0].strip() # Name of the profile
    tube_diameter = None            # LE tube diameter of the profile, in % of the chord 
    depth = -float('inf')           # Depth of the profile, in % of the chord 
    x_depth = None                  # Position of the maximum depth of the profile, in % of the chord 
    TE_angle_deg = None             # Angle of the TE

    # Compute tube diameter of the LE
    # Left empty for now because it is read with the ribs coordinate data in "reading_surfplan_txt"
    # It could also be determined here geometrically 
    #
    #

    # Read profile points to find maximum depth and its position
    for line in lines[1:]:
        x, y = map(float, line.split())
        if y > depth:
            depth = y
            x_depth = x

    # Calculate the TE angle 
    # TE angle is defined here as the angle between the horizontal and the TE extrado line
    # The TE extrado line is going from the TE to the 3rd point of the extrado from the TE
    if len(lines) > 4:
        (x1, y1) = map(float, lines[1].split())
        (x2, y2) = map(float, lines[3].split())
        delta_x = x2 - x1
        delta_y = y2 - y1
        TE_angle_rad = math.atan2(delta_y, delta_x)
        TE_angle_deg = 180 - math.degrees(TE_angle_rad)
    else:
        TE_angle_deg = None  # Not enough points to calculate the angle

    return {
        "name":profile_name, 
        "tube_diameter":tube_diameter, 
        "depth":100*depth, 
        "x_depth":100*x_depth, 
        "TE_angle":TE_angle_deg
    }


# Example usage:
filepath = 'data/V3/profiles/rib_1.dat'
profile = read_profile(filepath)
profile_name = profile["name"]
depth = profile["depth"]
x_depth = profile["x_depth"]
TE_angle = profile["TE_angle"]
print(f"Profile Name: {profile_name}")
print(f"Highest Point X Coordinate (x_depth): {x_depth} %")
print(f"Highest Point Y Coordinate (depth): {depth} %")
print(f"TE angle: {TE_angle:.2f}°")
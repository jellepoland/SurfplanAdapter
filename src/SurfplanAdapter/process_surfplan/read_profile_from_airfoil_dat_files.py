import matplotlib.pyplot as plt
import math
import os
from pathlib import Path
import numpy as np


def reading_profile_from_airfoil_dat_files(filepath, is_return_point_list=False):
    """
    Read main characteristics of an ILE profile from a .dat file.

    Parameters:
    filepath (str): The name of the file containing the profile data. Should be a .dat file

    The .dat file should follow the XFoil norm: points start at the trailing edge (TE),
    go to the leading edge (LE) through the extrado, and come back to the TE through the intrado.

    Returns:
    dict: A dictionary containing the profile name, tube diameter, max_camber, x_max_camber, and TE angle.
          The keys are "name", "tube_diameter", "max_camber", "x_max_camber", and "TE_angle".
    """
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Initialize variables
    profile_name = lines[0].strip()  # Name of the profile
    tube_diameter = None  # LE tube diameter of the profile, in % of the chord
    y_max_camber = -float("inf")  # max_camber of the profile, in % of the chord
    TE_angle_deg = None  # Angle of the TE

    # Compute tube diameter of the LE
    # Left empty for now because it is read with the ribs coordinate data in "reading_surfplan_txt"
    # It could also be determined here geometrically
    #
    #

    # Read profile points to find maximum max_camber and its position
    for line in lines[1:]:
        x, y = map(float, line.split())
        if y > y_max_camber:
            y_max_camber = y
            x_max_camber = x

    # Calculate the TE angle
    # TE angle is defined here as the angle between the horizontal and the TE extrado line
    # The TE extrado line is going from the TE to the 3rd point of the extrado from the TE
    point_list = []
    if len(lines) > 4:
        (x1, y1) = map(float, lines[1].split())
        (x2, y2) = map(float, lines[3].split())
        delta_x = x2 - x1
        delta_y = y2 - y1
        TE_angle_rad = math.atan2(delta_y, delta_x)
        TE_angle_deg = 180 - math.degrees(TE_angle_rad)
        # appending all points to the list
        for line in lines[1:]:
            x, y = map(float, line.split())
            point_list.append([x, y])

    else:
        TE_angle_deg = None  # Not enough points to calculate the angle

    profile_dict = {
        "name": profile_name,
        "tube_diameter": tube_diameter,
        "x_max_camber": x_max_camber,
        "y_max_camber": y_max_camber,
        "TE_angle": TE_angle_deg,
    }

    if is_return_point_list:
        return profile_dict, point_list
    else:
        return profile_dict


if __name__ == "__main__":
    from SurfplanAdapter.utils import PROJECT_DIR

    # defining paths
    filepath = (
        Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "profiles" / "prof_1.dat"
    )
    # Example usage:
    profile = reading_profile_from_airfoil_dat_files(filepath)
    profile_name = profile["name"]
    max_camber = profile["max_camber"]
    x_max_camber = profile["x_max_camber"]
    TE_angle = profile["TE_angle"]
    # print(f"Profile Name: {profile_name}")
    # print(f"Highest Point X Coordinate (x_max_camber): {x_max_camber} m")
    # print(f"Highest Point Y Coordinate (max_camber) CAMBER: {max_camber} m")
    # print(f"TE angle: {TE_angle:.2f}°")
    # print(f"profile:{profile}")

    points = profile["points"]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label="Profile")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.draw()
    # plt.show()

# def rotate_points(points, angle_degrees, center_x=-0.6, center_y=-1.28 / 2):
#     angle_rad = np.radians(angle_degrees)
#     rotation_matrix = np.array(
#         [
#             [np.cos(angle_rad), -np.sin(angle_rad)],
#             [np.sin(angle_rad), np.cos(angle_rad)],
#         ]
#     )
#     points_array = np.array(points)
#     centered_points = points_array - np.array([center_x, center_y])
#     rotated_points = np.dot(centered_points, rotation_matrix.T)
#     translated_points = rotated_points - rotated_points[-1]
#     return translated_points


# def add_circle(ax, x_offset=0, y_offset=0):
#     radius = 0.112452 / 2
#     circle_center = (x_offset, radius + y_offset)
#     circle = plt.Circle(circle_center, radius, fill=False, linestyle="-", color="blue")
#     ax.add_artist(circle)
#     return circle_center, radius


# def update_ax_limits(ax, point_list, circle_centers, radius):
#     # Find overall min and max for x and y coordinates
#     x_coords = [p[:, 0] for p in point_list]
#     y_coords = [p[:, 1] for p in point_list]
#     x_min = min([x.min() for x in x_coords])
#     x_max = max([x.max() for x in x_coords])
#     y_min = min([y.min() for y in y_coords])
#     y_max = max([y.max() for y in y_coords])

#     # Consider circle bounds
#     for center in circle_centers:
#         x_min = min(x_min, center[0] - radius)
#         x_max = max(x_max, center[0] + radius)
#         y_min = min(y_min, center[1] - radius)
#         y_max = max(y_max, center[1] + radius)

#     # Add a small margin
#     margin = 0.1
#     ax.set_xlim(x_min - margin, x_max + margin)
#     ax.set_ylim(y_min - margin, y_max + margin)

# # Create figure with two subplots side by side
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 4))

# # Plot configurations
# plot_configs = [
#     {
#         "left_angle": 90,
#         "right_angle": 90,
#         "title": "Straight tips",
#         "delta_x": 0.2,
#         "delta_y": 0,
#     },
#     {
#         "left_angle": 94,
#         "right_angle": 86,
#         "title": "Tips with beta=4°",
#         "delta_x": 0.2,
#         "delta_y": 0.04,
#     },
#     {
#         "left_angle": 98,
#         "right_angle": 82,
#         "title": "Tips with beta=8°",
#         "delta_x": 0.2,
#         "delta_y": 0.08,
#     },
#     {
#         "left_angle": 100,
#         "right_angle": 80,
#         "title": "Tips with beta=10°",
#         "delta_x": 0.2,
#         "delta_y": 0.1,
#     },
# ]

# for ax, config in zip([ax1, ax2, ax3, ax4], plot_configs):
#     delta_x = config["delta_x"]
#     delta_y = config["delta_y"]
#     left_angle = config["left_angle"]
#     right_angle = config["right_angle"]

#     # Calculate all points first
#     left_points = rotate_points(points, left_angle)
#     right_points = rotate_points(points, right_angle)

#     # Adjust positions
#     left_points_adjusted = np.copy(left_points)
#     left_points_adjusted[:, 0] -= delta_x
#     left_points_adjusted[:, 1] -= delta_y

#     right_points_adjusted = np.copy(right_points)
#     right_points_adjusted[:, 0] = -right_points_adjusted[:, 0] + delta_x
#     right_points_adjusted[:, 1] += delta_y

#     # Plot points
#     ax.plot(
#         left_points_adjusted[:, 0],
#         left_points_adjusted[:, 1],
#         label="Left tip",
#         color="blue",
#     )
#     ax.plot(
#         right_points_adjusted[:, 0],
#         right_points_adjusted[:, 1],
#         label="Right tip",
#         color="blue",
#     )

#     # Add circles and collect their centers
#     circle_centers = []
#     left_center, radius = add_circle(ax, -delta_x, -delta_y)
#     right_center, _ = add_circle(ax, delta_x, delta_y)
#     circle_centers.extend([left_center, right_center])

#     # Update axis limits considering all elements
#     update_ax_limits(
#         ax, [left_points_adjusted, right_points_adjusted], circle_centers, radius
#     )

#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.grid(True)
#     ax.set_aspect("equal", adjustable="box")
#     ax.legend()
#     ax.set_title(config["title"])

# plt.tight_layout()
# plt.show()

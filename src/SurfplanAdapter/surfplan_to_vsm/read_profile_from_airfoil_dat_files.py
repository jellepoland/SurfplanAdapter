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


# import math
# import numpy as np
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt


# def circle_error(x_center, points):
#     """
#     Calculate error for circle with center at (x_center, 0)
#     going through origin (0,0)
#     """
#     # Radius is distance from center to origin
#     radius = np.abs(x_center)

#     # Calculate distances from points to center
#     distances = np.sqrt((points[:, 0] - x_center) ** 2 + points[:, 1] ** 2)

#     # Calculate error as sum of squared differences from radius
#     error = np.sum((distances - radius) ** 2)

#     return error


# def fit_circle(points, x_min=0.01, x_max=0.3):
#     """
#     Fit circle with center on x-axis using points near the leading edge.
#     The circle passes through the origin (0,0).

#     Parameters:
#     points (array): Array of points to fit
#     x_min, x_max: Constraints for x-coordinate of circle center

#     Returns:
#     tuple: (x_center, radius)
#     """
#     # Try multiple initial guesses to avoid local minima
#     best_error = float("inf")
#     best_x_center = None

#     # Try different initial guesses spread across the allowed range
#     for x0 in np.linspace(x_min, x_max, 10):
#         try:
#             # Define objective function for optimization
#             objective = lambda x: circle_error(x[0], points)

#             # Initial guess for x_center
#             initial_guess = [x0]

#             # Constraints: x_center between x_min and x_max
#             bounds = [(x_min, x_max)]

#             # Perform optimization
#             result = minimize(
#                 objective, initial_guess, bounds=bounds, method="L-BFGS-B"
#             )

#             # Check if this is better than previous attempts
#             if result.fun < best_error:
#                 best_error = result.fun
#                 best_x_center = result.x[0]

#         except:
#             continue

#     if best_x_center is None:
#         raise ValueError("Circle fitting failed for all initial guesses")

#     # Get optimal x_center and calculate radius
#     x_center = best_x_center
#     radius = np.abs(x_center)

#     return x_center, radius


# def select_leading_edge_points(points, max_x=0.2, num_points=10):
#     """
#     Select points near the leading edge for circle fitting

#     Parameters:
#     points (array): Array of airfoil points
#     max_x: Maximum x-coordinate to consider
#     num_points: Number of points to use for fitting

#     Returns:
#     array: Selected points near leading edge
#     """
#     # Convert list of points to numpy array if needed
#     if not isinstance(points, np.ndarray):
#         points = np.array(points)

#     # Find points with small x values (near leading edge)
#     leading_edge_mask = (points[:, 0] <= max_x) & (points[:, 0] >= 0)
#     le_points = points[leading_edge_mask]

#     # If we found too few points, increase max_x and try again
#     if len(le_points) < num_points:
#         return select_leading_edge_points(
#             points, max_x=max_x * 1.5, num_points=num_points
#         )

#     # If we have too many points, select the closest ones to the origin
#     if len(le_points) > num_points:
#         distances = np.sqrt(le_points[:, 0] ** 2 + le_points[:, 1] ** 2)
#         closest_indices = np.argsort(distances)[:num_points]
#         return le_points[closest_indices]

#     return le_points


# def reading_profile_from_airfoil_dat_files(filepath):
#     """
#     Read airfoil profile from a .dat file and fit a leading edge circle.
#     """
#     with open(filepath, "r") as file:
#         lines = file.readlines()

#     # Initialize variables
#     profile_name = lines[0].strip()  # Name of the profile
#     tube_diameter = None  # LE tube diameter of the profile
#     max_camber = -float("inf")  # max_camber of the profile
#     x_max_camber = None  # Position of the maximum max_camber
#     TE_angle_deg = None  # Angle of the TE

#     # Read profile points
#     point_list = []
#     for line in lines[1:]:
#         try:
#             values = line.split()
#             if len(values) >= 2:
#                 x, y = map(float, values[:2])
#                 point_list.append([x, y])
#                 if y > max_camber:
#                     max_camber = y
#                     x_max_camber = x
#         except ValueError:
#             continue  # Skip lines that can't be converted

#     # Calculate the TE angle
#     if len(point_list) > 3:
#         x1, y1 = point_list[0]
#         x2, y2 = point_list[2]
#         delta_x = x2 - x1
#         delta_y = y2 - y1
#         TE_angle_rad = np.arctan2(delta_y, delta_x)
#         TE_angle_deg = 180 - np.degrees(TE_angle_rad)
#     else:
#         TE_angle_deg = None

#     # Convert point_list to numpy array
#     points_array = np.array(point_list)

#     # Fit leading edge circle
#     le_circle_center_x = None
#     le_circle_radius = None
#     le_points = None

#     if len(points_array) > 3:
#         try:
#             # Select points near the leading edge
#             le_points = select_leading_edge_points(
#                 points_array, max_x=0.15, num_points=8
#             )

#             # Fit circle with center on x-axis passing through origin
#             le_circle_center_x, le_circle_radius = fit_circle(
#                 le_points, x_min=0.01, x_max=0.2  # Minimum x-center  # Maximum x-center
#             )

#             tube_diameter = 2 * le_circle_radius
#         except Exception as e:
#             print(f"Error fitting leading edge circle: {e}")

#     return {
#         "name": profile_name,
#         "tube_diameter": tube_diameter,
#         "max_camber": max_camber,
#         "x_max_camber": x_max_camber,
#         "TE_angle": TE_angle_deg,
#         "points": point_list,
#         "le_circle_center_x": le_circle_center_x,
#         "le_circle_radius": le_circle_radius,
#         "le_points": le_points,
#     }


# def plot_airfoil_with_le_circle(profile_data):
#     """
#     Plot the airfoil profile with the fitted leading edge circle
#     """
#     points = np.array(profile_data["points"])

#     plt.figure(figsize=(10, 6))

#     # Plot the airfoil profile
#     plt.plot(points[:, 0], points[:, 1], "b-", linewidth=2, label="Airfoil Profile")

#     # Plot the origin
#     plt.scatter(0, 0, color="black", s=100, label="Origin (0,0)")

#     # If we have leading edge points, plot them
#     if profile_data["le_points"] is not None:
#         le_points = profile_data["le_points"]
#         plt.scatter(
#             le_points[:, 0],
#             le_points[:, 1],
#             color="red",
#             s=60,
#             label="Leading Edge Points Used for Fitting",
#         )

#     # If we have a fitted circle, plot it
#     if (
#         profile_data["le_circle_center_x"] is not None
#         and profile_data["le_circle_radius"] is not None
#     ):
#         x_center = profile_data["le_circle_center_x"]
#         y_center = 0  # Center is on x-axis
#         radius = profile_data["le_circle_radius"]

#         # Plot circle center
#         plt.scatter(
#             x_center,
#             y_center,
#             color="green",
#             s=100,
#             label=f"Circle Center ({x_center:.4f}, {y_center})",
#         )

#         # Plot the fitted circle
#         theta = np.linspace(0, 2 * np.pi, 100)
#         x_circle = x_center + radius * np.cos(theta)
#         y_circle = y_center + radius * np.sin(theta)
#         plt.plot(
#             x_circle,
#             y_circle,
#             "g--",
#             linewidth=2,
#             label=f"Fitted LE Circle (r={radius:.4f}, diameter={2*radius:.4f})",
#         )

#     # Add a zoomed inset for the leading edge
#     from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

#     axins = zoomed_inset_axes(plt.gca(), 4, loc="upper right")

#     # Plot the same content in the inset
#     axins.plot(points[:, 0], points[:, 1], "b-", linewidth=2)

#     if profile_data["le_points"] is not None:
#         le_points = profile_data["le_points"]
#         axins.scatter(le_points[:, 0], le_points[:, 1], color="red", s=60)

#     if (
#         profile_data["le_circle_center_x"] is not None
#         and profile_data["le_circle_radius"] is not None
#     ):
#         x_center = profile_data["le_circle_center_x"]
#         y_center = 0
#         radius = profile_data["le_circle_radius"]

#         axins.scatter(x_center, y_center, color="green", s=100)

#         theta = np.linspace(0, 2 * np.pi, 100)
#         x_circle = x_center + radius * np.cos(theta)
#         y_circle = y_center + radius * np.sin(theta)
#         axins.plot(x_circle, y_circle, "g--", linewidth=2)

#     # Set the limits for the zoomed inset
#     axins.set_xlim(-0.05, 0.15)
#     axins.set_ylim(-0.1, 0.1)
#     axins.grid(True)

#     # Connect the inset to the main plot
#     mark_inset(plt.gca(), axins, loc1=1, loc2=3, fc="none", ec="0.5")

#     # Display profile information
#     info_text = (
#         f"Profile: {profile_data['name']}\n"
#         f"max_camber: {profile_data['max_camber']:.4f}\n"
#         f"x_max_camber: {profile_data['x_max_camber']:.4f}\n"
#     )

#     if profile_data["TE_angle"] is not None:
#         info_text += f"TE Angle: {profile_data['TE_angle']:.2f}°\n"

#     if profile_data["tube_diameter"] is not None:
#         info_text += f"LE Tube Diameter: {profile_data['tube_diameter']:.4f}"

#     plt.annotate(
#         info_text,
#         xy=(0.65, 0.05),
#         xycoords="axes fraction",
#         bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
#     )

#     plt.title("Airfoil Profile with Leading Edge Circle Fit")
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.legend()
#     plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


# def plot_airfoil_with_le_circle(profile_data):
#     """
#     Plot the airfoil profile with the fitted leading edge circle

#     Args:
#         profile_data (dict): Dictionary containing airfoil profile information
#     """
#     try:
#         # Check if required data exists
#         if "points" not in profile_data or profile_data["points"] is None:
#             raise ValueError("Profile data must contain 'points' key with valid data")

#         points = np.array(profile_data["points"])

#         # Create figure
#         fig, ax = plt.subplots(figsize=(10, 6))

#         # Plot the airfoil profile
#         ax.plot(points[:, 0], points[:, 1], "b-", linewidth=2, label="Airfoil Profile")

#         # Plot the origin
#         ax.scatter(0, 0, color="black", s=100, label="Origin (0,0)")

#         # Plot leading edge points if available
#         if "le_points" in profile_data and profile_data["le_points"] is not None:
#             le_points = np.array(profile_data["le_points"])
#             ax.scatter(
#                 le_points[:, 0],
#                 le_points[:, 1],
#                 color="red",
#                 s=60,
#                 label="Leading Edge Points Used for Fitting",
#             )

#         # Plot fitted circle if data is available
#         if all(
#             key in profile_data and profile_data[key] is not None
#             for key in ["le_circle_center_x", "le_circle_radius"]
#         ):
#             x_center = profile_data["le_circle_center_x"]
#             y_center = 0  # Center is on x-axis
#             radius = profile_data["le_circle_radius"]

#             # Plot circle center
#             ax.scatter(
#                 x_center,
#                 y_center,
#                 color="green",
#                 s=100,
#                 label=f"Circle Center ({x_center:.4f}, {y_center})",
#             )

#             # Plot the fitted circle
#             theta = np.linspace(0, 2 * np.pi, 100)
#             x_circle = x_center + radius * np.cos(theta)
#             y_circle = y_center + radius * np.sin(theta)
#             ax.plot(
#                 x_circle,
#                 y_circle,
#                 "g--",
#                 linewidth=2,
#                 label=f"Fitted LE Circle (r={radius:.4f}, diameter={2*radius:.4f})",
#             )

#         # Add zoomed inset
#         axins = zoomed_inset_axes(ax, zoom=4, loc="upper right")

#         # Plot inset content
#         axins.plot(points[:, 0], points[:, 1], "b-", linewidth=2)

#         if "le_points" in profile_data and profile_data["le_points"] is not None:
#             axins.scatter(le_points[:, 0], le_points[:, 1], color="red", s=60)

#         if all(
#             key in profile_data and profile_data[key] is not None
#             for key in ["le_circle_center_x", "le_circle_radius"]
#         ):
#             axins.scatter(x_center, y_center, color="green", s=100)
#             axins.plot(x_circle, y_circle, "g--", linewidth=2)

#         # Set inset limits and grid
#         axins.set_xlim(-0.05, 0.15)
#         axins.set_ylim(-0.1, 0.1)
#         axins.grid(True)

#         # Connect inset to main plot
#         mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

#         # Build info text
#         info_text = (
#             f"Profile: {profile_data.get('name', 'N/A')}\n"
#             f"max_camber: {profile_data.get('max_camber', 0):.4f}\n"
#             f"x_max_camber: {profile_data.get('x_max_camber', 0):.4f}\n"
#         )

#         if "TE_angle" in profile_data and profile_data["TE_angle"] is not None:
#             info_text += f"TE Angle: {profile_data['TE_angle']:.2f}°\n"

#         if (
#             "tube_diameter" in profile_data
#             and profile_data["tube_diameter"] is not None
#         ):
#             info_text += f"LE Tube Diameter: {profile_data['tube_diameter']:.4f}"

#         # Add annotation
#         ax.annotate(
#             info_text,
#             xy=(0.65, 0.05),
#             xycoords="axes fraction",
#             bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
#         )

#         # Set plot properties
#         ax.set_title("Airfoil Profile with Leading Edge Circle Fit")
#         ax.set_xlabel("X Coordinate")
#         ax.set_ylabel("Y Coordinate")
#         ax.set_aspect("equal")
#         ax.grid(True)
#         ax.legend()

#         plt.tight_layout()
#         plt.show()

#     except Exception as e:
#         print(f"Error plotting airfoil: {str(e)}")
#         raise


# # Example usage
# if __name__ == "__main__":
#     # Replace with your actual file path
#     from SurfplanAdapter.utils import PROJECT_DIR

#     filepath = (
#         Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "profiles" / "prof_1.dat"
#     )

#     try:
#         profile_data = reading_profile_from_airfoil_dat_files(filepath)
#         print(f"Profile name: {profile_data['name']}")
#         print(f"max_camber: {profile_data['max_camber']}")
#         print(f"X max_camber: {profile_data['x_max_camber']}")
#         print(f"TE angle: {profile_data['TE_angle']} degrees")

#         if profile_data["tube_diameter"] is not None:
#             print(f"Tube diameter: {profile_data['tube_diameter']}")

#         # def add_circle(ax, x_offset=0, y_offset=0):
#         #     radius = 0.112452 / 2
#         #     circle_center = (x_offset, radius + y_offset)
#         #     circle = plt.Circle(
#         #         circle_center, radius, fill=False, linestyle="-", color="blue"
#         #     )
#         #     ax.add_artist(circle)
#         #     return circle_center, radius

#         # from SurfplanAdapter.plotting import plot_profiles

#         # plot_profiles(filepath, profile_folder)

#         # Plot the airfoil with the fitted circle
#         plot_airfoil_with_le_circle(profile_data)

#     except Exception as e:
#         print(f"Error processing file: {e}")

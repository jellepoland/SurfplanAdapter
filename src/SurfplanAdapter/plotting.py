import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
from SurfplanAdapter.surfplan_to_vsm.read_surfplan_txt import read_surfplan_txt
from SurfplanAdapter.surfplan_to_vsm.read_profile_from_airfoil_dat_files import (
    reading_profile_from_airfoil_dat_files,
)
from SurfplanAdapter.surfplan_to_vsm.transform_coordinate_system_surfplan_to_VSM import (
    transform_coordinate_system_surfplan_to_VSM,
)


def plot_ribs(ribs_coord):
    """
    Plot kite ribs in 3D.

    Parameters:
    ribs_coord (list of list of tuples): A list where each element is a list of two tuples,
                                        each tuple containing three floats representing
                                        the x, y, and z coordinates of the rib endpoints.

    Returns:
    None: This function does not return anything. It plots the ribs in a 3D space using matplotlib.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Check if there is any data to plot
    if len(ribs_coord) == 0:
        print("No data to plot.")
        return
    first_rib = True
    # Iterate through each rib in the ribs_coord
    for rib in ribs_coord:
        # Set the label for the first rib only
        label = "ribs" if first_rib else ""

        # Plot the rib as a line between LE and TE
        ax.plot(
            [rib[0][0], rib[1][0]],
            [rib[0][1], rib[1][1]],
            [rib[0][2], rib[1][2]],
            c="r",
            label=label,
        )
        # Scatter plot the LE point of the rib
        ax.scatter(rib[0][0], rib[0][1], rib[0][2], c="c", marker=".")
        first_rib = False

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.show()


# Plot the profile described in the input file and display caracteristics
# Input :
#   filepath : str
def plot_profiles(filepath):
    """
    Plot the profile described in the input file and display its characteristics.

    Parameters:
    filepath (str): The name of the file containing the profile data. Should be a .dat file

    Returns:
    None: This function does not return anything. It plots the profile and displays its characteristics.
    """
    # Open the file and read all lines
    with open(filepath, "r") as file:
        lines = file.readlines()

    # List to store (x, y) points of the profile
    points = []

    # Store profile points, skipping the first line (assumed to be the name of the profile)
    for line in lines[1:]:
        x, y = map(float, line.split())
        points.append((x, y))

    # Unzip points into x and y coordinates for plotting
    x_points, y_points = zip(*points)

    # Read profile characteristics from the file
    profile = reading_profile_from_airfoil_dat_files(filepath)
    profile_name = profile["name"]
    depth = profile["depth"]
    x_depth = profile["x_depth"]
    TE_angle = profile["TE_angle"]

    # Plot the profile points
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, y_points, marker="", linestyle="-", color="b")

    # Highlight the highest point in red
    plt.scatter(
        [x_depth / 100], [depth / 100], color="r", zorder=5, label="Highest Point"
    )

    # Annotate the highest point with its coordinates
    plt.annotate(
        f"({x_depth/100}, {depth/100})",
        xy=(x_depth / 100, depth / 100),
        xytext=(x_depth / 100, depth / 100 + 0.05),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )

    # Set plot labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(profile_name)

    # Create a legend with profile characteristics
    legend = f"Depth: {depth:.2f}%\nx_depth = {x_depth:.2f}%\nTE Angle: {TE_angle:.2f}°"
    plt.legend([legend], loc="best")
    # Enable grid for better readability
    plt.grid(True)
    # Set equal aspect ratio for the plot
    plt.gca().set_aspect("equal", adjustable="box")  # Set equal aspect ratio

    # Display the plot
    plt.show()


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
    filepath = Path(root_dir) / "data" / "default_kite" / "default_kite_3d.txt"
    filepath_profile = (
        Path(root_dir) / "data" / "default_kite" / "profiles" / "prof_5.dat"
    )

    ribs_data = read_surfplan_txt(filepath)
    ribs_coords_surfplan = [[rib["LE"], rib["TE"]] for rib in ribs_data]
    ribs_coords_vsm = [
        [
            transform_coordinate_system_surfplan_to_VSM(rib["LE"]),
            transform_coordinate_system_surfplan_to_VSM(rib["TE"]),
        ]
        for rib in ribs_data
    ]
    # Plot the data
    plot_ribs(ribs_coords_vsm)
    plot_profiles(filepath_profile)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from pathlib import Path
from SurfplanAdapter.process_surfplan.read_profile_from_airfoil_dat_files import (
    reading_profile_from_airfoil_dat_files,
)
from SurfplanAdapter.process_surfplan.transform_coordinate_system_surfplan_to_VSM import (
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
def plot_profiles(
    filepath, profile_save_dir, t, c, is_with_return_t_eta_kappa_delta_c=False
):
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

    # Plot the profile points
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, y_points, marker="", linestyle="-", color="black")

    # Highlight the highest point in red
    x_max_camber = profile["x_max_camber"]
    y_max_camber = profile["y_max_camber"]
    plt.scatter(
        x_max_camber,
        y_max_camber,
        color="black",
        zorder=3,
        label="max_camber: ({x:.3f}, {y:.3f})".format(x=x_max_camber, y=y_max_camber),
    )

    # Annotate the highest point with its coordinates
    # plt.annotate(
    #     f"({x_depth/100:.2f}, {depth/100:.2f})",
    #     xy=(x_depth / 100, depth / 100),
    #     xytext=(x_depth / 100, depth / 100 + 0.05),
    #     arrowprops=dict(facecolor="black", arrowstyle="->"),
    # )

    # Create a circle
    radius = t / 2
    circle = plt.Circle((radius, 0), radius, fill=False, linestyle="-", color="black")

    # Add the circle to the current axes
    plt.gca().add_patch(circle)
    plt.xlabel(f"$x/c$")
    plt.ylabel(f"$y/c$")
    plt.title(
        rf'{profile_name} --- t: {t:.3f} $\eta$: {profile["x_max_camber"]:.3f}, $\kappa$: {profile["y_max_camber"]:.3f}, $\delta$: {profile["TE_angle"]:.3f}deg, c: {c:.3f}m'
    )
    plt.legend(loc="lower center")
    # Enable grid for better readability
    plt.grid(True)
    # Set equal aspect ratio for the plot
    plt.gca().set_aspect("equal", adjustable="box")  # Set equal aspect ratio

    # Display the plot
    plt.savefig(Path(profile_save_dir) / f"{filepath.stem}.png")
    plt.close()

    if is_with_return_t_eta_kappa_delta_c:
        return (
            t,
            profile["x_max_camber"],
            profile["y_max_camber"],
            profile["TE_angle"],
            c,
        )


def plot_and_save_all_profiles(profile_save_dir, ribs_data):

    if not profile_save_dir.is_dir():
        raise ValueError(f"Directory {profile_save_dir} does not exist.")

    # Create a list to store the data for each profile
    profile_data = []

    i = 0
    for file_name in profile_save_dir.iterdir():
        # Check if the file name starts with "prof" and ends with ".dat"
        if file_name.name.startswith("prof") and file_name.name.endswith(".dat"):
            i += 1
            rib = ribs_data[i]
            # print(f"rib: {rib}")
            t, eta, kappa, delta, c = plot_profiles(
                file_name,
                profile_save_dir=profile_save_dir,
                t=rib["d_tube"],
                c=rib["chord"],
                is_with_return_t_eta_kappa_delta_c=True,
            )

            # Add data to our list
            profile_data.append(
                {
                    "profile_name": file_name.stem,
                    "t": t,
                    "eta": eta,
                    "kappa": kappa,
                    "delta": delta,
                    "c": c,
                }
            )

    # Convert the list to a pandas DataFrame
    if profile_data:
        df = pd.DataFrame(profile_data)

        # sort on last str in profile_name
        df["profile_number"] = df["profile_name"].str.split("_").str[-1].astype(int)
        df = df.sort_values("profile_number")

        # reshuffle the columns
        df = df[
            [
                "profile_number",
                "profile_name",
                "t",
                "eta",
                "kappa",
                "delta",
                "c",
            ]
        ]

        # Save the DataFrame to a CSV file
        csv_path = profile_save_dir / "profile_parameters.csv"
        df.to_csv(csv_path, index=False)
        print(f"Profile parameters saved to {csv_path}")
    else:
        print("No profiles found to save.")

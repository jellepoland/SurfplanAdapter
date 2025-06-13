import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from SurfplanAdapter.process_surfplan.read_profile_from_airfoil_dat_files import (
    reading_profile_from_airfoil_dat_files,
)
from SurfplanAdapter.process_surfplan.transform_coordinate_system_surfplan_to_VSM import (
    transform_coordinate_system_surfplan_to_VSM,
)


def plot_airfoils_3d_from_yaml(
    yaml_file_path, profile_base_dir, save_path=None, show_plot=True
):
    """
    Plot airfoils in 3D space using YAML file data.

    This function reads the generated YAML configuration file, loads the corresponding
    .dat files for each airfoil, scales them according to the chord length, and positions
    them correctly in 3D space using LE, TE, and VUP information.

    Parameters:
        yaml_file_path (str or Path): Path to the YAML configuration file
        profile_base_dir (str or Path): Base directory containing the profile .dat files
        save_path (str or Path, optional): Path to save the plot. If None, plot is not saved.
        show_plot (bool): Whether to display the plot

    Returns:
        None: This function displays/saves the 3D plot of airfoils
    """
    yaml_file_path = Path(yaml_file_path)
    profile_base_dir = Path(profile_base_dir)

    # Load YAML configuration
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract wing sections data
    wing_sections = config["wing_sections"]
    wing_sections_data = wing_sections["data"]
    headers = wing_sections["headers"]

    # Create mapping from headers to indices
    header_map = {header: idx for idx, header in enumerate(headers)}

    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    print(f"Plotting {len(wing_sections_data)} airfoil sections...")

    # Process each wing section
    for i, section_data in enumerate(wing_sections_data):
        # Extract coordinates from section data
        airfoil_id = section_data[header_map["airfoil_id"]]
        le_x = section_data[header_map["LE_x"]]
        le_y = section_data[header_map["LE_y"]]
        le_z = section_data[header_map["LE_z"]]
        te_x = section_data[header_map["TE_x"]]
        te_y = section_data[header_map["TE_y"]]
        te_z = section_data[header_map["TE_z"]]
        vup_x = section_data[header_map["VUP_x"]]
        vup_y = section_data[header_map["VUP_y"]]
        vup_z = section_data[header_map["VUP_z"]]

        # Create vectors
        le_point = np.array([le_x, le_y, le_z])
        te_point = np.array([te_x, te_y, te_z])
        vup_vector = np.array([vup_x, vup_y, vup_z])

        # Calculate chord vector and length
        chord_vector = te_point - le_point
        chord_length = np.linalg.norm(chord_vector)
        chord_unit = chord_vector / chord_length

        # Normalize VUP vector
        vup_unit = vup_vector / np.linalg.norm(vup_vector)

        # Create coordinate system for the airfoil
        # x_local: along chord (LE to TE)
        # y_local: perpendicular to chord in the VUP direction
        # z_local: perpendicular to both (right-hand rule)
        x_local = chord_unit
        y_local = vup_unit
        z_local = np.cross(x_local, y_local)
        z_local = z_local / np.linalg.norm(z_local)

        # Find corresponding .dat file
        dat_file_path = profile_base_dir / f"prof_{airfoil_id}.dat"

        if not dat_file_path.exists():
            print(
                f"Warning: Profile file {dat_file_path} not found, skipping airfoil {airfoil_id}"
            )
            continue

        try:
            # Read airfoil coordinates from .dat file
            airfoil_coords = []
            with open(dat_file_path, "r") as f:
                lines = f.readlines()

            # Skip the first line (profile name) and read coordinates
            for line in lines[1:]:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        airfoil_coords.append([x, y])

            if not airfoil_coords:
                print(f"Warning: No coordinates found in {dat_file_path}")
                continue

            airfoil_coords = np.array(airfoil_coords)

            # Scale airfoil coordinates by chord length
            # .dat file coordinates are normalized (0 to 1 in x-direction)
            airfoil_x = airfoil_coords[:, 0] * chord_length
            airfoil_y = airfoil_coords[:, 1] * chord_length
            airfoil_z = np.zeros_like(airfoil_x)  # Start with z=0 in local coordinates

            # Transform airfoil coordinates to 3D world coordinates
            world_coords = []
            for j in range(len(airfoil_x)):
                # Local airfoil coordinate
                local_coord = np.array([airfoil_x[j], airfoil_y[j], airfoil_z[j]])

                # Transform to world coordinates using the local coordinate system
                world_coord = (
                    le_point
                    + local_coord[0] * x_local
                    + local_coord[1] * y_local
                    + local_coord[2] * z_local
                )
                world_coords.append(world_coord)

            world_coords = np.array(world_coords)

            # Plot the airfoil as a line in 3D space
            ax.plot(
                world_coords[:, 0],
                world_coords[:, 1],
                world_coords[:, 2],
                "black",
                linewidth=1,
                alpha=0.7,
            )

            # Plot LE and TE points
            if i == 0:  # Only add labels for the first airfoil
                ax.scatter(
                    [le_x],
                    [le_y],
                    [le_z],
                    c="blue",
                    s=10,
                    alpha=0.8,
                    label="Leading Edge",
                )
                ax.scatter(
                    [te_x],
                    [te_y],
                    [te_z],
                    c="red",
                    s=10,
                    alpha=0.8,
                    label="Trailing Edge",
                )
            else:
                ax.scatter([le_x], [le_y], [le_z], c="blue", s=10, alpha=0.8)
                ax.scatter([te_x], [te_y], [te_z], c="red", s=10, alpha=0.8)

            # Plot chord line
            if i == 0:
                ax.plot(
                    [le_x, te_x],
                    [le_y, te_y],
                    [le_z, te_z],
                    "k--",
                    linewidth=0.5,
                    alpha=0.5,
                    label="Chord Line",
                )
            else:
                ax.plot(
                    [le_x, te_x],
                    [le_y, te_y],
                    [le_z, te_z],
                    "k--",
                    linewidth=0.5,
                    alpha=0.5,
                )

        except Exception as e:
            print(f"Error processing airfoil {airfoil_id}: {e}")
            continue

    # Set labels and title
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D Airfoil Sections from YAML Configuration")
    ax.legend()

    # Set equal aspect ratio using manual limits calculation
    # Get all the plotted data bounds
    all_coords = []
    for i, section_data in enumerate(wing_sections_data):
        airfoil_id = section_data[header_map["airfoil_id"]]
        dat_file_path = profile_base_dir / f"prof_{airfoil_id}.dat"
        if dat_file_path.exists():
            le_x = section_data[header_map["LE_x"]]
            le_y = section_data[header_map["LE_y"]]
            le_z = section_data[header_map["LE_z"]]
            te_x = section_data[header_map["TE_x"]]
            te_y = section_data[header_map["TE_y"]]
            te_z = section_data[header_map["TE_z"]]
            all_coords.extend([[le_x, le_y, le_z], [te_x, te_y, te_z]])

    if all_coords:
        all_coords = np.array(all_coords)
        # Calculate the range for each axis
        x_range = [all_coords[:, 0].min(), all_coords[:, 0].max()]
        y_range = [all_coords[:, 1].min(), all_coords[:, 1].max()]
        z_range = [all_coords[:, 2].min(), all_coords[:, 2].max()]

        # Calculate the center and maximum range
        x_center = np.mean(x_range)
        y_center = np.mean(y_range)
        z_center = np.mean(z_range)

        max_range = max(
            x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
        )

        # Set equal limits around the center
        margin = max_range * 0.1  # Add 10% margin
        half_range = (max_range + margin) / 2

        ax.set_xlim([x_center - half_range, x_center + half_range])
        ax.set_ylim([y_center - half_range, y_center + half_range])
        ax.set_zlim([z_center - half_range, z_center + half_range])

    # Set view angle (azimuth rotated by -90 degrees from default)
    ax.view_init(elev=20, azim=-120)  # Default azim is 30, so 30-90 = -60

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save plot if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_profiles(
    filepath,
    profile_save_dir,
    t,
    c,
    d_tube_from_dat=None,
    is_with_return_t_eta_kappa_delta_c=False,
):
    """
    Plot the profile described in the input file and display its characteristics.

    Parameters:
    filepath (str): The name of the file containing the profile data. Should be a .dat file
    profile_save_dir (str or Path): Directory to save the plot
    t (float): Tube diameter to chord ratio from Surfplan txt file (non-dimensional)
    c (float): Chord length in meters
    d_tube_from_dat (float, optional): Pre-calculated tube diameter from .dat file fitting
    is_with_return_t_eta_kappa_delta_c (bool): Whether to return profile parameters

    Returns:
    None or tuple: This function saves the plot. Optionally returns profile parameters.
    """
    # Read profile characteristics from the file
    profile = reading_profile_from_airfoil_dat_files(
        filepath, is_calculate_d_tube_from_dat=False
    )
    profile_name = profile["name"]

    # Open the file and read all lines for plotting
    with open(filepath, "r") as file:
        lines = file.readlines()

    # List to store (x, y) points of the profile
    points = []
    for line in lines[1:]:
        x, y = map(float, line.split())
        points.append((x, y))

    # Unzip points into x and y coordinates for plotting
    x_points, y_points = zip(*points)

    center_x = d_tube_from_dat / 2
    center_y = 0.0

    # Plot the profile points
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, y_points, marker="", linestyle="-", color="black")

    # Highlight the max camber point
    x_max_camber = profile["x_max_camber"]
    y_max_camber = profile["y_max_camber"]
    plt.scatter(
        x_max_camber,
        y_max_camber,
        color="black",
        zorder=3,
        label="max_camber: ({x:.3f}, {y:.3f})".format(x=x_max_camber, y=y_max_camber),
    )

    # Create original circle (from parameter t - from Surfplan txt)
    radius = t / 2
    circle = plt.Circle(
        (radius, 0),
        radius,
        fill=False,
        linestyle="-",
        color="red",
        label=f"d_tube_from_surfplan_txt: {t:.3f}",
    )
    plt.gca().add_patch(circle)

    # Create fitted circle from dat file (in red)
    if d_tube_from_dat > 0:
        radius_fitted = d_tube_from_dat / 2
        circle_fitted = plt.Circle(
            (center_x, center_y),
            radius_fitted,
            fill=False,
            linestyle="-",
            color="black",
            label=f"d_tube_from_dat: {d_tube_from_dat:.3f}",
        )
        plt.gca().add_patch(circle_fitted)

    # Calculate dimensional parameters for display
    tube_diameter_mm = t * c * 1000
    tube_diameter_fitted_mm = d_tube_from_dat * c * 1000
    x_max_camber_mm = profile["x_max_camber"] * c * 1000
    y_max_camber_mm = profile["y_max_camber"] * c * 1000

    plt.xlabel(f"$x/c$")
    plt.ylabel(f"$y/c$")
    plt.title(
        rf'{profile_name} --- t: {t:.3f} $\eta$: {profile["x_max_camber"]:.3f}, $\kappa$: {profile["y_max_camber"]:.3f}, $\delta$: {profile["TE_angle"]:.3f}deg, c: {c:.3f}m'
        f"\nTube: {tube_diameter_mm:.1f}mm, Fitted: {tube_diameter_fitted_mm:.1f}mm, Max camber pos: {x_max_camber_mm:.1f}mm, Max camber height: {y_max_camber_mm:.1f}mm"
    )
    plt.legend(loc="lower center")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")

    # Save the plot
    plt.savefig(Path(profile_save_dir) / f"{filepath.stem}.pdf")
    plt.close()

    if is_with_return_t_eta_kappa_delta_c:
        return (
            t,
            profile["x_max_camber"],
            profile["y_max_camber"],
            profile["TE_angle"],
            c,
        )


# def plot_and_save_all_profiles(profile_save_dir, ribs_data):

#     if not profile_save_dir.is_dir():
#         raise ValueError(f"Directory {profile_save_dir} does not exist.")

#     # Create a list to store the data for each profile
#     profile_data = []

#     i = 0
#     for file_name in profile_save_dir.iterdir():
#         # Check if the file name starts with "prof" and ends with ".dat"
#         if file_name.name.startswith("prof") and file_name.name.endswith(".dat"):
#             i += 1
#             rib = ribs_data[i]
#             # print(f"rib: {rib}")
#             t, eta, kappa, delta, c = plot_profiles(
#                 file_name,
#                 profile_save_dir=profile_save_dir,
#                 t=rib["d_tube"],
#                 c=rib["chord"],
#                 is_with_return_t_eta_kappa_delta_c=True,
#             )

#             # Add data to our list
#             profile_data.append(
#                 {
#                     "profile_name": file_name.stem,
#                     "t": t,
#                     "eta": eta,
#                     "kappa": kappa,
#                     "delta": delta,
#                     "c": c,
#                 }
#             )

#     # Convert the list to a pandas DataFrame
#     if profile_data:
#         df = pd.DataFrame(profile_data)

#         # sort on last str in profile_name
#         df["profile_number"] = df["profile_name"].str.split("_").str[-1].astype(int)
#         df = df.sort_values("profile_number")

#         # reshuffle the columns
#         df = df[
#             [
#                 "profile_number",
#                 "profile_name",
#                 "t",
#                 "eta",
#                 "kappa",
#                 "delta",
#                 "c",
#             ]
#         ]

#         # Save the DataFrame to a CSV file
#         csv_path = profile_save_dir / "profile_parameters.csv"
#         df.to_csv(csv_path, index=False)
#         print(f"Profile parameters saved to {csv_path}")
#     else:
#         print("No profiles found to save.")


# def plot_and_save_all_profiles_from_yaml(yaml_file_path, profile_base_dir):
#     """
#     Plot and save all airfoil profiles using data from YAML configuration file.

#     This function reads profile parameters from the YAML wing_airfoils section
#     and plots each .dat file found in the profile directory.

#     Parameters:
#         yaml_file_path (str or Path): Path to the YAML configuration file
#         profile_base_dir (str or Path): Directory containing the profile .dat files

#     Returns:
#         None: This function plots all profiles and saves them as PNG files
#     """
#     yaml_file_path = Path(yaml_file_path)
#     profile_base_dir = Path(profile_base_dir)

#     if not profile_base_dir.is_dir():
#         raise ValueError(f"Directory {profile_base_dir} does not exist.")

#     # Load YAML configuration
#     with open(yaml_file_path, "r") as f:
#         config = yaml.safe_load(f)

#     # Extract wing airfoils data
#     wing_airfoils = config["wing_airfoils"]
#     wing_airfoils_data = wing_airfoils["data"]
#     headers = wing_airfoils["headers"]

#     # Create mapping from headers to indices
#     header_map = {header: idx for idx, header in enumerate(headers)}

#     # Create a dictionary to map airfoil_id to profile parameters
#     airfoil_params = {}
#     for airfoil_data in wing_airfoils_data:
#         airfoil_id = airfoil_data[header_map["airfoil_id"]]
#         info_dict = airfoil_data[header_map["info_dict"]]

#         # Extract parameters from meta_parameters or direct info_dict
#         if "meta_parameters" in info_dict:
#             meta_params = info_dict["meta_parameters"]
#             t = meta_params.get(
#                 "t", 0.1
#             )  # tube diameter to chord ratio (non-dimensional)
#             chord = meta_params.get("chord", 1.0)  # chord length in meters
#         else:
#             # Fallback to direct parameters
#             t = info_dict.get("t", 0.1)
#             chord = info_dict.get("chord", 1.0)

#         airfoil_params[airfoil_id] = {"t": t, "chord": chord}

#     print(f"Found {len(airfoil_params)} airfoil parameter sets in YAML")

#     # Create a list to store the data for each profile
#     profile_data = []

#     # Iterate through all .dat files in the profile directory, sorted numerically
#     def extract_airfoil_number(file_path):
#         """Extract the numeric part from prof_X.dat filename for sorting"""
#         try:
#             return int(file_path.stem.split("_")[1])
#         except (IndexError, ValueError):
#             return float("inf")  # Put invalid filenames at the end

#     for file_path in sorted(
#         profile_base_dir.glob("prof_*.dat"), key=extract_airfoil_number
#     ):
#         profile_name = file_path.stem

#         # Extract airfoil_id from filename (e.g., prof_1.dat -> 1)
#         try:
#             airfoil_id = int(profile_name.split("_")[1])
#         except (IndexError, ValueError):
#             print(
#                 f"Warning: Could not extract airfoil_id from {profile_name}, skipping"
#             )
#             continue

#         # Get parameters for this airfoil
#         if airfoil_id not in airfoil_params:
#             print(
#                 f"Warning: No parameters found for airfoil_id {airfoil_id}, using defaults"
#             )
#             t = 0.1
#             chord = 1.0
#         else:
#             t = airfoil_params[airfoil_id]["t"]
#             chord = airfoil_params[airfoil_id]["chord"]

#         print(f"Plotting {profile_name} with t={t:.4f}, chord={chord:.4f}")

#         try:
#             # Plot the profile and get characteristics
#             t_returned, eta, kappa, delta, c_returned = plot_profiles(
#                 file_path,
#                 profile_save_dir=profile_base_dir,
#                 t=t,
#                 c=chord,
#                 is_with_return_t_eta_kappa_delta_c=True,
#             )

#             # Add data to our list
#             profile_data.append(
#                 {
#                     "profile_name": profile_name,
#                     "airfoil_id": airfoil_id,
#                     "t": t_returned,
#                     "eta": eta,
#                     "kappa": kappa,
#                     "delta": delta,
#                     "c": c_returned,
#                 }
#             )

#         except (FileNotFoundError, ValueError, KeyError) as e:
#             print(f"Error plotting {profile_name}: {e}")
#             continue
#         except IOError as e:
#             print(f"IO error when plotting {profile_name}: {e}")
#             continue

#     # # Convert the list to a pandas DataFrame and save
#     # if profile_data:
#     #     df = pd.DataFrame(profile_data)

#     #     # Sort by airfoil_id
#     #     df = df.sort_values("airfoil_id")

#     #     # Reorder columns
#     #     df = df[
#     #         [
#     #             "airfoil_id",
#     #             "profile_name",
#     #             "t",
#     #             "eta",
#     #             "kappa",
#     #             "delta",
#     #             "c",
#     #         ]
#     #     ]

#     #     # Save the DataFrame to a CSV file
#     #     csv_path = profile_base_dir / "profile_parameters_from_yaml.csv"
#     #     df.to_csv(csv_path, index=False)
#     #     print(f"Profile parameters saved to {csv_path}")
#     #     print(f"Generated {len(profile_data)} profile plots")
#     # else:
#     #     print("No profiles found to save.")


def plot_and_save_all_profiles_from_ribs_data(ribs_data, profile_base_dir):
    """
    Plot and save all airfoil profiles using ribs_data directly.

    This function extracts unique airfoil_ids from ribs_data and plots
    each corresponding .dat file found in the profile directory.

    Parameters:
        ribs_data: List of rib dictionaries containing airfoil_id and profile parameters
        profile_base_dir (str or Path): Directory containing the profile .dat files

    Returns:
        None: This function plots all profiles and saves them as PNG files
    """
    profile_base_dir = Path(profile_base_dir)

    if not profile_base_dir.is_dir():
        raise ValueError(f"Directory {profile_base_dir} does not exist.")

    # Extract unique airfoil_ids and their corresponding rib data
    unique_airfoils = {}
    for rib in ribs_data:
        airfoil_id = rib["airfoil_id"]
        if airfoil_id not in unique_airfoils:
            unique_airfoils[airfoil_id] = rib

    print(f"Found {len(unique_airfoils)} unique airfoils in ribs_data")

    # Create a list to store the data for each profile
    profile_data = []

    # Sort by airfoil_id to ensure consistent ordering
    for airfoil_id in sorted(unique_airfoils.keys()):
        rib = unique_airfoils[airfoil_id]

        # Construct the profile file path
        profile_name = f"prof_{airfoil_id}"
        file_path = profile_base_dir / f"{profile_name}.dat"

        if not file_path.exists():
            print(
                f"Warning: Profile file {file_path} not found, skipping airfoil_id {airfoil_id}"
            )
            continue

        # Extract parameters from rib data
        t = float(
            rib["d_tube_from_surfplan_txt"]
        )  # tube diameter to chord ratio from surfplan txt file
        chord = float(rib["chord"])
        d_tube_from_dat = float(
            rib.get("d_tube_from_dat", 0.0)
        )  # Use pre-calculated value

        print(
            f"Plotting {profile_name} with d_tube_from_surfplan_txt={t:.4f}, chord={chord:.4f}, d_tube_from_dat={d_tube_from_dat:.4f}"
        )

        try:
            # Plot the profile and get characteristics using pre-calculated d_tube_from_dat
            t_returned, eta, kappa, delta, c_returned = plot_profiles(
                file_path,
                profile_save_dir=profile_base_dir,
                t=t,
                c=chord,
                d_tube_from_dat=d_tube_from_dat,  # Pass the pre-calculated value
                is_with_return_t_eta_kappa_delta_c=True,
            )

            # Add data to our list
            profile_data.append(
                {
                    "profile_name": profile_name,
                    "airfoil_id": airfoil_id,
                    "t": t_returned,
                    "eta": eta,
                    "kappa": kappa,
                    "delta": delta,
                    "c": c_returned,
                }
            )

        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"Error plotting {profile_name}: {e}")
            continue
        except IOError as e:
            print(f"IO error when plotting {profile_name}: {e}")
            continue

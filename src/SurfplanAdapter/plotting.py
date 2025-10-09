import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# from SurfplanAdapter.process_surfplan.read_profile_from_airfoil_dat_files import (
#     reading_profile_from_airfoil_dat_files,
# )
from SurfplanAdapter.utils import (
    transform_coordinate_system_surfplan_to_VSM,
)
from SurfplanAdapter.find_airfoil_parameters.main_find_airfoil_parameters import (
    get_fitted_airfoil_parameters,
)


def plot_airfoils_3d_from_yaml(
    yaml_file_path, profile_base_dir, save_path=None, show_plot=True
):
    """
    Plot airfoils in 3D space using YAML file data, and also plot bridles.

    This function reads the generated YAML configuration file, loads the corresponding
    .dat files for each airfoil, scales them according to the chord length, and positions
    them correctly in 3D space using LE, TE, and VUP information. It also plots bridle nodes
    and bridle connections.

    Parameters:
        yaml_file_path (str or Path): Path to the YAML configuration file
        profile_base_dir (str or Path): Base directory containing the profile .dat files
        save_path (str or Path, optional): Path to save the plot. If None, plot is not saved.
        show_plot (bool): Whether to display the plot

    Returns:
        None: This function displays/saves the 3D plot of airfoils and bridles
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
    header_map = {header: idx for idx, header in enumerate(headers)}

    # Extract bridle nodes and connections
    bridle_nodes = config.get("bridle_nodes", {})
    bridle_nodes_data = bridle_nodes.get("data", [])
    bridle_nodes_headers = bridle_nodes.get("headers", [])
    bridle_nodes_map = {header: idx for idx, header in enumerate(bridle_nodes_headers)}

    bridle_connections = config.get("bridle_connections", {})
    bridle_connections_data = bridle_connections.get("data", [])
    bridle_connections_headers = bridle_connections.get("headers", [])
    bridle_connections_map = {
        header: idx for idx, header in enumerate(bridle_connections_headers)
    }

    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    print(f"Plotting {len(wing_sections_data)} airfoil sections...")

    # Plot airfoils
    for i, section_data in enumerate(wing_sections_data):
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

        le_point = np.array([le_x, le_y, le_z])
        te_point = np.array([te_x, te_y, te_z])
        vup_vector = np.array([vup_x, vup_y, vup_z])

        chord_vector = te_point - le_point
        chord_length = np.linalg.norm(chord_vector)
        chord_unit = chord_vector / chord_length
        vup_unit = vup_vector / np.linalg.norm(vup_vector)
        x_local = chord_unit
        y_local = vup_unit
        z_local = np.cross(x_local, y_local)
        z_local = z_local / np.linalg.norm(z_local)

        dat_file_path = profile_base_dir / f"prof_{airfoil_id}.dat"
        if not dat_file_path.exists():
            print(
                f"Warning: Profile file {dat_file_path} not found, skipping airfoil {airfoil_id}"
            )
            continue

        try:
            airfoil_coords = []
            with open(dat_file_path, "r") as f:
                lines = f.readlines()
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
            airfoil_x = airfoil_coords[:, 0] * chord_length
            airfoil_y = airfoil_coords[:, 1] * chord_length
            airfoil_z = np.zeros_like(airfoil_x)
            world_coords = []
            for j in range(len(airfoil_x)):
                local_coord = np.array([airfoil_x[j], airfoil_y[j], airfoil_z[j]])
                world_coord = (
                    le_point
                    + local_coord[0] * x_local
                    + local_coord[1] * y_local
                    + local_coord[2] * z_local
                )
                world_coords.append(world_coord)
            world_coords = np.array(world_coords)
            ax.plot(
                world_coords[:, 0],
                world_coords[:, 1],
                world_coords[:, 2],
                "black",
                linewidth=1,
                alpha=0.7,
            )
            # Plot LE and TE points
            if i == 0:
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

    # Plot bridle nodes
    if bridle_nodes_data:
        # Extract only numeric coordinates to avoid dtype issues with matplotlib
        x_idx = bridle_nodes_map.get("x", 1)
        y_idx = bridle_nodes_map.get("y", 2)
        z_idx = bridle_nodes_map.get("z", 3)

        # Extract coordinates as separate arrays to ensure proper numeric dtypes
        x_coords = np.array([float(row[x_idx]) for row in bridle_nodes_data])
        y_coords = np.array([float(row[y_idx]) for row in bridle_nodes_data])
        z_coords = np.array([float(row[z_idx]) for row in bridle_nodes_data])

        ax.scatter(
            x_coords,
            y_coords,
            z_coords,
            c="orange",
            s=10,
            alpha=0.7,
            label="Bridle Nodes",
        )

    # Plot bridle connections
    if bridle_connections_data and bridle_nodes_data:
        # Build a mapping from node id to coordinates
        node_id_idx = bridle_nodes_map.get("id", 0)
        x_idx = bridle_nodes_map.get("x", 1)
        y_idx = bridle_nodes_map.get("y", 2)
        z_idx = bridle_nodes_map.get("z", 3)
        node_coords = {
            int(row[node_id_idx]): np.array(
                [float(row[x_idx]), float(row[y_idx]), float(row[z_idx])]
            )
            for row in bridle_nodes_data
        }
        ci_idx = bridle_connections_map.get("ci", 1)
        cj_idx = bridle_connections_map.get("cj", 2)
        for conn in bridle_connections_data:
            ci = int(conn[ci_idx])
            cj = int(conn[cj_idx])
            if ci in node_coords and cj in node_coords:
                p1 = node_coords[ci]
                p2 = node_coords[cj]
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    c="orange",
                    alpha=0.7,
                    linewidth=1,
                    label=None,
                )

    # Set labels and title
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D Airfoil Sections and Bridles from YAML Configuration")
    ax.legend()

    # Set equal aspect ratio using manual limits calculation
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
    # Add bridle node coordinates to bounds
    if bridle_nodes_data:
        x_idx = bridle_nodes_map.get("x", 1)
        y_idx = bridle_nodes_map.get("y", 2)
        z_idx = bridle_nodes_map.get("z", 3)
        all_coords.extend(
            [
                [float(row[x_idx]), float(row[y_idx]), float(row[z_idx])]
                for row in bridle_nodes_data
            ]
        )
    if all_coords:
        all_coords = np.array(all_coords)
        x_range = [all_coords[:, 0].min(), all_coords[:, 0].max()]
        y_range = [all_coords[:, 1].min(), all_coords[:, 1].max()]
        z_range = [all_coords[:, 2].min(), all_coords[:, 2].max()]
        x_center = np.mean(x_range)
        y_center = np.mean(y_range)
        z_center = np.mean(z_range)
        max_range = max(
            x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
        )
        margin = max_range * 0.1
        half_range = (max_range + margin) / 2
        ax.set_xlim([x_center - half_range, x_center + half_range])
        ax.set_ylim([y_center - half_range, y_center + half_range])
        ax.set_zlim([z_center - half_range, z_center + half_range])

    ax.view_init(elev=20, azim=-120)
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_struc_geometry_yaml(yaml_path):
    # read yaml file
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # extract relevant data
    wing_particles = config["wing_particles"]["data"]
    wing_connections = config["wing_connections"]["data"]
    bridle_particles = config["bridle_particles"]["data"]
    bridle_connections = config["bridle_connections"]["data"]

    # create 3d plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Structural Geometry from YAML Configuration")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=-120)  # Default azim is 30, so 30-90 = -60

    # plot wing particles
    wing_particles = np.array(wing_particles)
    ax.scatter(
        wing_particles[:, 1],
        wing_particles[:, 2],
        wing_particles[:, 3],
        c="blue",
        s=20,
        alpha=0.6,
        label="Wing Particles",
    )
    # plot bridle particles
    bridle_particles = np.array(bridle_particles)
    ax.scatter(
        bridle_particles[:, 1],
        bridle_particles[:, 2],
        bridle_particles[:, 3],
        c="red",
        s=20,
        alpha=0.6,
        label="Bridle Particles",
    )
    # cluster particles
    all_particles = np.vstack((wing_particles, bridle_particles))

    # plot wing connections
    for conn in wing_connections:
        i, j = conn[1], conn[2]
        p1 = all_particles[i - 1]
        p2 = all_particles[j - 1]
        ax.plot(
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            [p1[3], p2[3]],
            c="blue",
            alpha=1,
            linewidth=1,
        )

    # plot bridle connections
    for conn in bridle_connections:
        i, j = conn[1], conn[2]
        p1 = all_particles[i - 1]
        p2 = all_particles[j - 1]
        ax.plot(
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            [p1[3], p2[3]],
            c="red",
            alpha=1,
            linewidth=1,
        )

    # Set equal aspect ratio using manual limits calculation
    # Get all the plotted data bounds
    if all_particles.size > 0:
        # Calculate the range for each axis
        x_range = [all_particles[:, 1].min(), all_particles[:, 1].max()]
        y_range = [all_particles[:, 2].min(), all_particles[:, 2].max()]
        z_range = [all_particles[:, 3].min(), all_particles[:, 3].max()]

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

    ax.legend()
    plt.title("struc_geometry.yaml")
    plt.show()

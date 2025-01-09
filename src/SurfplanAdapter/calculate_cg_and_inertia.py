import numpy as np
import pandas as pd
from pathlib import Path


def find_mass_distributions(
    file_path: Path,
    total_wing_mass: float,
    canopy_kg_p_sqm: float,
    le_to_strut_mass_ratio: float,
    sensor_mass: float,
):
    """
    Calculate the center of gravity (CG) and inertia of the wing based on panel and component masss.

    Parameters:
    file_path (Path): Path to the CSV file containing wing data.
    total_wing_mass (float): Total mass of the wing.
    canopy_kg_p_sqm (float): Canopy mass in grams per square meter.
    le_to_strut_mass_ratio (float): Ratio of mass between LE and struts.
    sensor_mass (float): mass of the sensor in kilograms.

    Returns:
    dict: A dictionary containing CG, total mass distribution, and individual component masss.
    """
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Extract leading edge (LE) and trailing edge (TE) points
    LE_points = data[["LE_x", "LE_y", "LE_z"]].to_numpy()
    TE_points = data[["TE_x", "TE_y", "TE_z"]].to_numpy()
    d_tube = data["d_tube"].to_numpy()
    is_strut = data["is_strut"].to_numpy()

    # Initialize variables
    total_canopy_mass = 0
    total_area = 0
    panels = []
    panel_canopy_mass_list = []

    # Create panels from consecutive LE and TE points
    for i in range(len(LE_points) - 1):
        # Panel vertices
        p1 = LE_points[i]
        p2 = TE_points[i]
        p3 = TE_points[i + 1]
        p4 = LE_points[i + 1]
        panels.append([p1, p2, p3, p4])

        # Calculate panel surface area (approximate as quadrilateral)
        edge1 = np.linalg.norm(p2 - p1)
        edge2 = np.linalg.norm(p3 - p2)
        diagonal = np.linalg.norm(p3 - p1)
        s = (edge1 + edge2 + diagonal) / 2
        area1 = np.sqrt(
            s * (s - edge1) * (s - edge2) * (s - diagonal)
        )  # First triangle
        edge3 = np.linalg.norm(p4 - p3)
        edge4 = np.linalg.norm(p4 - p1)
        diagonal = np.linalg.norm(p4 - p3)
        s = (edge3 + edge4 + diagonal) / 2
        area2 = np.sqrt(
            s * (s - edge3) * (s - edge4) * (s - diagonal)
        )  # Second triangle
        panel_area = area1 + area2
        total_area += panel_area

        # Calculate panel mass (canopy mass in kg)
        panel_canopy_mass = panel_area * canopy_kg_p_sqm  # g -> kg
        panel_canopy_mass_list.append(panel_canopy_mass)
        total_canopy_mass += panel_canopy_mass

    # Calculate mass of other components
    if total_canopy_mass > total_wing_mass:
        raise ValueError(
            "Total panel mass exceeds total wing mass.\nLower canopy mass OR increase total_mass."
        )
    non_canopy_mass = total_wing_mass - total_canopy_mass

    # Distribute the remaining mass
    non_canopy_mass_min_sensor = non_canopy_mass - sensor_mass
    le_mass = non_canopy_mass_min_sensor * le_to_strut_mass_ratio
    strut_mass = non_canopy_mass_min_sensor * (1 - le_to_strut_mass_ratio)

    #### Distributing the mass over the nodes
    # Assuming the mass is distributed uniformly over the length of the LE
    le_mass_per_node = le_mass / len(LE_points)
    # Assuming the mass is distributed uniformly over the length of the strut
    strut_mass_per_node = 0.5 * (strut_mass / len(LE_points))

    ## Find leading-edge points where the sensor mass is at
    n_ribs = len(LE_points)
    sensor_points_indices = [int(n_ribs // 2), int(n_ribs // 2) - 1]

    return (
        total_canopy_mass,
        panel_canopy_mass_list,
        le_mass_per_node,
        strut_mass_per_node,
        sensor_points_indices,
        LE_points,
        TE_points,
        is_strut,
        total_area,
    )


def distribute_mass_over_nodes(
    le_mass_per_node,
    strut_mass_per_node,
    sensor_mass,
    panel_canopy_mass_list,
    sensor_points_indices,
    LE_points,
    TE_points,
    is_strut,
):
    # Distribute the mass over the nodes
    # Initialize list of nodes with mass distribution
    nodes = []

    for i, le_point in enumerate(LE_points):
        node_mass = le_mass_per_node
        # Determine if the node is a sensor node
        if i in sensor_points_indices:
            node_mass += sensor_mass / len(sensor_points_indices)
        # Determine if strut
        if is_strut[i]:
            node_mass += strut_mass_per_node
        # Add canopy mass
        if i == len(LE_points) - 1:
            node_mass += 0.25 * panel_canopy_mass_list[i - 1]
        else:
            node_mass += 0.25 * panel_canopy_mass_list[i]

        nodes.append([le_point, node_mass])

    for i, te_point in enumerate(TE_points):
        node_mass = 0
        # Determine if strut
        if is_strut[i]:
            node_mass += strut_mass_per_node
        # Add canopy mass
        if i == len(TE_points) - 1:
            node_mass += 0.25 * panel_canopy_mass_list[i - 1]
        else:
            node_mass += 0.25 * panel_canopy_mass_list[i]

        nodes.append([te_point, node_mass])

    return nodes


def calculate_cg(nodes):
    """
    Calculate the center of gravity (CG) of the nodes.

    Parameters:
    nodes (list): A list of nodes, where each node is [position (array), mass (float)].

    Returns:
    tuple: The x, y, z coordinates of the center of gravity.
    """
    total_mass = sum(node[1] for node in nodes)
    x_cg = sum(node[0][0] * node[1] for node in nodes) / total_mass
    y_cg = sum(node[0][1] * node[1] for node in nodes) / total_mass
    z_cg = sum(node[0][2] * node[1] for node in nodes) / total_mass

    return x_cg, y_cg, z_cg


def plot_nodes(nodes, x_cg, y_cg, z_cg):

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # make a 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    node_masses = np.array([node[1] for node in nodes])  # Extract mass values

    # Normalize mass values for coloring
    node_positions = np.array(
        [node[0] for node in nodes]
    )  # Extract [x, y, z] positions

    mass_normalized = (node_masses - np.min(node_masses)) / (
        np.max(node_masses) - np.min(node_masses)
    )

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot with colors representing mass
    sc = ax.scatter(
        node_positions[:, 0],  # x-coordinates
        node_positions[:, 1],  # y-coordinates
        node_positions[:, 2],  # z-coordinates
        c=mass_normalized,  # Use normalized mass as the color
        cmap=cm.viridis,  # Colormap
        s=50,  # Marker size
    )

    # plot the cg
    ax.scatter(x_cg, y_cg, z_cg, c="r", marker="x", s=100, label="Center of Gravity")

    # Add color bar for the mass scale
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Normalized Mass")

    # Label axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Show the plot
    plt.show()


def calculate_inertia(nodes, desired_point):
    """
    Calculate the inertia tensor of the nodes about a desired point.

    Parameters:
    nodes (list): A list of nodes, where each node is [position (array), mass (float)].
    desired_point (array-like): The point [x, y, z] about which the inertia tensor is calculated.

    Returns:
    np.ndarray: The 3x3 inertia tensor.
    """
    desired_point = np.array(desired_point)  # Ensure it's a numpy array
    inertia_tensor = np.zeros((3, 3))  # Initialize 3x3 tensor

    for node in nodes:
        position = node[0]  # Node position [x, y, z]
        mass = node[1]  # Node mass

        # Vector from desired point to node
        r = position - desired_point
        r_x, r_y, r_z = r

        # Update the inertia tensor
        inertia_tensor[0, 0] += mass * (r_y**2 + r_z**2)  # Ixx
        inertia_tensor[1, 1] += mass * (r_x**2 + r_z**2)  # Iyy
        inertia_tensor[2, 2] += mass * (r_x**2 + r_y**2)  # Izz
        inertia_tensor[0, 1] -= mass * r_x * r_y  # Ixy
        inertia_tensor[0, 2] -= mass * r_x * r_z  # Ixz
        inertia_tensor[1, 2] -= mass * r_y * r_z  # Iyz

    # Symmetric off-diagonal elements
    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]

    return inertia_tensor


def main(
    file_path,
    total_wing_mass,
    canopy_kg_p_sqm,
    le_to_strut_mass_ratio,
    sensor_mass,
    is_show_plot=True,
):

    (
        total_canopy_mass,
        panel_canopy_mass_list,
        le_mass_per_node,
        strut_mass_per_node,
        sensor_points_indices,
        LE_points,
        TE_points,
        is_strut,
        total_area,
    ) = find_mass_distributions(
        file_path,
        total_wing_mass,
        canopy_kg_p_sqm,
        le_to_strut_mass_ratio,
        sensor_mass,
    )
    nodes = distribute_mass_over_nodes(
        le_mass_per_node,
        strut_mass_per_node,
        sensor_mass,
        panel_canopy_mass_list,
        sensor_points_indices,
        LE_points,
        TE_points,
        is_strut,
    )
    x_cg, y_cg, z_cg = calculate_cg(nodes)
    if is_show_plot:
        plot_nodes(
            nodes,
            x_cg,
            y_cg,
            z_cg,
        )
    desired_point = [0, 0, 0]
    inertia_tensor = calculate_inertia(nodes, desired_point)

    # printing
    print(f"----------- OUTPUT")
    # print(f"sensor_points_indices: {sensor_points_indices}\n")
    # print(
    #     f"Total canopy mass:      {total_canopy_mass:.2f} kg (area: {total_area:.2f} m^2 * {canopy_kg_p_sqm} kg/m^2)"
    # )
    # print(f"leading_edge_mass:      {le_mass_per_node*len(LE_points):.2f} kg")
    # print(f"strut_mass:             {strut_mass_per_node*len(LE_points):.2f} kg")
    # print(f"sensor_mass:            {sensor_mass} kg")
    # print(
    #     f"Sum of the above:      {total_canopy_mass+ le_mass_per_node*len(LE_points) + strut_mass_per_node*len(LE_points) + sensor_mass:.2f} kg"
    # )
    print(f"Total node mass:        {sum([node[1] for node in nodes]):.2f} kg")
    print(f"Total wing mass         {total_wing_mass} kg")
    print(f"\n")
    print(f"center of gravity: {x_cg:.2f}, {y_cg:.2f}, {z_cg:.2f}")
    print("Inertia tensor:")
    print(inertia_tensor)


if __name__ == "__main__":
    file_path = Path(
        "/home/jellepoland/ownCloud/phd/code/SurfplanAdapter/processed_data/TUDELFT_V3_LEI_KITE/geometry.csv"
    )
    total_wing_mass = 10.0
    canopy_kg_p_sqm = 0.2
    le_to_strut_mass_ratio = 0.7
    sensor_mass = 0.5
    is_show_plot = False
    print(f"-------------- INPUT")
    print(f"total_wing_mass: {total_wing_mass}")
    print(f"canopy_kg_p_sqm: {canopy_kg_p_sqm}")
    print(f"le_to_strut_mass_ratio: {le_to_strut_mass_ratio}")
    print(f"sensor_mass: {sensor_mass}\n")

    main(
        file_path,
        total_wing_mass,
        canopy_kg_p_sqm,
        le_to_strut_mass_ratio,
        sensor_mass,
        is_show_plot,
    )

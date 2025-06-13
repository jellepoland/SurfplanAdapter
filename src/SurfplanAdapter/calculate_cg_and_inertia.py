import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from SurfplanAdapter.utils import PROJECT_DIR


def find_mass_distributions(
    wing_sections,
    total_wing_mass: float,
    canopy_kg_p_sqm: float,
    le_to_strut_mass_ratio: float,
    sensor_mass: float,
):
    """
    Calculate the center of gravity (CG) and inertia of the wing based on panel and component masss.

    Parameters:
    wing_sections (list): List of wing sections data from YAML.
    total_wing_mass (float): Total mass of the wing.
    canopy_kg_p_sqm (float): Canopy mass in grams per square meter.
    le_to_strut_mass_ratio (float): Ratio of mass between LE and struts.
    sensor_mass (float): mass of the sensor in kilograms.

    Returns:
    dict: A dictionary containing CG, total mass distribution, and individual component masss.
    """
    # Extract leading edge (LE) and trailing edge (TE) points
    # wing_sections data format: [airfoil_id, LE_x, LE_y, LE_z, TE_x, TE_y, TE_z, VUP_x, VUP_y, VUP_z]
    LE_points = np.array(
        [
            [section[1], section[2], section[3]]  # LE_x, LE_y, LE_z
            for section in wing_sections
        ]
    )
    TE_points = np.array(
        [
            [section[4], section[5], section[6]]  # TE_x, TE_y, TE_z
            for section in wing_sections
        ]
    )
    # VUP points are now available if needed: [section[7], section[8], section[9]]
    # For now, using default values for d_tube and is_strut since they're not in the YAML format
    # TODO: Add these to the YAML generation if needed
    d_tube = np.array([0.01 for section in wing_sections])  # Default tube diameter
    is_strut = np.array([False for section in wing_sections])  # Default: no struts

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
        panel_canopy_mass = panel_area * canopy_kg_p_sqm
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
    n_LE_points = len(LE_points) + 2
    le_mass_per_node = le_mass / n_LE_points
    # Assuming the mass is distributed uniformly over the length of the strut
    # strut_mass_per_node = strut_mass / len(LE_points)
    n_strut_nodes = np.count_nonzero(is_strut)
    strut_mass_per_node = 0.5 * (strut_mass / n_strut_nodes)

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
    """
    Distribute the mass over the nodes so that:
      - LE mass is distributed among LE nodes (le_mass_per_node).
      - Strut mass is added to nodes flagged as 'is_strut' (strut_mass_per_node).
      - The sensor mass is distributed among a couple of LE nodes (sensor_points_indices).
      - The canopy mass from each panel is split among its 4 corner nodes, i.e.
        each corner node gets 1/4 of that panel's canopy mass.

    The trick is that an LE node i is corner of panel (i-1) and panel i
    (except at the boundaries). The same logic applies to TE nodes --
    EXCEPT for the outer two TE nodes (i=0, i=n_te-1), which we treat
    as if they are LE nodes for mass distribution.
    """

    import numpy as np

    nodes = []

    n_le = len(LE_points)  # Number of leading-edge nodes
    n_te = len(TE_points)  # Should match n_le if CSV is consistent

    #
    # 1) Distribute mass to LE nodes
    #
    for i, le_point in enumerate(LE_points):
        node_mass = 0.0
        # LE mass portion
        node_mass += le_mass_per_node
        # Sensor mass portion (if this LE node is flagged for sensor)
        if i in sensor_points_indices:
            node_mass += sensor_mass / len(sensor_points_indices)
        # Strut mass portion (only if flagged)
        if is_strut[i]:
            node_mass += strut_mass_per_node

        # ---- Canopy mass portion for LE node i ----
        # This LE node belongs to panel i-1 (if i > 0) and panel i (if i < n_le-1)
        if i > 0:
            node_mass += 0.25 * panel_canopy_mass_list[i - 1]
        if i < n_le - 1:
            node_mass += 0.25 * panel_canopy_mass_list[i]

        # Store node with mass
        nodes.append([le_point, node_mass])

    #
    # 2) Distribute mass to TE nodes
    #
    for i, te_point in enumerate(TE_points):
        node_mass = 0.0

        # If this is an outer TE node (i=0 or i=n_te-1), treat it like LE for mass
        # distribution. That means we add LE mass, possible sensor mass, etc.
        # But still do the "TE canopy corner" logic for panel distribution, so we
        # *also* get the 1/4 canopy mass from panels i and i-1 if inside range.
        if i == 0 or i == (n_te - 1):
            # Outer TE -> treat like LE
            node_mass += le_mass_per_node

        else:
            # Normal TE node logic (no LE mass, no sensor mass, but maybe strut)
            if is_strut[i]:
                node_mass += strut_mass_per_node

        # ---- Canopy mass portion for TE node i ----
        # If i>0, we add 0.25 from panel i-1
        if i > 0:
            node_mass += 0.25 * panel_canopy_mass_list[i - 1]
        # If i<n_te-1, we add 0.25 from panel i
        if i < n_te - 1:
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


def plot_nodes(
    nodes,
    x_cg,
    y_cg,
    z_cg,
    desired_point,
    LE_points=None,
    TE_points=None,
    is_strut=None,
):
    """
    Parameters
    ----------
    nodes : list
        A list of [ [x,y,z], mass ] for each node.
    x_cg, y_cg, z_cg : float
        Center of gravity coordinates.
    LE_points : array-like, shape (n_ribs, 3), optional
        Leading-edge points corresponding to each rib.
    TE_points : array-like, shape (n_ribs, 3), optional
        Trailing-edge points corresponding to each rib.
    is_strut : array-like of bool, shape (n_ribs,), optional
        Boolean flags indicating where struts exist.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    # Quick helper for setting 3D axes to equal scale
    def set_axes_equal_3d(ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.  This is one possible solution to Matplotlib's
        3D aspect ratio problem.
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        max_range = max(x_range, y_range, z_range)
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    # Create a new figure + 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Convert node data into arrays for plotting
    node_positions = np.array([node[0] for node in nodes])  # shape (N, 3)
    node_masses = np.array([node[1] for node in nodes])  # shape (N,)

    # Scatter plot of all nodes, colored by mass
    sc = ax.scatter(
        node_positions[:, 0],
        node_positions[:, 1],
        node_positions[:, 2],
        c=node_masses,
        cmap=cm.cool,
        s=50,
        alpha=0.9,
        label="Wing Nodes colored by mass",
    )

    # Plot the CG as a big red 'X'
    ax.scatter(
        x_cg,
        y_cg,
        z_cg,
        c="red",
        marker="x",
        s=100,
        label=f"Center of Gravity ({x_cg:.2f}, {y_cg:.2f}, {z_cg:.2f})",
    )

    # Plot the point around which the inertia tensor is calculated
    ax.scatter(
        desired_point[0],
        desired_point[1],
        desired_point[2],
        c="green",
        marker="x",
        s=100,
        label=f"Point of Inertia Calculation ({desired_point[0]:.2f}, {desired_point[1]:.2f}, {desired_point[2]:.2f})",
    )
    # Optional: draw the LE “backbone” if provided
    if LE_points is not None and len(LE_points) > 1:
        LE_points = np.array(LE_points)  # shape (n_ribs, 3)
        # add first and last TE_points to this list
        if TE_points is not None:
            LE_points_with_tips = np.vstack([TE_points[0], LE_points, TE_points[-1]])
        ax.plot(
            LE_points_with_tips[:, 0],
            LE_points_with_tips[:, 1],
            LE_points_with_tips[:, 2],
            c="black",
            linewidth=5,
            label="Leading Edge",
        )

    # Optional: draw each strut if we have both LE, TE, and is_strut info
    if (
        LE_points is not None
        and TE_points is not None
        and is_strut is not None
        and len(LE_points) == len(TE_points) == len(is_strut)
    ):
        for i in range(len(LE_points)):
            if is_strut[i]:
                ax.plot(
                    [LE_points[i, 0], TE_points[i, 0]],
                    [LE_points[i, 1], TE_points[i, 1]],
                    [LE_points[i, 2], TE_points[i, 2]],
                    c="black",
                    linewidth=3,
                    label="Strut" if i == 1 else "",
                )
            else:
                ax.plot(
                    [LE_points[i, 0], TE_points[i, 0]],
                    [LE_points[i, 1], TE_points[i, 1]],
                    [LE_points[i, 2], TE_points[i, 2]],
                    c="grey",
                    linewidth=0.5,
                    linestyle="-",
                    label="Rib lines" if i == 0 else "",
                )

    # Add TE line
    if TE_points is not None and len(TE_points) > 1:
        TE_points = np.array(TE_points)
        ax.plot(
            TE_points[:, 0],
            TE_points[:, 1],
            TE_points[:, 2],
            c="grey",
            linewidth=0.5,
            label="Trailing Edge",
        )

    # Add color bar for node masses
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("Node Mass (kg)")

    # Label axes and set 3D axes to equal scale
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(False)
    set_axes_equal_3d(ax)

    ax.legend()
    plt.tight_layout()
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
    yaml_file_path,
    total_wing_mass=10.0,
    canopy_kg_p_sqm=0.05,
    le_to_strut_mass_ratio=0.7,
    sensor_mass=0.5,
    desired_point=[0, 0, 0],
    is_show_plot=True,
):
    """
    Calculate CG and inertia using a config_kite.yaml file.
    """
    # Load geometry from YAML
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)
    # Extract wing_sections data
    wing_sections = config["wing_sections"]["data"]
    # Optionally: extract other info from YAML as needed

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
        wing_sections,
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
    inertia_tensor = calculate_inertia(nodes, desired_point)

    # printing
    print(f"\n--- INPUT ---")
    print(f"total_wing_mass: {total_wing_mass}")
    print(f"canopy_kg_p_sqm: {canopy_kg_p_sqm}")
    print(f"le_to_strut_mass_ratio: {le_to_strut_mass_ratio}")
    print(f"sensor_mass: {sensor_mass}")

    print(f"\n--- OUTPUT --- ")
    print(
        f"Total node mass:        {sum([node[1] for node in nodes]):.2f} kg (should be equal to input total_wing_mass)"
    )
    print(f"center of gravity: [{x_cg:.2f}, {y_cg:.2f}, {z_cg:.2f}] [m]")
    print(f"point around intertia is calculated: {desired_point} [m]")
    print("Inertia tensor:")
    print("Ixx: {:.2f}".format(inertia_tensor[0, 0]))
    print("Iyy: {:.2f}".format(inertia_tensor[1, 1]))
    print("Izz: {:.2f}".format(inertia_tensor[2, 2]))
    print("Ixy: {:.2f}".format(inertia_tensor[0, 1]))
    print("Ixz: {:.2f}".format(inertia_tensor[0, 2]))
    print("Iyz: {:.2f}".format(inertia_tensor[1, 2]))

    if is_show_plot:
        plot_nodes(
            nodes,
            x_cg,
            y_cg,
            z_cg,
            desired_point,
            LE_points=LE_points,
            TE_points=TE_points,
            is_strut=is_strut,  # so we can draw strut lines
        )

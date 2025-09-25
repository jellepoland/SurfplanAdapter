from pathlib import Path
from SurfplanAdapter.process_bridle_lines import (
    generate_bridle_connections_data,
    generate_bridle_lines_data,
    generate_bridle_nodes_data,
)
from SurfplanAdapter.process_wing import (
    generate_wing_sections_data,
    generate_wing_airfoils_data,
)
from SurfplanAdapter.generate_yaml import utils


def transform_struc_geometry_dict_to_yaml_format(struc_geometry_dict):
    """
    Transform the structural geometry dict to the desired YAML format.
    Adds general, mass, and material properties sections, then appends all geometry data.
    """
    yaml_data = {}

    # General section
    yaml_data["##############################"] = None
    yaml_data["## General ###################"] = None
    yaml_data["##############################"] = None
    yaml_data["   "] = None
    yaml_data["bridle_point_node"] = [0, 0, 0]  # [x,y,z] --> location of kcu

    # Mass section
    yaml_data["   "] = None
    yaml_data["## Mass"] = None
    yaml_data["pulley_mass"] = float(0.1)  # [kg]
    yaml_data["kcu_mass"] = float(8.4)  # [kg]

    # Material properties section
    yaml_data["   "] = None
    yaml_data["## Material properties"] = None
    yaml_data["dyneema"] = {
        "density": 724,  # [kg/m^3]
        "youngs_modulus": 550000000,  # [Pa]
        "damping_per_stiffness": 0.0,  # [/s]
    }

    # Wing section
    yaml_data["   "] = None
    yaml_data["###########################"] = None
    yaml_data["## Wing ###################"] = None
    yaml_data["###########################"] = None
    yaml_data["   "] = None

    # Add wing_particles, wing_connections, wing_elements
    for key in ["wing_particles", "wing_connections", "wing_elements"]:
        yaml_data["   "] = None
        yaml_data[key] = struc_geometry_dict[key]

    # Bridle section
    yaml_data["   "] = None
    yaml_data["#############################"] = None
    yaml_data["## Bridle ###################"] = None
    yaml_data["#############################"] = None
    yaml_data["   "] = None

    # Add bridle_particles
    yaml_data["   "] = None
    yaml_data["bridle_particles"] = struc_geometry_dict["bridle_particles"]

    # Add bridle_connections
    yaml_data["   "] = None
    yaml_data["bridle_connections"] = struc_geometry_dict["bridle_connections"]

    # Add bridle_lines
    yaml_data["   "] = None
    yaml_data["bridle_lines"] = struc_geometry_dict["bridle_lines"]

    return yaml_data


def main(
    ribs_data,
    bridle_lines,
    yaml_file_path,
    airfoil_type: str = "masure_regression",
):
    yaml_file_path = Path(yaml_file_path.parent / "struc_geometry.yaml")

    # Create wing dict
    wing_sections = generate_wing_sections_data.main(ribs_data)
    wing_airfoils = generate_wing_airfoils_data.main(ribs_data, airfoil_type)

    # Sort wing_airfoils["data"], such that we are left only with ribs where a LAP is present
    wing_airfoils_data_struts_only = []
    n_airfoils = len(wing_airfoils["data"])
    rib_indices_with_LAPS = []
    for i, entry in enumerate(wing_airfoils["data"]):
        entry_dict = entry[2]
        print(f"entry_dict: {entry_dict}")
        if entry_dict["is_strut"] is True:
            wing_airfoils_data_struts_only.append(entry)
            rib_indices_with_LAPS.append(i + 1)
        elif i == n_airfoils - 1:
            # Always include the last rib, which is the tip
            wing_airfoils_data_struts_only.append(entry)
            rib_indices_with_LAPS.append(i + 1)

    wing_airfoils["data"] = wing_airfoils_data_struts_only

    # Now filter wing_sections to only include ribs in rib_indices_with_LAPS
    wing_sections_data_filtered = []
    for i, section in enumerate(wing_sections["data"]):
        airfoil_id = section[0]
        if airfoil_id in rib_indices_with_LAPS:
            wing_sections_data_filtered.append(section)

    wing_sections["data"] = wing_sections_data_filtered

    # rewrite the wing_sections_data into a new format:
    wing_particles = {"headers": ["id", "x", "y", "z"]}
    wing_particles_data = []
    idx = 1
    for section in wing_sections_data_filtered:
        # append leading-edge and then then trailing-edge
        wing_particles_data.append([idx, section[1], section[2], section[3]])
        idx += 1
        wing_particles_data.append([idx, section[4], section[5], section[6]])
        idx += 1

    wing_particles["data"] = wing_particles_data

    # Then we create the wing_connections dict

    # Each panel consists of 4 points: [LE_i, TE_i, LE_{i+1}, TE_{i+1}]
    # Indices in wing_particles_data: [2*i, 2*i+1, 2*i+2, 2*i+3]
    n_panels = (len(wing_particles_data) // 2) - 1
    wing_connections = {"headers": ["name", "ci", "cj"], "data": []}

    # Left side panels
    for i in range(n_panels // 2):
        le_i = 2 * i + 1
        te_i = 2 * i + 2
        le_next = 2 * (i + 1) + 1
        te_next = 2 * (i + 1) + 2

        # Panel number for naming (1-based)
        panel_num = i + 1

        wing_connections["data"].extend(
            [
                [f"strut_{panel_num}", le_i, te_i],
                [f"le_{panel_num}", le_i, le_next],
                [f"te_{panel_num}", te_i, te_next],
                [f"dia_{panel_num}a", le_i, te_next],
                [f"dia_{panel_num}b", te_i, le_next],
            ]
        )

    # Center panel
    center_idx = n_panels // 2
    le_c = 2 * center_idx + 1
    te_c = 2 * center_idx + 2
    le_next_c = 2 * (center_idx + 1) + 1
    te_next_c = 2 * (center_idx + 1) + 2
    wing_connections["data"].extend(
        [
            [f"strut_{center_idx+1}", le_c, te_c],
            [f"le_{center_idx+1}", le_c, le_next_c],
            [f"te_{center_idx+1}", te_c, te_next_c],
            [f"dia_{center_idx+1}a", le_c, te_next_c],
            [f"dia_{center_idx+1}b", te_c, le_next_c],
        ]
    )

    # Right side panels (mirror)
    for i in range(center_idx + 1, n_panels + 1):
        mirror_panel = n_panels - i + 1
        le_i = 2 * i + 1
        te_i = 2 * i + 2
        le_prev = 2 * (i - 1) + 1
        te_prev = 2 * (i - 1) + 2

        # Use mirrored names
        wing_connections["data"].extend(
            [
                [f"strut_{mirror_panel}", le_i, te_i],
                [f"le_{mirror_panel}", le_i, le_prev],
                [f"te_{mirror_panel}", te_i, te_prev],
                [f"dia_{mirror_panel}b", le_i, te_prev],
                [f"dia_{mirror_panel}a", te_i, le_prev],
            ]
        )

    # Optionally, connect last strut at tip (for symmetry)
    if len(wing_particles_data) >= 2:
        last_le = len(wing_particles_data) - 1
        last_te = len(wing_particles_data)
        wing_connections["data"].append([f"strut_1", last_le, last_te])

    # Then we need to create wing_elements

    # Default parameters for elements
    # You can add these as arguments to the function if needed
    k_le = 2e3
    k_strut = 2e3
    k_te = 1e3
    k_dia = 1e3
    c_default = 0
    m_le = 1
    m_strut = 1
    m_te = 0.2
    m_dia = 0.1
    linktype_default = "default"

    wing_elements = {
        "headers": ["name", "l0", "k", "c", "m", "linktype"],
        "data": [],
    }

    # Helper to get coordinates from wing_particles_data by index
    def get_coords(idx):
        return [
            wing_particles_data[idx - 1][1],
            wing_particles_data[idx - 1][2],
            wing_particles_data[idx - 1][3],
        ]

    # Add leading-edge tubes (le_X)
    for conn in wing_connections["data"]:
        name, ci, cj = conn[0], conn[1], conn[2]
        if name.startswith("le_"):
            p1 = get_coords(ci)
            p2 = get_coords(cj)
            l0 = (
                (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
            ) ** 0.5
            wing_elements["data"].append(
                [name, l0, k_le, c_default, m_le, linktype_default]
            )

    # Add struts (strut_X)
    for conn in wing_connections["data"]:
        name, ci, cj = conn[0], conn[1], conn[2]
        if name.startswith("strut_"):
            p1 = get_coords(ci)
            p2 = get_coords(cj)
            l0 = (
                (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
            ) ** 0.5
            wing_elements["data"].append(
                [name, l0, k_strut, c_default, m_strut, linktype_default]
            )

    # Add trailing-edge wires (te_X)
    for conn in wing_connections["data"]:
        name, ci, cj = conn[0], conn[1], conn[2]
        if name.startswith("te_"):
            p1 = get_coords(ci)
            p2 = get_coords(cj)
            l0 = (
                (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
            ) ** 0.5
            wing_elements["data"].append(
                [name, l0, k_te, c_default, m_te, linktype_default]
            )

    # Add diagonal springs (dia_Xa and dia_Xb)
    for conn in wing_connections["data"]:
        name, ci, cj = conn[0], conn[1], conn[2]
        if name.startswith("dia_"):
            p1 = get_coords(ci)
            p2 = get_coords(cj)
            l0 = (
                (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
            ) ** 0.5
            wing_elements["data"].append(
                [name, l0, k_dia, c_default, m_dia, linktype_default]
            )

    # now deal with the bridle lines
    bridle_nodes = generate_bridle_nodes_data.main(bridle_lines)

    #### the bridle_nodes should be transformed to the new_format FIRST
    #### to preserve the coordinate-to-ID mapping that bridle_connections rely on

    # Find the last id used in wing_particles
    last_wing_particle_id = wing_particles_data[-1][0] if wing_particles_data else 0

    # Prepare bridle_particles in the new format
    # IMPORTANT: Preserve the original node ID mapping by using (original_id + offset)
    bridle_particles = {"headers": ["id", "x", "y", "z"], "data": []}
    offset = last_wing_particle_id

    for node in bridle_nodes["data"]:
        # node: [original_id, x, y, z, type]
        # Use original_id + offset to preserve coordinate-to-ID mapping
        new_id = node[0] + offset
        bridle_particles["data"].append([new_id, node[1], node[2], node[3]])

    # Now generate bridle_connections using the same offset
    bridle_connections = generate_bridle_connections_data.main(
        bridle_lines, bridle_nodes, len(wing_particles_data)
    )
    bridle_lines_yaml = generate_bridle_lines_data.main(bridle_lines)

    # Compose the final yaml_data dictionary
    struc_geometry_dict = {
        "wing_particles": wing_particles,
        "wing_connections": wing_connections,
        "wing_elements": wing_elements,
        "bridle_particles": bridle_particles,
        "bridle_connections": bridle_connections,
        "bridle_lines": bridle_lines_yaml,
    }

    ### now we need to transform all this to the correct yaml format
    yaml_data = transform_struc_geometry_dict_to_yaml_format(struc_geometry_dict)

    # # Add bridle data if available
    # if len(bridle_lines) > 0:
    #     yaml_data[" "] = None  # Empty line before bridle_nodes
    #     bridle_nodes = generate_bridle_nodes_data.main(bridle_lines)
    #     bridle_connections = generate_bridle_connections_data.main(
    #         bridle_lines, bridle_nodes
    #     )
    #     bridle_lines_yaml = generate_bridle_lines_data.main(bridle_lines)

    #     yaml_data = create_bridle_dict(
    #         yaml_data, bridle_nodes, bridle_lines_yaml, bridle_connections
    #     )

    # Save to YAML
    utils.save_to_yaml(yaml_data, yaml_file_path)

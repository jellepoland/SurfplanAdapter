import os
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

from SurfplanAdapter.utils import PROJECT_DIR
from SurfplanAdapter.process_surfplan import main_process_surfplan


def represent_list(self, data):
    # Force inline (flow) style for data arrays
    # Check if this is a data row (list with mixed types including dicts)
    if data and len(data) <= 10:  # Typical data row length
        # Special handling for rows that contain dictionaries
        if any(isinstance(item, dict) for item in data):
            # For wing_airfoils data rows with info_dict
            return self.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )
        # For simple numeric/string data rows
        elif all(isinstance(item, (int, float, str)) for item in data):
            return self.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )
    return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


def generate_wing_sections_data(processed_ribs_data):
    """
    Generate wing sections data for YAML output.

    Parameters:
        processed_ribs_data: List of processed rib dictionaries with airfoil_id already assigned

    Returns:
        dict: Wing sections data formatted for YAML
    """
    wing_sections_data = []

    for rib in processed_ribs_data:
        # Transform coordinates to VSM coordinate system
        LE = rib["LE"]
        TE = rib["TE"]
        VUP = rib["VUP"]

        # Use the airfoil_id that was already assigned during processing
        airfoil_id = rib["airfoil_id"]

        wing_sections_data.append(
            [
                airfoil_id,
                float(LE[0]),
                float(LE[1]),
                float(LE[2]),
                float(TE[0]),
                float(TE[1]),
                float(TE[2]),
                float(VUP[0]),
                float(VUP[1]),
                float(VUP[2]),
            ]
        )

    return {
        "headers": [
            "airfoil_id",
            "LE_x",
            "LE_y",
            "LE_z",
            "TE_x",
            "TE_y",
            "TE_z",
            "VUP_x",
            "VUP_y",
            "VUP_z",
        ],
        "data": wing_sections_data,
    }


def generate_wing_airfoils_data(ribs_data, airfoil_type="masure_regression"):
    """
    Generate wing airfoils data for YAML output.

    Parameters:
        ribs_data: List of rib dictionaries containing airfoil_id and profile parameters
        airfoil_type: Type of airfoil data ("polars", "breukels_regression", etc.)

    Returns:
        dict: Wing airfoils data formatted for YAML
    """
    wing_airfoils_data = []

    # Extract unique airfoil_ids and their corresponding rib data
    unique_airfoils = {}
    for rib in ribs_data:
        airfoil_id = rib["airfoil_id"]
        if airfoil_id not in unique_airfoils:
            unique_airfoils[airfoil_id] = rib

    # Sort by airfoil_id to ensure consistent ordering
    for airfoil_id in sorted(unique_airfoils.keys()):
        rib = unique_airfoils[airfoil_id]

        # Create common meta_parameters for all types
        meta_parameters = {
            "d_tube_from_surfplan_txt": float(rib["d_tube_from_surfplan_txt"]),
            "d_tube_from_dat": float(rib["d_tube_from_dat"]),
            "eta": float(rib["x_max_camber"]),  # eta is x-position of max camber
            "kappa": float(rib["y_max_camber"]),  # kappa is camber magnitude
            "delta": float(rib["TE_angle"]),  # delta is trailing edge angle
            "chord": float(rib["chord"]),
            "is_strut": bool(rib["is_strut"]),
        }

        if airfoil_type == "polars":
            # For polars, reference CSV file
            info_dict = {
                "csv_file_path": f"2D_polars_CFD/{airfoil_id}.csv",
                "meta_parameters": meta_parameters,
            }
        elif airfoil_type == "breukels_regression":
            # For Breukels regression, use t and kappa from rib data
            info_dict = {
                "t": float(
                    rib["d_tube_from_dat"]
                ),  # Use d_tube_from_dat for fitting accuracy
                "kappa": float(
                    rib["y_max_camber"]
                ),  # y_max_camber is the camber magnitude
                "meta_parameters": meta_parameters,
            }
        elif airfoil_type == "neuralfoil":
            # For NeuralFoil, reference the dat file and include all available parameters
            info_dict = {
                "dat_file_path": f"profiles/prof_{airfoil_id}.dat",
                "model_size": "xxxlarge",
                "xtr_lower": 0.1,
                "xtr_upper": 0.1,
                "n_crit": 9,
                "meta_parameters": meta_parameters,
            }
        elif airfoil_type == "masure_regression":
            # For Masure regression, use all available parameters and reference .dat file
            info_dict = {
                "dat_file_path": f"profiles/prof_{airfoil_id}.dat",
                "t": float(
                    rib["d_tube_from_dat"]
                ),  # Use d_tube_from_dat for fitting accuracy
                "eta": float(rib["x_max_camber"]),
                "kappa": float(rib["y_max_camber"]),
                "delta": float(rib["TE_angle"]),
                "lamba": 0.0,  # Default value, not typically available in rib data
                "phi": 0.0,  # Default value, not typically available in rib data
                "chord": float(rib["chord"]),
                "meta_parameters": meta_parameters,
            }
        else:
            # Default case - include .dat file and available parameters
            info_dict = {
                "dat_file_path": f"profiles/prof_{airfoil_id}.dat",
                "t": float(
                    rib["d_tube_from_dat"]
                ),  # Use d_tube_from_dat for fitting accuracy
                "eta": float(rib["x_max_camber"]),
                "kappa": float(rib["y_max_camber"]),
                "delta": float(rib["TE_angle"]),
                "chord": float(rib["chord"]),
                "meta_parameters": meta_parameters,
            }

        wing_airfoils_data.append([airfoil_id, airfoil_type, info_dict])

    return {
        "alpha_range": [-10, 31, 0.5],  # Default range in degrees
        "reynolds": 1e6,  # Default Reynolds number
        "headers": ["airfoil_id", "type", "info_dict"],
        "data": wing_airfoils_data,
    }


def generate_bridle_nodes_data(bridle_lines):
    """
    Generate bridle nodes data for YAML output.
    If all nodes have negative y-coordinates, create symmetrical nodes with positive y.

    Parameters:
        bridle_lines: List of bridle line data from main_process_surfplan

    Returns:
        dict: Bridle nodes data formatted for YAML
    """
    bridle_nodes_data = []
    node_id = 1

    # Create unique nodes from bridle line endpoints
    unique_points = set()
    for bridle_line in bridle_lines:
        if bridle_line and len(bridle_line) >= 5:
            p1, p2 = bridle_line[0], bridle_line[1]
            unique_points.add(tuple(p1))
            unique_points.add(tuple(p2))

    # Convert to sorted list for consistent ordering
    unique_points = sorted(list(unique_points))

    # Check if all y-coordinates are negative
    all_y_negative = all(point[1] < 0 for point in unique_points)

    # Add original nodes
    for point in unique_points:
        bridle_nodes_data.append(
            [
                node_id,
                float(point[0]),
                float(point[1]),
                float(point[2]),
                "knot",  # Default to knot, could be "pulley" based on analysis
            ]
        )
        node_id += 1

    # If all y-coordinates are negative, add symmetrical nodes with positive y
    if all_y_negative:
        print(
            f"All bridle nodes have negative y-coordinates. Adding {len(unique_points)} symmetrical nodes with positive y."
        )
        for point in unique_points:
            bridle_nodes_data.append(
                [
                    node_id,
                    float(point[0]),
                    float(-point[1]),  # Mirror the y-coordinate
                    float(point[2]),
                    "knot",
                ]
            )
            node_id += 1

    return {"headers": ["id", "x", "y", "z", "type"], "data": bridle_nodes_data}


def generate_bridle_lines_data(bridle_lines):
    """
    Generate bridle lines data for YAML output.

    Parameters:
        bridle_lines: List of bridle line data from main_process_surfplan
                     Each bridle_line is [point1, point2, name, length, diameter]

    Returns:
        dict: Bridle lines data formatted for YAML
    """
    bridle_lines_data = []

    for i, bridle_line in enumerate(bridle_lines):
        if bridle_line and len(bridle_line) >= 5:
            p1, p2, name, length, diameter = (
                bridle_line[0],
                bridle_line[1],
                bridle_line[2],
                bridle_line[3],
                bridle_line[4],
            )

            # Use provided length, or calculate as distance between points if not available
            rest_length = (
                length if length > 0 else np.linalg.norm(np.array(p2) - np.array(p1))
            )

            # Use diameter if available, otherwise default to 2mm
            line_diameter = diameter if diameter > 0 else 0.002

            # Convert diameter from mm to m (SurfPlan uses mm, YAML expects m)
            line_diameter_m = line_diameter / 1000.0

            bridle_lines_data.append(
                [
                    name,  # Use actual line name
                    float(rest_length),  # rest_length
                    float(line_diameter_m),  # diameter in meters
                    "dyneema",  # material
                    970,  # density
                ]
            )

    return {
        "headers": ["name", "rest_length", "diameter", "material", "density"],
        "data": bridle_lines_data,
    }


def generate_bridle_connections_data(bridle_lines, bridle_nodes_data):
    """
    Generate bridle connections data for YAML output.
    If symmetrical nodes exist (positive y-coordinates), create connections for both sides.

    Parameters:
        bridle_lines: List of bridle line data from main_process_surfplan
                     Each bridle_line is [point1, point2, name, length, diameter]
        bridle_nodes_data: Bridle nodes data to reference node IDs

    Returns:
        dict: Bridle connections data formatted for YAML
    """
    bridle_connections_data = []

    # Create mapping from points to node IDs
    point_to_node_id = {}
    for node_data in bridle_nodes_data["data"]:
        node_id, x, y, z = node_data[0], node_data[1], node_data[2], node_data[3]
        point_to_node_id[(x, y, z)] = node_id

    # Check if we have symmetrical nodes (both negative and positive y-coordinates)
    y_coords = [node_data[2] for node_data in bridle_nodes_data["data"]]
    has_negative_y = any(y < 0 for y in y_coords)
    has_positive_y = any(y > 0 for y in y_coords)
    has_symmetrical_nodes = has_negative_y and has_positive_y

    for i, bridle_line in enumerate(bridle_lines):
        if bridle_line and len(bridle_line) >= 5:
            p1, p2, name = bridle_line[0], bridle_line[1], bridle_line[2]

            # Find corresponding node IDs for original connections
            p1_tuple = (float(p1[0]), float(p1[1]), float(p1[2]))
            p2_tuple = (float(p2[0]), float(p2[1]), float(p2[2]))

            ci = point_to_node_id.get(p1_tuple, 0)
            cj = point_to_node_id.get(p2_tuple, 0)

            if ci > 0 and cj > 0:
                bridle_connections_data.append(
                    [
                        name,  # Use actual line name
                        ci,  # ci (start node)
                        cj,  # cj (end node)
                    ]
                )

                # If we have symmetrical nodes, create mirrored connections
                if has_symmetrical_nodes:
                    # Find the mirrored points (with y-coordinate sign flipped)
                    p1_mirrored = (float(p1[0]), float(-p1[1]), float(p1[2]))
                    p2_mirrored = (float(p2[0]), float(-p2[1]), float(p2[2]))

                    ci_mirrored = point_to_node_id.get(p1_mirrored, 0)
                    cj_mirrored = point_to_node_id.get(p2_mirrored, 0)

                    if ci_mirrored > 0 and cj_mirrored > 0:
                        bridle_connections_data.append(
                            [
                                name,  # Same line name for symmetrical connection
                                ci_mirrored,  # mirrored start node
                                cj_mirrored,  # mirrored end node
                            ]
                        )

    return {"headers": ["name", "ci", "cj", "ck"], "data": bridle_connections_data}


def main(
    path_surfplan_file: Path,
    save_dir: Path,
    profile_load_dir: Path,
    profile_save_dir: Path,
    airfoil_type: str = "masure_regression",
):
    """
    Generate a YAML file with wing geometry and airfoil data using SurfplanAdapter logic.

    Parameters:
        path_surfplan_file: Path to the Surfplan .txt file
        save_dir: Directory to save the YAML file
        profile_load_dir: Directory containing profile .dat files
        profile_save_dir: Directory containing profile parameters CSV
        airfoil_type: Type of airfoil data ("polars", "breukels_regression", "neuralfoil", etc.)
    """
    # Process Surfplan data using existing logic
    ribs_data, bridle_lines = main_process_surfplan.main(
        surfplan_txt_file_path=path_surfplan_file,
        profile_load_dir=profile_load_dir,
        profile_save_dir=profile_save_dir,
    )

    # Generate YAML data structures
    wing_sections = generate_wing_sections_data(ribs_data)
    wing_airfoils = generate_wing_airfoils_data(ribs_data, airfoil_type)

    # Create the complete YAML structure
    yaml_data = {
        "wing_sections": {
            "# ---------------------------------------------------------------": None,
            "# headers:": None,
            "#   - airfoil_id: integer, unique identifier for the airfoil (matches wing_airfoils)": None,
            "#   - LE_x: x-coordinate of leading edge": None,
            "#   - LE_y: y-coordinate of leading edge": None,
            "#   - LE_z: z-coordinate of leading edge": None,
            "#   - TE_x: x-coordinate of trailing edge": None,
            "#   - TE_y: y-coordinate of trailing edge": None,
            "#   - TE_z: z-coordinate of trailing edge": None,
            "# ---------------------------------------------------------------": None,
            **wing_sections,
        },
        "": None,  # Empty line before wing_airfoils
        "wing_airfoils": {
            "# ---------------------------------------------------------------": None,
            "# headers:": None,
            "#   - airfoil_id: integer, unique identifier for the airfoil": None,
            "#   - type: one of [neuralfoil, breukels_regression, masure_regression, polars]": None,
            "#   - info_dict: dictionary with parameters depending on 'type'": None,
            "#": None,
            "# info_dict fields by type:": None,
            "#   - breukels_regression:": None,
            "#       t: Tube diameter non-dimensionalized by chord (required)": None,
            "#       kappa: Maximum camber height/magnitude, non-dimensionalized by chord (required)": None,
            "#   - neuralfoil:": None,
            "#       dat_file_path: Path to airfoil .dat file (x, y columns)": None,
            '#       model_size: NeuralFoil model size (e.g., "xxxlarge")': None,
            "#       xtr_lower: Lower transition location (0=forced, 1=free)": None,
            "#       xtr_upper: Upper transition location": None,
            "#       n_crit: Critical amplification factor (see guidelines below)": None,
            "#         n_crit guidelines:": None,
            "#           Sailplane:           12–14": None,
            "#           Motorglider:         11–13": None,
            "#           Clean wind tunnel:   10–12": None,
            '#           Average wind tunnel: 9   (standard "e^9 method")': None,
            "#           Dirty wind tunnel:   4–8": None,
            "#   - polars:": None,
            "#       csv_file_path: Path to polar CSV file (columns: alpha [rad], cl, cd, cm)": None,
            "#   - masure_regression:": None,
            "#       t, eta, kappa, delta, lamba, phi: Regression parameters": None,
            "#   - inviscid:": None,
            "#       no further data is required": None,
            "# ---------------------------------------------------------------": None,
            **wing_airfoils,
        },
    }

    # Add bridle data if available
    if len(bridle_lines) > 0:
        bridle_nodes = generate_bridle_nodes_data(bridle_lines)
        bridle_lines_yaml = generate_bridle_lines_data(bridle_lines)
        bridle_connections = generate_bridle_connections_data(
            bridle_lines, bridle_nodes
        )

        yaml_data[" "] = None  # Empty line before bridle_nodes
        yaml_data["bridle_nodes"] = {
            "# ---------------------------------------------------------------": None,
            "# headers:": None,
            "#   - id: integer, unique identifier for the node": None,
            "#   - x: x-coordinate [m]": None,
            "#   - y: y-coordinate [m]": None,
            "#   - z: z-coordinate [m]": None,
            "#   - type: node type, either 'knot' or 'pulley'": None,
            "# ---------------------------------------------------------------": None,
            **bridle_nodes,
        }

        yaml_data["  "] = None  # Empty line before bridle_lines
        yaml_data["bridle_lines"] = {
            "# ---------------------------------------------------------------": None,
            "# headers:": None,
            "#   - name: string, line name": None,
            "#   - rest_length: measured rest length [m]": None,
            "#   - diameter: line diameter [m]": None,
            "#   - material: string, material type (e.g., dyneema)": None,
            "#   - density: material density [kg/m^3]": None,
            "# ---------------------------------------------------------------": None,
            **bridle_lines_yaml,
        }

        yaml_data["   "] = None  # Empty line before bridle_connections
        yaml_data["bridle_connections"] = {
            "# ---------------------------------------------------------------": None,
            "# headers:": None,
            "#   - name: string, line name": None,
            "#   - ci: integer, node id (start)": None,
            "#   - cj: integer, node id (end)": None,
            "#   - ck: integer, third node id (only for pulleys, else omitted or 0)": None,
            "# ---------------------------------------------------------------": None,
            **bridle_connections,
        }

    # Save YAML file
    os.makedirs(save_dir, exist_ok=True)
    yaml_file_path = Path(save_dir) / "config_kite.yaml"

    # Custom YAML representers to handle comments and formatting
    def represent_none(self, data):
        return self.represent_scalar("tag:yaml.org,2002:null", "")

    def represent_list(self, data):
        # Force inline (flow) style for data arrays
        # Check if this is a data row (list with mixed types including dicts)
        if data and len(data) <= 10:  # Typical data row length
            # Special handling for rows that contain dictionaries
            if any(isinstance(item, dict) for item in data):
                # For wing_airfoils data rows with info_dict
                return self.represent_sequence(
                    "tag:yaml.org,2002:seq", data, flow_style=True
                )
            # For simple numeric/string data rows
            elif all(isinstance(item, (int, float, str)) for item in data):
                return self.represent_sequence(
                    "tag:yaml.org,2002:seq", data, flow_style=True
                )
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)

    yaml.add_representer(type(None), represent_none)
    yaml.add_representer(list, represent_list)

    with open(yaml_file_path, "w") as f:
        yaml.dump(
            yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # Post-process to clean up comments (remove quotes and None values)
    with open(yaml_file_path, "r") as f:
        content = f.read()

    # Clean up comment formatting
    content = content.replace("'#", "#")
    content = content.replace("': null", "")
    content = content.replace("': ''", "")

    # Convert empty line keys to actual blank lines - handle all patterns
    import re

    # Remove lines that are just empty keys with various patterns
    content = re.sub(r"^\? ''\n:\n", "\n", content, flags=re.MULTILINE)
    content = re.sub(r"^\? ' '\n:\n", "\n", content, flags=re.MULTILINE)
    content = re.sub(r"^\? '  '\n:\n", "\n", content, flags=re.MULTILINE)
    content = re.sub(r"^\? '   '\n:\n", "\n", content, flags=re.MULTILINE)

    # Also handle simple key patterns
    content = re.sub(r"^'': *$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^' ': *$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^'  ': *$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^'   ': *$", "", content, flags=re.MULTILINE)

    with open(yaml_file_path, "w") as f:
        f.write(content)

    print(f'Generated YAML file and saved at "{yaml_file_path}"')
    print(f'Wing sections: {len(wing_sections["data"])}')
    print(f'Wing airfoils: {len(wing_airfoils["data"])}')
    if len(bridle_lines) > 0:
        print(f"Bridle lines: {len(bridle_lines)}")

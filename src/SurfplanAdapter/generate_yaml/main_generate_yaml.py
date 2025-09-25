import os
import yaml
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
from SurfplanAdapter.generate_yaml import create_struc_geometry_yaml


def create_wing_dict(wing_sections, wing_airfoils):
    return {
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
            "#       t, eta, kappa, delta, lambda, phi: Regression parameters": None,
            "#   - inviscid:": None,
            "#       no further data is required": None,
            "# ---------------------------------------------------------------": None,
            **wing_airfoils,
        },
    }


def create_bridle_dict(yaml_data, bridle_nodes, bridle_lines_yaml, bridle_connections):
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
    return yaml_data


def main(
    ribs_data: list,
    bridle_lines: list,
    yaml_file_path: Path = None,
    airfoil_type: str = "masure_regression",
    wing_yaml="aero_geometry.yaml",
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
    # Save YAML file
    os.makedirs(yaml_file_path.parent, exist_ok=True)

    # Create wing dict
    wing_sections = generate_wing_sections_data.main(ribs_data)
    wing_airfoils = generate_wing_airfoils_data.main(ribs_data, airfoil_type)
    yaml_data = create_wing_dict(wing_sections, wing_airfoils)
    # save wing only to aero_geometrygenerate_bridle_connections_data
    utils.save_to_yaml(yaml_data, f"{yaml_file_path.parent}/{wing_yaml}")

    # Add bridle data if available
    if len(bridle_lines) > 0:
        yaml_data[" "] = None  # Empty line before bridle_nodes
        bridle_nodes = generate_bridle_nodes_data.main(bridle_lines)
        bridle_connections = generate_bridle_connections_data.main(
            bridle_lines, bridle_nodes, 0
        )
        bridle_lines_yaml = generate_bridle_lines_data.main(bridle_lines)

        yaml_data = create_bridle_dict(
            yaml_data, bridle_nodes, bridle_lines_yaml, bridle_connections
        )
    # Save to YAML
    utils.save_to_yaml(yaml_data, yaml_file_path)

    # create the struc_geometry.yaml with only struts and bridle lines
    create_struc_geometry_yaml.main(ribs_data, bridle_lines, yaml_file_path)

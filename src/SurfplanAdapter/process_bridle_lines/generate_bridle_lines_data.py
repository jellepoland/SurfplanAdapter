import numpy as np


def main(bridle_lines):
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

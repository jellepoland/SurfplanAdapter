def main(bridle_lines):
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

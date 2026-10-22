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

    # Surfplan exports one half of the bridle (y < 0). Nodes on the symmetry plane
    # (y = 0, e.g. a single-line kite's bridle apex) belong to both halves: the half
    # is mirrored as long as no node lies on the other side, and those nodes are
    # not duplicated.
    one_sided = all(point[1] <= 0 for point in unique_points)
    off_plane = [point for point in unique_points if point[1] != 0.0]

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

    # Add the symmetrical half, with positive y
    if one_sided and off_plane:
        print(
            f"Bridle exported as one half. Adding {len(off_plane)} symmetrical nodes with positive y."
        )
        for point in off_plane:
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

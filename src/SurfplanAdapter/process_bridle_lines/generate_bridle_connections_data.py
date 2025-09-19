def main(bridle_lines, bridle_nodes_data):
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

def main(processed_ribs_data):
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

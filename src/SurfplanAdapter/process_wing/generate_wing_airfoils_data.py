def main(
    ribs_data,
    airfoil_type="masure_regression",
    reynolds=1e6,
    alpha_range=[-10, 31, 0.5],
    xtr_lower=0.1,
    xtr_upper=0.1,
    n_crit=9,
):
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

        if airfoil_type == "polars":
            # For polars, reference CSV file
            info_dict = {
                "csv_file_path": f"2D_polars_CFD/{airfoil_id}.csv",
            }
        elif airfoil_type == "breukels_regression":
            # For Breukels regression, use t and kappa from rib data
            info_dict = {
                "t": float(rib["d_tube_from_dat"]),
                "kappa": float(rib["y_max_camber"]),
            }
        elif airfoil_type == "neuralfoil":
            # For NeuralFoil, reference the dat file and include all available parameters
            info_dict = {
                "dat_file_path": f"profiles/prof_{airfoil_id}.dat",
                "model_size": "xxxlarge",
                "xtr_lower": xtr_lower,
                "xtr_upper": xtr_upper,
                "n_crit": n_crit,
            }
        elif airfoil_type == "masure_regression":
            # For Masure regression, use all available parameters and reference .dat file
            info_dict = {
                "dat_file_path": f"profiles/prof_{airfoil_id}.dat",
                "t": round(float(rib["d_tube_from_dat"]), 3),
                "eta": round(float(rib["x_max_camber"]), 3),
                "kappa": round(float(rib["y_max_camber"]), 3),
                "delta": round(float(rib["TE_angle"]), 3),
                "lambda": round(float(rib["te_tension"]), 3),
                "phi": round(float(rib["le_tension"]), 3),
                "chord": round(float(rib["chord"]), 3),
                "is_strut": rib["is_strut"],
            }
        else:
            raise ValueError(f"Unsupported airfoil_type: {airfoil_type}")

        wing_airfoils_data.append([airfoil_id, airfoil_type, info_dict])

    return {
        "alpha_range": alpha_range,  # Default range in degrees
        "reynolds": reynolds,  # Default Reynolds number
        "headers": ["airfoil_id", "type", "info_dict"],
        "data": wing_airfoils_data,
    }

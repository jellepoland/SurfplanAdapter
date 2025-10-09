from pathlib import Path
from SurfplanAdapter.find_airfoil_parameters import utils_lei_parametric
import numpy as np
from scipy.interpolate import interp1d
import math


def read_dat_file_into_points(dat_file_path: str) -> tuple:
    """
    Read a .dat airfoil file and extract coordinate points.

    Parameters:
    -----------
    dat_file_path : str
        Path to the .dat file containing airfoil coordinates

    Returns:
    --------
    tuple
        (point_list, x_points, y_points) where:
        - point_list: list of [x, y] coordinate pairs
        - x_points: list of x coordinates
        - y_points: list of y coordinates
        - name: name of the airfoil from the header line
    """
    with open(dat_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Extract points from .dat file
    point_list = []
    for line in lines[1:]:  # Skip header line
        x, y = map(float, line.split())
        point_list.append([x, y])

    x_points = [point[0] for point in point_list]
    y_points = [point[1] for point in point_list]

    return point_list, x_points, y_points, str(lines[0].strip())


def fit_circle_from_le_points(x_points, y_points):
    """
    Fit a circle from LE to the lowest point between x=0 and x=0.2.
    Returns the fitted diameter with center constrained to (radius, 0).

    This function is adapted from read_profile_from_airfoil_dat_files.py
    """
    # Find the leading edge (x=0 or closest to x=0)
    x_array = np.array(x_points)
    y_array = np.array(y_points)

    # Find LE point (closest to x=0)
    le_idx = np.argmin(np.abs(x_array))
    le_x, le_y = x_array[le_idx], y_array[le_idx]

    # Find the lowest point between x=0 and x=0.2
    mask_region = (x_array >= 0) & (x_array <= 0.2)

    if np.sum(mask_region) < 2:
        return 0.0  # Not enough points in the region

    # Get indices of points in the region
    region_indices = np.where(mask_region)[0]
    region_y = y_array[mask_region]

    # Find the index of the lowest point in this region
    lowest_idx_in_region = np.argmin(region_y)
    lowest_idx = region_indices[lowest_idx_in_region]

    # Get points from LE to the lowest point
    # Determine the range of indices to use
    start_idx = min(le_idx, lowest_idx)
    end_idx = max(le_idx, lowest_idx) + 1

    x_fit = x_array[start_idx:end_idx]
    y_fit = y_array[start_idx:end_idx]

    if len(x_fit) < 3:
        return 0.0  # Not enough points for circle fitting

    # Fit circle with center constrained to (radius, 0)
    # Circle equation: (x-radius)^2 + y^2 = radius^2
    # Rearranged: x^2 - 2*radius*x + radius^2 + y^2 = radius^2
    # Simplified: x^2 + y^2 = 2*radius*x
    # So: radius = (x^2 + y^2) / (2*x)

    # Calculate radius for each point and take the mean
    radii = []
    for i in range(len(x_fit)):
        if x_fit[i] > 0:  # Avoid division by zero
            r = (x_fit[i] ** 2 + y_fit[i] ** 2) / (2 * x_fit[i])
            radii.append(r)

    if not radii:
        return 0.0

    # Use median to be robust against outliers
    radius = np.median(radii)

    # Return diameter
    return 2 * radius


def extract_delta_from_points(x_points, y_points) -> float:
    """
    Extract delta parameter (trailing edge angle) from x and y points.

    Returns:
    -------
    float
        Trailing edge angle (delta parameter in degrees)
    """

    if len(x_points) > 3:
        x1, y1 = x_points[0], y_points[0]
        x2, y2 = x_points[2], y_points[2]
        delta_x = x2 - x1
        delta_y = y2 - y1

        if delta_x == 0:
            return 0.0

        te_angle_rad = math.atan2(delta_y, delta_x)
        te_angle_deg = 180 - math.degrees(te_angle_rad)
        return te_angle_deg
    else:
        return 0.0  # Not enough points to calculate the angle


def interpolate_surface_at_x_points(surface_coords, x_points):
    """
    Interpolate surface coordinates at specified x points.

    Parameters:
    -----------
    surface_coords : numpy.ndarray
        Array of shape (n_points, 2) with columns [x, y]
    x_points : numpy.ndarray
        Array of x coordinates where to interpolate y values

    Returns:
    --------
    numpy.ndarray
        Array of interpolated y values at the specified x points
    """
    # Sort surface by x coordinate to ensure proper interpolation
    sorted_indices = np.argsort(surface_coords[:, 0])
    sorted_surface = surface_coords[sorted_indices]

    # Create interpolation function
    # Use linear interpolation, extrapolate for points outside range
    interp_func = interp1d(
        sorted_surface[:, 0],
        sorted_surface[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    return interp_func(x_points)


def calculate_surface_error_for_input_points(
    input_profile_points, generated_profile_points, lambda_value
):
    """
    Calculate the surface error between input profile points and generated profile points.

    This function works directly with point arrays rather than .dat files,
    making it more efficient when you already have the input points loaded.

    Parameters:
    -----------
    input_profile_points : numpy.ndarray
        Array of input profile points with shape (n_points, 2)
    generated_profile_points : numpy.ndarray
        Array of generated profile points with shape (n_points, 2)
    lambda_value : float
        The lambda value used to generate the profile

    Returns:
    --------
    dict
        Dictionary containing error metrics for this lambda value
    """
    # Extract top surface from input points
    # Assume input points follow standard .dat format
    input_coords = np.array(input_profile_points)
    first_point = input_coords[0]

    if abs(first_point[0] - 1.0) < 0.01:  # Starts near TE (x ≈ 1.0)
        # Input file format: TE -> top -> LE -> bottom -> TE
        le_index = np.argmin(input_coords[:, 0])  # Find LE
        input_top_surface = input_coords[: le_index + 1]
    else:  # Starts near LE (x ≈ 0.0)
        # Generated file format: LE -> top -> TE -> bottom -> LE
        te_index = np.argmax(input_coords[:, 0])  # Find TE
        input_top_surface = input_coords[: te_index + 1]

    # Extract top surface from generated points
    # Assume generated points follow LE -> top -> TE -> bottom -> LE format
    te_index = np.argmax(generated_profile_points[:, 0])  # Find TE
    generated_top_surface = generated_profile_points[: te_index + 1]

    # Define x points for comparison (equally spaced from 0 to 1)
    x_min = max(0.0, input_top_surface[:, 0].min(), generated_top_surface[:, 0].min())
    x_max = min(1.0, input_top_surface[:, 0].max(), generated_top_surface[:, 0].max())
    x_points = np.linspace(x_min, x_max, 100)

    # Interpolate both surfaces at these x points
    input_y_interp = interpolate_surface_at_x_points(input_top_surface, x_points)
    generated_y_interp = interpolate_surface_at_x_points(
        generated_top_surface, x_points
    )

    # Calculate error metrics
    absolute_distances = np.abs(input_y_interp - generated_y_interp)
    total_distance = np.sum(absolute_distances)
    mean_distance = np.mean(absolute_distances)
    max_distance = np.max(absolute_distances)

    return {
        "lambda_value": lambda_value,
        "total_distance": total_distance,
        "mean_distance": mean_distance,
        "max_distance": max_distance,
    }


def find_optimal_lambda_from_profile_points(
    input_profile_points, base_parameters, lambda_values=None
):
    """
    Find optimal lambda value by testing different lambda values with profile points.

    This function generates profiles with different lambda values and compares them
    to the input profile points to find the best match.

    Parameters:
    -----------
    input_profile_points : numpy.ndarray or list
        Array/list of input profile points with shape (n_points, 2)
    base_parameters : dict
        Dictionary containing base airfoil parameters (t_val, eta_val, kappa_val, delta_val, phi_val)
        lambda_val will be varied during optimization
    lambda_values : list of float, optional
        List of lambda values to test (default: np.arange(0.1, 0.71, 0.02))

    Returns:
    --------
    dict
        Dictionary containing the optimal lambda value and error metrics
    """
    # Default lambda values to test
    if lambda_values is None:
        lambda_values = np.arange(0.1, 0.71, 0.02)  # 0.1 to 0.7 with 0.02 spacing
        lambda_values = np.round(
            lambda_values, 2
        )  # Round to avoid floating point errors

    # print(
    #     f"Testing {len(lambda_values)} lambda values from {lambda_values[0]:.2f} to {lambda_values[-1]:.2f}"
    # )

    results = []

    for lambda_val in lambda_values:
        # Create parameters with current lambda value
        test_parameters = base_parameters.copy()
        test_parameters["lambda_val"] = lambda_val

        try:
            # Generate profile with current lambda
            generated_points, _, _ = utils_lei_parametric.generate_profile(
                **test_parameters
            )

            # Calculate error for this lambda
            error_metrics = calculate_surface_error_for_input_points(
                input_profile_points, generated_points, lambda_val
            )

            results.append(error_metrics)

            # print(
            #     f"Lambda {lambda_val:.2f}: Total error = {error_metrics['total_distance']:.6f}"
            # )

        except Exception as e:
            print(f"Error testing lambda {lambda_val:.2f}: {e}")
            continue

    if not results:
        return {"error": "No valid lambda values could be tested"}

    # Find the best match (minimum total distance)
    best_match = min(results, key=lambda x: x["total_distance"])

    # print(f"\n🎯 Optimal lambda value: {best_match['lambda_value']:.2f}")
    # print(f"Total error: {best_match['total_distance']:.6f}")
    # print(f"Mean error: {best_match['mean_distance']:.6f}")

    return {
        "optimal_lambda": best_match["lambda_value"],
        "best_match": best_match,
        "all_results": results,
        "total_tests": len(results),
    }


def get_fitted_airfoil_parameters(
    input_dat_file: Path, constant_phi=0.65, base_lambda=0.4
):

    # Read .dat file points
    point_list, x_points, y_points, profile_name = read_dat_file_into_points(
        str(input_dat_file)
    )
    # find index of max camber point
    index_max_camber = np.argmax(y_points)

    base_parameters = {
        "t_val": fit_circle_from_le_points(x_points, y_points),
        "eta_val": x_points[index_max_camber],
        "kappa_val": y_points[index_max_camber],
        "delta_val": extract_delta_from_points(x_points, y_points),
        "lambda_val": base_lambda,
        "phi_val": constant_phi,
    }

    optimization_result = find_optimal_lambda_from_profile_points(
        input_profile_points=point_list, base_parameters=base_parameters
    )

    if "error" not in optimization_result:
        optimal_lambda = optimization_result["optimal_lambda"]
        # print(f"🎯 Optimal lambda value = {optimal_lambda:.3f}")
        base_parameters["lambda_val"] = optimal_lambda
    else:
        print(f"Lambda optimization failed: {optimization_result['error']}")
        print(f"Using default lambda value, λ = {base_lambda:.3f}")

    return base_parameters, point_list, profile_name

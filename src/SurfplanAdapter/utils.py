import numpy as np


def clean_numeric_line(line, delimiter=None):
    """
    Clean a line containing numeric data by handling multiple periods, converting commas to periods,
    and automatically detecting delimiters.

    Args:
        line (str): The input line to clean
        delimiter (str, optional): The delimiter used to split the line.
                                 If None, automatically detects between ";" and ","

    Returns:
        list: List of cleaned string parts that can be converted to float
    """
    # Auto-detect delimiter if not provided
    if delimiter is None:
        # Count occurrences of potential delimiters
        semicolon_count = line.count(";")

        # If we have semicolons, this is likely TUDELFT format (semicolon delimited, comma decimals)
        if semicolon_count > 0:
            delimiter = ";"
        # Otherwise use commas as delimiter (default_kite format)
        else:
            delimiter = ","

    # Split the line by the delimiter first
    parts = line.split(delimiter)
    cleaned_parts = []

    for part in parts:
        part = part.strip()
        if part and any(char.isdigit() for char in part):
            # For semicolon-delimited format, convert commas to periods for decimal numbers
            if delimiter == ";":
                part = part.replace(",", ".")

            # Handle multiple periods in numbers (malformed floats)
            if part.count(".") > 1:
                first_period = part.find(".")
                if first_period != -1:
                    before_period = part[: first_period + 1]
                    after_period = part[first_period + 1 :].replace(".", "")
                    part = before_period + after_period
            cleaned_parts.append(part)
        elif part:  # Keep non-empty non-numeric parts as is
            cleaned_parts.append(part)

    return cleaned_parts


def line_parser(line):
    """
    Parse a line from the .txt file from Surfplan.

    Parameters:
        line (str): The line to parse.

    Returns:
        list: A list of floats containing the parsed values.
    """
    cleaned_parts = clean_numeric_line(line)
    return list(map(float, cleaned_parts))


def transform_coordinate_system_surfplan_to_VSM(coord_surfplan):
    """
    Transform coordinate from Surfplan reference frame to VSM reference frame

    Surfplan reference frame :
    # z: along the chord / parallel to flow direction
    # x: left
    # y: upwards

    VSM reference frame :
    # Body EastNorthUp (ENU) Reference Frame (aligned with Earth direction)
    # x: along the chord / parallel to flow direction
    # y: left
    # z: upwards

    Parameters:
    coord_surfplan (tuple): a tuple of three floats representing the x, y, and z coordinates of the rib endpoint in Surfplan reference frame.

    Returns:
    coord_vsm (tuple): a tuple of three floats representing the x, y, and z coordinates of the rib endpoint in VSM reference frame.
    """

    # Rotation matrix
    R = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    coord_vsm = np.dot(R, coord_surfplan)

    return coord_vsm

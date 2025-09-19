from SurfplanAdapter.utils import transform_coordinate_system_surfplan_to_VSM


def main(filepath):
    """
    Read the bridle line data from a Surfplan .txt file.

    This function locates the "3d Bridle" section in the file, skips its header,
    and then parses each subsequent line to extract:
      - point1: [TopX, Y, Z]
      - point2: [BottomX, Y, Z]
      - name: the line name from the 'Name' column
      - length: the line length from the 'Length' column
      - diameter: the value in the 'Diameter' column

    Each bridle line is stored as:
        bridle_line = [point1, point2, name, length, diameter]
    and all such lines are collected in a list which is returned.

    Parameters:
        filepath (str): Path to the Surfplan .txt file.

    Returns:
        list: A list of bridle lines, each as [point1, point2, name, length, diameter].
              If a field is empty, default values are used.
    """
    bridle_lines = []
    in_bridle_section = False
    header_skipped = False

    with open(filepath, "r") as file:
        lines = file.readlines()

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Look for the start of the bridle section.
        if "3d Bridle" in line:
            in_bridle_section = True
            header_skipped = False  # Reset header skip for the new section.
            continue

        if in_bridle_section:
            # Skip the header line (which contains column names)
            if not header_skipped:
                header_skipped = True
                continue

            # If the line does not contain a semicolon, assume the bridle section has ended.
            if ";" not in line:
                break

            # For bridle lines, we need to preserve all columns including empty ones
            # Split by semicolon and clean each part, but keep all parts
            raw_parts = line.split(";")
            cleaned_parts = []
            for part in raw_parts:
                part = part.strip()
                if part and any(char.isdigit() for char in part):
                    # Convert comma to period for decimal numbers
                    part = part.replace(",", ".")
                    # Handle multiple periods in numbers (malformed floats)
                    if part.count(".") > 1:
                        first_period = part.find(".")
                        if first_period != -1:
                            before_period = part[: first_period + 1]
                            after_period = part[first_period + 1 :].replace(".", "")
                            part = before_period + after_period
                cleaned_parts.append(part)

            # Expecting at least 10 columns based on the header:
            # TopX, Y, Z, BottomX, Y, Z, Name, Length, Material, Diameter
            if len(cleaned_parts) < 10:
                continue

            try:
                # Extract point1 from the first three columns.
                point1 = [float(cleaned_parts[i]) for i in range(3)]
                # Extract point2 from columns 4-6.
                point2 = [float(cleaned_parts[i]) for i in range(3, 6)]
            except ValueError:
                continue

            # Extract name from column 7 (index 6)
            name = (
                cleaned_parts[6].strip()
                if len(cleaned_parts) > 6
                else f"line_{len(bridle_lines)+1}"
            )

            # Extract length from column 8 (index 7)
            length_str = cleaned_parts[7] if len(cleaned_parts) > 7 else "0"
            try:
                length = float(length_str) if length_str else 0.0
            except ValueError:
                length = 0.0

            # The diameter is expected to be in the 10th column (index 9).
            diam_str = cleaned_parts[9] if len(cleaned_parts) > 9 else "0"
            try:
                diameter = (
                    float(diam_str) if diam_str else 0.002
                )  # Default 2mm diameter
            except ValueError:
                diameter = 0.002

            bridle_line = [point1, point2, name, length, diameter]
            bridle_lines.append(bridle_line)

    if len(bridle_lines) > 0:
        bridle_lines = [
            [
                transform_coordinate_system_surfplan_to_VSM(bridle_line[0]),  # point1
                transform_coordinate_system_surfplan_to_VSM(bridle_line[1]),  # point2
                bridle_line[2],  # name (string)
                bridle_line[3],  # length (float)
                bridle_line[4],  # diameter (float)
            ]
            for bridle_line in bridle_lines
        ]

    return bridle_lines

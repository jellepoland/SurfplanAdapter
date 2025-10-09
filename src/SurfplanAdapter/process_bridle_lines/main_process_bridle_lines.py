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
    # Define which column indices contain numeric data that need cleaning
    # TopX,Y,Z (0,1,2), BottomX,Y,Z (3,4,5), Length (7), Diameter (9)
    NUMERIC_IDXS = {0, 1, 2, 3, 4, 5, 7, 9}

    def _num_clean(s: str) -> str:
        """Clean numeric strings: convert comma to dot and handle malformed floats."""
        s = s.strip()
        if not s:
            return s
        # decimal comma → dot
        s = s.replace(",", ".")
        # collapse multiple dots in malformed numeric tokens (e.g., '12.3.4' → '12.34')
        if s.count(".") > 1:
            i = s.find(".")
            s = s[: i + 1] + s[i + 1 :].replace(".", "")
        return s

    def _to_float(s, default):
        """Convert string to float with explicit default handling."""
        try:
            return float(s) if s not in (None, "") else default
        except ValueError:
            return default

    bridle_lines = []
    in_bridle_section = False
    header_skipped = False

    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
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

            # If the line is too short, we assume the section has ended.
            if len(line) < 3:
                break

            # For bridle lines, we need to preserve all columns including empty ones
            # Split by semicolon or comma, then clean each part
            if ";" in line:
                raw_parts = line.split(";")
            elif "," in line:
                raw_parts = line.split(",")
            else:
                raise ValueError(
                    f"Unexpected delimiter in bridle line at line {line_num + 1}: {line}"
                )
            cleaned_parts = []
            for i, part in enumerate(raw_parts):
                part = part.strip()
                if part and any(char.isdigit() for char in part):
                    # Convert comma to period for decimal numbers
                    if "," in part:
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

            # Extract name from column 7 (index 6) - keep exactly as-is
            name = (
                cleaned_parts[6].strip()
                if len(cleaned_parts) > 6 and cleaned_parts[6].strip()
                else f"line_{len(bridle_lines)+1}"
            )

            # Extract length and diameter using consistent default handling
            length = _to_float(
                cleaned_parts[7] if len(cleaned_parts) > 7 else None, 0.0
            )
            diameter = _to_float(
                cleaned_parts[9] if len(cleaned_parts) > 9 else None, 0.002
            )

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

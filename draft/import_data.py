import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from SurfplanAdapter.utils import clean_numeric_line

# Define the path to the file
file_path = "/home/jellepoland/ownCloud/phd/code/SurfplanAdapter/data/default_kite/default_kite_3d.txt"
# file_path = "/home/jellepoland/ownCloud/phd/code/SurfplanAdapter/data/TUDELFT_V3_KITE/TUDELFT_V3_KITE.txt"


def detect_file_format(file_path):
    """Detect the delimiter and decimal separator used in the file."""
    with open(file_path, "r") as file:
        # Read a few lines to detect format
        for _ in range(20):  # Look at first 20 lines
            line = file.readline().strip()
            if (
                line
                and any(char.isdigit() for char in line)
                and (";" in line or "," in line)
            ):
                # Count separators
                semicolon_count = line.count(";")
                comma_count = line.count(",")

                if semicolon_count > 0 and semicolon_count >= comma_count:
                    # TUDELFT format: semicolon delimiter, comma decimal
                    return ";", ","
                else:
                    # Default format: comma delimiter, period decimal
                    return ",", "."

    # Default fallback
    return ",", "."


def parse_section_manually(file_path, section_start_idx, num_rows, column_names):
    """Parse a section manually using our utility function."""
    with open(file_path, "r") as file:
        lines = file.readlines()

    data_lines = []
    start_idx = section_start_idx + 3  # Skip header lines

    for i in range(start_idx, min(start_idx + num_rows, len(lines))):
        line = lines[i].strip()
        if not line or not any(char.isdigit() for char in line):
            continue

        # Use our utility function to clean the line
        cleaned_parts = clean_numeric_line(line)

        # Convert numeric parts to float, keep strings as is
        parsed_values = []
        for part in cleaned_parts:
            if part.strip():  # Skip empty parts
                try:
                    parsed_values.append(float(part))
                except ValueError:
                    # Keep as string if conversion fails
                    parsed_values.append(part.strip())

        if len(parsed_values) >= len(column_names):
            data_lines.append(parsed_values[: len(column_names)])

    if data_lines:
        return pd.DataFrame(data_lines, columns=column_names)
    else:
        return pd.DataFrame(columns=column_names)


# Load file and detect format
delimiter, decimal = detect_file_format(file_path)
print(f"Detected format: delimiter='{delimiter}', decimal='{decimal}'")

with open(file_path, "r") as file:
    content = file.read()
    content = content.strip()
lines = content.split("\n")

hn_ribs = ["LE_X", "LE_Y", "LE_Z", "TE_X", "TE_Y", "TE_Z", "VU_X", "VU_Y", "VU_Z"]
hn_struts = ["X", "Y", "Z", "D"]
hn_bridles = [
    "XT",
    "YT",
    "ZT",
    "XB",
    "YB",
    "ZB",
    "Name",
    "Length",
    "Material",
    "Diameter",
]

df = {}

for idx, line in enumerate(lines):
    if line.startswith("3d rib"):
        # Find the number of ribs
        num_ribs = None
        for i in range(idx + 1, min(idx + 5, len(lines))):
            if lines[i].strip().isdigit():
                num_ribs = int(lines[i].strip())
                break

        if num_ribs:
            df[str(line)] = parse_section_manually(file_path, idx, num_ribs, hn_ribs)
        else:
            print(f"Could not find number of ribs for section: {line}")

    elif line.startswith("Strut"):
        # Find the number of strut sections
        num_sections = None
        for i in range(idx + 1, min(idx + 5, len(lines))):
            if lines[i].strip().isdigit():
                num_sections = int(lines[i].strip())
                break

        if num_sections:
            strut_df = parse_section_manually(file_path, idx, num_sections, hn_struts)
            df[str(line)] = strut_df
        else:
            print(f"Could not find number of sections for: {line}")

    elif line.startswith("3d Bridle"):
        # For bridles, read until end of file or empty line
        bridle_df = parse_section_manually(
            file_path, idx, 1000, hn_bridles
        )  # Large number to read all
        if not bridle_df.empty:
            df[str(line)] = bridle_df

# Print available sections
print("Available sections:")
for key in df.keys():
    print(f"  {key}: {len(df[key])} rows")
    if not df[key].empty:
        print(f"    Columns: {list(df[key].columns)}")

# Check if we have the required data
rib_section = None
bridle_section = None

for key in df.keys():
    if "3d rib" in key:
        rib_section = key
    elif "3d Bridle" in key:
        bridle_section = key

if rib_section is None or df[rib_section].empty:
    print("Error: No rib data found or rib data is empty")
    exit(1)

print(f"\nUsing rib section: {rib_section}")
print(f"Rib data shape: {df[rib_section].shape}")
print("First few rows of rib data:")
print(df[rib_section].head())

# Create the plot
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")

# Plot ribs
rib_data = df[rib_section]
if not rib_data.empty:
    # Plot Leading Edge
    ax.plot3D(
        rib_data["LE_X"],
        rib_data["LE_Y"],
        rib_data["LE_Z"],
        "blue",
        label="Leading Edge",
        linewidth=2,
    )

    # Plot Trailing Edge
    ax.plot3D(
        rib_data["TE_X"],
        rib_data["TE_Y"],
        rib_data["TE_Z"],
        "red",
        label="Trailing Edge",
        linewidth=2,
    )

    # Plot ribs (connecting LE to TE)
    for i in range(len(rib_data)):
        ax.plot3D(
            [rib_data["LE_X"].iloc[i], rib_data["TE_X"].iloc[i]],
            [rib_data["LE_Y"].iloc[i], rib_data["TE_Y"].iloc[i]],
            [rib_data["LE_Z"].iloc[i], rib_data["TE_Z"].iloc[i]],
            "yellow",
            alpha=0.6,
        )

# Plot bridles if available
if bridle_section and not df[bridle_section].empty:
    print(f"\nUsing bridle section: {bridle_section}")
    bridle_data = df[bridle_section]
    print(f"Bridle data shape: {bridle_data.shape}")

    # Check if we have the required columns
    required_cols = ["XT", "YT", "ZT", "XB", "YB", "ZB"]
    available_cols = list(bridle_data.columns)

    if all(col in available_cols for col in required_cols):
        for i in range(len(bridle_data)):
            # Plot bridle line
            ax.plot3D(
                [bridle_data["XT"].iloc[i], bridle_data["XB"].iloc[i]],
                [bridle_data["YT"].iloc[i], bridle_data["YB"].iloc[i]],
                [bridle_data["ZT"].iloc[i], bridle_data["ZB"].iloc[i]],
                "gray",
                alpha=0.7,
            )
            # Plot symmetric bridle line
            ax.plot3D(
                [bridle_data["XT"].iloc[i] * (-1), bridle_data["XB"].iloc[i] * (-1)],
                [bridle_data["YT"].iloc[i], bridle_data["YB"].iloc[i]],
                [bridle_data["ZT"].iloc[i], bridle_data["ZB"].iloc[i]],
                "gray",
                alpha=0.7,
            )
    else:
        print(f"Warning: Missing required bridle columns. Available: {available_cols}")
else:
    print("No bridle data found or bridle data is empty")

# Set labels and show plot
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.set_title(f"Kite Visualization: {Path(file_path).stem}")

plt.tight_layout()
plt.show()

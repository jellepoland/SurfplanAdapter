# Process Bridle Lines Module

## Overview

The `process_bridle_lines` module parses bridle line data from Surfplan exports, handling node connections, line properties, and structural attachments with proper node ID offset management.

## Main Components

### `main_process_bridle_lines.py`

Primary entry point for processing bridle line data.

**Key Functions:**
- `main(surfplan_txt_file_path)`: Extracts and processes bridle line information
  - Parses line connections and node IDs
  - Extracts line material properties
  - Handles attachment points to wing structure
  - Manages node ID offsets for split YAML generation

**Output:**
- Returns `bridle_lines` dictionary with connection data and properties

### `generate_bridle_connections_data.py`

Generates bridle connection data with proper node ID mapping.

**Key Features:**
- Maps bridle nodes to structural particles
- Applies offset for split geometry (wing particles vs bridle particles)
- Handles mirrored connections for symmetric kites
- Validates connection integrity

## Data Structure

The module produces a `bridle_lines` dictionary:

```python
{
    'connections': [
        {
            'name': str,          # line identifier
            'ci': int,            # start node ID (with offset)
            'cj': int,            # end node ID (with offset)
        },
        ...
    ],
    'properties': {
        'line_1': {
            'diameter': float,    # mm
            'material': str,      # line material type
            'length': float,      # m
        },
        ...
    }
}
```

## Node ID Management

**Critical Feature:** The module applies proper node ID offsets when generating `struc_geometry.yaml`:

- Wing particles: IDs 1 to N (where N = number of wing sections)
- Bridle particles: IDs (N+1) to (N+M) (where M = number of bridle nodes)

This ensures bridle connections reference the correct particle IDs in the structural geometry file.

## Usage Example

```python
from pathlib import Path
from SurfplanAdapter.process_bridle_lines import main_process_bridle_lines

surfplan_file = Path("data/TUDELFT_V3_KITE/TUDELFT_V3_KITE.txt")

bridle_lines = main_process_bridle_lines.main(surfplan_file)
```

## Recent Fixes

- **Node ID offset correction**: Fixed mirrored connections to properly apply `len_wing_sections` offset
- **Header consistency**: Removed unused "ck" field from connection headers
- **Text encoding**: Robust handling of various text encodings in Surfplan files
- **Numeric field cleaning**: Selective cleaning for numeric vs text columns

## Notes

- Handles symmetric kite designs with mirrored connections
- Validates node ID consistency
- Supports various Surfplan export formats
- Critical for structural analysis integration

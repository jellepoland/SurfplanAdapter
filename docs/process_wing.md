# Process Wing Module

## Overview

The `process_wing` module extracts wing geometry data from Surfplan `.txt` export files, including rib positions, chord lengths, twist angles, dihedral angles, and airfoil profile associations.

## Main Components

### `main_process_wing.py`

Primary entry point for processing wing geometry.

**Key Functions:**
- `main(surfplan_txt_file_path, profile_load_dir, profile_save_dir)`: Main processing pipeline
  - Parses Surfplan .txt file for wing sections and ribs
  - Extracts geometric parameters (position, chord, twist, dihedral)
  - Associates airfoil profiles with each rib
  - Fits parametric LEI model to CAD-sliced profiles
  - Saves processed data and parametric profiles

**Output:**
- Returns `ribs_data` dictionary with all rib information
- Saves parametric airfoil profiles to `profile_save_dir`

## Data Structure

The module produces a `ribs_data` dictionary with the following structure:

```python
{
    'rib_1': {
        'position': [x, y, z],  # 3D coordinates
        'chord': float,          # chord length
        'twist': float,          # twist angle (degrees)
        'dihedral': float,       # dihedral angle (degrees)
        'profile_id': int,       # airfoil profile ID
        'airfoil_params': {      # LEI parametric model
            't': float,          # tube diameter
            'eta': float,        # camber position
            'kappa': float,      # camber height
            'delta': float,      # TE angle
            'lambda': float,     # TE tension
            'phi': float         # LE tension
        }
    },
    ...
}
```

## Usage Example

```python
from pathlib import Path
from SurfplanAdapter.process_wing import main_process_wing

surfplan_file = Path("data/TUDELFT_V3_KITE/TUDELFT_V3_KITE.txt")
profile_load_dir = Path("data/TUDELFT_V3_KITE/profiles")
profile_save_dir = Path("processed_data/TUDELFT_V3_KITE/profiles")

ribs_data = main_process_wing.main(
    surfplan_txt_file_path=surfplan_file,
    profile_load_dir=profile_load_dir,
    profile_save_dir=profile_save_dir
)
```

## Notes

- Reads Surfplan-specific .txt format
- Handles multiple wing sections (left/right symmetry)
- Automatically fits parametric model to CAD profiles
- Validates geometric consistency

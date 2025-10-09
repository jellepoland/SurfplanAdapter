# Plotting Module

## Overview

The `plotting` module provides visualization tools for kite geometry, including 3D airfoil sections, structural networks, and parametric vs CAD profile comparisons.

## Main Components

### `plotting.py`

Core visualization functions.

**Key Functions:**

#### `plot_airfoils_3d_from_yaml(yaml_file_path, profile_base_dir, save_path, show_plot)`
- Generates 3D visualization of all wing sections
- Plots airfoils at their spatial positions
- Shows leading and trailing edges
- Displays chord and twist distribution
- Creates interactive matplotlib 3D plot

#### `plot_struc_geometry_yaml(yaml_file_path, save_path, show_plot)`
- Visualizes structural network from `struc_geometry.yaml`
- Shows all particles (wing + bridle)
- Plots connections as lines
- Color-codes different element types
- 3D network visualization

### `find_airfoil_parameters/plot_airfoils_comparison.py`

Airfoil comparison and validation plots.

**Key Functions:**

#### `plot_all_airfoils(yaml_path, output_path, surfplan_airfoils_dir)`
- Creates multi-panel comparison plot
- Overlays parametric and CAD-sliced profiles
- One subplot per airfoil
- Smart profile mapping logic
- PDF output for documentation

**Profile Mapping Logic:**

When `N_CAD < N_YAML` (e.g., 12 CAD profiles, 18 YAML profiles):

```python
for airfoil_id in range(1, N_YAML + 1):
    if airfoil_id < N_CAD:
        # Direct comparison
        plot_yaml(airfoil_id)
        overlay_cad(airfoil_id)
    elif airfoil_id == N_YAML:
        # Tip comparison
        plot_yaml(N_YAML)
        overlay_cad(N_CAD)  # Use last CAD profile
    else:
        # Parametric only (middle profiles)
        plot_yaml(airfoil_id)
        # No overlay
```

This allows:
1. Direct validation for profiles 1 to (N_CAD - 1)
2. Parametric-only visualization for middle extrapolations
3. Tip validation comparing last YAML to last CAD

## Visualization Features

### 3D Airfoil Plot
- Shows spatial arrangement of wing sections
- Visualizes sweep, dihedral, and twist
- Interactive rotation and zoom
- Chord length scaling visible
- Leading edge markers

### Structure Network Plot
- All particles as nodes
- All connections as edges
- Color-coded by type:
  - Blue: Wing structure
  - Red: Bridle lines
  - Green: Attachment points
- 3D spatial layout

### Comparison Plot
- Black line: Parametric (generated)
- Blue line: CAD-sliced (Surfplan)
- Gray fill: Parametric profile area
- Parameter annotations on each subplot
- Equal aspect ratio for proper shape

## Usage Examples

### 3D Airfoil Visualization

```python
from SurfplanAdapter.plotting import plot_airfoils_3d_from_yaml
from pathlib import Path

plot_airfoils_3d_from_yaml(
    yaml_file_path=Path("processed_data/TUDELFT_V3_KITE/config_kite.yaml"),
    profile_base_dir=Path("processed_data/TUDELFT_V3_KITE/profiles"),
    save_path=Path("results/3d_airfoil_plot.png"),
    show_plot=True
)
```

### Airfoil Comparison

```python
from SurfplanAdapter.find_airfoil_parameters.plot_airfoils_comparison import plot_all_airfoils
from pathlib import Path

plot_all_airfoils(
    yaml_path=Path("processed_data/TUDELFT_V3_KITE/aero_geometry.yaml"),
    output_path=Path("processed_data/TUDELFT_V3_KITE/airfoils_comparison.pdf"),
    surfplan_airfoils_dir=Path("data/TUDELFT_V3_KITE/profiles")
)
```

### Structure Visualization

```python
from SurfplanAdapter.plotting import plot_struc_geometry_yaml
from pathlib import Path

plot_struc_geometry_yaml(
    yaml_file_path=Path("processed_data/TUDELFT_V3_KITE/struc_geometry.yaml"),
    save_path=Path("results/structural_network.png"),
    show_plot=False  # Save only, don't block
)
```

## Plot Styling

The module uses `VSM.plot_styling.set_plot_style()` for consistent formatting:
- LaTeX-style font rendering
- Publication-quality figures
- Consistent colors and line widths
- Grid and axis styling
- Legend formatting

## Output Formats

Supported output formats:
- **PNG**: Raster images for quick viewing
- **PDF**: Vector graphics for publications
- **SVG**: Vector graphics for editing
- **Interactive**: Matplotlib GUI for exploration

## Recent Fixes

- **NumPy data type handling**: Fixed 3D projection errors with mixed types
- **File sorting**: Numeric sort for prof_1.dat, prof_12.dat (not lexicographic)
- **Line ending normalization**: Handles CRLF and LF in .dat files
- **Bounds calculation**: Proper min/max with overlay data

## Notes

- All plots use VSM styling for consistency
- 3D plots support interactive rotation
- PDF outputs suitable for publication
- Handles missing data gracefully
- Integrated into main processing pipeline

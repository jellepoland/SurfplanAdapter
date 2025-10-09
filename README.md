# SurfplanAdapter: Kite Design Processing and Analysis Tool

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Purpose

SurfplanAdapter is a Python toolkit designed to process kite design files from [SurfPlan](http://www.surfplan.com.au/sp/) and prepare them for aerodynamic and structural analysis. The tool extracts geometric data, parametrizes airfoil profiles using a leading-edge inflatable (LEI) model, and generates configuration files compatible with the [Vortex-Step-Method (VSM)](https://github.com/awegroup/Vortex-Step-Method) for aerodynamic simulations.

Key capabilities include:
- **Surfplan file parsing**: Extract wing geometry, airfoil profiles, and bridle line data from .txt exports
- **Parametric airfoil generation**: Fit LEI airfoil parameters using the Masure regression model
- **YAML configuration generation**: Create VSM-compatible aerodynamic and structural geometry files
- **Visualization tools**: Compare parametric vs CAD-sliced airfoils and visualize 3D geometry
- **Mass properties calculation**: Compute center of gravity and inertia tensors

## Main Workflow

The typical workflow consists of:

1. **Export from SurfPlan**: Export your kite design as a .txt file along with airfoil profile .dat files

2. **Process Surfplan files**: Run `scripts/process_surfplan_files.py` to:
   - Extract wing geometry and rib data
   - Parse bridle line connections
   - Fit parametric airfoil models to CAD profiles
   - Generate `config_kite.yaml`, `aero_geometry.yaml`, and `struc_geometry.yaml`
   - Create comparison plots and 3D visualizations

3. **Calculate mass properties**: Use `scripts/calculate_cg_and_inertia.py` to:
   - Compute center of gravity location
   - Calculate inertia tensor
   - Visualize mass distribution

4. **Run aerodynamic analysis**: Use the generated YAML files with VSM for aerodynamic simulations

## Project Structure

```
SurfplanAdapter/
├── data/                          # Input Surfplan files and airfoil profiles
│   ├── TUDELFT_V3_KITE/          # Example: TU Delft V3 kite design
│   │   ├── TUDELFT_V3_KITE.txt   # Surfplan export
│   │   └── profiles/              # CAD-sliced airfoil .dat files
│   ├── default_kite/
│   └── V9_60J-Inertia/
├── processed_data/                # Output directory for processed geometry
│   └── TUDELFT_V3_KITE/
│       ├── config_kite.yaml       # Combined configuration
│       ├── aero_geometry.yaml     # Aerodynamic geometry (for VSM)
│       ├── struc_geometry.yaml    # Structural geometry (for FEM/PSS)
│       ├── profiles/              # Parametric airfoil profiles
│       └── *.pdf                  # Visualization outputs
├── results/                       # Analysis results and plots
├── scripts/                       # Main workflow scripts
│   ├── process_surfplan_files.py # Main processing pipeline
│   ├── calculate_cg_and_inertia.py
│   └── run_vsm.py
├── src/SurfplanAdapter/           # Core modules
│   ├── process_wing/              # Wing geometry extraction
│   ├── process_bridle_lines/      # Bridle line parsing
│   ├── find_airfoil_parameters/   # Parametric airfoil fitting
│   ├── generate_yaml/             # YAML file generation
│   ├── plotting.py                # Visualization utilities
│   └── calculate_cg_and_inertia.py
└── docs/                          # Documentation
```

### Module Details

- **[process_wing](docs/process_wing.md)**: Extracts rib positions, chord lengths, twist angles, and airfoil profiles from Surfplan files
- **[process_bridle_lines](docs/process_bridle_lines.md)**: Parses bridle line data with node ID mapping and connection management
- **[find_airfoil_parameters](docs/find_airfoil_parameters.md)**: Fits 6-parameter LEI model (t, η, κ, δ, λ, φ) to airfoil profiles
- **[generate_yaml](docs/generate_yaml.md)**: Creates configuration files for aerodynamic and structural solvers
- **[plotting](docs/plotting.md)**: 3D visualization and comparison plots

## Installation Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/jellepoland/SurfplanAdapter.git
    ```

2. Navigate to the repository folder:
    ```bash
    cd SurfplanAdapter
    ```
    
3. Create a virtual environment:
   
   Linux or Mac:
    ```bash
    python3 -m venv venv
    ```
    
    Windows:
    ```bash
    python -m venv venv
    ```
    
4. Activate the virtual environment:

   Linux or Mac:
    ```bash
    source venv/bin/activate
    ```

    Windows:
    ```bash
    .\venv\Scripts\activate
    ```

5. Install the required dependencies:

   For users:
    ```bash
    pip install .
    ```
        
   For developers:
    ```bash
    pip install -e .[dev]
    ```

6. To deactivate the virtual environment:
    ```bash
    deactivate
    ```

## Quick Start

### Process a Surfplan Design

```python
from pathlib import Path
from scripts.process_surfplan_files import main

# Process the TU Delft V3 kite example
main(kite_name="TUDELFT_V3_KITE", airfoil_type="masure_regression")
```

This will:
- Read `data/TUDELFT_V3_KITE/TUDELFT_V3_KITE.txt`
- Extract geometry and fit airfoil parameters
- Generate YAML files in `processed_data/TUDELFT_V3_KITE/`
- Create visualization plots

### Calculate Mass Properties

```python
from scripts.calculate_cg_and_inertia import main

main(
    kite_name="TUDELFT_V3_KITE",
    total_wing_mass=10.0,  # kg
    canopy_kg_p_sqm=0.05,  # kg/m²
    le_to_strut_mass_ratio=0.7,
    sensor_mass=0.5,  # kg
)
```

## Dependencies

Core dependencies (see [pyproject.toml](pyproject.toml) for complete list):
- **numpy**: Numerical computations and array operations
- **matplotlib**: Plotting and visualization
- **pyyaml**: YAML file handling
- **scipy**: Scientific computing utilities

Optional dependencies:
- **VSM** (Vortex-Step-Method): For aerodynamic analysis
- **pytest**: For running tests (dev)

## LEI Airfoil Parametrization

The tool uses a 6-parameter model for leading-edge inflatable (LEI) airfoils, based on the work of [K.R.G. Masure](https://resolver.tudelft.nl/uuid:865d59fc-ccff-462e-9bac-e81725f1c0c9):

- **t**: Leading edge tube diameter (normalized by chord)
- **η**: Chordwise camber position (0 to 1)
- **κ**: Maximum camber height (normalized by chord)
- **δ**: Trailing edge reflex angle (degrees)
- **λ**: Trailing edge camber tension (0 to 1)
- **φ**: Leading edge curvature tension (0 to 1)

These parameters are automatically fitted to CAD-sliced profiles from Surfplan.

## Usage Example

```bash
# 1. Export your kite design from SurfPlan
#    - Export as .txt file
#    - Export airfoil profiles as .dat files

# 2. Place files in data directory
mkdir -p data/my_kite/profiles
cp my_kite.txt data/my_kite/
cp prof_*.dat data/my_kite/profiles/

# 3. Process the design
cd scripts
python process_surfplan_files.py
# (Modify kite_name="my_kite" in the script)

# 4. Results will be in processed_data/my_kite/
```

## Visualization Outputs

The processing pipeline automatically generates:
- **3d_airfoil_plot.png**: 3D view of all wing sections
- **airfoils_in_aero_geometry.pdf**: Comparison of parametric vs CAD profiles
- **struc_geometry visualization**: Network plot of structural connections

## Contributing Guide

Please report issues and create pull requests using:
```
https://github.com/jellepoland/SurfplanAdapter
```

We welcome contributions! Here's how:

1. **Create an issue** on GitHub describing the bug/feature
2. **Create a branch** from the issue
   ```bash
   git checkout -b issue_number-description
   ```
3. **Implement your changes**
4. **Run tests** to verify nothing broke
   ```bash
   pytest
   ```
5. **Commit with descriptive message**
   ```bash
   git commit -m "#<issue_number> <description>"
   ```
6. **Push to GitHub**
   ```bash
   git push origin issue_number-description
   ```
7. **Create a pull request** targeting `main` branch
8. After merge, **close the issue**

## Citation

If you use this project in your research, please cite:
```bibtex
@software{surfplanadapter2025,
  author = {Poland, Jelle and Mooijman, Tom and Tan, Corentin},
  title = {SurfplanAdapter},
  year = {2025},
  url = {https://github.com/jellepoland/SurfplanAdapter}
}
```

Citation details can also be found in [CITATION.cff](CITATION.cff).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## WAIVER

Technische Universiteit Delft hereby disclaims all copyright interest in the package written by the Author(s).

Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering

### Copyright

Copyright (c) 2024-2025 Jelle Poland (TU Delft)

Copyright (c) 2024 Tom Mooijman (Kitepower)

Copyright (c) 2024 Corentin Tan (BeyondTheSea)

## Help and Documentation

- [AWE Group Developer Guide](https://awegroup.github.io/developer-guide/)
- [Changelog](changelog.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)

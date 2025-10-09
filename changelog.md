# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-09

### Added
- Initial release of SurfplanAdapter
- Core functionality to process Surfplan kite design files (.txt format)
- Wing geometry processing with airfoil profile extraction and parametrization
- Bridle line processing with node ID mapping and connection management
- LEI (Leading Edge Inflatable) parametric airfoil generation using Masure regression model
- Support for 6 airfoil parameters: t (tube diameter), η (camber position), κ (camber height), δ (trailing edge angle), λ (trailing edge camber tension), φ (leading edge tension)
- YAML configuration file generation for aerodynamic and structural geometry
- Automatic splitting of combined YAML into `aero_geometry.yaml` and `struc_geometry.yaml`
- 3D visualization tools for airfoils and structural geometry
- Airfoil comparison plotting between parametric and CAD-sliced profiles
- Center of gravity (CG) and inertia calculation module
- Comprehensive plotting utilities with matplotlib integration

### Features
- **Process Wing Module**: Extracts rib geometry, airfoil profiles, and wing sections from Surfplan files
- **Process Bridle Lines Module**: Parses bridle line data with robust text encoding handling and node ID offset calculations
- **Find Airfoil Parameters Module**: Fits parametric LEI airfoil model to CAD-sliced profiles
- **Generate YAML Module**: Creates VSM-compatible configuration files for aerodynamic and structural simulations
- **Plotting Module**: 
  - 3D airfoil visualization with matplotlib
  - Comparison plots overlaying parametric and CAD profiles
  - Structural geometry network visualization
  - Automatic handling of mismatched profile counts (uses last CAD profile for comparison with tip)

### Configuration
- Support for multiple kite designs (TUDELFT_V3_KITE, default_kite, V9_60J-Inertia examples included)
- Flexible profile directory structure with automatic detection
- Smart numeric sorting of profile files (handles prof_1.dat through prof_12.dat correctly)

### Documentation
- Example workflow in `scripts/process_surfplan_files.py`
- Detailed parameter finding framework documentation
- Implementation summary with architecture overview

### Technical Details
- Python 3.8+ compatibility
- NumPy-based geometric calculations with curvature validation
- Robust CSV/text file parsing with encoding detection
- Automatic line ending normalization (CRLF to LF)
- Node ID offset management for split YAML generation (wing particles vs bridle particles)

### Known Issues
- None reported for initial release

### Dependencies
- numpy: Numerical computations and array operations
- matplotlib: Plotting and visualization
- pyyaml: YAML file handling
- VSM (Vortex-Step-Method): Aerodynamic calculations (external dependency)

[0.1.0]: https://github.com/jellepoland/SurfplanAdapter/releases/tag/v0.1.0

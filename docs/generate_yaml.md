# Generate YAML Module

## Overview

The `generate_yaml` module creates configuration files in YAML format for aerodynamic (VSM) and structural (FEM/PSS) simulations. It processes wing and bridle data and generates standardized configuration files.

## Main Components

### `main_generate_yaml.py`

Primary entry point for YAML generation.

**Key Functions:**
- `main(ribs_data, bridle_lines, yaml_file_path, airfoil_type)`: Main generation pipeline
  - Combines wing and bridle data
  - Creates unified `config_kite.yaml`
  - Generates split `aero_geometry.yaml` and `struc_geometry.yaml`
  - Validates data consistency

### `create_struc_geometry_yaml.py`

Generates structural geometry file from combined configuration.

**Key Features:**
- Splits combined YAML into structural format
- Applies proper node ID offsets for bridle particles
- Preserves original node IDs with offset mapping
- Generates bridle connection references

**Critical Fix:**
```python
# Bridle particles get offset to avoid collision with wing particle IDs
bridle_particle_id = original_id + len_wing_sections
```

### `create_aero_geometry_yaml.py`

Generates aerodynamic geometry file for VSM.

**Key Features:**
- Extracts wing sections and airfoil data
- Formats for VSM input requirements
- Includes parametric airfoil parameters
- Handles chord, twist, dihedral distributions

## YAML File Structure

### `config_kite.yaml` (Combined)

Complete kite configuration:

```yaml
wing_sections:
  headers: [id, x, y, z, chord, twist, dihedral, airfoil_id]
  data:
    - [1, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 1]
    - ...

wing_airfoils:
  headers: [id, type, params]
  data:
    - [1, "masure_regression", {t: 0.077, eta: 0.175, ...}]
    - ...

bridle_particles:
  headers: [id, x, y, z]
  data:
    - [1, 0.5, 0.0, -2.0]
    - ...

bridle_connections:
  headers: [name, ci, cj]
  data:
    - ["A1_A2", 1, 2]
    - ...
```

### `aero_geometry.yaml` (VSM)

Aerodynamic-only data:
- Wing sections (position, chord, twist, dihedral)
- Airfoil definitions (parametric or profile references)
- No structural connections

### `struc_geometry.yaml` (FEM/PSS)

Structural-only data:
- All particles (wing + bridle, with proper ID offsets)
- Connection topology (springs/cables)
- Material properties
- No aerodynamic parameters

## Node ID Offset Management

**Problem:** When splitting YAML, bridle node IDs must reference particle IDs, not original connection IDs.

**Solution:**
```python
# Wing particles: IDs 1 to N
wing_particles = wing_sections  # IDs 1, 2, 3, ..., N

# Bridle particles: IDs (N+1) to (N+M)
bridle_particles = [
    {id: original_id + N, x: ..., y: ..., z: ...}
    for original_id, coords in bridle_nodes
]

# Bridle connections reference offset IDs
bridle_connections = [
    {name: ..., ci: ci_original + N, cj: cj_original + N}
    for connection in connections
]
```

## Usage Example

```python
from pathlib import Path
from SurfplanAdapter.generate_yaml import main_generate_yaml

yaml_file_path = Path("processed_data/TUDELFT_V3_KITE/config_kite.yaml")

main_generate_yaml.main(
    ribs_data=ribs_data,           # from process_wing
    bridle_lines=bridle_lines,     # from process_bridle_lines
    yaml_file_path=yaml_file_path,
    airfoil_type="masure_regression"
)
```

This generates:
- `config_kite.yaml` - Complete configuration
- `aero_geometry.yaml` - For VSM aerodynamic analysis
- `struc_geometry.yaml` - For FEM/PSS structural analysis

## Validation

The module validates:
- Node ID uniqueness
- Connection reference integrity  
- Geometric consistency
- Parameter bounds
- YAML syntax correctness

## Notes

- Supports multiple airfoil types (parametric, profile lookup)
- Handles symmetric and asymmetric kite designs
- Compatible with VSM and FEM/PSS input formats
- Critical for coupled aero-structural simulations

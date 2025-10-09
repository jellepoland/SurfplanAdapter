# Find Airfoil Parameters Module

## Overview

The `find_airfoil_parameters` module fits a 6-parameter Leading Edge Inflatable (LEI) airfoil model to CAD-sliced profiles from Surfplan. Based on the parametrization developed by [K.R.G. Masure](https://resolver.tudelft.nl/uuid:865d59fc-ccff-462e-9bac-e81725f1c0c9).

## LEI Airfoil Parametrization

### Six Parameters

1. **t** - Tube diameter (normalized by chord): 0.05 to 0.15
   - Leading edge inflatable tube size
   
2. **η (eta)** - Camber position (chordwise): 0.0 to 0.5
   - Location of maximum camber along chord
   
3. **κ (kappa)** - Camber height (normalized by chord): 0.0 to 0.3
   - Maximum camber magnitude
   
4. **δ (delta)** - Trailing edge reflex angle (degrees): -10° to +10°
   - TE upward/downward deflection
   
5. **λ (lambda)** - Trailing edge camber tension: 0.0 to 1.0
   - Controls camber line curvature near TE
   
6. **φ (phi)** - Leading edge tension: 0.0 to 1.0
   - Controls LE curvature and shape

### Geometric Construction

The airfoil is constructed using:
- **Cubic Bézier curves** for upper and lower surfaces
- **Camber line** defined by parametric spline
- **Seam allowance** at LE for inflatable tube
- **Curvature validation** to prevent straight-line segments

## Main Components

### `utils_lei_parametric.py`

Core geometric generation functions.

**Key Functions:**
- `generate_profile(t, eta, kappa, delta, lambda_val, phi)`: Main profile generator
  - Creates upper and lower surface curves
  - Applies camber and twist
  - Validates curvature constraints
  - Returns coordinate array and metadata

**Recent Enhancements:**
- Minimum curvature enforcement (>0.01 threshold)
- LE_tension limit validation
- Collinearity detection for control points
- Special case handling for extreme parameters

### `main_find_airfoil_parameters.py`

Parameter optimization and fitting.

**Key Functions:**
- `fit_airfoil_to_profile(profile_points)`: Optimization routine
  - Minimizes geometric difference to CAD profile
  - Uses scipy.optimize for parameter search
  - Validates physical constraints
  - Returns optimal parameter set

### `plot_airfoils_comparison.py`

Visualization and comparison tools.

**Key Functions:**
- `plot_all_airfoils(yaml_path, output_path, surfplan_airfoils_dir)`: Main plotting function
  - Overlays parametric and CAD profiles
  - Handles mismatched profile counts
  - Smart file mapping (last CAD profile reused for tip comparison)
  - Generates comprehensive PDF reports

## Usage Example

```python
from SurfplanAdapter.find_airfoil_parameters import utils_lei_parametric

# Generate parametric airfoil
points, profile_name, seam_a = utils_lei_parametric.generate_profile(
    t_val=0.077,      # 7.7% tube diameter
    eta_val=0.175,    # camber at 17.5% chord
    kappa_val=0.095,  # 9.5% camber height
    delta_val=7.2,    # 7.2° TE angle
    lambda_val=0.1,   # low TE tension
    phi_val=0.65      # moderate LE tension
)
```

## Comparison Plotting Logic

When Surfplan has 12 CAD profiles and YAML defines 18:

- **Profiles 1-11**: YAML N overlaid with CAD N (direct comparison)
- **Profile 12**: YAML only (no overlay - reserved for tip comparison)
- **Profiles 13-17**: YAML only (parametric extrapolation)
- **Profile 18**: YAML 18 overlaid with CAD 12 (tip vs last CAD)

This allows validation of parametric tip extrapolation against the outermost CAD slice.

## Parameter Finding Framework

See [parameter_finding_framework.md](parameter_finding_framework.md) for detailed optimization methodology, constraints, and convergence criteria.

## Notes

- Cubic Bézier implementation ensures smooth curves
- Curvature validation prevents non-physical shapes
- Optimization handles multiple local minima
- Supports LEI-specific geometric constraints

# LEI Airfoil Parameter Finding Framework

## Overview

This document summarizes the restructured LEI airfoil parameter finding framework. The code has been organized into individual modules for each of the 6 airfoil parameters, providing a systematic approach to parameter extraction and optimization.

## Package Structure

```
src/SurfplanAdapter/find_airfoil_parameters/
├── __init__.py                           # Package imports and documentation
├── utils_lei_parametric.py               # LEI airfoil generation utilities
├── finding_lambda_masure_regression.py   # Legacy lambda optimization (moved to find_camber_tension__lambda.py)
├── find_le_diameter__t.py                # Tube diameter parameter (t)
├── find_max_camber_x__eta.py             # Max camber x-position parameter (eta)
├── find_max_camber_y__kappa.py           # Max camber height parameter (kappa)
├── find_reflex_angle__delta.py           # Reflex angle parameter (delta)
├── find_camber_tension__lambda.py        # Camber tension parameter (lambda) - IMPLEMENTED
└── find_le_curvature__phi.py             # Leading edge curvature parameter (phi)
```

## Parameter Descriptions

### 1. **t** (Tube Diameter) - `find_le_diameter__t.py`
- **Physical meaning**: Leading edge inflatable tube diameter
- **Range**: Typically 0.05 to 0.25 (normalized by chord)
- **Effect**: Controls overall airfoil thickness and leading edge bluntness
- **Status**: ⚠️ Partially implemented (function stubs created)

### 2. **eta** (η, Max Camber X-Position) - `find_max_camber_x__eta.py`
- **Physical meaning**: Chordwise position of maximum camber (x/c)
- **Range**: 0.1 to 0.8 (fraction of chord)
- **Effect**: Controls pressure distribution and aerodynamic center location
- **Status**: ⚠️ Partially implemented (function stubs created)

### 3. **kappa** (κ, Max Camber Height) - `find_max_camber_y__kappa.py`
- **Physical meaning**: Height of maximum camber (y/c)
- **Range**: 0.0 to 0.15 (fraction of chord)
- **Effect**: Directly affects lift coefficient and moment characteristics
- **Status**: ⚠️ Partially implemented (function stubs created)

### 4. **delta** (δ, Reflex Angle) - `find_reflex_angle__delta.py`
- **Physical meaning**: Trailing edge reflex angle in degrees
- **Range**: -10° to +20°
- **Effect**: Controls pitching moment and longitudinal stability
- **Status**: ⚠️ Partially implemented (function stubs created)

### 5. **lambda** (λ, Camber Tension) - `find_camber_tension__lambda.py`
- **Physical meaning**: Bézier curve tension for camber line
- **Range**: 0.1 to 0.4
- **Effect**: Controls camber line shape and smoothness
- **Status**: ✅ **FULLY IMPLEMENTED** with comprehensive optimization pipeline

### 6. **phi** (φ, Leading Edge Curvature) - `find_le_curvature__phi.py`
- **Physical meaning**: Leading edge curvature parameter
- **Range**: 0° to 60°
- **Effect**: Controls leading edge radius and pressure recovery
- **Status**: ⚠️ Partially implemented (function stubs created)

## Implementation Status

### ✅ Completed Features
- **Lambda optimization**: Complete pipeline with 31 test values (0.1-0.4, spacing 0.01)
- **Surface comparison**: Top surface extraction and interpolation
- **Plotting and visualization**: Profile comparison plots
- **Data management**: Automatic saving to `processed_data` structure
- **Code organization**: Proper module structure with imports

### ⚠️ Partially Implemented
- **Parameter extraction stubs**: All 6 modules have detailed function signatures and documentation
- **Validation functions**: Range checking for all parameters
- **Optimization frameworks**: Template functions for each parameter
- **Import structure**: Package `__init__.py` with proper exports

### 🔄 Next Steps
1. **Complete t parameter implementation**: Focus on tube diameter extraction from leading edge geometry
2. **Implement eta parameter**: Camber line analysis and maximum camber location finding
3. **Implement kappa parameter**: Maximum camber height calculation
4. **Implement delta parameter**: Trailing edge angle analysis
5. **Implement phi parameter**: Leading edge curvature fitting

## Key Functions Available

### Working Functions (Lambda Module)
```python
from SurfplanAdapter.find_airfoil_parameters import find_optimal_lambda

# Find optimal lambda parameter
optimal_lambda = find_optimal_lambda(
    input_dat_file="path/to/airfoil.dat",
    output_dir="processed_data/test_case"
)
```

### Framework Functions (All Modules)
```python
from SurfplanAdapter.find_airfoil_parameters import (
    # Extraction functions
    extract_tube_diameter_from_dat_file,
    extract_eta_from_camber_line,
    extract_kappa_from_camber_analysis,
    extract_delta_from_trailing_edge,
    extract_phi_from_leading_edge,
    
    # Optimization functions
    optimize_t_parameter,
    optimize_eta_parameter,
    optimize_kappa_parameter,
    optimize_delta_parameter,
    optimize_phi_parameter,
    
    # Validation functions
    validate_t_parameter,
    validate_eta_parameter,
    validate_kappa_parameter,
    validate_delta_parameter,
    validate_phi_parameter,
)
```

## Example Usage

See `examples/parameter_extraction_example.py` for a complete demonstration of the intended interface.

## Code Quality

- **Documentation**: Comprehensive docstrings for all functions
- **Type hints**: Full type annotations using `typing` module
- **Error handling**: Proper exception handling and validation
- **Modularity**: Clean separation of concerns across modules
- **Extensibility**: Easy to add new analysis methods

## Dependencies

- **NumPy**: Numerical operations and array handling
- **SciPy**: Interpolation and optimization algorithms
- **Matplotlib**: Plotting and visualization
- **Pathlib**: Modern file path handling

## Testing

- **Lambda module**: Tested and working with real airfoil data
- **Other modules**: Function stubs ready for implementation and testing
- **Integration**: Package structure supports comprehensive testing

## Migration Notes

- **Old location**: `finding_lambda_masure_regression.py` → **New location**: `find_camber_tension__lambda.py`
- **Import updates**: All imports updated to use new module structure
- **No breaking changes**: Existing functionality preserved and enhanced

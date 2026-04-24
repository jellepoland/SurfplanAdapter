# SurfplanAdapter: Kite Design Processing and Analysis Tool

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Purpose

SurfplanAdapter converts design files from [SurfPlan](http://www.surfplan.com.au/sp/) to yaml format, designed for use in aerodynamic and structural analysis of kites. The yaml files are used in:
- [Vortex-Step-Method](https://github.com/awegroup/Vortex-Step-Method) for aerodynamic simulations.
- [kite_fem](https://github.com/awegroup/kite_fem) for structural analysis.
- [Particle_System_Simulator](https://github.com/awegroup/Particle_System_Simulator) for structural analysis.
- [ASKITE](https://github.com/awegroup/ASKITE) for coupled aero-structural analysis, integrating the toolchains mentioned above.
  
## Installation Instruction
### 1. Install by running
Linux: 
   
    ```bash
    git clone git@github.com:jellepoland/SurfplanAdapter.git && \
    cd SurfplanAdapter && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install -e .[dev]
      ```
      
Windows:
    
    ```bash
    git clone git@github.com:jellepoland/SurfplanAdapter.git; `
    cd SurfplanAdapter; `
    python -m venv venv; `
    .\venv\Scripts\Activate.ps1; `
    pip install -e .[dev]
    ```
### 2. Verify working, by running the workflow for the [TUDELFT_V3_KITE](https://awegroup.github.io/TUDELFT_V3_KITE/)
  ```bash
  python -m scripts.process_surfplan_files --kite_name=TUDELFT_V3_KITE
  ```

## Processing your own Surfplan Design

1. Export your kite design from SurfPlan:
   - Export the main design as a `.txt` file.
   - Export each airfoil profile as a `.dat` file (through the XFLR5 output).

2. Create a new kite folder in `data/` (e.g., `data/MY_NEW_KITE`) and place your exported files

3. Adjust the `config.yaml` in your kite folder to specify any custom settings, e.g. canopy density.

4. Run the processing pipeline from the repository root:

    ```bash
    python -m scripts.process_surfplan_files --kite_name=MY_NEW_KITE
    ```

## Citation

If you use this project in your research, please cite:
```bibtex
@software{surfplanadapter2026,
  author = {Poland, Jelle and Mooijman, Tom and Tan, Corentin and Romain, Lambert},
  title = {SurfplanAdapter},
  year = {2026},
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

Copyright (c) 2026 Jelle Poland (TU Delft), Romain Lambert ([beyond the sea](https://beyond-the-sea.com/en/))

Copyright (c) 2025 Jelle Poland (TU Delft)

Copyright (c) 2024 Jelle Poland (TU Delft), Corentin Tan ([beyond the sea](https://beyond-the-sea.com/en/)), Tom Mooijman ([Kitepower](https://thekitepower.com/))

# Description
Reads out [SurfPlan](http://www.surfplan.com.au/sp/) native export, transforms the data structure, and runs an aerodynamic analysis using the [Vortex-Step-Method (VSM)](https://github.com/ocayon/Vortex-Step-Method) on your kite design.

## Installation Instructions
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install

1. Clone the repository:
    ```bash
    git clone https://github.com/jellepoland/SurfplanAdapter.git
    ```

2. Navigate to the repository folder:
    ```bash
    cd SurfplanAdapter/
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

    Windows
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

## Contributing Guide
We welcome contributions to this project! Whether you're reporting a bug, suggesting a feature, or writing code, here’s how you can contribute:

1. **Create an issue** on GitHub
2. **Create a branch** from this issue
   ```bash
   git checkout -b issue_number-new-feature
   ```
3. --- Implement your new feature---
4. Verify nothing broke using **pytest**
```
  pytest
```
5. **Commit your changes** with a descriptive message
```
  git commit -m "#<number> <message>"
```
6. **Push your changes** to the github repo:
   git push origin branch-name
   
7. **Create a pull-request**, with `base:develop`, to merge this feature branch
8. Once the pull request has been accepted, **close the issue**


## :eyes: Usage
Inside the examples folder, the [main_run_VSM.py](examples/TUDELFT_V3_LEI_KITE/main_run_VSM.py) script is present that goes through the main steps for the [TUDELFT_V3_LEI_KITE](https://github.com/awegroup/TUDELFT_V3_LEI_KITE) SurfPlan files, in the following order.  
1. Loading the surfplan files from the data folder. 
2. Transforming the data into the appropriate format and store it inside the processed_data folder as .csv
3. Running an aerodynamic analysis using the [Vortex-Step-Method (VSM)](https://github.com/ocayon/Vortex-Step-Method)
4. Storing the outcome in the results folder.

To use this code on your kite designs from SurfPlan, one needs to:
1. Acquire the .txt and rib profiles in .dat format from the SurfPlan export menu
2. Install this code and insert the SurfPlan exported files into the data folder
3. Adjust the path inside the `main_run_VSM.py` code to where your files are stored and press run!

## Citation
If you use this project in your research, please consider citing it. 
Citation details can be found in the [CITATION.cff](CITATION.cff) file included in this repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Copyright
Copyright (c) 2024 Jelle Poland (TU Delft), Tom Mooijman (Kitepower), Corentin Tan (BeyondTheSea)

## :gem: Help and Documentation
[AWE Group | Developer Guide](https://awegroup.github.io/developer-guide/)



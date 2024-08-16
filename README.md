# Description
SurfplanAdapter, that calls VSM and runs aerodynamic analysis on kite designs.

## :gear: Installation

### User Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
git clone https://github.com/jellepoland/SurfplanAdapter.git
cd SurfplanAdapter/
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Dependencies

- Sphinx Dependencies, see [requirements](requirements.txt)


## :eyes: Usage

Inside the examples folder, a script is present that runs an aerodynamic analysis using the [Vortex-Step-Method](https://github.com/ocayon/Vortex-Step-Method), on your Surfplan model.

The script will (a) process the files into the desired format and store this in processed_data and (b) read out these files and run the VSM to generate plots and values and store these in the results folder.

## :wave: Contributing (optional)

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## :warning: License and Waiver

Specify the license under which your software is distributed, and include the copyright notice:

> Technische Universiteit Delft hereby disclaims all copyright interest in the program “NAME PROGRAM” (one line description of the content or function) written by the Author(s).
> 
> Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering
> 
> Copyright (c) [YEAR] [NAME SURNAME].

## :gem: Help and Documentation
[AWE Group | Developer Guide](https://awegroup.github.io/developer-guide/)



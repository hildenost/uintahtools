# uintahtools
Collection of command line scripts to automate running simulations in Uintah MPM code

Quick start:
```bash
# Generates new input files based on settings in the settings file
uintahtools generate inputfile.ups settings.yaml
# Runs all input files in given folder, simultaneously
uintahtools run folder/
```
## Features
- Easy input file modification
- Easy simulation test suite generation (no need to manually copy, create, change input files to vary one or several parameters)
- Starting multiple simulations with just one command

## Installation
Install by running
```bash
python setup.py install
```
or
```bash
pip install .
```

## Licence
This project is licenced under the MIT licence.

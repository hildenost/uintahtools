"""Simple plotting script for Plotting Udas from UPSes.

Provide the x variable and the y variable to be plotted along with the uda-folder
to make a simple 2D scatter plot with matplotlib. Output points are also stored in 
a dat-file.

"""
import os
import re
from functools import partial
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import subprocess
from pathlib import Path
from ruamel.yaml import YAML
from uintahtools import CONFIG

# Creating global variable PUDA
yaml = YAML()
load = yaml.load(Path(CONFIG))
PUDA = "/".join([os.path.dirname(load['uintahpath']), "puda"])

def header(var):
    """Creates column headers based on extracted variable."""
    FIXEDCOLS = ["time", "patch", "matl", "partId"]
    HEADERS = {
        "p.x": ["x", "y", "z"],
        "p.porepressure": ["pw"],
    }
    if var not in HEADERS:
        print("Sorry, the variable {var} is not implemented yet. No headers assigned for {var}".format(var=var))
        return FIXEDCOLS + [var]
    return FIXEDCOLS + HEADERS[var]

def normalize(var, varmax, varmin=0, flip=False):
    """Function to normalize var with regards to wrt.
    
    Normalizes to the range [0, 1] where var_min scales to 0 by default,
    and the range can be flipped. Resulting values can lie outside the range.

    """
    return (varmax - var)/(varmax-varmin) if flip else (var-varmin)/(varmax-varmin)
    
def run_puda_timesteps(uda):
    cmd = [PUDA, "-timesteps", uda]
    # Running the command, fetching the output and decode it to string
    return subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")

def parse_timesteps(output):
    """Parse timesteps, return a dict of {timestep: simtime}."""
    result = re.findall(
        "(?P<timestep>\d+): (?P<simtime>.*)",
        output,
        re.MULTILINE)
    return {int(timestep): float(simtime) for timestep, simtime in result}

def get_timestep(times, timedict):
    """For a given list of times, return the timestep number.
    
    Will return the closest timestep found in timedict.

    """
    idx = np.searchsorted(sorted(timedict.values()), times)
    return [str(i) for i in idx] if isinstance(times, list) else [str(idx)]

def construct_cmd(var, uda, timestep=None):
    """Creating the command line instruction to extract variable.
    
    If time is provided, the timestep options are included in the command. Time can
    be a list as [mintime, maxtime], and returns results inbetween as well. Time can
    be a single float value, in which case only the snapshot is returned.

    This function should be invoked in a loop if one wishes to extract a given set
    of time snapshots.
    """
    cmd = [PUDA, "-partvar", var, uda]
    if timestep:
        # Converting the time float to timestep integer
        if not isinstance(timestep, list):
            timestep = [timestep]
        cmd[-1:-1] = ["-timesteplow", str(min(timestep)), "-timestephigh", str(max(timestep))]
    return cmd

def udaplot(x, y, uda):
    """Main function.

    Steps:
        1. Extract XVAR from uda <-|
        2. Extract YVAR from uda <-|-Both should pipe the output to the read_table function
      x 3. Store XVAR and YVAR in their respective dataframes
      x 4. Set column names
      x 5. Merge dataframes
      x 6. Drop duplicates (removes the need for line select)
      x 7. Column select: time, XVAR, YVAR
      x 8. Normalize variables
    """
    print("Plotting x:", x, " vs  y:", y, " contained in ", uda)
    
    # Extracting columns
    # subprocess.call(["./uintahtools/terzaghi", x, y])
    # print("Done with bashing about")
    read_table = partial(pd.read_table, header=None,
                                        skiprows=2,
                                        # nrows=100, #Uncomment for testing purposes
                                        sep="\s+"
                                        )
    timeseries = [0.02, 0.05, 0.1, 0.2, 0.5, 1]

    output = run_puda_timesteps(uda)
    timedict = parse_timesteps(output)

    cmds = [construct_cmd("p.x", uda, timestep)
            for timestep in get_timestep(timeseries, timedict)]
    
    for cmd in cmds:
        print(cmd)
    # subprocess.call(construct_cmd("p.x", uda, time=[0.02, 1.1]))
    # df1 = read_table("ys.dat", names=header(y))
    # df2 = read_table("xs.dat", names=header(x))
    
    # selected = ['time', 'y', 'pw']
    
    # df = pd.merge(df1, df2).filter(selected).drop_duplicates(selected)
    
    # pwnorm = partial(normalize, varmax=-10000)
    # ynorm = partial(normalize, varmax=1, flip=True)
    
    # df['pw'] = df['pw'].map(lambda x: pwnorm(x))
    # df['y'] = df['y'].map(lambda x: ynorm(x))
    
    # print(df)
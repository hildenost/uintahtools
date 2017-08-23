"""Simple plotting script for Plotting outputs from UPSes.

Provide the x variable and the y variable to be plotted along with the uda-folder
to make a simple 2D scatter plot with matplotlib. Output points are also stored in 
a dat-file.

"""
import os
import re
import subprocess
from io import StringIO
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from ruamel.yaml import YAML

from uintahtools import CONFIG

# Creating global variable PUDA
yaml = YAML()
load = yaml.load(Path(CONFIG))
PUDA = "/".join([os.path.dirname(load['uintahpath']), "puda"])

def header(var):
    """Create column headers based on extracted variable."""
    FIXEDCOLS = ["time", "patch", "matl", "partId"]
    HEADERS = {
        "p.x": ["x", "y", "z"],
        "p.porepressure": ["pw"],
    }
    if var not in HEADERS:
        print("Sorry, the variable {var} is not implemented yet. No headers assigned for {var}".format(var=var))
        return FIXEDCOLS + [var]
    return FIXEDCOLS + HEADERS[var]

def normalize(var, varmax=1, varmin=0, flip=False):
    """Normalize var to the range [0, 1].
    
    The range can be flipped. Resulting values can lie outside the range.

    """
    return (varmax - var)/(varmax-varmin) if flip else (var-varmin)/(varmax-varmin)
    
def run_puda(uda):
    cmd = [PUDA, "-timesteps", uda]
    return run_cmd(cmd)

def construct_cmd(var, uda, timestep=None):
    """Create the command line instruction to extract variable.
    
    If timestep is provided, the timestep options are included in the command. Time can
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

def run_cmd(cmd):
    """Shortcut for the long and winding subprocess call output decoder."""
    return subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")

def timesteps_parse(output):
    """Parse timesteps, return a dict of {timestep: simtime}."""
    result = re.findall(
        "(?P<timestep>\d+): (?P<simtime>.*)",
        output,
        re.MULTILINE)
    return {int(timestep): float(simtime) for timestep, simtime in result}

def timesteps_get(times, timedict):
    """For a given list of times, return the index of the best-fitting timestep."""
    idx = np.searchsorted(sorted(timedict.values()), times)
    return [str(i) for i in idx] if isinstance(times, list) else [str(idx)]

def udaplot(x, y, uda):
    """Main function.

    Steps:
      x 1. Extract XVAR from uda <-|
      x 2. Extract YVAR from uda <-|-Both should pipe the output to the read_table function
      x 3. Store XVAR and YVAR in their respective dataframes
      x 4. Set column names
      x 5. Merge dataframes
      x 6. Drop duplicates (removes the need for line select)
      x 7. Column select: time, XVAR, YVAR
      x 8. Normalize variables
        9. REFACTOR AND TEST
    """
    print("Plotting x:", x, " vs  y:", y, " contained in ", uda)
    
    read_table = partial(pd.read_table, header=None,
                                        skiprows=2,
                                        # nrows=100, #Uncomment for testing purposes
                                        sep="\s+"
                                        )

    timeseries = [0.02, 0.05, 0.1, 0.2, 0.5, 1]

    timesteps = timesteps_get(timeseries, timesteps_parse(run_puda(uda)))

    dfs = []
    for testvar in (x, y):
        cmds = [construct_cmd(testvar, uda, timestep)
                for timestep in timesteps]
        
        lots_of_xes = [StringIO(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")) for cmd in cmds]

        df = DataFrame()
        for x in lots_of_xes:
            dftemp = read_table(x, names=header(testvar))
            df = dftemp if df.empty else df.append(dftemp)

        dfs.append(df)

    selected = ['time', 'y', 'pw']
    df = pd.merge(*dfs).filter(selected).drop_duplicates(selected)
    
    df['pw'] = df['pw'].map(lambda x: normalize(x, varmax=-1e4))
    df['y'] = df['y'].map(lambda x: normalize(x, flip=True))
    
    print(df)
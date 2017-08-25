"""Simple plotting script for Plotting outputs from UPSes.

Provide the x variable and the y variable to be plotted along with the uda-folder
to make a simple 2D scatter plot with matplotlib. Output points are also stored in 
a dat-file.

"""
import os
import re
import subprocess
from functools import partial
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ruamel.yaml import YAML

from uintahtools import CONFIG
from uintahtools.terzaghi.terzaghi import terzaghi

# Creating global variable PUDA
yaml = YAML()
load = yaml.load(Path(CONFIG))
PUDA = "/".join([os.path.dirname(load['uintahpath']), "puda"])

def header(var):
    """Create column headers based on extracted variable."""
    fixedcols = ["time", "patch", "matl", "partId"]
    headers = {
        "p.x": ["x", "y", "z"],
        "p.porepressure": ["p.porepressure"],
    }
    if var not in headers:
        print("Sorry, the variable {var} is not implemented yet. No headers assigned for {var}".format(var=var))
        return fixedcols + [var]
    return fixedcols + headers[var]

def normalize(var, varmax=1, varmin=0, flip=False):
    """Normalize var to the range [0, 1].
    
    The range can be flipped. Resulting values can lie outside the range.

    """
    return (varmax - var)/(varmax-varmin) if flip else (var-varmin)/(varmax-varmin)

def cmd_make(var, uda, timestep=None):
    """Assemble the command line instruction to extract variable.
    
    If timestep is provided, the timestep options are included in the command. Time can
    be a list as [mintime, maxtime], and returns results inbetween as well. Time can
    be a single float value, in which case only the snapshot is returned.

    This function should be invoked in a loop if one wishes to extract a given set
    of time snapshots.
    """
    cmdargs = ["-partvar", var]
    if timestep:
        if not isinstance(timestep, list):
            timestep = [timestep]
        cmdargs.extend(["-timesteplow", str(min(timestep)), "-timestephigh", str(max(timestep))])
    return [PUDA, *cmdargs, uda]

def cmd_run(cmd):
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

def extracted(variable, uda, timestep):
    """Extract the variable and wrap it in a stream."""
    return StringIO(cmd_run(cmd_make(variable, uda, timestep)))

def dataframe_assemble(variable, timesteps, uda):
    """Create and return dataframe from extracting the variable at given timesteps from the UDA folder."""
    def table_read(variable, uda, timestep):
        """Wrapping pd.read_table for readability."""
        return pd.read_table(extracted(variable, uda, timestep),
                        header=None,
                        names=header(variable),
                        skiprows=2,
                        sep="\s+")
    dfs = (table_read(variable, uda, timestep) for timestep in timesteps)
    return pd.concat(dfs)

def dataframe_create(x, y, uda, timesteps):
    """Create the final dataframe.

    Extracting the variables from the given timesteps, concatenating and merging dataframes
    to the final shape of selected_cols.

    """
    settings = {
        "y": {'flip': False},
        x: {'varmax': -1e4},
    }

    dfs = [dataframe_assemble(var, timesteps, uda) for var in (x,y)]
    df = pd.merge(*dfs).filter([x, "y", "time"]).drop_duplicates([x, "y", "time"])
    for col in (x, "y"):
        df[col] = df[col].map(lambda t: normalize(t, **settings[col]))
    return df

def plot_analytical(func, ax, timeseries, samples=40, maxj=15):
    """Compute and plot analytical solution."""
    add_to_plot = partial(ax.plot, color="red", linestyle="solid", linewidth=0.2)
    for timefactor in timeseries:
        zs = [z/samples for z in range(samples+1)]
        pores = [func(timefactor, z, maxj) for z in zs]
        add_to_plot(pores, zs)

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

    timeseries = [0.02, 0.05, 0.1, 0.2, 0.5, 1]

    timesteps = timesteps_get(
        times=timeseries,
        timedict=timesteps_parse(cmd_run([PUDA, "-timesteps", uda]))
        )
    
    df = dataframe_create(x, y, uda, timesteps)

    fig, ax = plt.subplots()
    plot_analytical(terzaghi, ax, timeseries)

    # Plotting the dataframe
    norm = colors.BoundaryNorm(boundaries=timeseries, ncolors=len(timeseries))
    df.plot.scatter(x=x, y="y", ax=ax, c="time", norm=norm, cmap="Set3", alpha=0.8)

    plt.show()
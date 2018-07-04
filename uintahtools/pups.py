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
import seaborn as sns
sns.set(color_codes=True)
from ruamel.yaml import YAML

from uintahtools import CONFIG
from uintahtools.udaframe import UdaFrame
from uintahtools.udaplot import UdaPlot
from uintahtools.settings import Settings
from uintahtools.uda import Uda

# Creating global variable PUDA
yaml = YAML()
load = yaml.load(Path(CONFIG))
PUDA = "/".join([os.path.dirname(load['uintahpath']), "puda"])
PARTEXTRACT = "/".join([os.path.dirname(load['uintahpath']), "partextract"])
LINEEXTRACT = "/".join([os.path.dirname(load['uintahpath']), "lineextract"])

FIGSIZE = (5, 3.8)


def header(var):
    """Create column headers based on extracted variable."""
    fixedcols = ["time", "patch", "matl", "partId"]
    headers = {
        "p.x": ["x", "y", "z"],
        "p.porepressure": ["p.porepressure"],
        "p.stress": ["sigma11", "sigma12", "sigma13",
                     "sigma21", "sigma22", "sigma23",
                     "sigma31", "sigma32", "sigma33"]
    }
    if var not in headers:
        print(
            "Sorry, the variable {var} is not implemented yet. No headers assigned for {var}".
            format(var=var))
        return fixedcols + [var]
    return fixedcols + headers[var]


def normalize(var, varmax=1, varmin=0, flip=False, offset=0):
    """Normalize var to the range [0, 1].

    The range can be flipped. Resulting values can lie outside the range.

    """
    return (varmax - var) / (varmax - varmin) if flip else (var - varmin) / (varmax - varmin) + offset


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
        cmdargs.extend([
            "-timesteplow",
            str(min(timestep)), "-timestephigh",
            str(max(timestep))
        ])
    return [PUDA, *cmdargs, uda]


def cmd_run(cmd):
    """Shortcut for the long and winding subprocess call output decoder."""
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True).stdout.decode("utf-8")


def extracted(variable, uda, timestep):
    """Extract the variable and wrap it in a stream."""
    try:
        process = cmd_run(cmd_make(variable, uda, timestep))
    except subprocess.CalledProcessError:
        print("The timestep {timestep} was not found in {uda}"
              " and was therefore ignored.".format(timestep=timestep, uda=uda))
        return None
    return StringIO(process)


def get_particle_outputs(uda, var, partid):
    """Create dataframe of time vs porepressure at given depth."""
    cmd = [PARTEXTRACT, "-partvar", var, "-partid", partid, uda]
    output = cmd_run(cmd)
    result = pd.read_table(
        StringIO(output),
        header=None,
        names=header(var),
        index_col="time",
        skiprows=2,
        sep="\s+",
        usecols=["time", "partId", var])
    return result


def get_particleposes(uda):
    """Return a dataframe of particleID y where duplicates of y is dropped."""
    cmd = [PARTEXTRACT, "-partvar", "p.x", "-timestep", "0", uda]
    output = cmd_run(cmd)
    result = pd.read_table(
        StringIO(output),
        header=None,
        names=header("p.x"),
        index_col="partId",
        skiprows=2,
        sep="\s+").drop_duplicates("y").filter(["y"])
    return result


def get_particleIDs(uda):
    """Return a sample of 3 evenly spread particle IDs.

    In future, one migth add the option to provide a point
    in Cartesian coordinates and return the closest particle ID. And also
    provide any number of samples.

    """
    pdf = get_particleposes(uda)
    n = len(pdf)
    sampleidxs = [n // 2, n - 1]
    ndecimals = 6
    return {str(pid): round(y, ndecimals)
            for i, (pid, y) in enumerate(pdf.itertuples())
            if i in sampleidxs}


def lineextract(uda):
    x = "0.05"
    z = "0"
    y = ["0", "1"]
    cmd = [LINEEXTRACT, "-v", "p.porepressure", "-startPt",
           x, y[0], z, "-endPt", x, y[1], z, "-uda", uda]
    output = cmd_run(cmd)
    result = pd.read_table(
        StringIO(output),
        header=None,
        names=header("p.porepressure"),
        skiprows=4,
        sep="\s+")
    print(result)


def dataframe_assemble(variable, timesteps, uda):
    """Create and return dataframe from extracting the variable at given timesteps from the UDA folder."""

    def table_read(variable, uda, timestep):
        """Wrapping pd.read_table for readability."""
        result = extracted(variable, uda, timestep)
        return pd.read_table(
            result, header=None, names=header(variable), skiprows=2,
            sep="\s+") if result is not None else pd.DataFrame(
                columns=header(variable))

    dfs = (table_read(variable, uda, timestep) for timestep in timesteps)
    return pd.concat(dfs)


def dataframe_create(x, y, uda, timesteps):
    """Create the final dataframe.

    Extracting the variables from the given timesteps, concatenating and merging dataframes
    to the final shape of selected_cols.

    """
    settings = {
        "y": {
            'flip': False
        },
        x: {
            'varmax': -1e4
        },
    }

    dfs = [dataframe_assemble(var, timesteps, uda) for var in (x, y)]
    df = pd.merge(*dfs).filter([x, "y",
                                "time"]).drop_duplicates([x, "y", "time"])
    for col in (x, "y"):
        df[col] = df[col].map(lambda t: normalize(t, **settings[col]))
    return df


def variablelist(uda):
    """Return a dict of tracked variables and their types."""
    result = re.findall(
        "(?P<varname>[pg]\..+): (?:Particle|NC)Variable<(?P<vartype>.*)>",
        cmd_run([PUDA, "-listvariables", uda]), re.MULTILINE)
    return dict(result)


def udaplot(x, y, udapath, output=None, compare=False):
    """Module pups main plotting function.

    From a given set of timepoints, the provided variables are extracted
    from the provided uda folder and plotted along with the reference
    results.

    Not implemented yet:
        [ ] Reference plot could be from external file
        [ ] Reference plot could be extracted from another uda file
        [ ] Listing the variables this uda has tracked
        [ ] If no variables provided, interactively choose
            [ ] Tab completion
        [ ] Make p-q-p_w-plot
            [ ] For given particles
            [ ] For all particles

    """
    print("Plotting x:", x, " vs  y:", y, " contained in ", udapath)

    # if compare and len(uda) > 1:
    #     print("Comparing ", len(uda))
    #     uda2 = uda[1]
    # else:
    #     compare = False
    udapath = udapath[0]
    # print("Plotting x:", x, " vs  y:", y, " contained in ", udapath)

    if (x, y) == ("p.porepressure", "p.x"):
        key = "terzaghi"
    elif (x, y) == ("p.porepressure", "time"):
        key = "terzaghi_time"
    elif (x, y) == ("p.porepressure", "momentum"):
        key = "porepressure_momentum"
    else:
        print("Plot type not recognized.")
        exit()

    settings = Settings()
    settings.configure(key, override=False)

    uda = Uda(udapath, key, settings[key])

    print("Dataframe creating...")
    df2 = UdaFrame(uda)
    print("Dataframe instance: ", df2)
    df = dataframe_create(x, y, uda.uda, uda.timesteps)
    print("Dataframe created")

    udaplot = UdaPlot.create(key, df, uda)
    udaplot.plot()
    udaplot.display_plot(output)

    # New dataframe for selected depths.
    # Collects depth, porepressure and time.
    # Time on x-axis, porepressure on y.
    # ys = timesteps_get(ysamples, df["y"])
    # dfs = [dataframe_assemble(var, ys, uda) for var in (x, y)]
    # print(dfs)
    # df = pd.merge(*dfs).filter(selected+["time"]).drop_duplicates(selected+["time"])

    # PARTEXTRACT -partvar p.porepressure -partid PARTID uda
    # partids = get_particleIDs(uda)
    # print(partids)

    # dfs = [get_particle_outputs(uda, x, partid) for partid in partids]

    # fig, ax = plt.subplots()
    # plot_analytical(terzaghi, ax, zs=partids.values(), time=True)
    # for df in dfs:
    #     df[x] = df[x].map(lambda t: normalize(t, varmax=1e4))
    #     df.plot(use_index=True, y=x,
    #             ax=ax,
    #             linewidth=0.4,
    #             grid=True,
    #             # c="gray",
    #             # alpha=0.5,
    #             # logx=True,
    #             ylim=(0, 1))
    # plt.show()
    # Fixed for now
    # Hent alltid ut min og max
    # That's how it should be done.
    # So need a function to retrieve the partid of particles at specified
    # depth. I cannot make sense of lineextract
    # Success!!
    # LINEEXTRACT -v p.porepressure -istart 3 0 0 -iend 3 40 0 -uda uda
    # lineextract(uda)
    # print(df)

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
PARTEXTRACT = "/".join([os.path.dirname(load['uintahpath']), "partextract"])
LINEEXTRACT = "/".join([os.path.dirname(load['uintahpath']), "lineextract"])


def header(var):
    """Create column headers based on extracted variable."""
    fixedcols = ["time", "patch", "matl", "partId"]
    headers = {
        "p.x": ["x", "y", "z"],
        "p.porepressure": ["p.porepressure"],
    }
    if var not in headers:
        print("Sorry, the variable {var} is not implemented yet. No headers assigned for {var}".format(
            var=var))
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
        cmdargs.extend(["-timesteplow", str(min(timestep)),
                        "-timestephigh", str(max(timestep))])
    return [PUDA, *cmdargs, uda]


def cmd_run(cmd):
    """Shortcut for the long and winding subprocess call output decoder."""
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.decode("utf-8")


def timesteps_parse(output):
    """Parse timesteps, return a dict of {timestep: simtime}."""
    result = re.findall(
        "(?P<timestep>\d+): (?P<simtime>.*)",
        output,
        re.MULTILINE)
    return {int(timestep): float(simtime) for timestep, simtime in result}


def timesteps_get(timedict, times=None, frequency=None):
    """For a given list of times, return the index of the best-fitting timestep."""
    idx = np.searchsorted(timedict, times)
    return [str(i) for i in idx] if isinstance(times, list) else [str(idx)]


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
    result = pd.read_table(StringIO(output),
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
    result = pd.read_table(StringIO(output),
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
    result = pd.read_table(StringIO(output),
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
        return pd.read_table(result,
                             header=None,
                             names=header(variable),
                             skiprows=2,
                             sep="\s+") if result is not None else pd.DataFrame(columns=header(variable))
    dfs = (table_read(variable, uda, timestep) for timestep in timesteps)
    return pd.concat(dfs)


def dataframe_create(x, y, uda, timesteps):
    """Create the final dataframe.

    Extracting the variables from the given timesteps, concatenating and merging dataframes
    to the final shape of selected_cols.

    """
    settings = {
        "y": {'flip': False, 'offset': 0},
        x: {'varmax': -1e4},
    }

    dfs = [dataframe_assemble(var, timesteps, uda) for var in (x, y)]
    df = pd.merge(*dfs).filter([x, "y", "time"]
                               ).drop_duplicates([x, "y", "time"])
    for col in (x, "y"):
        df[col] = df[col].map(lambda t: normalize(t, **settings[col]))
    return df


def pqplot(uda):
    """Create a stress p-q-plot.

    (If material model is Cam clay, the yield surface start and stop is also drawn.)

    If pore pressures present, total p also plotted.
    """

    # Step 1: Get stresses from uda-file
    stresses = extracted("p.stress", uda, None)

    print("Did command run?")
    print(stresses)

    print(timesteps_parse(cmd_run([PUDA, "-timesteps", uda])))


def plot_analytical(func, ax, timeseries=[], zs=[], samples=40, maxj=50, time=False):
    """Compute and plot analytical solution.

    Two options:
        1.  porepressure vs depth (z)
        2.  porepressure vs time (t)

    """
    add_to_plot = partial(ax.plot,  color="gray", alpha=0.8,
                          linestyle="solid", linewidth=2, zorder=1)
    if not zs:
        zs = [z / samples for z in range(samples + 1)]
    if not timeseries:
        timeseries = np.logspace(-5, 1, num=samples)
    if time:
        for z in zs:
            pores = [func(t, z, maxj) for t in timeseries]
            add_to_plot(timeseries, pores)
    else:
        legend_entry = False
        for timefactor in timeseries:
            pores = [func(timefactor, z, maxj) for z in zs]

            if legend_entry:
                add_to_plot(pores, zs)
            else:
                add_to_plot(pores, zs, label="analytical")
                legend_entry = True


def variablelist(uda):
    """Return a dict of tracked variables and their types."""
    result = re.findall(
        "(?P<varname>[pg]\..+): (?:Particle|NC)Variable<(?P<vartype>.*)>",
        cmd_run([PUDA, "-listvariables", uda]),
        re.MULTILINE
    )
    return dict(result)


def annotate(plt, timeseries, df):
    """Annotate the isochrones."""
    # Creating labels
    # pos = [(0.22, 0.15),
    #        (0.27, 0.25),
    #        (0.51, 0.33),
    #        (0.655, 0.34),
    #        (0.87, 0.35),
    #        (0.87, 0.5),
    #        (0.87, 0.6),
    #        (0.87, 0.7),
    #        (0.8, 0.85)
    #        ]
    pos = [(0.2, 0.15),
           (0.23, 0.25),
           (0.37, 0.33),
           (0.655, 0.34),
           (0.87, 0.35),
           (0.87, 0.5),
           (0.87, 0.6),
           (0.87, 0.7),
           (0.8, 0.85)
           ]
    for i, time in enumerate(reversed(timeseries)):
        label = "$T = " + str(time) + "$"
        plt.figtext(*pos[i], label, horizontalalignment="left")

    return


def swap_uda_extension(uda, ext):
    """Uda results are in a folder, so in-built functions are of no use."""
    return ".".join((uda.split(".")[0], ext))


def udaplot(x, y, uda, output=None):
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
        [ ] Plot labels
            [ ] Labels on the time series
        [ ] Make p-q-p_w-plot
            [ ] For given particles
            [ ] For all particles

    """
    print("Plotting x:", x, " vs  y:", y, " contained in ", uda)

    if (x, y) == ("p", "q"):
        print("Creating a pqplot")

        pqplot(uda)

        exit()

    timeseries = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

    timesteps = timesteps_get(
        times=timeseries,
        timedict=sorted(timesteps_parse(
            cmd_run([PUDA, "-timesteps", uda])).values())
    )

    df = dataframe_create(x, y, uda, timesteps)

    fig = plt.figure(figsize=(7.5, 3.8))
    ax = fig.add_subplot(111)

    # Plotting the reference solution
    plot_analytical(terzaghi, ax, timeseries)

    time_set = set(df["time"])

    for time in time_set:
        print(time)
        if time < 0.2:
            subset = df[df.time == time]
            print("SMALL!!")
            print(subset)
            subset.plot.scatter(x=x, y="y", ax=ax, color="red",
                                edgecolor="black", zorder=2, legend=False)

    # Plotting the dataframe
    df.plot.scatter(x=x, y="y", ax=ax, color="none",
                    edgecolor="black", zorder=2, label="MPM-FVM")

    # Removing plot frame
    for side in {'right', 'top'}:
        ax.spines[side].set_visible(False)

    ax.set_xbound(lower=0, upper=1.5)
    ax.set_ybound(lower=0, upper=1)

    # Adding labels
    xlabel = "Normalized pore pressure $p/p_0$"
    ylabel = "Normalized depth $z/H$"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adding annotations
    annotate(plt, timeseries, df)

    # Adding legend
    plt.legend(bbox_to_anchor=(0.7, 0), loc=4)

    # df.to_clipboard(excel=True)

    if (output):
        if (len(output) == 1):
            outfile = swap_uda_extension(uda, "pdf")
        else:
            outfile = output[1]
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()

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

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
<<<<<<< HEAD
        print(
            "Sorry, the variable {var} is not implemented yet. No headers assigned for {var}".
            format(var=var))
||||||| merged common ancestors
        print("Sorry, the variable {var} is not implemented yet. No headers assigned for {var}".format(var=var))
=======
        print("Sorry, the variable {var} is not implemented yet. No headers assigned for {var}".format(
            var=var))
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
        return fixedcols + [var]
    return fixedcols + headers[var]

<<<<<<< HEAD

def normalize(var, varmax=1, varmin=0, flip=False):
||||||| merged common ancestors
def normalize(var, varmax=1, varmin=0, flip=False):
=======

def normalize(var, varmax=1, varmin=0, flip=False, offset=0):
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    """Normalize var to the range [0, 1].

    The range can be flipped. Resulting values can lie outside the range.

    """
<<<<<<< HEAD
    return (varmax - var) / (varmax - varmin) if flip else (var - varmin) / (
        varmax - varmin)

||||||| merged common ancestors
    return (varmax - var)/(varmax-varmin) if flip else (var-varmin)/(varmax-varmin)
=======
    return (varmax - var) / (varmax - varmin) if flip else (var - varmin) / (varmax - varmin) + offset

>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9

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
<<<<<<< HEAD
        cmdargs.extend([
            "-timesteplow",
            str(min(timestep)), "-timestephigh",
            str(max(timestep))
        ])
||||||| merged common ancestors
        cmdargs.extend(["-timesteplow", str(min(timestep)), "-timestephigh", str(max(timestep))])
=======
        cmdargs.extend(["-timesteplow", str(min(timestep)),
                        "-timestephigh", str(max(timestep))])
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    return [PUDA, *cmdargs, uda]


def cmd_run(cmd):
    """Shortcut for the long and winding subprocess call output decoder."""
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True).stdout.decode("utf-8")



def timesteps_parse(output):
    """Parse timesteps, return a dict of {timestep: simtime}."""
    result = re.findall("(?P<timestep>\d+): (?P<simtime>.*)", output,
                        re.MULTILINE)
    return {int(timestep): float(simtime) for timestep, simtime in result}

<<<<<<< HEAD

def timesteps_get(timedict, times=None, every=None):
||||||| merged common ancestors
def timesteps_get(times, timedict):
=======

def timesteps_get(times, timedict):
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
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
<<<<<<< HEAD
    result = pd.read_table(
        StringIO(output),
        header=None,
        names=header(var),
        index_col="time",
        skiprows=2,
        sep="\s+",
        usecols=["time", "partId", var])
||||||| merged common ancestors
    result = pd.read_table(StringIO(output),
        header=None,
        names=header(var),
        index_col="time",
        skiprows=2,
        sep="\s+",
        usecols=["time", "partId", var])
=======
    result = pd.read_table(StringIO(output),
                           header=None,
                           names=header(var),
                           index_col="time",
                           skiprows=2,
                           sep="\s+",
                           usecols=["time", "partId", var])
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    return result


def get_particleposes(uda):
    """Return a dataframe of particleID y where duplicates of y is dropped."""
    cmd = [PARTEXTRACT, "-partvar", "p.x", "-timestep", "0", uda]
    output = cmd_run(cmd)
<<<<<<< HEAD
    result = pd.read_table(
        StringIO(output),
        header=None,
        names=header("p.x"),
        index_col="partId",
        skiprows=2,
        sep="\s+").drop_duplicates("y").filter(["y"])
||||||| merged common ancestors
    result = pd.read_table(StringIO(output),
        header=None,
        names=header("p.x"),
        index_col="partId",
        skiprows=2,
        sep="\s+").drop_duplicates("y").filter(["y"])
=======
    result = pd.read_table(StringIO(output),
                           header=None,
                           names=header("p.x"),
                           index_col="partId",
                           skiprows=2,
                           sep="\s+").drop_duplicates("y").filter(["y"])
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    return result


def get_particleIDs(uda):
    """Return a sample of 3 evenly spread particle IDs.

    In future, one migth add the option to provide a point
    in Cartesian coordinates and return the closest particle ID. And also
    provide any number of samples.

    """
    pdf = get_particleposes(uda)
    n = len(pdf)
<<<<<<< HEAD
    sampleidxs = [0, n // 2, n - 1]
||||||| merged common ancestors
    sampleidxs = [0, n//2, n-1]
=======
    sampleidxs = [n // 2, n - 1]
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    ndecimals = 6
<<<<<<< HEAD
    return {
        str(pid): round(y, ndecimals)
        for i, (pid, y) in enumerate(pdf.itertuples()) if i in sampleidxs
    }

||||||| merged common ancestors
    return {str(pid): round(y, ndecimals)
                for i, (pid, y) in enumerate(pdf.itertuples())
                if i in sampleidxs}
=======
    return {str(pid): round(y, ndecimals)
            for i, (pid, y) in enumerate(pdf.itertuples())
            if i in sampleidxs}

>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9

def lineextract(uda):
    x = "0.05"
    z = "0"
    y = ["0", "1"]
<<<<<<< HEAD
    cmd = [
        LINEEXTRACT, "-v", "p.porepressure", "-startPt", x, y[0], z, "-endPt",
        x, y[1], z, "-uda", uda
    ]
||||||| merged common ancestors
    cmd = [LINEEXTRACT, "-v", "p.porepressure", "-startPt",
            x, y[0], z, "-endPt", x, y[1], z, "-uda", uda]
=======
    cmd = [LINEEXTRACT, "-v", "p.porepressure", "-startPt",
           x, y[0], z, "-endPt", x, y[1], z, "-uda", uda]
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    output = cmd_run(cmd)
<<<<<<< HEAD
    result = pd.read_table(
        StringIO(output),
        header=None,
        names=header("p.porepressure"),
        skiprows=4,
        sep="\s+")
||||||| merged common ancestors
    result = pd.read_table(StringIO(output),
        header=None,
        names=header("p.porepressure"),
        skiprows=4,
        sep="\s+")
=======
    result = pd.read_table(StringIO(output),
                           header=None,
                           names=header("p.porepressure"),
                           skiprows=4,
                           sep="\s+")
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    print(result)


def dataframe_assemble(variable, timesteps, uda):
    """Create and return dataframe from extracting the variable at given timesteps from the UDA folder."""

    def table_read(variable, uda, timestep):
        """Wrapping pd.read_table for readability."""
        result = extracted(variable, uda, timestep)
<<<<<<< HEAD
        return pd.read_table(
            result, header=None, names=header(variable), skiprows=2,
            sep="\s+") if result is not None else pd.DataFrame(
                columns=header(variable))

||||||| merged common ancestors
        return pd.read_table(result,
                        header=None,
                        names=header(variable),
                        skiprows=2,
                        sep="\s+") if result is not None else pd.DataFrame(columns=header(variable))
=======
        return pd.read_table(result,
                             header=None,
                             names=header(variable),
                             skiprows=2,
                             sep="\s+") if result is not None else pd.DataFrame(columns=header(variable))
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    dfs = (table_read(variable, uda, timestep) for timestep in timesteps)
    return pd.concat(dfs)


def dataframe_create(x, y, uda, timesteps):
    """Create the final dataframe.

    Extracting the variables from the given timesteps, concatenating and merging dataframes
    to the final shape of selected_cols.

    """
    settings = {
<<<<<<< HEAD
        "y": {
            'flip': False
        },
        x: {
            'varmax': -1e4
        },
||||||| merged common ancestors
        "y": {'flip': False},
        x: {'varmax': -1e4},
=======
        "y": {'flip': False, 'offset': 1},
        x: {'varmax': 1e4},
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    }

<<<<<<< HEAD
    dfs = [dataframe_assemble(var, timesteps, uda) for var in (x, y)]
    df = pd.merge(*dfs).filter([x, "y",
                                "time"]).drop_duplicates([x, "y", "time"])
||||||| merged common ancestors
    dfs = [dataframe_assemble(var, timesteps, uda) for var in (x,y)]
    df = pd.merge(*dfs).filter([x, "y", "time"]).drop_duplicates([x, "y", "time"])
=======
    dfs = [dataframe_assemble(var, timesteps, uda) for var in (x, y)]
    df = pd.merge(*dfs).filter([x, "y", "time"]
                               ).drop_duplicates([x, "y", "time"])
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    for col in (x, "y"):
        df[col] = df[col].map(lambda t: normalize(t, **settings[col]))
    return df

<<<<<<< HEAD

def andregradsligning(a, b, c):
    x = (-b + sqrt(b**2 - 4 * a * c)) / (2 * a)
    return x


def plot_analytical(func,
                    ax,
                    timeseries=[],
                    zs=[],
                    samples=40,
                    maxj=15,
                    time=False):
||||||| merged common ancestors
def plot_analytical(func, ax, timeseries=[], zs=[], samples=40, maxj=15, time=False):
=======

def plot_analytical(func, ax, timeseries=[], zs=[], samples=40, maxj=50, time=False):
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    """Compute and plot analytical solution.

    Two options:
        1.  porepressure vs depth (z)
        2.  porepressure vs time (t)

    """
<<<<<<< HEAD
    add_to_plot = partial(
        ax.plot, color="red", linestyle="solid", linewidth=0.4)
||||||| merged common ancestors
    add_to_plot = partial(ax.plot, color="red", linestyle="solid", linewidth=0.4)
=======
    add_to_plot = partial(ax.plot,  color="gray", alpha=0.8,
                          linestyle="solid", linewidth=2, zorder=1)
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
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
        cmd_run([PUDA, "-listvariables", uda]), re.MULTILINE)
    return dict(result)

<<<<<<< HEAD

||||||| merged common ancestors
=======

def annotate(plt, timeseries, df):
    """Annotate the isochrones."""
    # Creating labels
    pos = [(0.22, 0.15),
           (0.27, 0.25),
           (0.51, 0.33),
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


>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
def udaplot(x, y, uda):
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

    """
    print("Plotting x:", x, " vs  y:", y, " contained in ", uda)

    timeseries = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

    timesteps = timesteps_get(
<<<<<<< HEAD
        timedict=sorted(
            timesteps_parse(cmd_run([PUDA, "-timesteps", uda])).values()),
        times=timeseries)

||||||| merged common ancestors
        times=timeseries,
        timedict=sorted(timesteps_parse(
            cmd_run([PUDA, "-timesteps", uda])).values())
        )
    
=======
        times=timeseries,
        timedict=sorted(timesteps_parse(
            cmd_run([PUDA, "-timesteps", uda])).values())
    )

>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    df = dataframe_create(x, y, uda, timesteps)

    fig = plt.figure(figsize=(5, 3.8))
    ax = fig.add_subplot(111)

    # Plotting the reference solution
    plot_analytical(terzaghi, ax, timeseries)

    # Plotting the dataframe
<<<<<<< HEAD
    norm = colors.BoundaryNorm(boundaries=timeseries, ncolors=len(timeseries))
    df.plot.scatter(
        x=x,
        y="y",
        ax=ax,
        c="time",
        norm=norm,
        cmap="Set3",
        alpha=0.8,
        xlim=(0, 1.2))

    # df.to_clipboard(excel=True)
||||||| merged common ancestors
    norm = colors.BoundaryNorm(boundaries=timeseries, ncolors=len(timeseries))
    df.plot.scatter(x=x, y="y", ax=ax, c="time", norm=norm, cmap="Set3", alpha=0.8, xlim=(0,1.2))
    
    # df.to_clipboard(excel=True)
=======
    df.plot.scatter(x=x, y="y", ax=ax, color="none",
                    edgecolor="black", zorder=2, label="MPM-FVM")

    # Removing plot frame
    for side in {'right', 'top'}:
        ax.spines[side].set_visible(False)

    ax.set_xbound(lower=0)
    ax.set_ybound(lower=0, upper=1)

    # Adding labels
    xlabel = "Normalized pore pressure $p/p_0$"
    ylabel = "Normalized depth $z/H$"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adding annotations
    annotate(plt, timeseries, df)
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9

    # Adding legend
    plt.legend(bbox_to_anchor=(0.7, 0), loc=4)

    # df.to_clipboard(excel=True)
    plt.show()

    # plt.savefig("terzaghicoarse.pdf", dpi=300)

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
<<<<<<< HEAD
    dfs[1][x] = dfs[1][x].map(lambda t: normalize(t, varmax=-1e4))
    dfs[1].plot(
        use_index=True,
        y=x,
        ax=ax,
        linewidth=0.2,
        grid=True,
        c="gray",
        alpha=0.5,
        # logx=True,
        ylim=(0, 5))
    plt.show()
||||||| merged common ancestors
    dfs[1][x] = dfs[1][x].map(lambda t: normalize(t, varmax=-1e4))
    dfs[1].plot(use_index=True, y=x,
            ax=ax,
            linewidth=0.2,
            grid=True,
            c="gray",
            alpha=0.5,
            # logx=True,
            ylim=(0, 5))
    plt.show()
=======
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
>>>>>>> 9bcbde12372d0721c506ec1a6093d18d17e557f9
    # Fixed for now
    # Hent alltid ut min og max
    # That's how it should be done.
    # So need a function to retrieve the partid of particles at specified
    # depth. I cannot make sense of lineextract
    # Success!!
    # LINEEXTRACT -v p.porepressure -istart 3 0 0 -iend 3 40 0 -uda uda
    # lineextract(uda)
    # print(df)

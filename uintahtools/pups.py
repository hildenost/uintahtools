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
from uintahtools.terzaghi.terzaghi import terzaghi

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


def timesteps_parse(output):
    """Parse timesteps, return a dict of {timestep: simtime}."""
    result = re.findall("(?P<timestep>\d+): (?P<simtime>.*)", output,
                        re.MULTILINE)
    return {int(timestep): float(simtime) for timestep, simtime in result}


def timesteps_get(timedict, times=None, every=None, samples=None):
    """Generate a list of timestep indices.

    If a list of timesteps is provided, return the index of the best-fitting timestep.
    If instead the caller wants every nth timestep, the list of timesteps is
    generated from the frequency.
    If the caller wants a total of N samples, the list of timesteps returned contains
    N + 1 timesteps, to include the very first timestep.

    """

    if samples is not None:
        stride = len(timedict) // samples
        return [str(i) for i in range(0, len(timedict), stride)]
    elif every is not None:
        max_time_index = len(timedict) - 1
        indices = [str(i) for i in range(0, max_time_index, every)]

        # Inclusive interval: including the maximum time step index,
        # even when it does not fit with the stride
        if not str(max_time_index) in indices:
            indices.append(str(max_time_index))

        return indices
    elif times is not None:
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
    """Compute and plot analytical solution.

    Two options:
        1.  porepressure vs depth (z)
        2.  porepressure vs time (t)

    """
    add_to_plot = partial(
        ax.plot, color="red", linestyle="solid", linewidth=0.4)
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


def volumetric_stress():
    """Create DataFrame.eval() expression for volumetric stress."""
    return "p = (sigma11+sigma22+sigma33)/3.0"


def equivalent_stress(s11, s22, s33, s12, s13, s23):
    """Return compute equivalent stress for symmetric stress tensor."""
    stress = np.array([[s11, s12, s13],
                       [s12, s22, s23],
                       [s13, s23, s33]])
    stress_dev = stress - 1.0 / 3.0 * np.trace(stress) * np.eye(3)
    J2 = 0.5 * np.trace(np.dot(stress_dev, stress_dev))
    return np.sqrt(3.0 * J2)


def plotpq(uda):
    """Plot of p-q-stress paths."""

    print("Creating a p-q-stresspath-plot")

    # Step 1: Generate list of timesteps
    timesteps = timesteps_get(
        timedict=sorted(timesteps_parse(
            cmd_run([PUDA, "-timesteps", uda])).values()),
        every=200
    )

    # Step 2: Create dataframe of p.stress, particleId and time
    dfs = [dataframe_assemble(var, timesteps, uda)
           for var in ("p.stress", "p.x")]
    headers = ["time", "y", "partId", "sigma11", "sigma12", "sigma13",
               "sigma21", "sigma22", "sigma23",
               "sigma31", "sigma32", "sigma33"]
    df = pd.merge(*dfs).filter(headers).drop_duplicates(["time", "y"])

    # Step 3: Compute volumetric stress p and equivalent stress q
    df.eval(volumetric_stress(), inplace=True)
    df['q'] = np.vectorize(equivalent_stress)(df['sigma11'], df['sigma22'], df['sigma33'],
                                              df['sigma12'], df['sigma13'], df['sigma23'])

    # Step 4: Plot all p-q-paths in dataframe by particleId
    # Tidying up dataframe
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)

    # Plotting the dataframe
    # norm = colors.BoundaryNorm(
    # boundaries=timeseries, ncolors=len(timeseries))
    df.plot.scatter(
        x="p",
        y="q",
        ax=ax,
        c="y",
        # norm=norm,
        cmap="Set3",
        alpha=0.8)

    # # Adding legend
    # plt.legend(bbox_to_anchor=(0.7, 0), loc=4)

    # # df.to_clipboard(excel=True)
    plt.show()


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

    if (x, y) == ("p", "q"):
        plotpq(uda)
    else:

        timeseries = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

        timesteps = timesteps_get(
            timedict=sorted(
                timesteps_parse(cmd_run([PUDA, "-timesteps", uda])).values()),
            times=timeseries)

        df = dataframe_create(x, y, uda, timesteps)

        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)

        # Plotting the reference solution
        plot_analytical(terzaghi, ax, timeseries)

        # Plotting the dataframe
        norm = colors.BoundaryNorm(
            boundaries=timeseries, ncolors=len(timeseries))
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

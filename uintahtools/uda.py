import os
import re
import subprocess

from collections import namedtuple
from io import StringIO

import numpy as np
from ruamel.yaml import YAML
from pathlib import Path


from uintahtools import CONFIG


yaml = YAML()
load = yaml.load(Path(CONFIG))
PUDA = "/".join([os.path.dirname(load['uintahpath']), "puda"])


class Uda:
    """Class for manipulation of and data extraction from the Uintah result folder .uda/."""

    def __init__(self, uda, key, timesteps=None, init=None, every=None, samples=None):
        self.uda = uda
        self.timesteps = self.get_timesteps(timesteps, every, samples)
        self.timeseries = timesteps
        self.key = key
        self.vars = Variable(key)

        self.udainit = Uda(init, key, timesteps=[0]) if init else None

    def __str__(self):
        return self.uda

    def extracted(self, variable, timestep):
        """Extract the variable and wrap it in a stream."""
        try:
            process = cmd_run(cmd_make(variable, self.uda, timestep))
        except subprocess.CalledProcessError:
            print("The timestep {timestep} was not found in {uda}"
                  " and was therefore ignored.".format(timestep=timestep, uda=self.uda))
            return None
        return StringIO(process)

    def swap_extension(self, extension, number=False):
        return ".".join((self.uda.split(".")[0], extension))

    def generate_timedict(self):
        return sorted(self.timesteps_parse(cmd_run([PUDA, "-timesteps", self.uda])).values())

    def timesteps_parse(self, output):
        """Parse timesteps, return a dict of {timestep: simtime}."""
        result = re.findall("(?P<timestep>\d+): (?P<simtime>.*)", output,
                            re.MULTILINE)
        return {int(timestep): float(simtime) for timestep, simtime in result}

    def get_timesteps(self, timesteps=None, every=None, samples=None):
        """Generate a list of timestep indices.

        If a list of timesteps is provided, return the index of the best-fitting timestep.
        If instead the caller wants every nth timestep, the list of timesteps is
        generated from the frequency.
        If the caller wants a total of N samples, the list of timesteps returned contains
        N + 1 timesteps, to include the very first timestep.

        """
        timedict = self.generate_timedict()

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
        elif timesteps is not None:
            idx = np.searchsorted(timedict, timesteps)
            return [str(i) for i in idx] if isinstance(timesteps, list) else [str(idx)]


class Variable():
    fixedcols = ["time", "patch", "matl", "partId"]
    headers = {
        "p.x": ["x", "y", "z"],
        "p.porepressure": ["p.porepressure"],
        "p.stress": ["sigma11", "sigma12", "sigma13",
                     "sigma21", "sigma22", "sigma23",
                     "sigma31", "sigma32", "sigma33"]
    }
    Var = namedtuple("Var", ["udavar", "header", "settings"])
    Vars = namedtuple("Vars", ["x", "y"])

    def __init__(self, plottype):
        if (plottype == "terzaghi"):
            self.vars = self.TerzaghiVariables()
        elif (plottype == "porepressure_momentum"):
            self.vars = self.MomentumVariables()
        elif (plottype == "beam.deflection"):
            self.vars = self.DeflectionVariables()

    def get_headers(self):
        return [var.header for var in self.vars]

    @staticmethod
    def get_uda_headers(var):
        if (isinstance(var, str)):
            return Variable.fixedcols + Variable.headers[var]
        return Variable.fixedcols + Variable.headers[var.udavar]

    def __repr__(self):
        return self.vars.__repr__()

    def __iter__(self):
        return self.vars.__iter__()

    @staticmethod
    def DeflectionVariables():
        xx = Variable.Var(udavar="p.x",
                          header="x", settings={})
        yy = Variable.Var(udavar="p.x", header="y", settings={})
        return Variable.Vars(xx, yy)

    @staticmethod
    def TerzaghiVariables():
        xx = Variable.Var(udavar="p.porepressure",
                          header="p.porepressure", settings={"varmax": 1e4})
        yy = Variable.Var(udavar="p.x", header="y", settings={
                          "flip": False, "offset": 1})
        return Variable.Vars(xx, yy)

    @staticmethod
    def MomentumVariables():
        xx = Variable.Var(udavar="p.x",
                          header="x", settings={})
        yy = Variable.Var(udavar="p.porepressure",
                          header="p.porepressure", settings={})
        return Variable.Vars(xx, yy)


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

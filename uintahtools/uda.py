PUDA = "/".join([os.path.dirname(load['uintahpath']), "puda"])


class Uda:
    """Class for manipulation of and data extraction from the Uintah result folder .uda/."""

    def __init__(self, uda):
        self.uda = uda

    def __str__(self):
        return self.uda

    def generate_timedict(self):
        return self.timesteps_parse(cmd_run([PUDA, "-timesteps", self.uda])).values())

    def timesteps_parse(self, output):
        """Parse timesteps, return a dict of {timestep: simtime}."""
        result=re.findall("(?P<timestep>\d+): (?P<simtime>.*)", output,
                            re.MULTILINE)
        return sorted({int(timestep): float(simtime) for timestep, simtime in result})


def cmd_make(var, uda, timestep = None):
    """Assemble the command line instruction to extract variable.

    If timestep is provided, the timestep options are included in the command. Time can
    be a list as [mintime, maxtime], and returns results inbetween as well. Time can
    be a single float value, in which case only the snapshot is returned.

    This function should be invoked in a loop if one wishes to extract a given set
    of time snapshots.
    """
    cmdargs=["-partvar", var]
    if timestep:
        if not isinstance(timestep, list):
            timestep=[timestep]
        cmdargs.extend([
            "-timesteplow",
            str(min(timestep)), "-timestephigh",
            str(max(timestep))
        ])
    return [PUDA, *cmdargs, uda]


def cmd_run(cmd):
    """Shortcut for the long and winding subprocess call output decoder."""
    return subprocess.run(
        cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
        check = True).stdout.decode("utf-8")

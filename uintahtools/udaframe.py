"""Module of all things dataframes."""
from collections import namedtuple

from cycler import cycler
import pandas as pd


class UdaFrame(pd.DataFrame):

    def __init__(self, uda):
        super().__init__()
        self.vars = Variable(uda.key)
        df = self.dataframe_create(uda)
        super(UdaFrame, self).__init__(df)

    def plot_df(self, ax=None):
        pass

    def table_read(self, variable, uda, timestep):
        """Wrapping pd.read_table for readability."""
        result = uda.extracted(variable.udavar, timestep)
        return pd.read_table(
            result, header=None, names=self.vars.get_uda_headers(variable), skiprows=2,
            sep="\s+") if result is not None else pd.DataFrame(
                columns=self.vars.get_uda_headers(variable))

    def dataframe_assemble(self, variable, uda):
        """Create and return dataframe from extracting the variable at given timesteps from the UDA folder."""
        dfs = (self.table_read(variable, uda, timestep)
               for timestep in uda.timesteps)
        return pd.concat(dfs)

    @staticmethod
    def normalize(var, varmax=1, varmin=0, flip=False, offset=0):
        """Normalize var to the range [0, 1].

        The range can be flipped. Resulting values can lie outside the range.

        """
        return (varmax - var) / (varmax - varmin) if flip else (var - varmin) / (varmax - varmin) + offset

    def dataframe_create(self, uda):
        """Create the final dataframe.

        Extracting the variables from the given timesteps, concatenating and merging dataframes
        to the final shape of selected_cols.

        """
        dfs = [self.dataframe_assemble(var, uda)
               for var in self.vars]

        return pd.merge(*dfs)


class TerzaghiFrame(UdaFrame):
    def dataframe_create(self, uda):
        df = super().dataframe_create(uda)

        df = df.filter(self.vars.get_headers() +
                       ["time"]).drop_duplicates(self.vars.get_headers() + ["time"])
        for var in self.vars:
            df[var.header] = df[var.header].map(
                lambda t: self.normalize(t, **var.settings))
        return df

    def plot_df(self, ax):
        self.plot.scatter(x=self.vars.vars.x.header, y=self.vars.vars.y.header,
                          ax=ax, color="none", edgecolor="black", zorder=2, label="MPM-FVM")


class PorePressureMomentumFrame(UdaFrame):
    class Beam:
        def __init__(self, b, l, h, E):
            self.b = b
            self.l = l
            self.h = h
            self.E = E
            self.I = self.second_moment_of_area(b, h)

        def second_moment_of_area(self, b, h):
            return b * h * h * h / 12.0

    def plot_df(self, ax=None):
        print(self.columns)
        print(self.index)
        mp_cycler = cycler("linestyle", ['-', '--', ':', '-.'])
        ax.set_prop_cycle(mp_cycler)
        self.plot.line(ax=ax, color="black",
                       zorder=2, label="MPM-FVM")
        for column in reversed(self.columns):
            ax.fill_between(
                x=self.index, y1=self[column], alpha=0.5, color="gray")

    def dataframe_create(self, uda):
        df = super().dataframe_create(uda)
        self.df = df.filter(["time", "partId", "p.porepressure"])

        sections = self.initialize_group_sections(uda)
        momentum = self.compute_pore_pressure_momentum(sections)
        return pd.DataFrame(data=momentum)

    @staticmethod
    def initialize_group_sections(uda):
        beam = PorePressureMomentumFrame.Beam(b=0.1, l=1.0, h=0.3, E=10e6)
        result = uda.extracted("p.x", uda.timesteps[0])

        names = Variable.get_uda_headers("p.x")

        df = pd.read_table(
            result, header=None, names=names, skiprows=2,
            sep="\s+") if result is not None else pd.DataFrame(
                columns=names)

        y_mean = -beam.h / 2

        def demean(y): return y - y_mean
        df["y"] = df["y"].apply(demean)
        return df.groupby("x")

    def compute_pore_pressure_momentum(self, grouped):
        timegroup = self.df.groupby("time")
        pressure_momentum = {}
        for time, data in timegroup:
            pressure_momentum[time] = {}
            data.set_index(["partId"], inplace=True)
            for x, group in grouped:
                group.set_index(["partId"], inplace=True)
                pressure_momentum[time][x] = 0.0
                for pId, row in group.iterrows():
                    porepressure = data.get_value(pId, "p.porepressure")
                    y = row["y"]
                    pressure_momentum[time][x] -= porepressure * y
        return pressure_momentum


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
    def TerzaghiVariables():
        xx = Variable.Var(udavar="p.porepressure",
                          header="p.porepressure", settings={"varmax": -1e4})
        yy = Variable.Var(udavar="p.x", header="y", settings={"flip": False})
        return Variable.Vars(xx, yy)

    @staticmethod
    def MomentumVariables():
        xx = Variable.Var(udavar="p.x",
                          header="x", settings={})
        yy = Variable.Var(udavar="p.porepressure",
                          header="p.porepressure", settings={})
        return Variable.Vars(xx, yy)

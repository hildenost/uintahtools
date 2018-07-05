"""Module of all things dataframes."""
from cycler import cycler
import pandas as pd
import matplotlib.pyplot as plt


class UdaFrame(pd.DataFrame):

    def __init__(self, uda):
        super().__init__()
        df = self.dataframe_create(uda)
        super(UdaFrame, self).__init__(df)

    def plot_df(self, ax=None):
        pass

    def table_read(self, variable, uda, timestep):
        """Wrapping pd.read_table for readability."""
        result = uda.extracted(variable.udavar, timestep)
        return pd.read_table(
            result, header=None, names=uda.vars.get_uda_headers(variable), skiprows=2,
            sep="\s+") if result is not None else pd.DataFrame(
                columns=uda.vars.get_uda_headers(variable))

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
               for var in uda.vars]
        return pd.merge(*dfs)


class TerzaghiFrame(UdaFrame):
    def dataframe_create(self, uda):
        df = super().dataframe_create(uda)

        df = df.filter(uda.vars.get_headers() +
                       ["time"]).drop_duplicates(uda.vars.get_headers() + ["time"])
        for var in uda.vars:
            df[var.header] = df[var.header].map(
                lambda t: self.normalize(t, **var.settings))

        print(df.head())
        return df

    def plot_df(self, ax):
        self.plot.scatter(x=self.uda.vars.vars.x.header, y=self.uda.vars.vars.y.header,
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
        mp_cycler = cycler("linestyle", ['-', '--', ':', '-.'])
        ax.set_prop_cycle(mp_cycler)

        self.groupby("time").plot(x="x", y="momentum", ax=ax)
        # , ax=ax, color="black",
        #  zorder=2, label="MPM-FVM")
        # ax.fill_between(
        # x="x", y1="momentum", alpha=0.2, color="gray")

    def dataframe_create(self, uda):
        df = super().dataframe_create(uda)
        df = df.filter(["time", "partId", "p.porepressure"])

        sections = self.initialize_group_sections(uda)
        momentum = self.compute_pore_pressure_momentum(sections, df)
        return momentum

    @staticmethod
    def initialize_group_sections(uda):
        beam = PorePressureMomentumFrame.Beam(b=0.1, l=1.0, h=0.3, E=10e6)
        result = uda.extracted("p.x", uda.timesteps[0])

        print(uda.timesteps)

        names = uda.vars.get_uda_headers("p.x")
        df = pd.read_table(
            result, header=None, names=names, skiprows=2,
            sep="\s+") if result is not None else pd.DataFrame(
                columns=names)

        y_mean = -beam.h / 2

        def demean(y): return y - y_mean
        df["y"] = df["y"].apply(demean)

        print(df)
        return df.groupby("x")

    @staticmethod
    def compute_pore_pressure_momentum(grouped, df):
        timegroup = df.groupby("time")
        momentum = {"time": [], "x": [], "momentum": []}
        for time, data in timegroup:
            data.set_index(["partId"], inplace=True)
            for x, group in grouped:
                group.set_index(["partId"], inplace=True)
                mom = 0.0
                for pId, row in group.iterrows():
                    porepressure = data.at[pId, "p.porepressure"]
                    y = row["y"]
                    mom -= porepressure * y
                momentum["time"].append(time)
                momentum["x"].append(x)
                momentum["momentum"].append(mom)
        return pd.DataFrame(data=momentum)

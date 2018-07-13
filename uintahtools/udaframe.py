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

        return df

    def plot_df(self, ax):
        self.plot.scatter(x=self.uda.vars.vars.x.header, y=self.uda.vars.vars.y.header,
                          ax=ax, color="none", edgecolor="black", zorder=2, label="MPM-FVM")


class BeamDeflectionFrame(UdaFrame):
    def dataframe_create(self, uda):
        df = super().dataframe_create(uda)

        beam_middle = self.get_beam_middle_partIds(uda, 0.3)

        deflection = self.compute_deflection(beam_middle, df)

        return deflection

    def plot_df(self, ax):

        mp_cycler = cycler("linestyle", ['-', '--', ':', '-.'])
        ax.set_prop_cycle(mp_cycler)

        grouped = self.groupby("time")

        for label, df in grouped:
            df.plot(x="x", y="deflection", ax=ax,
                    color="black",
                    zorder=2, label="$T = " + str(round(label - 0.03, 3)) + "$")
        return ax

        # self.plot.scatter(x="x", y="deflection",
        #   ax=ax)
        #   , color="none", edgecolor="black", zorder=2, label="MPM-FVM")

    @staticmethod
    def compute_deflection(positiondf, df):
        timegroup = df.groupby("time")
        positiongroup = positiondf.groupby("x")

        deflection_data = {"time": [], "x": [], "deflection": []}

        for time, data in timegroup:
            data.set_index(["partId"], inplace=True)
            if time < 0.02:
                continue
            for x, group in positiongroup:
                group.set_index(["partId"], inplace=True)
                for pId, row in group.iterrows():
                    try:
                        y = data.at[pId, "y"] + 0.15
                    except KeyError:
                        print(
                            "Particle ID {pId} not found in both dataframes.".format(pId=pId))
                        exit()
                    y0 = row["y"]
                    deflection = y - y0
                deflection_data["time"].append(time)
                deflection_data["deflection"].append(deflection)
                deflection_data["x"].append(data.at[pId, "x"])

        return pd.DataFrame(data=deflection_data)

    @staticmethod
    def get_beam_middle_partIds(uda_incoming, beam_height):
        uda = uda_incoming.udainit if uda_incoming.udainit else uda_incoming

        result = uda.extracted("p.x", uda.timesteps[0])

        names = uda.vars.get_uda_headers("p.x")
        df = pd.read_table(
            result, header=None, names=names, skiprows=2,
            sep="\s+") if result is not None else pd.DataFrame(
                columns=names)

        y_mean = -beam_height / 2

        def demean(y): return y - y_mean
        df["y"] = df["y"].apply(demean)

        # Locate the y's closest to 0
        # drop duplicate x's
        return df.reindex(df.y.abs().sort_values().index).drop_duplicates("x")


class Beam:
    def __init__(self, b, l, h, E):
        self.b = b
        self.l = l
        self.h = h
        self.E = E
        self.I = self.second_moment_of_area(b, h)
        self.A = self.area(b, h)

    def area(self, b, h):
        return b * h

    def second_moment_of_area(self, b, h):
        return b * h * h * h / 12.0


class PorePressureMomentumFrame(UdaFrame):

    @staticmethod
    def insert_boundary_values(df, column, value):
        """Inserting a row of predefined value at boundaries.

        Boundaries are currently hardcoded to be at positions 0 and 1.

        """
        xmin = 0
        xmax = 1

        grouped = df.groupby("time")
        temp = pd.DataFrame()
        for __, group in grouped:
            tmp = group.iloc[[0], :].copy()
            tmp[column] = value

            tmp["x"] = xmin
            temp = temp.append(tmp, ignore_index=True)

            tmp["x"] = xmax
            temp = temp.append(tmp, ignore_index=True)

        return pd.concat([df, temp], ignore_index=True).sort_values(
            by=["time", "x"])

    @staticmethod
    def normalise_momentum(column, beam):
        """Normalising the momentum along the beam.

        norm_M = momentum * beam_length / (elastic_modulus * second_moment_of_area) * beam_area
        """
        def norm_M(M):
            return M * beam.l / (beam.E * beam.I) * beam.b

        return column.apply(norm_M)

    def plot_df(self, ax=None):
        mp_cycler = cycler("linestyle", ['-', '--', ':', '-.'])
        ax.set_prop_cycle(mp_cycler)

        grouped = self.groupby("time")

        for label, df in grouped:
            df.plot(x="x", y="momentum", ax=ax,
                    color="black",
                    zorder=2, label="$T =" + str(round(label - 0.03, 3)) + "$")

            ax.fill_between(
                x=df.x, y1=df.momentum, alpha=0.2, color=(0, 0.11, 0.86))

    def dataframe_create(self, uda):
        df = super().dataframe_create(uda)
        df = df.filter(["time", "partId", "p.porepressure"])

        beam = Beam(b=0.1, l=1.0, h=0.3, E=10e6)

        sections = self.initialize_group_sections(uda, beam)
        momentum = self.compute_pore_pressure_momentum(sections, df)

        # dropping the initial time step
        momentum = momentum.ix[~(momentum.time < uda.timeseries[1])]

        momentum.momentum = self.normalise_momentum(momentum.momentum, beam)
        momentum = self.insert_boundary_values(
            momentum, "momentum", 0.0)

        return momentum

    @staticmethod
    def initialize_group_sections(uda_incoming, beam):
        uda = uda_incoming.udainit if uda_incoming.udainit else uda_incoming

        result = uda.extracted("p.x", uda.timesteps[0])

        names = uda.vars.get_uda_headers("p.x")
        df = pd.read_table(
            result, header=None, names=names, skiprows=2,
            sep="\s+") if result is not None else pd.DataFrame(
                columns=names)

        y_mean = -beam.h / 2

        def demean(y): return y - y_mean
        df["y"] = df["y"].apply(demean)

        return df

    @staticmethod
    def compute_pore_pressure_momentum(positiondf, df):
        timegroup = df.groupby("time")
        positiongroup = positiondf.groupby("x")

        momentum = {"time": [], "x": [], "momentum": []}

        for time, data in timegroup:
            data.set_index(["partId"], inplace=True)
            for x, group in positiongroup:
                group.set_index(["partId"], inplace=True)
                mom = 0.0
                ys = []
                xs = []
                for pId, row in group.iterrows():
                    try:
                        porepressure = data.at[pId, "p.porepressure"]
                    except KeyError:
                        print(
                            "Particle ID {pId} not found in both dataframes.".format(pId=pId))
                        exit()
                    y = row["y"]
                    ys.append(y * porepressure)
                    xs.append(y)
                mom = -trapezoidal_rule(ys, xs)

                momentum["time"].append(time)
                momentum["x"].append(x)
                momentum["momentum"].append(mom)
        return pd.DataFrame(data=momentum)


def trapezoidal_rule(function, x):
    result = 0.0
    for k in range(1, len(x)):
        result += ((function[k] + function[k - 1]) / 2.0 * (x[k] - x[k - 1]))
    return result

from functools import partial

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("white")

from uintahtools.udaframe import UdaFrame, TerzaghiFrame, PorePressureMomentumFrame, Variable
from uintahtools.terzaghi.terzaghi import terzaghi


FIGSIZE = (5, 3.8)


class UdaPlot:

    def __init__(self, uda):
        self.uda = uda
        # self.df = UdaFrame(uda)

    @staticmethod
    def create(type, uda):
        if type == "terzaghi":
            return TerzaghiPlot(uda)
        elif type == "porepressure_momentum":
            return PorePressureMomentumPlot(uda)
        assert 0, "Bad shape creation: " + type

    def plot(self):
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)

        # self.plot_analytical(ax)
        self.df.plot_df(ax)
        # self.df.plot_df()

        # Removing plot frame
        # for side in ('right', 'top'):
        #     ax.spines[side].set_visible(False)

        ax.set_xbound(lower=0, upper=1)
        # ax.set_ybound(lower=0, upper=1)

        self.add_labels(ax)
        # self.annotate()

        self.add_legend()

    def add_legend(self):
        pass

    def add_labels(self, ax):
        xlabel, ylabel = self.labels()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plot_analytical(self, ax):
        pass

    def labels(self):
        xlabel, ylabel = "X", "Y"
        return xlabel, ylabel

    def annotate(self):
        raise NotImplementedError

    def display_plot(self, output):
        if (output):
            if (len(output) == 1):
                outfile = self.uda.swap_extension("pdf")
            else:
                outfile = output[1]
            plt.savefig(outfile, dpi=300)
        else:
            plt.show()


class PorePressureMomentumPlot(UdaPlot):

    def __init__(self, uda):
        self.df = PorePressureMomentumFrame(uda)
        super(PorePressureMomentumPlot, self).__init__(uda)

    def add_legend(self):
        plt.legend(loc=0)

    def annotate(self):
        pass


class TerzaghiPlot(UdaPlot):

    def __init__(self, uda):
        super().__init__(uda)
        self.df = TerzaghiFrame(uda)

    def add_legend(self):
        plt.legend(bbox_to_anchor=(0.7, 0), loc=4)

    def labels(self):
        xlabel = "Normalized pore pressure $p/p_0$"
        ylabel = "Normalized depth $z/H$"
        return xlabel, ylabel

    def annotate(self):
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
        for i, time in enumerate(reversed(self.uda.timeseries)):
            label = "$T = " + str(time) + "$"
            plt.figtext(*pos[i], label, horizontalalignment="left")

    def plot_analytical(self,
                        ax,
                        zs=[],
                        samples=40,
                        maxj=25,
                        time=False):
        """Compute and plot analytical solution.

        Two options:
            1.  porepressure vs depth (z)
            2.  porepressure vs time (t)

        """
        func = terzaghi
        timeseries = self.uda.timeseries

        add_to_plot = partial(
            ax.plot, color="gray", alpha=0.8,
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

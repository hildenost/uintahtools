from functools import partial

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as colors
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import seaborn as sns
sns.set_style("white")

from uintahtools.udaframe import UdaFrame, TerzaghiFrame, PorePressureMomentumFrame, BeamDeflectionFrame, Beam, BeamContourFrame
from uintahtools.uda import Variable
from uintahtools.terzaghi.terzaghi import terzaghi
#from uintahtools.elastica.large_deflection_fdm import small_deflection, large_deflection


class UdaPlot:

    def __init__(self, uda):
        self.uda = uda
        self.FIGSIZE = (5, 3.8)

        # self.df = UdaFrame(uda)

    @staticmethod
    def create(type, uda):
        if type == "terzaghi":
            return TerzaghiPlot(uda)
        elif type == "porepressure_momentum":
            return PorePressureMomentumPlot(uda)
        elif type == "beam.deflection":
            return BeamDeflectionPlot(uda)
        elif type == "beam.contour":
            print("Creating beam contour plot")
            return BeamContourPlot(uda)
        assert 0, "Bad shape creation: " + type

    def plot(self):
        fig = plt.figure(figsize=self.FIGSIZE)
        fig.subplots_adjust(left=0.15)
        ax = fig.add_subplot(111)

        ax = self.df.plot_df(ax)
        # self.plot_analytical(ax)

        load = 54e3
        number_of_cells = 100
        beam = Beam(b=0.1, l=1, h=0.3, E=10e6)

        xs, ys = small_deflection(load * beam.b, number_of_cells, beam)

        ax.plot(xs, ys, color="gray", alpha=0.8, linestyle="solid",
                lw=2, zorder=1, label="analytical")
        ax.legend(fancybox=True, framealpha=0.8, frameon=True)
        ax.yaxis.grid(True, linestyle="--")
        # self.df.plot_df()

        # Removing plot frame
        # for side in ('right', 'top'):
        #     ax.spines[side].set_visible(False)

        ax.set_xbound(lower=0, upper=1)
        # ax.set_ybound(upper=0)

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
            outfile = self.uda.swap_extension(
                "pdf", number=True) if output == "std" else output
            plt.savefig(outfile, dpi=300)
        else:
            plt.show()


class BeamContourPlot(UdaPlot):

    def __init__(self, uda):
        self.df = BeamContourFrame(uda)
        super().__init__(uda)

    def plot(self):
        groups = self.df.groupby(by="time", as_index=False)
        N = 6
        for i, (name, group) in enumerate(groups):
            if i == 0:
                continue
            plt.figure(num=i + 1, dpi=300)
            ax = plt.gca()
            ax.set_xlim(0, 1.2)
            ax.set_ylim(-0.9, 0.1)
            ax.set_xticks(np.arange(0, 1.2, 0.1))
            ax.set_yticks(np.arange(-0.9, 0.1, 0.1))
            ax.set_aspect('equal')
            # Plot grid.
            ax.grid(c='k', ls='-', alpha=0.3)
            triang = tri.Triangulation(group.x, group.y)
            print(triang.x)
            treshold = 0.5
            for [v1, v2, v3] in triang.triangles:
                print(v1, v2, v3, end="\t")
                print(triang.x[v1], triang.x[v2], triang.x[v3])
                if abs(triang.x[v1] - triang.x[v2]) > treshold or abs(triang.x[v2] - triang.x[v3]) > treshold or abs(triang.x[v1] - triang.x[v3]) > treshold:
                    print("\t\t THIS TRIANGLE SHOULD BE MASKED!!!")

            # triang.set_mask()
            ax.triplot(triang, 'bo-', markersize=0.5, lw=0.1)
            cs = plt.tricontour(group.x, group.y,
                                group["p.porepressure"], N, colors='k', linewidths=0.7, extend="neither")
            plt.clabel(cs, fmt="%1.2g", fontsize=7.5)  # manual=True
            plt.tricontourf(group.x, group.y,
                            group["p.porepressure"], N, cmap=plt.get_cmap("binary"), alpha=0.5, extend="neither")
#            plt.scatter(group.x, group.y,
#                        c=group["p.porepressure"], s=0.6, cmap=plt.get_cmap("binary"))


class BeamDeflectionPlot(UdaPlot):

    def __init__(self, uda):
        self.df = BeamDeflectionFrame(uda)
        super().__init__(uda)
        self.FIGSIZE = (5, 3.8)

    def labels(self):
        xlabel = "Position from fixed end $x$"
        ylabel = "Beam deflection $y$"
        return xlabel, ylabel

    def plot_analytical(self, ax):
        add_to_plot = partial(
            ax.plot, color="gray", alpha=0.8,
            linestyle="solid", linewidth=2, zorder=1)

        load = 54e3
        number_of_cells = 100
        beam = Beam(b=0.1, l=1, h=0.3, E=10e6)

        xs, ys = small_deflection(load * beam.b, number_of_cells, beam)

        add_to_plot(xs, ys, label="analytical")


class PorePressureMomentumPlot(UdaPlot):

    def __init__(self, uda):
        self.df = PorePressureMomentumFrame(uda)
        super(PorePressureMomentumPlot, self).__init__(uda)
        self.FIGSIZE = (5, 6)

    def plot_analytical(self, ax):
        pass

    def plot(self):
        fig = plt.figure(figsize=self.FIGSIZE)
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
        plt.legend(loc=0)

    def labels(self):
        xlabel = "Position along beam $x/L$"
        ylabel = "Normalized pore pressure momentum $M_p^*$"
        return xlabel, ylabel

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

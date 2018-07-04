"""Module of all things dataframes."""
from collections import namedtuple

import pandas as pd


class UdaFrame(pd.DataFrame):

    def __init__(self, uda):
        super().__init__()
        self.vars = Variable(uda.key)
        df = self.dataframe_create(uda)
        super(UdaFrame, self).__init__(df)

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

        df = pd.merge(*dfs).filter(self.vars.get_headers() +
                                   ["time"]).drop_duplicates(self.vars.get_headers() + ["time"])
        for var in self.vars:
            df[var.header] = df[var.header].map(
                lambda t: self.normalize(t, **var.settings))
        return df


class Variable(dict):
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
        return Variable.fixedcols + Variable.headers[var.udavar]

    def __getitem__(self, key):
        return self.vars[key]

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
        x = "x"
        y = "p.porepressure"
        return {"p.porepressure": x, "p.x": y}

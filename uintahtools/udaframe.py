"""Module of all things dataframes."""
import pandas as pd


class UdaFrame(pd.DataFrame):

    def __init__(self, uda):
        super().__init__()

        print(self.dataframe_assemble(
            "p.porepressure", uda.timesteps, uda).head())

    def table_read(self, variable, uda, timestep):
        """Wrapping pd.read_table for readability."""
        result = uda.extracted(variable, timestep)
        return pd.read_table(
            result, header=None, names=self.header(variable), skiprows=2,
            sep="\s+") if result is not None else pd.DataFrame(
                columns=self.header(variable))

    def dataframe_assemble(self, variable, timesteps, uda):
        """Create and return dataframe from extracting the variable at given timesteps from the UDA folder."""

        dfs = (self.table_read(variable, uda, timestep)
               for timestep in timesteps)
        return pd.concat(dfs)

    @staticmethod
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

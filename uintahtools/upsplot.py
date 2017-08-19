"""Simple plotting script.

Provide the x variable and the y variable to be plotted along with the uda-folder
to make a simple 2D scatter plot with matplotlib. Output points are also stored in 
a dat-file.

"""
from functools import partial
from pandas import Series, DataFrame
import pandas as pd
import subprocess


def header(var):
    """Creates column headers based on extracted variable."""
    FIXEDCOLS = ["time", "patch", "matl", "partId"]
    HEADERS = {
        "p.x": ["x", "y", "z"],
        "p.porepressure": ["pw"],
    }
    if var not in HEADERS:
        print("Sorry, the variable {var} is not implemented yet. No headers assigned for {var}".format(var=var))
        return FIXEDCOLS + [var]
    return FIXEDCOLS + HEADERS[var]

def udaplot(x, y, uda):
    """Main function.

    Steps:
        1. Extract XVAR from uda
        2. Extract YVAR from uda
      x 3. Store XVAR and YVAR in their respective dataframes
      x 4. Set column names
        4. Merge dataframes
        5. Extract the columns needed: time, XVAR, YVAR
    """
    print("Plotting x:", x, " vs  y:", y, " contained in ", uda)
    
    # Extracting columns
    # subprocess.call(["./uintahtools/terzaghi", x, y])
    print("Done with bashing about")
    read_table = partial(pd.read_table, header=None,
                                        skiprows=2,
                                        nrows=15, #Uncomment for testing purposes
                                        sep="\s+"
                                        )
    df = read_table("ys.dat", names=header(y))
    print(df)
    df = read_table("xs.dat", names=header(x))
    print(df)

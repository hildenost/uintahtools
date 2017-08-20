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

def normalize(var, varmax, varmin=0, flip=False):
    """Function to normalize var with regards to wrt.
    
    Normalizes to the range [0, 1] where var_min scales to 0 by default,
    but this can be flipped. Resulting values can lie outside the range.

    """
    return (varmax - var)/(varmax-varmin) if flip else (var-varmin)/(varmax-varmin)

def udaplot(x, y, uda):
    """Main function.

    Steps:
        1. Extract XVAR from uda
        2. Extract YVAR from uda
      x 3. Store XVAR and YVAR in their respective dataframes
      x 4. Set column names
      x 5. Merge dataframes
      x 6. Drop duplicates (removes the need for line select)
      x 7. Column select: time, XVAR, YVAR
      x 8. Normalize variables
    """
    print("Plotting x:", x, " vs  y:", y, " contained in ", uda)
    
    # Extracting columns
    # subprocess.call(["./uintahtools/terzaghi", x, y])
    # print("Done with bashing about")
    read_table = partial(pd.read_table, header=None,
                                        skiprows=2,
                                        # nrows=100, #Uncomment for testing purposes
                                        sep="\s+"
                                        )
    df1 = read_table("ys.dat", names=header(y))
    df2 = read_table("xs.dat", names=header(x))
    
    selected = ['time', 'y', 'pw']
    
    df = pd.merge(df1, df2).filter(selected).drop_duplicates(selected)
    
    pwnorm = partial(normalize, varmax=-10000)
    ynorm = partial(normalize, varmax=1, flip=True)
    
    df['pw'] = df['pw'].map(lambda x: pwnorm(x))
    df['y'] = df['y'].map(lambda x: ynorm(x))
    
    print(df)
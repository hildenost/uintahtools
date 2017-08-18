"""Simple plotting script.

Provide the x variable and the y variable to be plotted along with the uda-folder
to make a simple 2D scatter plot with matplotlib. Output points are also stored in 
a dat-file.

"""
from pandas import Series, DataFrame
import pandas as pd
import subprocess

def udaplot(x, y, uda):
    print("Plotting x:", x, " vs  y:", y, " contained in ", uda)
    
    # Extracting columns
    subprocess.call(["./uintahtools/terzaghi", x, y])

    df = pd.read_table("xs.dat")
    print(df)

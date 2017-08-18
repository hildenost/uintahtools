"""Simple plotting script.

Provide the x variable and the y variable to be plotted along with the uda-folder
to make a simple 2D scatter plot with matplotlib. Output points are also stored in 
a dat-file.

"""
from pandas import Series, Dataframe
import pandas as pd
import subprocess

def udaplot(x, y):
    print("Plotting x:", x, " vs  y:", y)
    subprocess.call(["./uintahtools/terzaghi", x, y])

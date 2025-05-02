import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
import os
import sys

# Add project root to sys.path for clean imports
sys.path.append(os.path.abspath(".."))
__all__ = ['sp', 'np' , 'sns' , 'plt' , 'pd']

def ensure_column_vector(arr):
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def plotLineGraph(xlabel, ylabel, title, xdata, ydata):
    # Plotting
    plt.plot(xdata, ydata, marker='o', linestyle='-', color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()
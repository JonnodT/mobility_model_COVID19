import pandas as pd
import numpy as np
import datetime
x = pd.Series([1,2,3], index=pd.date_range('2001', freq='D', periods=3))
y = pd.Series([3,4,4,10,100,120,125,190], index=pd.date_range('2000', freq='D', periods=8)).index
z = pd.Series([1,2,3], index=pd.date_range('2003', freq='D', periods=3))

def kz(series, window, iterations):
    """KZ filter implementation
    series is a pandas series
    window is the filter window m in the units of the data (m = 2q+1)
    iterations is the number of times the moving average is evaluated
    """
    z = series.copy()
    for i in range(iterations):
        z = z.rolling(window=window, min_periods=1, center=True).mean()
    return z


def square(a):
    b = a - 1
    b[b < 0] = 0
    return (a * b) / 2


TEST_RANGE = pd.date_range(start='2020-09-20', end="2020-11-29", freq="7D")
print(TEST_RANGE)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(6,15))
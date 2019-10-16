import torch
import numpy as np
from torch import nn
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 6)
sns.set()

from pandas.plotting import register_matplotlib_converters, autocorrelation_plot
register_matplotlib_converters()

def plot_forecasts(forecasts_mean, forecasts_std, y, full_y):
	#plt.plot(np.hstack((y[:-1], forecasts_mean)), label='Forecasts')
	T = len(y)
	plt.plot(np.arange(T, T+len(forecasts_mean)), forecasts_mean, color="g", label='Forecasts')
	plt.plot(full_y, label='Data')
	plt.fill_between(np.arange(T, T+len(forecasts_mean)), forecasts_mean-forecasts_std,
						 forecasts_mean+forecasts_std,
						 facecolor="green", alpha=0.2)
	plt.legend(loc='best')
	plt.show()
	#print(y[-11:], forecasts_mean, full_y[-11:])
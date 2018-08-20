"""
Gaussian HMM of stock data
--------------------------

This script shows how to use Gaussian HMM on stock price data from
voptdb.
"""

from __future__ import print_function

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

from hmmlearn.hmm import GaussianHMM

import pickle

print(__doc__)


def load_by_pickle(file_name):
    """load dat is encoded by "bytes" type

    :param file_name: file name for loading
    :return: load data
    """
    if file_name[-4:] == ".pkl":
        print(file_name)
        pass
    else:
        file_name = file_name + ".pkl"
        print(file_name)
    file_object = open(file_name, 'rb')
    load_data = pickle.load(file_object, encoding="bytes")
    return load_data


dic_price = load_by_pickle('./data/price_data.pkl')

###############################################################################
# Get quotes from Yahoo! finance

# Unpack quotes
dates = np.array(dic_price["close"]["005930"].index)
close_v = np.array(dic_price["close"]["005930"])
volume = np.array(dic_price["volume"]["005930"])[1:]

# Take diff of close value. Note that this makes
# ``len(diff) = len(close_t) - 1``, therefore, other quantities also
# need to be shifted by 1.
diff = np.diff(close_v)
dates = dates[1:]
close_v = close_v[1:]

# Pack diff and volume for training.
X = np.column_stack([diff, volume])

###############################################################################
# Run Gaussian HMM
print("fitting to HMM and decoding ...", end="")

# Make an HMM instance and execute fit
model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print("done")

###############################################################################
# Print trained parameters and plot
print("Transition matrix")
print(model.transmat_)
print()

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()

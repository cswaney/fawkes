from fawkes.models import DiscreteNetworkPoisson
import pandas as pd
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

"""
Estimate the continuous-time Network Poisson model using Nasdaq data.

1. Import event data. CONVERT TO DISCRETE-TIME EVENT DATA.
2. Iterate Gibbs sampling algorithm.
3. Save samples to HDF5.
"""

N = 12
B = 3
L = 60
nsamples= 5000
t0 = 34200 + 3600
tN = 57600 - 3600
dt = 1
T = tN - t0
_, name = sys.argv  # e.g. 'PFE'
with h5.File('/Volumes/datasets/ITCH/events/events.hdf5'.format(L), 'r') as hdf:
    dates = [date for date in hdf[name].keys()]

for date in dates:
    print("Performing sampling for name {} and date {}".format(name, date))

    # Import the event data
    with h5.File('/Volumes/datasets/ITCH/events/events.hdf5', 'r') as hdf:
        events = pd.DataFrame(hdf['{}/{}'.format(name, date)][:], columns=('timestamp', 'event'))
        events = events[ (events['timestamp'] > t0) & (events['timestamp'] < tN) ]
        events['event'] = events['event'].astype('int')
        events['timestamp'] = events['timestamp'] - t0
        events = events.reset_index(drop=True)

    # Convert event data
    S = np.zeros((T, N), dtype='int')
    for i in np.arange(N):
        subset = events[ events['event'] == i ]
        t = subset['timestamp'].astype('int')  # round down
        c = pd.value_counts(t)
        S[c.index, i] = c.values

    assert S.sum() == len(events), "Number of events in `S` differs from `events`."

    # Create a network
    model = DiscreteNetworkPoisson(N=N, L=L, B=B, dt=dt)

    # Gibbs sampling
    print("Beginning sampling... (M = {})".format(len(events)))
    start = time.time()
    lambda0, W, theta = model.sample_ext(S, size=nsamples)
    print("Done. Elapsed time: {} s\n".format(time.time() - start))

    # Save samples.
    with h5.File('/Volumes/datasets/ITCH/samples/discrete_B={}_L={}.hdf5'.format(B, L), 'a') as hdf:
        hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'lambda0'), data=lambda0)
        hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'W'), data=W)
        hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'theta'), data=theta)

print("Done.")

from fawkes.models import NetworkPoisson
import pandas as pd
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

"""
Estimate the continuous-time Network Poisson model using Nasdaq data.

1. Import event data.
2. Iterate Gibbs sampling algorithm.
3. Save samples to HDF5.
"""

N = 12
T = (57600 - 3600) - (34200 - 3600)
dt_max = 60
nsamples= 5000
t0 = 34200 + 3600
tN = 57600 - 3600
_, name = sys.argv  # e.g. 'PFE'
with h5.File('/Volumes/datasets/ITCH/events/events.hdf5'.format(dt_max), 'r') as hdf:
    dates = [date for date in hdf[name].keys()]

for date in dates:
    print("Performing sampling for name {} and date {}".format(name, date))

    # Import the event data
    with h5.File('/Volumes/datasets/ITCH/events/events.hdf5', 'r') as hdf:
        events = pd.DataFrame(hdf['{}/{}'.format(name, date)][:], columns=('timestamp', 'event'))
        events = events[ (events['timestamp'] > t0) & (events['timestamp'] < tN) ]
        events['event'] = events['event'].astype('int')
        events['timestamp'] = events['timestamp'] - t0
        events = (events['timestamp'].values, events['event'].values)

    # Create a network
    model = NetworkPoisson(N=N, dt_max=dt_max)

    # Gibbs sampling
    print("Beginning sampling... (M = {})".format(len(events[0])))
    start = time.time()
    lambda0, W, mu, tau = model.sample_ext(events, T=T, size=nsamples)
    print("Done. Elapsed time: {} s\n".format(time.time() - start))

    # Save samples.
    with h5.File('/Volumes/datasets/ITCH/samples/samples_dt_max={}.hdf5'.format(dt_max), 'a') as hdf:
        hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'lambda0'), data=lambda0)
        hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'W'), data=W)
        hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'mu'), data=mu)
        hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'tau'), data=tau)

print("Done.")

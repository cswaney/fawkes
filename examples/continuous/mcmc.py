from fawkes.models import NetworkPoisson
import pandas as pd
import h5py as h5
import numpy as np
np.seterr(divide='ignore', invalid='ignore')  # TODO: how to deal with these?
import matplotlib.pyplot as plt
import time
import sys

"""Estimates the continuous-time Network Poisson model for specified group of stocks."""

N = 12
t0 = 34200 + 3600
tN = 57600 - 3600
T = tN - t0
dt_max = 60
nsamples= 2500
max_events = 60000

_, group, date = sys.argv
read_path = '/Volumes/datasets/ITCH/events/large2007.hdf5'
write_path = '/Volumes/datasets/ITCH/samples/large2007_dt_max={}.hdf5'.format(dt_max)
groups = [['A', 'B', 'C', 'D', 'E'],
          ['F', 'G', 'H', 'I', 'J', 'K'],
          ['L', 'M', 'N', 'O', 'P', 'Q', 'R'],
          ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']]

names = []
with h5.File(read_path, 'r') as hdf:
    for name in hdf.keys():
        if (date in hdf[name]) and (name[0] in groups[int(group) - 1]):
            names.append(name)

for name in names:

    print("Performing sampling for name {}, date {}...".format(name, date))

    # Import the event data
    with h5.File(read_path, 'r') as hdf:
        try:
            df = pd.DataFrame(hdf['{}/{}'.format(name, date)][:], columns=('timestamp', 'event'))
            df = df[ (df['timestamp'] > t0) & (df['timestamp'] < tN) ]
            df['event'] = df['event'].astype('int')
            df['timestamp'] = df['timestamp'] - t0
            dups = df.duplicated(keep='last')
            df = df[~dups].reset_index(drop=True)
            events = (df['timestamp'].values, df['event'].values)
        except:
            print('Unable to find event data; skipping\n')
            events = None
        if len(events[0]) == 0:
            print('Unable to find event data; skipping\n')
            events = None
        if len(events[0]) > max_events:
            print('Event count exceeds max_events; skipping\n')
            with open('/Users/colinswaney/Desktop/output_date={}_grp={}.txt'.format(date, group), 'a') as fout:
                fout.write('{}\n'.format(name))
            events = None

        # Generate MCMC sample
        if (events is not None):
            # Create a network
            model = NetworkPoisson(N=N, dt_max=dt_max)
            # Gibbs sampling
            print("Beginning sampling... (M = {})".format(len(events[0])))
            start = time.time()
            lambda0, W, mu, tau = model.sample_ext(events, T=T, size=nsamples)
            print("Finished MCMC sampling.")
            # Save samples.
            with h5.File(write_path, 'a') as hdf:
                print('Writing sample to HDF5...')
                hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'lambda0'), data=lambda0)
                hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'W'), data=W)
                hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'mu'), data=mu)
                hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'tau'), data=tau)
                print('Done.\n')

print("Finished.")

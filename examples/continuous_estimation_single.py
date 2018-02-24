from fawkes.models import NetworkPoisson
import pandas as pd
import h5py as h5
import numpy as np
np.seterr(divide='ignore', invalid='ignore')  # TODO: how to deal with these?
import matplotlib.pyplot as plt
import time
import sys

# TODO: Record information so that you can re-create the subsample selected.

N = 12
t0 = 34200 + 3600
tN = 57600 - 3600
T = tN - t0
dt_max = 5
nsamples= 2500
max_obs = 30000

def import_names(date):
    names = []
    for i in range(1,5):
        df = pd.read_csv('/Users/colinswaney/Desktop/output_date={}_grp={}.txt'.format(date, i), header=None)
        names += list(df[0].values)
    return names

date = '072413'
names = import_names(date)
# _, name, date = sys.argv
root_path = '/Volumes/datasets/ITCH/'
read_path = root_path + 'events/large2007.hdf5'
write_path = root_path + 'samples/large2007_dt_max={}_truncated.hdf5'.format(dt_max)

for name in names:

    print("Performing sampling for name {}, date {}...".format(name, date))

    # Importing
    with h5.File(read_path, 'r') as hdf:
        try:
            df = pd.DataFrame(hdf['{}/{}'.format(name, date)][:], columns=('timestamp', 'event'))
            df = df[ (df['timestamp'] > t0) & (df['timestamp'] < tN) ]
            df['event'] = df['event'].astype('int')
            df['timestamp'] = df['timestamp'] - t0
            dups = df.duplicated(keep='last')
            df = df[~dups].reset_index(drop=True)
            nobs = len(df)
            if nobs > max_obs:
                print('Number of events exceeds max_N; truncating a random subsample.')
                n = np.random.randint(low=0, high=nobs-max_obs)
                df = df.iloc[n:n+max_obs,:]
                df = df.reset_index(drop=True)
                t0 = df['timestamp'][0]
                tN = df['timestamp'][max_obs - 1]
                T = tN - t0
                df['timestamp'] = df['timestamp'] - t0
            events = (df['timestamp'].values, df['event'].values)
            import_success = True
        except:
            print('Unable to find event data; skipping\n')
            import_success = False
        if len(events[0]) == 0:
            print('Unable to find event data; skipping\n')
            import_success = False

    # MCMC Sampling
    if import_success:
        model = NetworkPoisson(N=N, dt_max=dt_max)
        print("Beginning sampling... (M = {})".format(len(events[0])))
        start = time.time()
        lambda0, W, mu, tau = model.sample_ext(events, T=T, size=nsamples)
        print("Finished MCMC sampling.")
        with h5.File(write_path, 'a') as hdf:
            print('Writing sample to HDF5...')
            hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'lambda0'), data=lambda0)
            hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'W'), data=W)
            hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'mu'), data=mu)
            hdf.create_dataset(name='{}/{}/{}'.format(name, date, 'tau'), data=tau)
            print('Done.\n')

print("Finished.")

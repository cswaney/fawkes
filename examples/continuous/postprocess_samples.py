from fawkes.models import NetworkPoisson
import pandas as pd
import numpy as np
import h5py as h5
import sys
import os

"""Creates HDF5 datasets of estimates and stability from MCMC samples."""

def import_samples(path, name, date, burn):
    print("Importing data for name {} and date {}...".format(name, date))
    try:
        with h5.File(path, 'r') as hdf:
            lambda0 = hdf['/{}/{}/lambda0'.format(name, date)][:]
            W = hdf['/{}/{}/W'.format(name, date)][:]
            mu = hdf['/{}/{}/mu'.format(name, date)][:]
            tau = hdf['/{}/{}/tau'.format(name, date)][:]
        print('> Successfully imported {} samples.'.format(W.shape[-1]))
    except:
        print('> Unable to find sample data; returning None')
        return None
    return lambda0[:, burn:], W[:, :, burn:], mu[:, :, burn:], tau[:, :, burn:]

def post_process(name, date, dt_max, burn=500):

    N = 12
    model = NetworkPoisson(N, dt_max)
    sample_path = '/Volumes/datasets/ITCH/samples/large2007_dt_max={}.hdf5'.format(dt_max)

    # Import samples
    samples = import_samples(sample_path, name, date, burn)
    if samples is not None:
        # Unpack samples
        lambda0, W, mu, tau = samples
        # Compute point estimates
        lambda0 = np.median(lambda0, axis=1)
        W = np.median(W, axis=2).reshape(N * N)  # row major
        mu = np.median(mu, axis=2).reshape(N * N)  # row major
        tau = np.median(tau, axis=2).reshape(N * N)  # row major
        estimates = [name, date] + list(np.concatenate([lambda0, W, mu, tau]))
        # Check stability
        model.lamb = lambda0
        model.W = W.reshape((N,N))
        model.mu = mu.reshape((N,N))
        model.tau = tau.reshape((N,N))
        _, maxeig = model.check_stability(return_value=True)
        eigenvalue = [name, date, maxeig]
        return estimates, eigenvalue

root = '/Volumes/datasets/ITCH/'
dates = [date for date in os.listdir('{}/csv/'.format(root)) if date != '.DS_Store']
names = [name.lstrip(' ') for name in pd.read_csv('{}/SP500.txt'.format(root))['Symbol']]
names.sort()

mcmc = []
eig = []
dt_max = 60
for name in names:
    for date in dates:
        estimate, eigenvalue = post_process(name, date)
        mcmc.append(estimate)
        eig.append(eigenvalue)

# Write to HDF5
with h5.File() as hdf:
    hdf['mcmc'] = mcmc
    hdf['eig'] = eig

from fawkes.models import NetworkPoisson, HomogeneousPoisson
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pandas as pd
import os
import sys
import time


"""Compute and compare the likelihood using MCMC estimates."""


def import_samples(path, name, date, burn=0):
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

def import_events(name, date, size=None, t0=37800, tN=54000):
    with h5.File('/Volumes/datasets/ITCH/events/large2007.hdf5', 'r') as hdf:
        try:
            df = pd.DataFrame(hdf['{}/{}'.format(name, date)][:], columns=('timestamp', 'event'))
        except:
            print('Unable to find event data; skipping\n')
            return None
        df = df[ (df['timestamp'] > t0) & (df['timestamp'] < tN) ]
        df['event'] = df['event'].astype('int')
        df['timestamp'] = df['timestamp'] - t0
        dups = df.duplicated(keep='last')
        df = df[~dups].reset_index(drop=True)
        nobs = len(df)
        if (size is not None) and (nobs < size):
            print('Insufficient event data (n={}); skipping\n'.format(nobs))
            return None
        elif (size is not None):
            idx = np.random.randint(low=0, high=nobs-size)
            df = df.iloc[idx:idx + size,:]
            df = df.reset_index(drop=True)
            t0 = df['timestamp'][0]
            tN = df['timestamp'][size - 1]
            df['timestamp'] = df['timestamp'] - t0
        events = (df['timestamp'].values, df['event'].values)
        return events, tN - t0

def get_estimates(sample, method='median', threshold=None, log=False, norm=False):
    lambda0, W, mu, tau = sample
    if method == 'median':
        lambda0 = np.median(lambda0, axis=1)
        W = np.median(W, axis=2)
        mu = np.median(mu, axis=2)
        tau = np.median(tau, axis=2)
    if threshold is not None:
        if log and norm:
            W[np.log(W / lambda0) < threshold] = 0
        elif log:
            W[np.log(W) < threshold] = 0
        elif norm:
            W[(W / lambda0) < threshold] = 0
        else:
            W[W < threshold] = 0
    return lambda0, W, mu, tau

def sample_events(events, size):
    times, nodes = events
    nobs = len(times)
    if nobs < size:
        print('Insufficient event data (n={}); skipping\n'.format(nobs))
        return None
    idx = np.random.randint(low=0, high=nobs-size)
    times = times[idx:idx + size]
    nodes = nodes[idx:idx + size]
    t0 = times[0]
    tN = times[-1]
    T = tN - t0
    times = times - t0
    return (times, nodes), T


# name = 'A'
# name = 'LLY'
name = 'SIG'
mcmc_date = '072413'  # test: '072513'
event_date = '072513'
norm = False
log = True
L = 100  # simulation size
M = 180  # sample size
N = 12  # network size
dt_max = 60
model_net = NetworkPoisson(N=N, dt_max=dt_max)
model_hom = HomogeneousPoisson(N=N)
df = []
read_path = '/Volumes/datasets/ITCH/samples/large2007_dt_max=60.hdf5'
write_path = '/Users/colinswaney/Desktop/threshold_name={}_date={}_M={}_norm={}.txt'.format(name, event_date, M, norm)
with h5.File(read_path, 'r') as hdf:
    start = time.time()
    try:
        lambda0, W, mu, tau = import_samples(read_path, name, mcmc_date)
    except:
        print('Unable to import samples; skipping')
    for threshold in np.arange(-8, 1, .5):
        lambda0_, W_, mu_, tau_ = get_estimates((lambda0, W, mu, tau),
                                                threshold=threshold,
                                                log=log,
                                                norm=norm)
        events, T = import_events(name, event_date)
        model_net.lamb = lambda0_
        model_net.W = W_
        model_net.mu = mu_
        model_net.tau = tau_
        model_hom.lambda0 = np.median(model_hom.sample(events, T, size=2500), axis=0)
        print('Computing likelihood on random subsamples (threshold={}, M={})...'.format(threshold, M))
        start_sub = time.time()
        for i in np.arange(L):
            sample, T = sample_events(events, size=M)
            ll_net, _ = model_net.compute_likelihood(sample, T)
            ll_hom, _ = model_hom.compute_likelihood(sample, T)
            bits = (ll_net - ll_hom) / M
            df.append([name, threshold, ll_net, ll_hom, bits, T])
            print('[{:d}] ll_net={:.2f}, ll_hom={:.2f}, bits/spike={:.2f} (T={:.2f})'.format(i, ll_net, ll_hom, bits, T))
        stop_sub = time.time()
        print('Done! Elapsed time: {:.2f} sec'.format(stop_sub - start_sub))
df = pd.DataFrame(df, columns=['name', 'threshold', 'll_net', 'll_hom', 'bits', 'T'])
df.to_csv(write_path)
stop = time.time()
print('Average differences (bits/sec): {}'.format(df['bits'].mean()))
print('Average sample: {}'.format(df['T'].mean()))
print('Elapsed time: {} sec'.format(stop - start))

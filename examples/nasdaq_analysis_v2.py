from fawkes.models import NetworkPoisson
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import os
import sys


def import_samples(path, name, date, dt_max, burn):
    print("Import data for name {} and date {}.".format(name, date))
    with h5.File(path, 'r') as hdf:
        lambda0 = hdf['/{}/{}/lambda0'.format(name, date)][:]
        W = hdf['/{}/{}/W'.format(name, date)][:]
        mu = hdf['/{}/{}/mu'.format(name, date)][:]
        tau = hdf['/{}/{}/tau'.format(name, date)][:]
    return lambda0[:, burn:], W[:, :, burn:], mu[:, :, burn:], tau[:, :, burn:]

def import_events(path, name, date, t0, tN):
    with h5.File(path, 'r') as hdf:
        events = pd.DataFrame(hdf['{}/{}'.format(name, date)][:], columns=('timestamp', 'event'))
        events = events[ (events['timestamp'] > t0) & (events['timestamp'] < tN) ]
        events['event'] = events['event'].astype('int')
        events['timestamp'] = events['timestamp'] - t0
    return events.reset_index(drop=True)

def calculate_homogeneous(events, N, T):
    lambda0 = np.zeros(N)
    for i in range(N):
        cnt = len(events[events['event'] == i])
        lambda0[i] = cnt / T
    return lambda0

def calculate_inhomogeneous(lambda0):
    return np.median(lambda0, axis=1)

def construct_table(lambda0, W, mu, tau):
    pass

def plot_posterior(lambda0, null=None):
    cnts, bins, _ = plt.hist(lambda0, bins=30)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.xticks(fontsize=8)
    if null is not None:
        plt.vlines(null, 0, np.max(cnts) * 1.1, linewidth=0.25, color='C1')

def plot_posteriors(lambda0, nulls=None):
    N, _ = lambda0.shape
    for i in range(N):
        plt.subplot(N, 1, i + 1)
        if nulls is not None:
            plot_posterior(lambda0[i, :], nulls[i])
        else:
            plot_posterior(lambda0[i, :])
    plt.show()
    plt.clf()

def plot_density(lambda0, N=100, null=None):
    grid = np.linspace(lambda0.min(), lambda0.max(), N)
    kde = gaussian_kde(lambda0).evaluate(grid)
    plt.plot(grid, kde)
    plt.fill_between(grid, kde)
    if null is not None:
        plt.vlines(null, 0, np.max(cnts) * 1.1, linewidth=0.25, color='C1')

def plot_densities(lambda0, nulls):
    N, _ = lambda0.shape
    for i in range(N):
        plt.subplot(N, 1, i + 1)
        if nulls is not None:
            plot_density(lambda0[i, :], nulls[i])
        else:
            plot_density(lambda0[i, :])
    plt.show()
    plt.clf()

def plot_weights(W):
    plt.imshow(W, cmap='Blues', vmin=0, vmax=0.5)
    cbar = plt.colorbar(pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel("Child nodes", labelpad=20, fontsize=8)
    tick_labels = ['Add 0 (Bid)',
                   'Add 1 (Bid)',
                   'Add 2 (Bid)',
                   'Add 0 (Ask)',
                   'Add 1 (Ask)',
                   'Add 2 (Ask)',
                   'Delete 1 (Bid)',
                   'Delete 2 (Bid)',
                   'Delete 1 (Ask)',
                   'Delete 2 (Ask)',
                   'Execute (Bid)',
                   'Execute (Ask)']
    plt.xticks(np.arange(0, 12), tick_labels, rotation=90, fontsize=8)
    plt.ylabel("Parent nodes", labelpad=20, fontsize=8)
    plt.yticks(np.arange(0, 12), tick_labels, fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.clf()

def logit_normal(dt, mu, tau, dt_max):
    assert (dt < dt_max).all(), "dt must be less than dt_max."
    assert (dt > 0).all(), "dt must be greater than zero."
    Z = dt * (dt_max - dt) / dt_max * (tau / (2 * np.pi)) ** (-0.5)
    x = dt / dt_max
    s = np.log(x / (1 - x))
    return (1 / Z) * np.exp( -tau / 2 * (s - mu) ** 2 )

def plot_impulse(parent, child, W, mu, tau, dt_max):
    eps = 0.001
    dt = np.linspace(0 + eps, dt_max - eps, 100)
    values = logit_normal(dt, mu[parent, child], tau[parent, child], dt_max)
    plt.plot(dt, values, linewidth=0.5)
    plt.xlim([0, dt_max])
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')

def plot_impulses(W, mu, tau, dt_max):
    N, M = W.shape
    for i in range(N):
        for j in range(M):
            plt.subplot(N, M, i * M + j + 1)
            plot_impulse(i, j, W, mu, tau, dt_max)
    plt.show()
    plt.clf()

def calculate_averages(path, name, dates):
    lambda0_medians = []
    W_medians = []
    mu_medians = []
    tau_medians = []
    for date in dates:
        lambda0, W, mu, tau = import_samples(path, name, date, dt_max, burn)
        N, _ = lambda0.shape
        lambda0_medians.append(np.median(lambda0, axis=1).reshape((N, 1)))
        W_medians.append(np.median(W, axis=2).reshape((N, N, 1)))
        mu_medians.append(np.median(mu, axis=2).reshape((N, N, 1)))
        tau_medians.append(np.median(tau, axis=2).reshape((N, N, 1)))
    lambda0_avg = np.mean(np.concatenate(lambda0_medians, axis=1), axis=1)
    W_avg = np.mean(np.concatenate(W_medians, axis=2), axis=2)
    mu_avg = np.mean(np.concatenate(mu_medians, axis=2), axis=2)
    tau_avg = np.mean(np.concatenate(tau_medians, axis=2), axis=2)
    return lambda0_avg, W_avg, mu_avg, tau_avg

def calculate_estimates(dates):
    lambda0_ = []
    W_ = []
    mu_ = []
    tau_ = []
    for date in dates:
        lambda0, W, mu, tau = import_samples(name, date, dt_max, burn)
        N, _ = lambda0.shape
        lambda0_.append(np.median(lambda0, axis=1).reshape((N, 1)))
        W_.append(np.median(W, axis=2).reshape((N, N, 1)))
        mu_.append(np.median(mu, axis=2).reshape((N, N, 1)))
        tau_.append(np.median(tau, axis=2).reshape((N, N, 1)))
    lambda0_ = np.concatenate(lambda0_, axis=1)
    W_ = np.concatenate(W_, axis=2)
    mu_ = np.concatenate(mu_, axis=2)
    tau_ = np.concatenate(tau_, axis=2)
    return lambda0_, W_, mu_, tau_

def calculate_quantiles(dates):
    lambda0_5 = []; lambda0_95 = [];
    W_5 = []; W_95 = [];
    mu_5 = []; mu_95 = [];
    tau_5 = []; tau_95 = [];
    for date in dates:
        lambda0, W, mu, tau = import_samples(name, date, dt_max, burn)
        N, _ = lambda0.shape
        lambda0_5.append(np.percentile(lambda0, 0.05, axis=1).reshape((N, 1)))
        lambda0_95.append(np.percentile(lambda0, 0.95, axis=1).reshape((N, 1)))
        W_5.append(np.percentile(W, 0.05, axis=2).reshape((N, N, 1)))
        W_95.append(np.percentile(W, 0.95, axis=2).reshape((N, N, 1)))
        mu_5.append(np.percentile(mu, 0.05, axis=2).reshape((N, N, 1)))
        mu_95.append(np.percentile(mu, 0.95, axis=2).reshape((N, N, 1)))
        tau_5.append(np.percentile(tau, 0.05, axis=2).reshape((N, N, 1)))
        tau_95.append(np.percentile(tau, 0.95, axis=2).reshape((N, N, 1)))
    lambda0_5 = np.concatenate(lambda0_5, axis=1); lambda0_95 = np.concatenate(lambda0_95, axis=1)
    W_5 = np.concatenate(W_5, axis=2); W_95 = np.concatenate(W_95, axis=2)
    mu_5 = np.concatenate(mu_5, axis=2); mu_95 = np.concatenate(mu_95, axis=2)
    tau_5 = np.concatenate(tau_5, axis=2); tau_95 = np.concatenate(tau_95, axis=2)
    return lambda0_5, lambda0_95, W_5, W_95, mu_5, mu_95, tau_5, tau_95

def plot_series(estimates, quantiles):
    lower, upper = quantiles
    L = len(estimates)
    plt.plot(estimates, linewidth=0.5)
    plt.fill_between(np.arange(L), y1=lower, y2=upper, alpha=0.20)
    plt.show()
    plt.clf()


N = 12
dt_max = 60
burn = 2000
t0 = 34200 + 3600
tN = 57600 - 3600
T = tN - t0
_, name = sys.argv
# sample_path = '/Volumes/datasets/ITCH/samples/samples_dt_max={}.hdf5'.format(dt_max)
# event_path = '/Volumes/datasets/ITCH/events/events.hdf5'
sample_path = '/Users/colinswaney/Desktop/continuous_dt_max=60.hdf5'
event_path = '/Users/colinswaney/Desktop/events.hdf5'
with h5.File(sample_path, 'r') as hdf:
    dates = [date for date in hdf[name].keys()]
date = dates[0]
lambda0, W, mu, tau = import_samples(sample_path, name, date, dt_max, burn)


"""Daily Analysis"""

# Calculate homogeneous rates.
events = import_events(event_path, name, date, t0, tN)
lambda0_h0 = calculate_homogeneous(events, T)
lambda0_h1 = calculate_inhomogeneous(lambda0)

# Plot posteriors for lambda0.
plot_posteriors(lambda0, lambda0_h0)
plot_densities(lambda0, lambda0_h0)

# Plot median of posteriors for W.
W_hat = np.median(W, axis=2)
plot_weights(W_hat)

# Plot median impulse response.
mu_hat = np.median(mu, axis=2)
tau_hat = np.median(tau, axis=2)
plot_impulses(W_hat, mu_hat, tau_hat, dt_max)


"""Monthly Analysis"""
lambda0_avg, W_avg, mu_avg, tau_avg = calculate_averages(sample_path, name, dates)
plot_weights(W_avg)
plot_impulses(W_avg, mu_avg, tau_avg, dt_max)


"""Time Series Analysis"""
lambda0_, W_, mu_, tau_ = calculate_estimates(dates)
lambda0_5, lambda0_95, W_5, W_95, mu_5, mu_95, tau_5, tau_95 = calculate_quantiles(daets)

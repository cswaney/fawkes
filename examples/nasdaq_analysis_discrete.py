from fawkes.models import DiscreteNetworkPoisson
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pandas as pd
import os
import sys


def import_samples(path, name, date, burn):
    print("Import data for name {} and date {}.".format(name, date))
    with h5.File(path, 'r') as hdf:
        lambda0 = hdf['/{}/{}/lambda0'.format(name, date)][:]
        W = hdf['/{}/{}/W'.format(name, date)][:]
        theta = hdf['/{}/{}/theta'.format(name, date)][:]
    return lambda0[:, burn:], W[:, :, burn:], theta[:, :, :, burn:]

def import_spikes(path, name, date, t0, tN):
    with h5.File(path, 'r') as hdf:
        events = pd.DataFrame(hdf['{}/{}'.format(name, date)][:], columns=('timestamp', 'event'))
        events = events[ (events['timestamp'] > t0) & (events['timestamp'] < tN) ]
        events['event'] = events['event'].astype('int')
        events['timestamp'] = events['timestamp'] - t0
        spikes = np.zeros((T, N), dtype='int')
        for i in np.arange(N):
            subset = events[ events['event'] == i ]
            t = subset['timestamp'].astype('int')  # round down
            c = pd.value_counts(t)
            spikes[c.index, i] = c.values
        assert spikes.sum() == len(events), "Number of events in `S` differs from `events`."
    return spikes

def calculate_homogeneous(spikes, T):
    return spikes.sum(axis=0) / T

def calculate_inhomogeneous(lambda0):
    return np.median(lambda0, axis=1)

# TODO
def construct_table(lambda0, W, theta, tau):
    pass

def plot_bias(lambda0, path):

    def plot_pair(bids, asks, label, xlim):
        bid_cnts, bid_bins, _ = plt.hist(bids, bins=25, alpha=0.75, color='C0')
        ask_cnts, ask_bins, _ = plt.hist(asks, bins=25, alpha=0.75, color='C3')
        plt.ylabel(label, fontsize=8)
        plt.ylim([0, np.maximum(np.max(bid_cnts), np.max(ask_cnts)) * 1.1])
        plt.yticks([])
        plt.xticks(fontsize=8)

    plt.subplot(3, 2, 1)
    plot_pair(lambda0[0,:], lambda0[3,:], 'ADDS, Level 0', [0.01, 0.03])
    plt.subplot(3, 2, 3)
    plot_pair(lambda0[1,:], lambda0[4,:], 'ADDS, Level 1', [0, 0.015])
    plt.subplot(3, 2, 5)
    plot_pair(lambda0[2,:], lambda0[5,:], 'ADDS, Level 2', [0, 0.01])
    plt.subplot(3, 2, 2)
    plot_pair(lambda0[6,:], lambda0[7,:], 'CANCELS, Level 1', [0, 0.01])
    plt.subplot(3, 2, 4)
    plot_pair(lambda0[8,:], lambda0[9,:], 'CANCELS, Level 2', [0, 0.005])
    plt.subplot(3, 2, 6)
    plot_pair(lambda0[10,:], lambda0[11,:], 'EXECUTES', [0.015, 0.025])
    plt.tight_layout()
    plt.savefig(path + 'bias.eps')
    plt.savefig(path + 'bias.pdf')
    plt.show()
    plt.clf()

def plot_self_connections(W, path):

    diagonal = np.diagonal(W).transpose()

    def plot_pair(bids, asks, label, xlim):
        bid_cnts, bid_bins, _ = plt.hist(bids, bins=25, alpha=0.75, color='C0')
        ask_cnts, ask_bins, _ = plt.hist(asks, bins=25, alpha=0.75, color='C3')
        plt.ylabel(label, fontsize=8)
        plt.ylim([0, np.maximum(np.max(bid_cnts), np.max(ask_cnts)) * 1.1])
        plt.yticks([])
        plt.xticks(fontsize=8)
        plt.xlim(xmin=0)

    plt.subplot(3, 2, 1)
    plot_pair(diagonal[0,:], diagonal[3,:], 'ADDS, Level 0', [0.01, 0.03])
    plt.subplot(3, 2, 3)
    plot_pair(diagonal[1,:], diagonal[4,:], 'ADDS, Level 1', [0, 0.015])
    plt.subplot(3, 2, 5)
    plot_pair(diagonal[2,:], diagonal[5,:], 'ADDS, Level 2', [0, 0.01])
    plt.subplot(3, 2, 2)
    plot_pair(diagonal[6,:], diagonal[7,:], 'CANCELS, Level 1', [0, 0.01])
    plt.subplot(3, 2, 4)
    plot_pair(diagonal[8,:], diagonal[9,:], 'CANCELS, Level 2', [0, 0.005])
    plt.subplot(3, 2, 6)
    plot_pair(diagonal[10,:], diagonal[11,:], 'EXECUTES', [0.015, 0.025])
    plt.tight_layout()
    plt.savefig(path + 'diagonal.eps')
    plt.savefig(path + 'diagonal.pdf')
    plt.show()
    plt.clf()

def plot_weights(W, path, ext=None, cmap=None):

    if ext is None:
        ext = ''

    if cmap is not None:
        plt.imshow(W, cmap=cmap)
    else:
        plt.imshow(W, cmap='PiYG')
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
    plt.savefig(path + 'weights{}.eps'.format(ext))
    plt.savefig(path + 'weights{}.pdf'.format(ext))
    plt.show()
    plt.clf()

def plot_impulses_1(theta, W, N, L, B, dt, path, ext=None):

    def plot_impulse(model, parent, child, theta, W, L):
        plt.plot(np.arange(1, L + 1), np.dot(model.phi, theta[:, parent, child])[1:],
                 linewidth=0.5, color='C0', alpha=W)
        plt.xlim([1, L + 1])
        plt.yticks([])
        plt.xticks([])
        plt.axis('off')

    if ext is None:
        ext = ''

    model = DiscreteNetworkPoisson(N=N, L=L, B=B, dt=dt)

    N, M = W.shape
    w = np.max(W)
    for i in range(N):
        for j in range(M):
            plt.subplot(N, M, i * M + j + 1)
            plot_impulse(model, i, j, theta, W[i, j] / w, L)
    plt.savefig(path + 'impulse{}.eps'.format(ext))
    plt.savefig(path + 'impulse{}.pdf'.format(ext))
    plt.show()
    plt.clf()

def plot_impulses_2(theta, W, N, L, B, dt, path, ext=None, thresh=None):

    if thresh is None:
        thresh = np.inf

    def plot_impulse(model, parent, child, theta, W, L):
        plt.plot(np.arange(1, L + 1), np.dot(model.phi, theta[:, parent, child])[1:],
                 linewidth=0.5, color='C0', alpha=W)
        plt.xlim([1, L + 1])
        plt.yticks([])
        plt.xticks([])
        # plt.axis('off')

    if ext is None:
        ext = ''

    model = DiscreteNetworkPoisson(N=N, L=L, B=B, dt=dt)

    N, M = W.shape
    for i in range(N):
        for j in range(M):
            if W[i, j] > thresh:
                plt.subplot(N, M, i * M + j + 1)
                plot_impulse(model, i, j, theta, 1, L)
                if j == 0:
                    plt.ylabel(i)
                if i == M - 1:
                    plt.xlabel(j)
    plt.savefig(path + 'impulse{}.eps'.format(ext))
    plt.savefig(path + 'impulse{}.pdf'.format(ext))
    plt.show()
    plt.clf()

def calculate_averages(path, name, dates, burn):
    lambda0_medians = []
    W_medians = []
    theta_medians = []
    for date in dates:
        lambda0, W, theta = import_samples(path, name, date, burn)
        N, _ = lambda0.shape
        lambda0_medians.append(np.median(lambda0, axis=1).reshape((N, 1)))
        W_medians.append(np.median(W, axis=2).reshape((N, N, 1)))
        theta_medians.append(np.median(theta, axis=3).reshape((B, N, N, 1)))
    lambda0_avg = np.mean(np.concatenate(lambda0_medians, axis=1), axis=1)
    W_avg = np.mean(np.concatenate(W_medians, axis=2), axis=2)
    theta_avg = np.mean(np.concatenate(theta_medians, axis=3), axis=3)
    return lambda0_avg, W_avg, theta_avg

def calculate_estimates(path, name, dates, burn):
    lambda0_ = []
    W_ = []
    theta_ = []
    for date in dates:
        lambda0, W, theta = import_samples(path, name, date, burn)
        N, _ = lambda0.shape
        lambda0_.append(np.median(lambda0, axis=1).reshape((N, 1)))
        W_.append(np.median(W, axis=2).reshape((N, N, 1)))
        theta_.append(np.median(theta, axis=3).reshape((B, N, N, 1)))
    lambda0_ = np.concatenate(lambda0_, axis=1)
    W_ = np.concatenate(W_, axis=2)
    theta_ = np.concatenate(theta_, axis=3)
    return lambda0_, W_, theta_

def calculate_quantiles(path, name, dates, burn):
    lambda0_5 = []; lambda0_95 = [];
    W_5 = []; W_95 = [];
    theta_5 = []; theta_95 = [];
    for date in dates:
        lambda0, W, theta = import_samples(path, name, date, burn)
        N, _ = lambda0.shape
        lambda0_5.append(np.percentile(lambda0, 5, axis=1).reshape((N, 1)))
        lambda0_95.append(np.percentile(lambda0, 95, axis=1).reshape((N, 1)))
        W_5.append(np.percentile(W, 5, axis=2).reshape((N, N, 1)))
        W_95.append(np.percentile(W, 95, axis=2).reshape((N, N, 1)))
        theta_5.append(np.percentile(theta, 5, axis=3).reshape((B, N, N, 1)))
        theta_95.append(np.percentile(theta, 95, axis=3).reshape((B, N, N, 1)))
    lambda0_5 = np.concatenate(lambda0_5, axis=1); lambda0_95 = np.concatenate(lambda0_95, axis=1)
    W_5 = np.concatenate(W_5, axis=2); W_95 = np.concatenate(W_95, axis=2)
    theta_5 = np.concatenate(theta_5, axis=3); theta_95 = np.concatenate(theta_95, axis=3)
    return lambda0_5, lambda0_95, W_5, W_95, theta_5, theta_95

def plot_intensities(spikes, lambda0, W, theta, path):
    """Plot the intensity given event data and fitted parameters."""

    seconds = pd.date_range(start='01-02-2013 10:30:00', end='01-02-2013 14:59:59', freq='S')
    hours = pd.date_range(start='01-02-2013 10:30:00', end='01-02-2013 14:59:59', freq='H')

    def plot_intensity_pair(Y, pair, label):
        i, j = pair
        plt.fill_between(seconds, y1=0, y2=Y[:, i], alpha=0.5, color='C0')
        plt.fill_between(seconds, y1=0, y2=-Y[:, j], alpha=0.5, color='C3')
        plt.ylabel(label, fontsize=8)
        plt.yticks(fontsize=8)
        plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], fontsize=8)
        plt.xlim([seconds[0], seconds[-1]])
        # plt.legend(['Bids', 'Asks'], fontsize=8)

    # Make a model
    N, = lambda0.shape
    params = {'bias': lambda0, 'weights': W, 'impulse': theta}
    model = DiscreteNetworkPoisson(N=N, L=L, B=B, dt=dt, params=params)

    # Compute intensity
    T, _ = spikes.shape
    convolved = model.convolve(spikes)
    Lambda = model.calculate_intensity(spikes, convolved)

    # Plot
    plt.subplot(3, 2, 1)
    plot_intensity_pair(Lambda, (0, 3), label='ADDS, Level 0')
    plt.subplot(3, 2, 3)
    plot_intensity_pair(Lambda, (1, 4), label='ADDS, Level 1')
    plt.subplot(3, 2, 5)
    plot_intensity_pair(Lambda, (2, 5), label='ADDS, Level 2')
    plt.subplot(3, 2, 2)
    plot_intensity_pair(Lambda, (6, 7), label='CANCELS, Level 1')
    plt.subplot(3, 2, 4)
    plot_intensity_pair(Lambda, (8, 9), label='CANCELS, Level 2')
    plt.subplot(3, 2, 6)
    plot_intensity_pair(Lambda, (10, 11), label='EXECUTES')

    # Save figures
    plt.tight_layout()
    plt.savefig(path + 'intensity_{}.eps'.format(date))
    plt.savefig(path + 'intensity_{}.pdf'.format(date))
    plt.show()
    plt.clf()

def plot_series(estimate, lower, upper, color):
    L = len(estimate)
    plt.plot(np.arange(L), estimate, linewidth=0.5, color=color)
    plt.fill_between(np.arange(L), y1=lower, y2=upper, alpha=0.20, color=color)

def plot_bias_series(estimates, uppers, lowers, dates, path):

    def plot_pair(pair, estimates, uppers, lowers, label):
        i, j = pair
        bid, ask = estimates[i, :], estimates[j, :]
        bid_95, ask_95 = uppers[i, :], uppers[j, :]
        bid_5, ask_5 = lowers[i, :], lowers[j, :]
        plot_series(bid, bid_95, bid_5, color='C0')
        plot_series(ask, ask_95, ask_5, color='C3')
        plt.ylabel(label, fontsize=8)
        plt.xlim([0, len(bid) - 1])
        plt.yticks(fontsize=8)
        idx = np.arange(2, 18, 5).astype('int')
        labels = pd.to_datetime(np.array(dates)[idx])
        plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    plt.subplot(3, 2, 1)
    plot_pair((0, 3), estimates, uppers, lowers, label='ADDS, Level 0')
    plt.subplot(3, 2, 3)
    plot_pair((1, 4), estimates, uppers, lowers, label='ADDS, Level 1')
    plt.subplot(3, 2, 5)
    plot_pair((2, 5), estimates, uppers, lowers, label='ADDS, Level 2')
    plt.subplot(3, 2, 2)
    plot_pair((6, 7), estimates, uppers, lowers, label='CANCELS, Level 1')
    plt.subplot(3, 2, 4)
    plot_pair((8, 9), estimates, uppers, lowers, label='CANCELS, Level 2')
    plt.subplot(3, 2, 6)
    plot_pair((10, 11), estimates, uppers, lowers, label='EXECUTES')
    plt.tight_layout()
    plt.savefig(path + 'bias_series.eps')
    plt.savefig(path + 'bias_series.pdf')
    plt.show()
    plt.clf()

def plot_self_connections_series(estimates, uppers, lowers, dates, path):

    estimates = np.diagonal(estimates).transpose()
    uppers = np.diagonal(uppers).transpose()
    lowers = np.diagonal(lowers).transpose()

    def plot_pair(pair, estimates, uppers, lowers, label):
        i, j = pair
        bid, ask = estimates[i, :], estimates[j, :]
        bid_95, ask_95 = uppers[i, :], uppers[j, :]
        bid_5, ask_5 = lowers[i, :], lowers[j, :]
        plot_series(bid, bid_95, bid_5, color='C0')
        plot_series(ask, ask_95, ask_5, color='C3')
        plt.ylabel(label, fontsize=8)
        plt.xlim([0, len(bid) - 1])
        plt.yticks(fontsize=8)
        idx = np.arange(2, 18, 5).astype('int')
        labels = pd.to_datetime(np.array(dates)[idx])
        plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    plt.subplot(3, 2, 1)
    plot_pair((0, 3), estimates, uppers, lowers, label='ADDS, Level 0')
    plt.subplot(3, 2, 3)
    plot_pair((1, 4), estimates, uppers, lowers, label='ADDS, Level 1')
    plt.subplot(3, 2, 5)
    plot_pair((2, 5), estimates, uppers, lowers, label='ADDS, Level 2')
    plt.subplot(3, 2, 2)
    plot_pair((6, 7), estimates, uppers, lowers, label='CANCELS, Level 1')
    plt.subplot(3, 2, 4)
    plot_pair((8, 9), estimates, uppers, lowers, label='CANCELS, Level 2')
    plt.subplot(3, 2, 6)
    plot_pair((10, 11), estimates, uppers, lowers, label='EXECUTES')
    plt.tight_layout()
    plt.savefig(path + 'self_connections_series.eps')
    plt.savefig(path + 'self_connections_series.pdf')
    plt.show()
    plt.clf()

def plot_endogeneity_series(estimates, uppers, lowers, dates, path):

    def plot_pair(pair, estimates, uppers, lowers, label):
        i, j = pair
        bid, ask = estimates[i, :], estimates[j, :]
        bid_95, ask_95 = uppers[i, :], uppers[j, :]
        bid_5, ask_5 = lowers[i, :], lowers[j, :]
        plot_series(bid, bid_95, bid_5, color='C0')
        plot_series(ask, ask_95, ask_5, color='C3')
        plt.ylabel(label, fontsize=8)
        plt.xlim([0, len(bid) - 1])
        plt.yticks(fontsize=8)
        idx = np.arange(2, 18, 5).astype('int')
        labels = pd.to_datetime(np.array(dates)[idx])
        plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    plt.subplot(3, 2, 1)
    plot_pair((0, 3), estimates, uppers, lowers, label='ADDS, Level 0')
    plt.subplot(3, 2, 3)
    plot_pair((1, 4), estimates, uppers, lowers, label='ADDS, Level 1')
    plt.subplot(3, 2, 5)
    plot_pair((2, 5), estimates, uppers, lowers, label='ADDS, Level 2')
    plt.subplot(3, 2, 2)
    plot_pair((6, 7), estimates, uppers, lowers, label='CANCELS, Level 1')
    plt.subplot(3, 2, 4)
    plot_pair((8, 9), estimates, uppers, lowers, label='CANCELS, Level 2')
    plt.subplot(3, 2, 6)
    plot_pair((10, 11), estimates, uppers, lowers, label='EXECUTES')

    plt.tight_layout()
    plt.savefig(path + 'endogeneity_series.eps')
    plt.savefig(path + 'endogeneity_series.pdf')
    plt.show()
    plt.clf()


N = 12
L = 10
B = 10
dt = 1
burn = 2000
t0 = 34200 + 3600
tN = 57600 - 3600
T = tN - t0
name = 'GOOG'
# sample_path = '/Volumes/datasets/ITCH/samples/discrete.hdf5'
# event_path = '/Volumes/datasets/ITCH/events/events.hdf5'
# data_path = '/Volumes/datasets/ITCH/hdf5/'
sample_path = '/Users/colinswaney/Desktop/discrete_B={}_L={}.hdf5'.format(B, L)
event_path = '/Users/colinswaney/Desktop/events.hdf5'
img_path = '/Users/colinswaney/Desktop/Figures/discrete_B={}_L={}/{}/'.format(B, L, name)
with h5.File(sample_path, 'r') as hdf:
    dates = [date for date in hdf[name].keys()]
date = dates[0]
lambda0, W, theta = import_samples(sample_path, name, date, burn)


"""Order Book Analysis"""
# messages, books = import_data(data_path, name, dates)

# Calculate average volume

# Calculate average spread

# Calculate average prices


"""Daily Analysis"""
# Calculate homogeneous rates.
spikes = import_spikes(event_path, name, date, t0, tN)
lambda0_null = calculate_homogeneous(spikes, T)
lambda0_hat = calculate_inhomogeneous(lambda0)
endogeneous = (lambda0_null - lambda0_hat) / lambda0_null
exogeneous = 1 - endogeneous

# Plot posteriors for lambda0.
plot_bias(lambda0, path=img_path)

# Plot posteriors for W (diagonal only).
plot_self_connections(W, path=img_path)

# Plot medians of posteriors for W.
W_hat = np.median(W, axis=2)
plot_weights(W_hat, cmap='Blues', path=img_path)

# Plot normalized medians of posteriors for W.
W_hat_normed = W_hat / lambda0_hat
plot_weights(np.log(W_hat_normed), cmap='RdBu', path=img_path, ext='_normed')

# Plot median impulse response.
theta_hat = np.median(theta, axis=3)
plot_impulses_2(theta_hat, np.log(W_hat_normed), N, L, B, dt, path=img_path, thresh=np.log(1))

# Plot the fitted intensity
# plot_intensities(spikes, lambda0_hat, W_hat, theta_hat, path=img_path)


"""Monthly Analysis"""
averages = calculate_averages(sample_path, name, dates, burn)
lambda0_avg, W_avg, theta_avg = averages

# Plot medians of posteriors for W.
plot_weights(W_avg, cmap='Blues', ext='_avg', path=img_path)

# Plot normalized medians of posteriors for W.
W_avg_normed = W_avg / lambda0_avg
plot_weights(np.log(W_avg_normed), cmap='RdBu', ext='_avg_normed', path=img_path)

# Plot median impulse response.
plot_impulses_2(theta_avg, np.log(W_avg_normed), N, L, B, dt, path=img_path, ext='_avg', thresh=np.log(2))


"""Time Series Analysis"""
estimates = calculate_estimates(sample_path, name, dates, burn)
lambda0_50, W_50, theta_50 = estimates
quantiles = calculate_quantiles(sample_path, name, dates, burn)
lambda0_5, lambda0_95, W_5, W_95, theta_5, theta_95 = quantiles

# Time series of bias
plot_bias_series(lambda0_50, lambda0_5, lambda0_95, dates, path=img_path)

# Time series of self-connections
plot_self_connections_series(W_50, W_5, W_95, dates, path=img_path)

# Time series endogeneity
endog_50 = np.log(np.diagonal(W_50).transpose() / lambda0_50)
endog_5 = np.log(np.diagonal(W_5).transpose() / lambda0_5)
endog_95 = np.log(np.diagonal(W_95).transpose() / lambda0_95)
plot_endogeneity_series(endog_50, endog_5, endog_95, dates, path=img_path)

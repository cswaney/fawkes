from fawkes.models import NetworkPoisson
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pandas as pd
import os
import sys

def import_data(path, name, dates):
    mkt_open = 34200
    mkt_close = 57600
    msg_columns = ['sec', 'nano', 'type', 'event', 'side', 'price', 'shares', 'refno', 'newrefno']
    book_columns = ['sec', 'nano']
    num_cols = 10
    book_columns += ['bidprc_{}'.format(i + 1) for i in range(num_cols)]
    book_columns += ['askprc_{}'.format(i + 1) for i in range(num_cols)]
    book_columns += ['bidvol_{}'.format(i + 1) for i in range(num_cols)]
    book_columns += ['askvol_{}'.format(i + 1) for i in range(num_cols)]
    messages_list = []
    books_list = []
    for date in dates:
        print("Loading data for date {}".format(date))
        with h5.File(path + 'itch-{}.hdf5'.format(date), 'r') as hdf:
            try:
                messages = hdf['/messages/{}'.format(name)][:]
                books = hdf['/orderbooks/{}'.format(name)][:]
                messages = pd.DataFrame(messages, columns=msg_columns)
                books = pd.DataFrame(books, columns=book_columns)
                books = books.loc[:, ['sec', 'nano', 'bidprc_1', 'askprc_1', 'bidvol_1', 'askvol_1']]
                messages = messages[ (messages['sec'] >= mkt_open) & (messages['sec'] < mkt_close)]
                books = books[ (books['sec'] >= mkt_open) & (books['sec'] < mkt_close)]
                messages['date'] = date
                books['date'] = date
                messages_list.append(messages)
                books_list.append(books)
            except:
                print("Unable to find data. Skipping.")
                return None
    messages = pd.concat(messages_list, axis=0)
    books = pd.concat(books_list, axis=0)
    books['price'] = (books['bidprc_1'] + books['askprc_1']) / (2 * 10000)
    return messages, books

def compute_daily_prices(books):
    return books.groupby('date').mean()['price']

def compute_daily_volumes(messages):
    volumes = messages.groupby(['type', 'date']).sum()['shares']
    adds = volumes[2]
    executes = volumes[5]
    deletes = volumes[4]
    return adds, executes, deletes

def import_samples(path, name, date, burn):
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

# TODO
def print_table(data, lower, upper, decimals=2, sep=" "):
    """Create a table from a two-dimensional array."""
    N, M = data.shape
    for i in range(2 * N):
        for j in range(M):
            if i % 2 == 0:
                d = data[int(i / 2), j]
                if j < M - 1:
                    print('{:.{}f}{}&'.format(d, decimals, sep), end=sep)
                else:
                    print('{:.{}f}'.format(d, decimals), end='\n')
            else:
                l = lower[int(i / 2), j]
                u = upper[int(i / 2), j]
                if j < M - 1:
                    print('({:.{}f} {:.{}f}){}&'.format(l, 2, u, 2, sep), end=sep)
                else:
                    print('({:.{}f}, {:.{}f})'.format(l, 2, u, 2), end='\n')

def construct_table(lambda0, W, mu, tau):
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
    plot_pair(lambda0[0,:], lambda0[2,:], 'Add message', [0.01, 0.03])
    plt.subplot(3, 2, 2)
    plot_pair(lambda0[1,:], lambda0[3,:], 'Add message (+)', [0, 0.015])
    plt.subplot(3, 2, 3)
    plot_pair(lambda0[4,:], lambda0[6,:], 'Execute message', [0, 0.01])
    plt.subplot(3, 2, 4)
    plot_pair(lambda0[5,:], lambda0[7,:], 'Execute message (-)', [0, 0.01])
    plt.subplot(3, 2, 5)
    plot_pair(lambda0[8,:], lambda0[10,:], 'Delete message', [0, 0.005])
    plt.subplot(3, 2, 6)
    plot_pair(lambda0[9,:], lambda0[11,:], 'Delete message (-)', [0.015, 0.025])
    plt.tight_layout()
    plt.savefig(path + 'bias_{}.pdf'.format(name))
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
    plot_pair(diagonal[0,:], diagonal[2,:], 'Add message', [0.01, 0.03])
    plt.subplot(3, 2, 2)
    plot_pair(diagonal[1,:], diagonal[3,:], 'Add message (+)', [0, 0.015])
    plt.subplot(3, 2, 3)
    plot_pair(diagonal[4,:], diagonal[6,:], 'Execute message', [0, 0.01])
    plt.subplot(3, 2, 4)
    plot_pair(diagonal[5,:], diagonal[7,:], 'Execute message (-)', [0, 0.01])
    plt.subplot(3, 2, 5)
    plot_pair(diagonal[8,:], diagonal[10,:], 'Delete message', [0, 0.005])
    plt.subplot(3, 2, 6)
    plot_pair(diagonal[9,:], diagonal[11,:], 'Delete message (-)', [0.015, 0.025])
    plt.tight_layout()
    plt.savefig(path + 'diagonal_{}.pdf'.format(name))
    plt.clf()

def plot_weights(W, vrange, path, ext=None, cmap=None):

    if ext is None:
        ext = ''

    vmin, vmax = vrange
    plt.imshow(W, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel("Child nodes", labelpad=20, fontsize=8)
    tick_labels = ['Add (bid)',
                   'Add (bid+)',
                   'Add (ask)',
                   'Add (ask-)',
                   'Execute (bid)',
                   'Execute (bid-)',
                   'Execute (ask)',
                   'Execute (ask+)',
                   'Delete (bid)',
                   'Delete (bid-)',
                   'Delete (ask-)',
                   'Delete (ask+)']
    plt.xticks(np.arange(0, 12), tick_labels, rotation=90, fontsize=8)
    plt.ylabel("Parent nodes", labelpad=20, fontsize=8)
    plt.yticks(np.arange(0, 12), tick_labels, fontsize=8)
    plt.tight_layout()
    plt.savefig(path + 'weights{}_{}.pdf'.format(ext, name))
    plt.clf()

def logit_normal(dt, mu, tau, dt_max):
    assert (dt < dt_max).all(), "dt must be less than dt_max."
    assert (dt > 0).all(), "dt must be greater than zero."
    Z = dt * (dt_max - dt) / dt_max * (tau / (2 * np.pi)) ** (-0.5)
    x = dt / dt_max
    s = np.log(x / (1 - x))
    return (1 / Z) * np.exp( -tau / 2 * (s - mu) ** 2 )

def plot_impulses(W, mu, tau, dt_max, x_min, x_max, path, ext=None, thresh=None):

    if ext is None:
        ext = ''

    if thresh is not None:

        def plot_impulse(parent, child, mu, tau, dt_max):
            assert x_max < dt_max, "xmax should be less than dt_max"
            dt = np.linspace(x_min, x_max, 100)
            values = logit_normal(dt, mu[parent, child], tau[parent, child], dt_max)
            plt.plot(dt, values, linewidth=0.5, color='C0')
            plt.xlim([x_min, x_max])
            plt.yticks([])
            plt.xticks([])
            plt.axis('off')

        N, M = W.shape
        for i in range(N):
            for j in range(M):
                if W[i, j] > thresh:
                    plt.subplot(N, M, i * M + j + 1)
                    plot_impulse(i, j, mu, tau, dt_max)
        plt.savefig(path + 'impulses{}_{}.png'.format(ext, name))
        plt.show()
        plt.clf()

    elif thresh is None:

        A = W

        def plot_impulse(parent, child, mu, tau, dt_max, alpha):
            assert x_max < dt_max, "xmax should be less than dt_max"
            dt = np.linspace(x_min, x_max, 100)
            values = logit_normal(dt, mu[parent, child], tau[parent, child], dt_max)
            plt.plot(dt, values, linewidth=0.5, color='C0', alpha=np.minimum(1, alpha))
            plt.xlim([x_min, x_max])
            plt.yticks([])
            plt.xticks([])
            plt.axis('off')

        N, M = W.shape
        for i in range(N):
            for j in range(M):
                plt.subplot(N, M, i * M + j + 1)
                plot_impulse(i, j, mu, tau, dt_max, alpha=A[i, j])
        plt.savefig(path + 'impulses{}_{}.png'.format(ext, name))
        plt.show()
        plt.clf()

def calculate_averages(path, name, dates, burn):
    lambda0_medians = []
    W_medians = []
    mu_medians = []
    tau_medians = []
    for date in dates:
        lambda0, W, mu, tau = import_samples(path, name, date, burn)
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

def calculate_estimates(path, name, dates, burn):
    lambda0_ = []
    W_ = []
    mu_ = []
    tau_ = []
    for date in dates:
        lambda0, W, mu, tau = import_samples(path, name, date, burn)
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

def calculate_quantiles(path, name, dates, burn):
    lambda0_5 = []; lambda0_95 = [];
    W_5 = []; W_95 = [];
    mu_5 = []; mu_95 = [];
    tau_5 = []; tau_95 = [];
    for date in dates:
        lambda0, W, mu, tau = import_samples(path, name, date, burn)
        N, _ = lambda0.shape
        lambda0_5.append(np.percentile(lambda0, 5, axis=1).reshape((N, 1)))
        lambda0_95.append(np.percentile(lambda0, 95, axis=1).reshape((N, 1)))
        W_5.append(np.percentile(W, 5, axis=2).reshape((N, N, 1)))
        W_95.append(np.percentile(W, 95, axis=2).reshape((N, N, 1)))
        mu_5.append(np.percentile(mu, 5, axis=2).reshape((N, N, 1)))
        mu_95.append(np.percentile(mu, 95, axis=2).reshape((N, N, 1)))
        tau_5.append(np.percentile(tau, 5, axis=2).reshape((N, N, 1)))
        tau_95.append(np.percentile(tau, 95, axis=2).reshape((N, N, 1)))
    lambda0_5 = np.concatenate(lambda0_5, axis=1); lambda0_95 = np.concatenate(lambda0_95, axis=1)
    W_5 = np.concatenate(W_5, axis=2); W_95 = np.concatenate(W_95, axis=2)
    mu_5 = np.concatenate(mu_5, axis=2); mu_95 = np.concatenate(mu_95, axis=2)
    tau_5 = np.concatenate(tau_5, axis=2); tau_95 = np.concatenate(tau_95, axis=2)
    return lambda0_5, lambda0_95, W_5, W_95, mu_5, mu_95, tau_5, tau_95

# TODO
def plot_intensities(events, lambda0, W, mu, tau, path):
    """Plot the intensity given event data and fitted parameters.

    lambda0: N
    W, mu, tau: N x N

    """

    seconds = pd.date_range(start='01-02-2013 10:30:00', end='01-02-2013 14:59:59', freq='S')
    hours = pd.date_range(start='01-02-2013 10:30:00', end='01-02-2013 14:59:59', freq='H')

    def plot_intensity_pair(Y, pair, label):
        """Y is a T x N matrix of intensities."""
        i, j = pair
        plt.fill_between(seconds, y1=0, y2=Y[:, i], alpha=0.5, color='C0')
        plt.fill_between(seconds, y1=0, y2=Y[:, j], alpha=0.5, color='C0')
        plt.ylabel(label, fontsize=8)
        plt.yticks(fontsize=8)
        plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], fontsize=8)
        plt.xlim([seconds[0], seconds[-1]])

    # Make a model
    N, = lambda0.shape
    params = {'lamb': lambda0, 'weights': W, 'mu': mu, 'tau': tau}
    model = NetworkPoisson(N=N, dt_max=dt_max, params=params)

    # Compute intensity
    grid = np.linspace(0, T, T + 1)  # dt = 1
    Lambda = model.compute_intensity(events, grid)

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
    # plt.savefig(path + 'intensity_{}.eps'.format(date))
    # plt.savefig(path + 'intensity_{}.pdf'.format(date))
    plt.show()
    plt.clf()

def plot_series(estimate, lower, upper, color):
    L = len(estimate)
    plt.plot(np.arange(L), estimate, linewidth=0.5, color=color)
    plt.fill_between(np.arange(L), y1=lower, y2=upper, alpha=0.20, color=color)

def plot_bias_series(estimates, upper, lower, dates, path):

    # ADD, Level 0
    ax = plt.subplot(3, 2, 1)
    plot_series(estimates[0,:], upper[0,:], lower[0,:], color='C0')
    plot_series(estimates[3,:], upper[3,:], lower[3,:], color='C3')
    plt.ylabel('ADDS, Level 0', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # ADD, Level 1
    ax = plt.subplot(3, 2, 3)
    plot_series(estimates[1,:], upper[1,:], lower[1,:], color='C0')
    plot_series(estimates[4,:], upper[4,:], lower[4,:], color='C3')
    plt.ylabel('ADDS, Level 1', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # ADD, Level 2
    ax = plt.subplot(3, 2, 5)
    plot_series(estimates[2,:], upper[2,:], lower[2,:], color='C0')
    plot_series(estimates[5,:], upper[5,:], lower[5,:], color='C3')
    plt.ylabel('ADDS, Level 2', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # DELETES, Level 1
    ax = plt.subplot(3, 2, 2)
    plot_series(estimates[6,:], upper[6,:], lower[6,:], color='C0')
    plot_series(estimates[8,:], upper[8,:], lower[8,:], color='C3')
    plt.ylabel('DELETES, Level 1', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # DELETES, Level 2
    ax = plt.subplot(3, 2, 4)
    plot_series(estimates[7,:], upper[7,:], lower[7,:], color='C0')
    plot_series(estimates[9,:], upper[9,:], lower[9,:], color='C3')
    plt.ylabel('DELETES, Level 2', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # EXECUTES
    ax = plt.subplot(3, 2, 6)
    plot_series(estimates[10,:], upper[10,:], lower[10,:], color='C0')
    plot_series(estimates[11,:], upper[11,:], lower[11,:], color='C3')
    plt.ylabel('EXECUTES', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    plt.tight_layout()
    # plt.savefig(path + 'bias_series.eps')
    # plt.savefig(path + 'bias_series.pdf')
    plt.show()
    plt.clf()

def plot_self_connections_series(estimates, upper, lower, dates, path):

    estimates = np.diagonal(estimates).transpose()
    upper = np.diagonal(upper).transpose()
    lower = np.diagonal(lower).transpose()

    # ADD, Level 0
    ax = plt.subplot(3, 2, 1)
    plot_series(estimates[0,:], upper[0,:], lower[0,:], color='C0')
    plot_series(estimates[3,:], upper[3,:], lower[3,:], color='C3')
    plt.ylabel('ADDS, Level 0', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    # plt.ylim(ymax=np.maximum(np.max(upper[0,:]), np.max(upper[3,:])))
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # ADD, Level 1
    ax = plt.subplot(3, 2, 3)
    plot_series(estimates[1,:], upper[1,:], lower[1,:], color='C0')
    plot_series(estimates[4,:], upper[4,:], lower[4,:], color='C3')
    plt.ylabel('ADDS, Level 1', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    # plt.ylim(ymax=np.maximum(np.max(upper[1,:]), np.max(upper[4,:])))
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # ADD, Level 2
    ax = plt.subplot(3, 2, 5)
    plot_series(estimates[2,:], upper[2,:], lower[2,:], color='C0')
    plot_series(estimates[5,:], upper[5,:], lower[5,:], color='C3')
    plt.ylabel('ADDS, Level 2', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    # plt.ylim(ymax=np.maximum(np.max(upper[2,:]), np.max(upper[5,:])))
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # DELETES, Level 1
    ax = plt.subplot(3, 2, 2)
    plot_series(estimates[6,:], upper[6,:], lower[6,:], color='C0')
    plot_series(estimates[8,:], upper[8,:], lower[8,:], color='C3')
    plt.ylabel('DELETES, Level 1', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    # plt.ylim(ymax=np.maximum(np.max(upper[6,:]), np.max(upper[8,:])))
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # DELETES, Level 2
    ax = plt.subplot(3, 2, 4)
    plot_series(estimates[7,:], upper[7,:], lower[7,:], color='C0')
    plot_series(estimates[9,:], upper[9,:], lower[9,:], color='C3')
    plt.ylabel('DELETES, Level 2', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    # plt.ylim(ymax=np.maximum(np.max(upper[7,:]), np.max(upper[9,:])))
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # EXECUTES
    ax = plt.subplot(3, 2, 6)
    plot_series(estimates[10,:], upper[10,:], lower[10,:], color='C0')
    plot_series(estimates[11,:], upper[11,:], lower[11,:], color='C3')
    plt.ylabel('EXECUTES', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    # plt.ylim(ymax=np.maximum(np.max(upper[10,:]), np.max(upper[11,:])))
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    plt.tight_layout()
    # plt.savefig(path + 'diagonal_series.eps')
    # plt.savefig(path + 'diagonal_series.pdf')
    plt.show()
    plt.clf()

def plot_endogeneity_series(estimates, upper, lower, dates, path):

    # ADD, Level 0
    ax = plt.subplot(3, 2, 1)
    plot_series(estimates[0,:], upper[0,:], lower[0,:], color='C0')
    plot_series(estimates[3,:], upper[3,:], lower[3,:], color='C3')
    plt.ylabel('ADDS, Level 0', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # ADD, Level 1
    ax = plt.subplot(3, 2, 3)
    plot_series(estimates[1,:], upper[1,:], lower[1,:], color='C0')
    plot_series(estimates[4,:], upper[4,:], lower[4,:], color='C3')
    plt.ylabel('ADDS, Level 1', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # ADD, Level 2
    ax = plt.subplot(3, 2, 5)
    plot_series(estimates[2,:], upper[2,:], lower[2,:], color='C0')
    plot_series(estimates[5,:], upper[5,:], lower[5,:], color='C3')
    plt.ylabel('ADDS, Level 2', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # DELETES, Level 1
    ax = plt.subplot(3, 2, 2)
    plot_series(estimates[6,:], upper[6,:], lower[6,:], color='C0')
    plot_series(estimates[8,:], upper[8,:], lower[8,:], color='C3')
    plt.ylabel('DELETES, Level 1', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # DELETES, Level 2
    ax = plt.subplot(3, 2, 4)
    plot_series(estimates[7,:], upper[7,:], lower[7,:], color='C0')
    plot_series(estimates[9,:], upper[9,:], lower[9,:], color='C3')
    plt.ylabel('DELETES, Level 2', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    # EXECUTES
    ax = plt.subplot(3, 2, 6)
    plot_series(estimates[10,:], upper[10,:], lower[10,:], color='C0')
    plot_series(estimates[11,:], upper[11,:], lower[11,:], color='C3')
    plt.ylabel('EXECUTES', fontsize=8)
    plt.xlim([0, estimates.shape[1] - 1])
    plt.yticks(fontsize=8)
    idx = np.arange(2, 18, 5).astype('int')
    labels = pd.to_datetime(np.array(dates)[idx])
    plt.xticks(idx, [l.strftime('%m/%d/%Y') for l in labels], fontsize=8)

    plt.tight_layout()
    # plt.savefig(path + 'endog_series.eps')
    # plt.savefig(path + 'endog_series.pdf')
    plt.show()
    plt.clf()

# TODO
def calculate_frequencies(path, name, dates):
    events = []
    for date in dates:
        print("date={}".format(date))
        events.append(import_events(event_path, name, date, t0, tN))
    events = pd.concat(events, axis=0)
    counts = pd.value_counts(events['event']).sort_index()
    seconds = len(dates) * (tN - t0)
    return counts / seconds

def calculate_bounds(lowers, uppers):
    out = []
    i = 0
    for l,u in zip(lowers, uppers):
        print("{}    ({:.{}f}, {:.{}f})".format(i, l, 3, u, 3))
        i += 1

N = 12
dt_max = 5
burn = 2000
t0 = 34200 + 3600
tN = 57600 - 3600
T = tN - t0
name = 'PFE'
sample_path = '/Users/colinswaney/Desktop/continuous_dt_max={}.hdf5'.format(dt_max)
event_path = '/Users/colinswaney/Desktop/events.hdf5'
img_path = '/Users/colinswaney/Desktop/Figures/continuous_dt_max={}/{}/'.format(dt_max, name)
with h5.File(sample_path, 'r') as hdf:
    dates = [date for date in hdf[name].keys()]
date = dates[0]
lambda0, W, mu, tau = import_samples(sample_path, name, date, burn)


"""Order Book Analysis"""
# messages, books = import_data(data_path, name, dates)

# Calculate average volume

# Calculate average spread

# Calculate average prices


"""Daily Analysis"""
# Calculate homogeneous rates.
events = import_events(event_path, name, date, t0, tN)
lambda0_null = calculate_homogeneous(events, N, T)
lambda0_hat = calculate_inhomogeneous(lambda0)
endogeneous = (lambda0_null - lambda0_hat) / lambda0_null
exogeneous = 1 - endogeneous

# Plot posteriors for lambda0.
plot_bias(lambda0, path=img_path)

# Plot posteriors for W (diagonal only).
plot_self_connections(W, path=img_path)

# Plot medians of posteriors for W.
# W_hat = np.median(W, axis=2)
# plot_weights(W_hat, (0, 1), cmap='Blues', path=img_path)

# Plot normalized medians of posteriors for W.
# W_hat_normed = W_hat / lambda0_hat
# plot_weights(np.log(W_hat_normed), (-5, 5), cmap='RdBu', path=img_path, ext='_normed')

# Plot median impulse response.
# mu_hat = np.median(mu, axis=2)
# tau_hat = np.median(tau, axis=2)
# plot_impulses_1(np.ones((N, N)), mu_hat, tau_hat, dt_max, x_max=dt_max - 0.1, path=img_path)

# Plot the fitted intensity
# plot_intensities(events, lambda0_hat, W_hat, mu_hat, tau_hat, path=img_path)


"""Monthly Analysis"""
averages = calculate_averages(sample_path, name, dates, burn)
lambda0_avg, W_avg, mu_avg, tau_avg = averages

# Plot medians of posteriors for W.
plot_weights(W_avg, (0, 1), cmap='Blues', ext='_matrix', path=img_path)

# Plot normalized medians of posteriors for W.
W_avg_normed = W_avg / lambda0_avg
plot_weights(np.log(W_avg_normed), (-5, 5), cmap='RdBu', ext='_matrix_normed', path=img_path)

# Plot median impulse response.
plot_impulses(W_avg, mu_avg, tau_avg, dt_max, x_min=0.01, x_max=0.1, path=img_path)
# plot_impulses(np.log(W_avg), mu_avg, tau_avg, dt_max, x_min=0.01, x_max=0.1, ext='_normed', path=img_path, thresh=0.1)

# Calculate frequency of events


"""Time Series Analysis"""
estimates = calculate_estimates(sample_path, name, dates, burn)
lambda0_50, W_50, mu_50, tau_50 = estimates
quantiles = calculate_quantiles(sample_path, name, dates, burn)
lambda0_5, lambda0_95, W_5, W_95, mu_5, mu_95, tau_5, tau_95 = quantiles

# Time series of bias
plot_bias_series(lambda0_50, lambda0_5, lambda0_95, dates, path=img_path)

# Time series of self-connections
plot_self_connections_series(W_50, W_5, W_95, dates, path=img_path)

# Time series endogeneity
# endog_50 = np.log(np.diagonal(W_50).transpose() / lambda0_50)
# endog_5 = np.log(np.diagonal(W_5).transpose() / lambda0_5)
# endog_95 = np.log(np.diagonal(W_95).transpose() / lambda0_95)
# plot_endogeneity_series(endog_50, endog_5, endog_95, dates, path=img_path)

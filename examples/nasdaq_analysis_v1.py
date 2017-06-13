from fawkes.models import NetworkPoisson
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import os

name = 'GOOG'
date = '070113'
dt_max = 60
N = 12
burn = 2000
with h5.File('/Volumes/datasets/ITCH/samples/samples_dt_max={}.hdf5'.format(dt_max), 'r') as hdf:
    dates = [date for date in hdf[name].keys()]

def import_data(name, date, dt_max, burn):
    print("Import data for name {} and date {}.".format(name, date))
    with h5.File('/Volumes/datasets/ITCH/samples/samples_dt_max={}.hdf5'.format(dt_max), 'r') as hdf:
        lambda0 = hdf['/{}/{}/lambda0'.format(name, date)][:]
        W = hdf['/{}/{}/W'.format(name, date)][:]
        mu = hdf['/{}/{}/mu'.format(name, date)][:]
        tau = hdf['/{}/{}/tau'.format(name, date)][:]
    return lambda0[:, burn:], W[:, :, burn:], mu[:, :, burn:], tau[:, :, burn:]
lambda0, W, mu, tau = import_data(name, date, dt_max, burn)

"""Plot the posterior distribution of all of the bias vector elements."""

def bias_plot(lambda0, burn=0):

    means = np.mean(lambda0[:, burn:], axis=1)

    # Level 0, BID
    ax = plt.subplot(3, 2, 1)
    cnts, bins, _ = plt.hist(lambda0[0, burn:], bins=30, alpha=0.50)
    plt.vlines(means[0], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 0', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 0, ASK
    plt.subplot(3, 2, 1)
    cnts, bins, _ = plt.hist(lambda0[3, burn:], bins=30, alpha=0.50)
    plt.vlines(means[3], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, BID
    plt.subplot(3, 2, 2)
    cnts, bins, _ = plt.hist(lambda0[1, burn:], bins=30, alpha=0.50)
    plt.vlines(means[1], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3, 2, 2)
    cnts, bins, _ = plt.hist(lambda0[4, burn:], bins=30, alpha=0.50)
    plt.vlines(means[4], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, BID
    plt.subplot(3, 2, 3)
    cnts, bins, _ = plt.hist(lambda0[2, burn:], bins=30, alpha=0.50)
    plt.vlines(means[2], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 2', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, ASK
    plt.subplot(3, 2, 3)
    cnts, bins, _ = plt.hist(lambda0[5, burn:], bins=30, alpha=0.50)
    plt.vlines(means[5], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_lambda0_add.pdf')

    #### DELETES ####

    # Level 1, BID
    plt.subplot(3, 2, 4)
    cnts, bins, _ = plt.hist(lambda0[6, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[6], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 5])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3, 2, 4)
    cnts, bins, _ = plt.hist(lambda0[8, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[8], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    # plt.ylim([0, np.max(cnts) + 1])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        # labelbottom='off'
        labelbottom='on')
    plt.xticks(fontsize=8)
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, BID
    plt.subplot(3, 2, 5)
    cnts, bins, _ = plt.hist(lambda0[7, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[7], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 2', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 5])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, ASK
    plt.subplot(3, 2, 5)
    cnts, bins, _ = plt.hist(lambda0[9, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[9], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    # plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_lambda0_delete.pdf')


    #### EXECUTES ####

    # Level 1, BID
    plt.subplot(3, 2, 6)
    cnts, bins, _ = plt.hist(lambda0[10, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[10], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3, 2, 6)
    cnts, bins, _ = plt.hist(lambda0[11, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[11], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    # plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    plt.show()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_lambda0_execute.pdf')

    plt.show()
    plt.clf()
bias_plot(lambda0)

def bias_series(name, date, dt_max, burn=0):
    medians = []
    lower = []
    upper = []
    for date in dates:
        lambda0, W, mu, tau = import_data(name, date, dt_max, burn)
        medians.append(np.percentile(lambda0, 0.50, axis=1).reshape((N, 1)))
        lower.append(np.percentile(lambda0, 0.05, axis=1).reshape((N, 1)))
        upper.append(np.percentile(lambda0, 0.95, axis=1).reshape((N, 1)))
    medians = np.concatenate(medians, axis=1).transpose()
    lower = np.concatenate(lower, axis=1).transpose()
    upper = np.concatenate(upper, axis=1).transpose()
    L, _ = medians.shape
    plt.plot(medians[:, 0], linewidth=0.5)  # add 0
    plt.fill_between(x=np.arange(L), y1=lower[:, 0].reshape(L), y2=upper[:, 0].reshape(L), alpha=0.20)
    plt.plot(medians[:, 1], linewidth=0.5)  # add 0
    plt.fill_between(x=np.arange(L), y1=lower[:, 1].reshape(L), y2=upper[:, 1].reshape(L), alpha=0.20)
    plt.show()
    # plt.plot(medians[2:4, :].transpose())  # add 1
    # plt.plot(medians[4:6, :].transpose())  # add 2
    # plt.plot(medians[6:8, :].transpose())  # cancel 1
    # plt.plot(medians[8:10, :].transpose())  # cancel 2
    # plt.plot(medians[10:12, :].transpose())  # execute
    plt.clf()
bias_series(name, date, dt_max, burn)


"""
First, plot MAP estimates of the connection matrix. Next, plot the posterior
distribution of the diagonal elements of the connection matrix (i.e. the
self-connections).

Third, for each date, plot the MAP estimates of trade self-connections.
"""

def weights_plot(W, name, date, dt_max, burn=0):
    W = W[:, :, burn:]
    plt.imshow(W.mean(axis=2), cmap='Blues', vmin=0, vmax=0.5)
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
    # plt.savefig('/Users/colinswaney/Desktop/Figures/posterior_W_{}_{}_dt_max={}.pdf'.format(name, date, dt_max))
    plt.show()
    plt.clf()
weights_plot(W, name, date, dt_max)

def self_connections_plot(W, name, date, dt_max, burn=0):

    diagonal = np.diagonal(W)[burn:, :]
    means = np.mean(diagonal, axis=0)

    # Level 0, BID
    plt.subplot(3, 2, 1)
    cnts, bins, _ = plt.hist(diagonal[:, 0], bins=30, alpha=0.50)
    plt.vlines(means[0], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 0', fontsize=8)
    # plt.xlim([0.10, 0.70])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 0, ASK
    plt.subplot(3, 2, 1)
    cnts, bins, _ = plt.hist(diagonal[:, 3], bins=30, alpha=0.50)
    plt.vlines(means[3], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0.10, 0.70])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, BID
    plt.subplot(3, 2, 2)
    cnts, bins, _ = plt.hist(diagonal[:, 1], bins=30, alpha=0.50)
    plt.vlines(means[1], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0.10, 0.70])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3, 2, 2)
    cnts, bins, _ = plt.hist(diagonal[:, 4], bins=30, alpha=0.50)
    plt.vlines(means[4], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0.10, 0.70])
    # plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, BID
    plt.subplot(3, 2, 3)
    cnts, bins, _ = plt.hist(diagonal[:, 2], bins=30, alpha=0.50)
    plt.vlines(means[2], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 2', fontsize=8)
    # plt.xlim([0.10, 0.70])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, ASK
    plt.subplot(3, 2, 3)
    cnts, bins, _ = plt.hist(diagonal[:, 5], bins=30, alpha=0.50)
    plt.vlines(means[5], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0.10, 0.70])
    # plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_W_add.pdf')
    # plt.clf()

    #### DELETES ####

    # Level 1, BID
    plt.subplot(3, 2, 4)
    cnts, bins, _ = plt.hist(diagonal[:, 6], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[6], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0.1, 0.6])
    plt.ylim([0, np.max(cnts) + 5])
    # plt.xticks(np.arange(0.1, 0.7, .1), np.arange(0.1, 0.7, .1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3, 2, 4)
    cnts, bins, _ = plt.hist(diagonal[:, 8], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[8], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0.1, 0.6])
    # plt.xticks(np.arange(0.1, 0.7, .1), np.arange(0.1, 0.7, .1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, BID
    plt.subplot(3, 2, 5)
    cnts, bins, _ = plt.hist(diagonal[:, 7], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[7], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 2', fontsize=8)
    # plt.xlim([0.1, 0.6])
    plt.ylim([0, np.max(cnts) + 5])
    # plt.xticks(np.arange(0.1, 0.7, .1), np.arange(0.1, 0.7, .1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, ASK
    plt.subplot(3, 2, 5)
    cnts, bins, _ = plt.hist(diagonal[:, 9], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[9], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0.1, 0.6])
    # plt.xticks(np.arange(0.1, 0.7, .1), np.arange(0.1, 0.7, .1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_W_delete.pdf')
    # plt.clf()

    #### EXECUTES ####

    # Level 1, BID
    plt.subplot(3,2,6)
    cnts, bins, _ = plt.hist(diagonal[:, 10], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[10], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 5])
    # plt.xticks(np.arange(0.0, 0.35, 0.05), np.arange(0.0, 0.35, 0.05), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3,2,6)
    cnts, bins, _ = plt.hist(diagonal[:, 11], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[11], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    # plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.35, 0.05), np.arange(0.0, 0.35, 0.05), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_W_execute.pdf')
    plt.show()
    plt.clf()
self_connections_plot(W, name, date, dt_max)

def weights_series(name, date, dt_max, burn=0):
    medians = []
    lower = []
    upper = []
    for date in dates:
        lambda0, W, mu, tau = import_data(name, date, dt_max, burn)
        # weights_plot(W, name, date, dt_max)
        medians.append(np.percentile(np.diagonal(W), 0.50, axis=0).reshape((N, 1)))
        lower.append(np.percentile(np.diagonal(W), 0.05, axis=0).reshape((N, 1)))
        upper.append(np.percentile(np.diagonal(W), 0.95, axis=0).reshape((N, 1)))
    medians = np.concatenate(medians, axis=1).transpose()
    lower = np.concatenate(lower, axis=1).transpose()
    upper = np.concatenate(upper, axis=1).transpose()
    L, _ = medians.shape
    plt.plot(medians[:, 0])  # add 0
    plt.fill_between(x=np.arange(L), y1=lower[:, 0].reshape(L), y2=upper[:, 0].reshape(L), alpha=0.10)
    plt.plot(medians[:, 1])  # add 0
    plt.fill_between(x=np.arange(L), y1=lower[:, 1].reshape(L), y2=upper[:, 1].reshape(L), alpha=0.10)
    plt.show()
    # plt.plot(medians[2:4, :].transpose())  # add 1
    # plt.plot(medians[4:6, :].transpose())  # add 2
    # plt.plot(medians[6:8, :].transpose())  # cancel 1
    # plt.plot(medians[8:10, :].transpose())  # cancel 2
    # plt.plot(medians[10:12, :].transpose())  # execute
    plt.clf()
weights_series(name, date, dt_max, burn)

def plot_posterior(sample, title, burn=0):
    cnts, bins, _ = plt.hist(sample[burn:], bins=30, alpha=0.50)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.show()
    plt.clf()
plot_posterior(lambda0[0, :], 'Add to Bid Level 0')


"""Calculate and plot the impulse responses based on MAP estimates of mu and tau."""

def logit_normal(dt, mu, tau, dt_max):
    assert (dt < dt_max).all(), "dt must be less than dt_max."
    assert (dt > 0).all(), "dt must be greater than zero."
    Z = dt * (dt_max - dt) / dt_max * (tau / (2 * np.pi)) ** (-0.5)
    x = dt / dt_max
    s = np.log(x / (1 - x))
    return (1 / Z) * np.exp( -tau / 2 * (s - mu) ** 2 )

def plot_impulse(parent, child, W, mu, tau, dt_max, xmax, ymax=None, burn=0):
    eps = 0.001
    dt = np.linspace(0 + eps, xmax - eps, 100)
    mu_map = np.median(mu[parent, child, burn:])
    tau_map = np.median(tau[parent, child, burn:])
    values = logit_normal(dt, mu_map, tau_map, dt_max)
    weights = np.median(W[parent, child, burn:]) / np.max(np.median(W, axis=2))
    plt.plot(dt, values, alpha=weights, linewidth=0.5)
    if ymax is not None:
        plt.ylim([0, ymax])
    else:
        plt.ylim([0, np.max(values) * 1.1])
    plt.xlim([0, xmax])
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')
    return np.max(values)

# Impulse response matrix for different timescales
M = np.zeros((12, 12))
for xmax in [60, 10, 1, 0.1, 0.01]:
    for i in range(12):
        for j in range(12):
            plt.subplot(12, 12, (i * 12)+ j + 1)
            if xmax == 60:
                m = plot_impulse(i, j, W, mu, tau, dt_max, xmax)
                M[i, j] = m * 1.1
            else:
                _ = plot_impulse(i, j, W, mu, tau, dt_max, xmax, M[i, j])
    print('Saving figure for xmax={}...'.format(xmax))
    plt.savefig('/Users/colinswaney/Dropbox/Research/hft-hawkes/results/{}/impulse_xmax={}.pdf'.format(name, xmax))
    plt.clf()

# Same thing, but averaging estimates over the month
W_medians = []
mu_medians = []
tau_medians = []
for date in dates:
    _, W, mu, tau = import_data(name, date, dt_max, burn)
    W_medians.append(np.median(W, axis=2).reshape((12, 12, 1)))
    mu_medians.append(np.median(mu, axis=2).reshape((12, 12, 1)))
    tau_medians.append(np.median(tau, axis=2).reshape((12, 12, 1)))
W_avg = np.mean(np.concatenate(W_medians, axis=2), axis=2)
mu_avg = np.mean(np.concatenate(mu_medians, axis=2), axis=2)
tau_avg = np.mean(np.concatenate(tau_medians, axis=2), axis=2)

def plot_impulse(parent, child, W, mu, tau, dt_max, xmax, ymax=None):

    """
        W: N x N
        mu: N x N
        tau: N x N
    """

    eps = 0.001
    dt = np.linspace(0 + eps, xmax - eps, 100)
    values = logit_normal(dt, mu[parent, child], tau[parent, child], dt_max)
    weights = np.median(W[parent, child]) / np.max(W)
    plt.plot(dt, values, linewidth=0.5)
    plt.xlim([0, xmax])
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')
    return np.max(values)

# the impulse matrix
xmax = 60
for i in range(12):
    for j in range(12):
        plt.subplot(12, 12, (i * 12)+ j + 1)
        _ = plot_impulse(i, j, W_avg, mu_avg, tau_avg, dt_max, xmax)
plt.savefig('/Users/colinswaney/Dropbox/Research/hft-hawkes/results/{}/impulse_xmax={}.pdf'.format(name, xmax))
plt.clf()

# the weight matrix
def weights_plot(W, name=None, date=None, dt_max=None, burn=0):
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
    plt.savefig('/Users/colinswaney/Desktop/connections_avg.pdf')
    plt.clf()
weights_plot(W_avg)


"""Plot the intensity based on MAP estimates of the parameters and the observed event data."""

def estimate(lambda0, W, mu, tau, est='median'):
    if est == 'median':
        lambda0Hat = np.median(lambda0, axis=-1)
        WHat = np.median(W, axis=-1)
        muHat = np.median(mu, axis=-1)
        tauHat = np.median(tau, axis=1)
    return lambda0Hat, WHat, muHat, tauHat

# Make a network with estimated parameters
lambda0_map, W_map, mu_map, tau_map = estimate(lambda0, W, mu, tau, 'median')
params = {'bias': lambda0_map, 'weights': W_map, 'mu': mu_map, 'tau': tau_map}
model = NetworkPoisson(N=N, dt_max=dt_max, params=params)


"""Perform kernel density estimation on Gibbs sample and plot."""
from scipy.stats import gaussian_kde
def plot_density(x):
    # x = lambda0[0, :]
    grid = np.linspace(x.min(), x.max(), 1000)
    kde = gaussian_kde(x).evaluate(grid)
    plt.plot(grid, kde)
    plt.fill_between(grid, kde)
    plt.show()
    plt.clf()

"""
For each date, compute median of Gibbs sample, then calculate the time series
average of the medians.
"""






# Import event data
with h5.File() as hdf:
    events = hdf['{}/{}'.format(name, date)][:]

# Calculate the intensity
Lambda = net.calculate_intensity(events)

# Timestamps for  xlabels
seconds = pd.date_range(start='01-02-2013 10:30:00', end='01-02-2013 14:59:59', freq='S')
hours = pd.date_range(start='01-02-2013 10:30:00', end='01-02-2013 14:59:59', freq='H')

def add_intensity_plot():
    plt.subplot(3, 1, 1)
    plt.fill_between(seconds, 0, Lambda[:, 0])
    plt.fill_between(seconds, 0, -Lambda[:, 3])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off')
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 0', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)

    plt.subplot(3, 1, 2)
    plt.fill_between(seconds, 0, Lambda[:, 1])
    plt.fill_between(seconds, 0, -Lambda[:, 4])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off')
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 1', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)

    plt.subplot(3, 1, 3)
    plt.fill_between(seconds, 0, Lambda[:, 2])
    plt.fill_between(seconds, 0, -Lambda[:, 5])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 2', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    plt.savefig('/Users/colinswaney/Desktop/intensity_fit_add.pdf')
    plt.clf()

def delete_intensity_plot():
    ax = plt.subplot(2, 1, 1)
    plt.fill_between(seconds, 0, Lambda[:, 6])
    plt.fill_between(seconds, 0, -Lambda[:, 8])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off')
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 1', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)

    plt.subplot(2, 1, 2)
    plt.fill_between(seconds, 0, Lambda[:, 7])
    plt.fill_between(seconds, 0, -Lambda[:, 9])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 2', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    plt.savefig('/Users/colinswaney/Desktop/intensity_fit_delete.pdf')
    plt.clf()

def execute_intensity_plot():
    plt.fill_between(seconds, 0, Lambda[:, 10])
    plt.fill_between(seconds, 0, -Lambda[:, 11])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 2', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    plt.savefig('/Users/colinswaney/Desktop/intensity_fit_execute.pdf')
    plt.clf()

def intensity_subplot():

    fig, ax = plt.subplots()
    plt.fill_between(seconds, 0, Lambda[:, 0])
    plt.fill_between(seconds, 0, -Lambda[:, 3])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    left='off',
                    right='off',
                    labelbottom='off')
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylim([-1.5, 1.5])
    plt.ylabel('Level 0', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()

    # left, bottom, width, height = (0.70, 0.70, .20, .20)
    # ax_inner = fig.add_axes([left, bottom, width, height])
    # ax_inner.fill_between(np.arange(0, 60), 0, Lambda[1000:1060, 0])
    # ax_inner.fill_between(np.arange(0, 60), 0, -Lambda[1000:1060, 3])
    # ax_inner.tick_params(axis='both',
    #                      which='both',
    #                      bottom='off',
    #                      top='off',
    #                      left='off',
    #                      right='off',
    #                      labelbottom='off',
    #                      labelleft='off')

    ax2 = ax.twinx()
    trades = pd.DataFrame(spikes[:, -2:], index=seconds, columns=['buys', 'sells']).reset_index(drop=True)
    trades['min'] = (trades.index.values / 60).astype(int)
    grouped = trades.groupby('min').mean()
    minutes = pd.date_range(start='01-02-2013 10:30:00', end='01-02-2013 14:59:59', freq='T')
    b, = ax2.plot(minutes, grouped['buys'], linewidth=1, color='k')
    # plt.fill_between(buys.index, 0, buys['shares'])
    s, = ax2.plot(minutes, -grouped['sells'], linewidth=1, color='k')
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_xlim([seconds[0], seconds[-1]])
    # plt.fill_between(sells.index, 0, -sells['shares'])
    # plt.ylabel('Trades', fontsize=8)
    # ax.yaxis.set_ticks_position('right')
    # plt.yticks(fontsize=8)
    # plt.xticks(np.arange(MKT_OPEN / 60, MKT_CLOSE / 60, 3600 / 60), [h.strftime('%H:%M:%S') for h in hours])
    # plt.xticks(rotation=45, fontsize=8)
    # plt.xlim([MKT_OPEN / 60, MKT_CLOSE / 60])
    # plt.grid(linestyle='--', linewidth=0.25)

    plt.show()
    plt.clf()

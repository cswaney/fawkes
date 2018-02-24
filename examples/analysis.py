from fawkes.models import NetworkPoisson
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pandas as pd
import os
import sys


"""Single-day analysis of MCMC samples."""


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

# 1. Estimates
def get_estimates(sample, method='median'):
    lambda0, W, mu, tau = sample
    if method == 'median':
        lambda0 = np.median(lambda0, axis=1)
        W = np.median(W, axis=2)
        mu = np.median(mu, axis=2)
        tau = np.median(tau, axis=2)
    return lambda0, W, mu, tau

def flatten(lambda0, W, mu, tau, name, date, eig):
    N, _ = W.shape
    W = W.flatten()
    mu = mu.flatten()
    tau = tau.flatten()
    return [name, date] + list(np.concatenate([lambda0, W, mu, tau])) + [eig]

def unflatten(values):
    name = values.pop(0)
    date = values.pop(0)
    lambda0 = np.array(values[0:12]).reshape((12,))
    W = np.array(values[12:12+144]).reshape((12,12))
    mu = np.array(values[12+144:12+288]).reshape((12,12))
    tau = np.array(values[12+288+144:12+432]).reshape((12,12))
    return lambda0, W, mu, tau

# 2. Plots
def plot_bias(lambda0, path, name, date):

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
    # plt.show()
    plt.savefig(path + '/bias/bias_{}_{}.pdf'.format(name, date))
    plt.clf()

def plot_diagonal(W, path, name, date):

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
    plt.savefig(path + '/diagonal/diagonal_{}_{}.pdf'.format(name, date))
    plt.clf()

def plot_weights(W, path, name, date):
    """Create an image plot of a weight matrix estimate.

        W: N x N array.
    """

    plt.imshow(W, cmap='coolwarm')
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
    plt.savefig(path + 'weights/weights_{}_{}.pdf'.format(name, date))
    plt.clf()

def logit_normal(dt, mu, tau, dt_max):
    """The logit-normal impulse response function.

        dt: (N,) array.
    """
    assert (dt < dt_max).all(), "dt must be less than dt_max."
    assert (dt > 0).all(), "dt must be greater than zero."
    Z = dt * (dt_max - dt) / dt_max * (tau / (2 * np.pi)) ** (-0.5)
    x = dt / dt_max
    s = np.log(x / (1 - x))
    return (1 / Z) * np.exp( -tau / 2 * (s - mu) ** 2 )

def plot_impulses(W, mu, tau, dt_max, path, name, date):

    eps = 10 ** -6
    def plot_impulse(parent, child, W, mu, tau, dt_max):
        dt = np.linspace(0 + eps, dt_max - eps, 1000)
        values = W[parent, child] * logit_normal(dt, mu[parent, child], tau[parent, child], dt_max)
        plt.plot(dt, values, alpha=min(W[parent, child], 1))
        plt.xlim([0, dt_max])
        plt.yticks([])
        plt.xticks([])
        plt.axis('off')

    N, _ = W.shape
    for i in range(N):
        for j in range(N):
            plt.subplot(N, N, i * N + j + 1)
            plot_impulse(i, j, W, mu, tau, dt_max)
    plt.savefig(path + 'impulse/impulse_{}_{}.pdf'.format(name, date))
    plt.tight_layout()
    plt.clf()

# 3. Stability
def check_stability(lambda0, W, mu, tau, dt_max):
    """Check if the model is stable for given parameter estimates."""
    N, _ = W.shape
    model = NetworkPoisson(N=N, dt_max=dt_max)
    model.lamb = lambda0
    model.W = W
    model.mu = mu
    model.tau = tau
    return model.check_stability(return_value=True)

# 4. Likelihood
def compute_likelihood(data, lambda0, W, mu, tau):
    pass

# Script
read_path = '/Volumes/datasets/ITCH/samples/large2007_dt_max=60.hdf5'
data_path = '/Users/colinswaney/Desktop/data/estimates.txt'
img_path = '/Users/colinswaney/Desktop/plots/'
# name = 'A'
date = '072413'
dt_max = 60
df = []
with h5.File(read_path, 'r') as hdf:
    for name in hdf.keys():
        try:
            lambda0, W, mu, tau = import_samples(read_path, name, date)
        except:
            print('Unable to import samples; skipping')
        lambda0_, W_, mu_, tau_ = get_estimates((lambda0, W, mu, tau))
        _, eig = check_stability(lambda0_, W_, mu_, tau_, dt_max)
        df.append(flatten(lambda0_, W_, mu_, tau_, name, date, eig))
        plot_bias(lambda0, img_path, name, date)
        plot_diagonal(W, img_path, name, date)
        plot_weights(W_, img_path, name, date)
        plot_impulses(W_, mu_, tau_, dt_max, img_path, name, date)

columns = ['name', 'date']
columns += ['lambda0_{}'.format(i + 1) for i in range(12)]
x = []; y = []; z = []
for i in range(12):
    for j in range(12):
        x.append('W_{}_{}'.format(i + 1, j + 1))
        y.append('mu_{}_{}'.format(i + 1, j + 1))
        z.append('tau_{}_{}'.format(i + 1, j + 1))
columns = columns + x + y + z + ['eig']
df = pd.DataFrame(df, columns=columns)
df.to_csv(data_path, index=False)

# Aggregate analysis
df = pd.read_csv('/Users/colinswaney/Desktop/data/estimates.txt')
W = df.loc[:, 'W_1_1':'W_12_12']
mu = df.loc[:, 'mu_1_1':'mu_12_12']
tau = df.loc[:, 'tau_1_1':'tau_12_12']

W_mean = np.mean(W, axis=0).values.reshape((12,12))
mu_mean = np.mean(mu, axis=0).values.reshape((12,12))
tau_mean = np.mean(tau, axis=0).values.reshape((12,12))

W_median = np.percentile(W, 50, axis=0).reshape((12,12))
mu_W_median = np.percentile(mu, 50, axis=0).reshape((12,12))
tau_W_median = np.percentile(tau, 50, axis=0).reshape((12,12))

W_25 = np.percentile(W, 25, axis=0).reshape((12,12))
mu_25 = np.percentile(mu, 25, axis=0).reshape((12,12))
tau_25 = np.percentile(tau, 25, axis=0).reshape((12,12))

W_75 = np.percentile(W, 75, axis=0).reshape((12,12))
mu_75 = np.percentile(mu, 75, axis=0).reshape((12,12))
tau_75 = np.percentile(tau, 75, axis=0).reshape((12,12))

plt.imshow(W_median, cmap='coolwarm'); plt.colorbar(); plt.show()
plt.imshow(W_25, cmap='coolwarm'); plt.colorbar(); plt.show()
plt.imshow(W_75, cmap='coolwarm'); plt.colorbar(); plt.show()


# Histograms
W_ = W.values.reshape((-1,12,12))
for i in range(0,4):
    for j in range(0,4):
        plt.subplot(4, 4, (j + 1) + i * 4)
        plt.hist(W_[:, 4 + i, j], bins=30, edgecolor='white')
        plt.title('W_{}_{}'.format(5 + i, 1 + j))
plt.tight_layout()
plt.show()
for i in range(0,4):
    for j in range(0,4):
        plt.subplot(4, 4, (j + 1) + i * 4)
        plt.hist(W_[:, 4 + i, 8 + j], bins=30, edgecolor='white')
        plt.title('W_{}_{}'.format(5 + i, 9 + j))
plt.tight_layout()
plt.show()

def plot_impulse(parent, child, W, Mu, Tau, dt_max, tmin, tmax, N, log=False):
    w = W[parent, child]
    mu = Mu[parent, child]
    tau = Tau[parent, child]
    dt = np.linspace(tmin, tmax, N)
    values = w * logit_normal(dt, mu, tau, dt_max)
    if log:
        tmin = np.log(tmin)
        tmax = np.log(tmax)
        plt.plot(np.log(dt), values, linewidth=1.)
        plt.fill_between(np.log(dt), 0, values, alpha=0.25)
    else:
        plt.plot(dt, values, linewidth=1.)
        plt.fill_between(dt, 0, values, alpha=0.25)
    plt.title('w={:.2f}, mu={:.2f}, tau={:.2f}'.format(w, mu, tau))
    plt.ylim(ymin=0)
    if log:
        plt.xlim(tmin, tmax)
    else:
        plt.xlim(0, tmax)
    plt.show()

plot_impulse(7, 0, W_median, mu_median, tau_median, 60, 0.00001, 1, 10 ** 7)

# Correlation matrix
S_W = np.dot(W, W.transpose())
x = np.sqrt(np.diagonal(S_W).reshape((-1,1)))
S_W = S_W / np.dot(x, x.transpose())
plt.imshow(S_W, cmap='coolwarm')
plt.colorbar()
plt.show()

S_W_lower = (S_W * np.tri(401)).reshape(-1)
S_W_lower = S_W_lower[S_W_lower > 0]
plt.hist(S_W_lower); plt.show()

S_W_mean = S_W.mean(axis=1)
plt.hist(S_W_mean); plt.show()


# PCA  #
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(W)  # demean and standardize? - W.mean(axis=0)
W_1, W_2 = pca.components_
plt.subplot(211)
plt.imshow(W_1.reshape((12,12)), cmap='coolwarm'); plt.colorbar()
plt.subplot(212)
plt.imshow(W_2.reshape((12,12)), cmap='coolwarm'); plt.colorbar()
plt.show()

W_ld = pca.fit_transform(W)  # 401 x 2
plt.scatter(W_ld[:,0], W_ld[:,1])

# Clustering (K-means)
from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=2, random_state=0, n_init=20).fit(W)
W_1, W_2 = kmeans.cluster_centers_
plt.subplot(211)
plt.imshow(W_1.reshape((12,12)), cmap='coolwarm'); plt.colorbar()
plt.subplot(212)
plt.imshow(W_2.reshape((12,12)), cmap='coolwarm'); plt.colorbar()
plt.show()

W_std = W / W.std(axis=0)
kmeans = KMeans(n_clusters=2, random_state=0, n_init=20).fit(W_std)
W_1, W_2 = kmeans.cluster_centers_
plt.subplot(211)
plt.imshow(W_1.reshape((12,12)), cmap='coolwarm'); plt.colorbar()
plt.subplot(212)
plt.imshow(W_2.reshape((12,12)), cmap='coolwarm'); plt.colorbar()
plt.show()

# Weights relative to background rates

from fawkes.models import NetworkPoisson, HomogeneousPoisson
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pandas as pd
import os
import sys


"""Analysis of MCMC samples for the entire collection of stocks."""


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

def plot_bias(lambda0, name, date, path=None):

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
    if path is None:
        plt.show()
    else:
        plt.savefig(path + '/bias/bias_{}_{}.pdf'.format(name, date))
        plt.clf()

def plot_diagonal(W, name, date, path=None):

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
    if path is None:
        plt.show()
    else:
        plt.savefig(path + '/diagonal/diagonal_{}_{}.pdf'.format(name, date))
        plt.clf()

def plot_weights(W, name, date, path=None):
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
    if path is None:
        plt.show()
    else:
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

# def plot_impulses(W, mu, tau, dt_max, path, name, date):
#
#     eps = 10 ** -6
#     def plot_impulse(parent, child, W, mu, tau, dt_max):
#         dt = np.linspace(0 + eps, dt_max - eps, 1000)
#         values = W[parent, child] * logit_normal(dt, mu[parent, child], tau[parent, child], dt_max)
#         plt.plot(dt, values, alpha=min(W[parent, child], 1))
#         plt.xlim([0, dt_max])
#         plt.yticks([])
#         plt.xticks([])
#         plt.axis('off')
#
#     N, _ = W.shape
#     for i in range(N):
#         for j in range(N):
#             plt.subplot(N, N, i * N + j + 1)
#             plot_impulse(i, j, W, mu, tau, dt_max)
#     plt.savefig(path + 'impulse/impulse_{}_{}.pdf'.format(name, date))
#     plt.tight_layout()
#     plt.clf()

def plot_impulse(parent, child, W, Mu, Tau, dt_max, tmin, tmax, N, **kwargs):

    """Keywords:
        - log: take log of tmin, tmax
        - color, alpha
    """

    log = False
    log10 = False
    color = 'C0'
    alpha = 1.
    for key, value in kwargs.items():
        if key == 'color':
            color = value
        if key == 'alpha':
            alpha = value
        if key == 'log':
            log = value
        if key == 'log10':
            log10 = value

    w = W[parent, child]
    mu = Mu[parent, child]
    tau = Tau[parent, child]
    if log:
        tmin = np.log(tmin)
        tmax = np.log(tmax)
    elif log10:
        tmin = np.log10(tmin)
        tmax = np.log10(tmax)
    dt = np.linspace(tmin, tmax, N)
    if log:
        values = w * logit_normal(np.exp(dt), mu, tau, dt_max)
    elif log10:
        values = w * logit_normal(10 ** dt, mu, tau, dt_max)
    else:
        values = w * logit_normal(dt, mu, tau, dt_max)
    plt.fill_between(dt, 0, values, alpha=alpha, color=color)
    plt.ylim(ymin=0)
    if log or log10:
        plt.xlim(tmin, tmax)
    else:
        plt.xlim(0, tmax)
    plt.show()

def check_stability(lambda0, W, mu, tau, dt_max):
    """Check if the model is stable for given parameter estimates."""
    N, _ = W.shape
    model = NetworkPoisson(N=N, dt_max=dt_max)
    model.lamb = lambda0
    model.W = W
    model.mu = mu
    model.tau = tau
    return model.check_stability(return_value=True)

def save_estimates(date, dt_max, read_path, write_path, img_path):
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
            plot_bias(lambda0, name, date, path=img_path)
            plot_diagonal(W, name, date, path=img_path)
            plot_weights(W_, name, date, path=img_path)

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
    df.to_csv(write_path, index=False)

# Create estimates (OPTIONAL)
dt_max = 60
date = '072413'
read_path = '/Volumes/datasets/ITCH/samples/large2007_dt_max=60.hdf5'
write_path = '/Users/colinswaney/Desktop/net/data/estimates.txt'
img_path = '/Users/colinswaney/Desktop/net/plots/'
save_estimates(date, dt_max, read_path, write_path, img_path)


# Import estimates
df = pd.read_csv('/Users/colinswaney/Desktop/net/data/estimates.txt')
lambda0 = df.loc[:, 'lambda0_1':'lambda0_12']
W = df.loc[:, 'W_1_1':'W_12_12']
mu = df.loc[:, 'mu_1_1':'mu_12_12']
tau = df.loc[:, 'tau_1_1':'tau_12_12']
lambda0_median = np.percentile(lambda0, 50, axis=0)
W_median = np.percentile(W, 50, axis=0).reshape((12,12))
mu_median = np.percentile(mu, 50, axis=0).reshape((12,12))
tau_median = np.percentile(tau, 50, axis=0).reshape((12,12))
W_25 = np.percentile(W, 25, axis=0).reshape((12,12))
mu_25 = np.percentile(mu, 25, axis=0).reshape((12,12))
tau_25 = np.percentile(tau, 25, axis=0).reshape((12,12))
W_75 = np.percentile(W, 75, axis=0).reshape((12,12))
mu_75 = np.percentile(mu, 75, axis=0).reshape((12,12))
tau_75 = np.percentile(tau, 75, axis=0).reshape((12,12))


# Composite weights: plots
plt.subplot(121)
plt.title('Raw')
plot_weights(W_median, 'composite', '072413')
plt.subplot(122)
plt.title('Normalized')
plot_weights(W_median / lambda0_median, 'composite (normalized)', '072413')
plt.tight_layout()


# Composite impulse: plots (raw)
N = 10 ** 3
tmin = 10 ** -7
tmax = 10.
plt.subplot(221)
plt.title('Executes & Adds')
plot_impulse(5, 2, W_median, mu_median, tau_median, dt_max, tmin, tmax, N, log10=True, color='C3', alpha=1.)
plot_impulse(7, 0, W_median, mu_median, tau_median, dt_max, tmin, tmax, N, log10=True, color='C0', alpha=1.)
plt.ylim([0, 15000])
plt.xticks([-6, -3, 0], [r'1$\mu$s', r'1$m$s', '1s'])
plt.legend([r'$h_{6,3}$', r'$h_{8,1}$'])
plt.subplot(222)
plt.title('Executes & Deletes')
plot_impulse(5, 8, W_median, mu_median, tau_median, dt_max, tmin, tmax, N, log10=True, color='C3', alpha=1.)
plot_impulse(7, 10, W_median, mu_median, tau_median, dt_max, tmin, tmax, N, log10=True, color='C0', alpha=1.)
plt.xticks([-6, -3, 0], [r'1$\mu$s', r'1$m$s', '1s'])
plt.legend([r'$h_{6,9}$', r'$h_{8,11}$'])
plt.tight_layout()
tmin = 10 ** -8
tmax = 10.
plt.subplot(223)
plt.title('Executes & Adds')
plot_impulse(5, 3, W_median, mu_median, tau_median, dt_max, tmin, tmax, N, log10=True, color='C3', alpha=1.)
plot_impulse(7, 1, W_median, mu_median, tau_median, dt_max, tmin, tmax, N, log10=True, color='C0', alpha=1.)
plt.xticks([-6, -3, 0], [r'1$\mu$s', r'1$m$s', '1s'])
plt.legend([r'$h_{6,6}$', r'$h_{8,2}$'])
tmin = 10 ** -5
tmax = 10.
plt.subplot(224)
plt.title('Adds & Deletes')
plot_impulse(1, 9, W_median, mu_median, tau_median, dt_max, tmin, tmax, N, log10=True, color='C3', alpha=1.)
plot_impulse(3, 11, W_median, mu_median, tau_median, dt_max, tmin, tmax, N, log10=True, color='C0', alpha=1.)
plt.xticks([-3, 0], [r'1$m$s', '1s'])
plt.legend([r'$h_{2,10}$', r'$h_{4,12}$'])
plt.tight_layout()


# Similarity/Correlation matrix
S_W = np.dot(W, W.transpose())
x = np.sqrt(np.diagonal(S_W).reshape((-1,1)))
S_W = S_W / np.dot(x, x.transpose())
plt.imshow(S_W, cmap='coolwarm')
plt.colorbar()
plt.show()


# PCA
from sklearn.decomposition import PCA
W_normed = W.values - W.values.mean(axis=0)
W_normed = W_normed / W_normed.std(axis=0)
pca = PCA(n_components=144)
pca.fit(W_normed)
W_1, W_2 = pca.components_[0], pca.components_[1]
plt.subplot(121)
plt.title('1st component')
plot_weights(W_1.reshape((12,12)), 'PCA (normalized)', '072413')
plt.subplot(122)
plt.title('2nd component')
plot_weights(W_2.reshape((12,12)), 'PCA (normalized)', '072413')
plt.show()

plt.subplot(121)
plt.bar(np.arange(1,13), pca.explained_variance_ratio_[:12])
plt.xlabel('Component')
plt.ylabel('Explained variance ratio')
plt.subplot(122)
W_lowdim = pca.fit_transform(W)  # 401 x 2
plt.scatter(W_ld[:,0], W_ld[:,1])
plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.tight_layout()


# PCA (normalized by lambda0)
from sklearn.decomposition import PCA
W_normed = W.values - np.repeat(lambda0.values, 12, axis=1)
W_normed = W_normed - W_normed.mean(axis=0)
W_normed = W_normed / W_normed.std(axis=0)
pca = PCA(n_components=144)
pca.fit(W_normed)
W_1, W_2 = pca.components_[0], pca.components_[1]
plt.subplot(121)
plt.title('1st component')
plot_weights(W_1.reshape((12,12)), 'PCA (normalized)', '072413')
plt.subplot(122)
plt.title('2nd component')
plot_weights(W_2.reshape((12,12)), 'PCA (normalized)', '072413')
plt.show()

plt.subplot(121)
plt.bar(np.arange(1,13), pca.explained_variance_ratio_[:12])
plt.xlabel('Component')
plt.ylabel('Explained variance ratio')
plt.subplot(122)
W_lowdim = pca.fit_transform(W)  # 401 x 2
plt.scatter(W_ld[:,0], W_ld[:,1])
plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.tight_layout()


# Clustering (K-means)
from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=3, random_state=0, n_init=20).fit(W_lowdim)
centroids = kmeans.cluster_centers_
labels = kmeans.predict(W_lowdim)
C1 = W_lowdim[labels == 0]; C2 = W_lowdim[labels == 1]; C3 = W_lowdim[labels == 2]
plt.scatter(C1[:,0], C1[:,1]); plt.scatter(C2[:,0], C2[:,1]); plt.scatter(C3[:,0], C3[:,1])

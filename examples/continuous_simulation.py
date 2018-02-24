from fawkes.models import NetworkPoisson, HomogeneousPoisson
import numpy as np
import matplotlib.pyplot as plt

"""Simulation of a simple Hawkes network for basic debugging."""

def plot_sample(sample, theta, name, burn=0):
    """
        sample: (N,) random sample of parameter.
        theta: true parameter value.
        name: name of the parameter.
        burn: size of burn-in (samples not included).
    """

    assert burn < len(sample), "Burn-in exceeds number of samples."
    ndim = len(sample.shape) - 1
    if ndim == 1:
        ncol, = sample.shape[:-1]
        for i in range(ncol):
            plt.subplot(1, ncol, i+1)
            cnts, bins, _ = plt.hist(sample[i], bins=30)
            plt.title("posterior: {}_{}".format(name, i))
            plt.vlines(theta[i], 0, np.max(cnts) + 10, colors='red')
            plt.ylim([0, np.max(cnts) + 10])
    elif ndim == 2:
        nrow, ncol = sample.shape[:-1]
        for i in range(nrow):
            for j in range(ncol):
                plt.subplot(nrow, ncol, i * ncol + (j+1))
                cnts, bins, _ = plt.hist(sample[i,j], bins=30)
                plt.title("posterior: {}_{},{}".format(name, i, j))
                plt.vlines(theta[i,j], 0, np.max(cnts) + 10, colors='red')
                plt.ylim([0, np.max(cnts) + 10])
    plt.tight_layout()
    plt.show()

# Set simulation parameters
N = 2
T = 1000.0
dt_max = 5.00
S = 2000
burn = 500

# Construct a network
lambda0 = np.array([1., 1.])
W = np.array([[0.5, 0.1], [0.1, 0.5]])
mu = -1.0 * np.ones((N,N))
tau = 1.0 * np.ones((N,N))
params = {'lamb': lambda0, 'weights': W, 'mu': mu, 'tau': tau}
model = NetworkPoisson(N=N, dt_max=dt_max, params=params)

# Generate some data
print('Generating data...')
data = model.generate_data(T=T)
print("N = {}".format(N))
print("T = {}".format(T))
print("M = {}".format(len(data[0])))
print("S = {}".format(S))

# Plot the data
model.plot_data(data)

# Sample the posterior (cython extension)
lambda0_, W_, mu_, tau_ = model.sample_ext(data, T=T, size=S, method='cython')

# Plot the sample
plot_sample(lambda0_, lambda0, 'lambda0')
plot_sample(W_, W, 'W')
plot_sample(mu_, mu, 'mu')
plot_sample(tau_, tau, 'tau')
# model.plot_impulse(mu=np.median(mu_[1, 0, :]), tau=np.median(tau_[1, 0, :]))

# Report stability
model.check_stability()
model.check_stability(A=np.ones((2,2)), W=np.median(W_, axis=2))

# Compute likelihood
theta_ = (np.median(lambda0_, axis=1),
          np.median(W_, axis=2),
          np.median(mu_, axis=2),
          np.median(tau_, axis=2))
ll_fit = model.compute_likelihood(data, T, theta_)
sample = (lambda0_, W_, mu_, tau_)
ll_pred = model.compute_pred_likelihood(data, T, sample)
print('log-likelihood (fit): {:.3f}'.format(ll_fit))
print('log-likelihood (pred): {:.3f}'.format(ll_pred))


# """Comparison with HomogeneousPoisson"""
model_homo = HomogeneousPoisson(N, dt_max, {'lambda0': model.lamb})
lambda0_homo = model_homo.sample(data, T, size=S)

def plot_sample(sample, theta, name, burn=0):
    """
        sample: (N,) random sample of parameter.
        theta: true parameter value.
        name: name of the parameter.
        burn: size of burn-in (samples not included).
    """

    assert burn < len(sample), "Burn-in exceeds number of samples."
    M,n = sample.shape
    for i in range(n):
        plt.subplot(1, n, i+1)
        cnts, bins, _ = plt.hist(sample[:,i], bins=20)
        plt.title("posterior: {}_{}".format(name, i))
        plt.vlines(theta[i], 0, np.max(cnts) + 10, colors='red')
        plt.ylim([0, np.max(cnts) + 10])
    plt.tight_layout()
    plt.show()

plot_sample(lambda0_homo, model_homo.lambda0, 'lambda0')

ll_fit = model_homo.compute_likelihood(data, T, np.median(lambda0_homo, axis=0))
ll_pred = model_homo.compute_pred_likelihood(data, T, S)
print('log-likelihood (fit): {:.3f}'.format(ll_fit))
print('log-likelihood (pred): {:.3f}'.format(ll_pred))

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

# Select parameters
N = 2
T = 1000.0
dt_max = 1.0
S = 2000
burn = 500
print('Running simulation...')
print('Parameters:\nN={:d}\nT={:.2f}\ndt_max={:.2f}\nS={:d}\n'.format(N, T, dt_max, S))

# Construct network
model = NetworkPoisson(N=N, dt_max=dt_max)
stable = False
while not stable:
    model.init_parameters()
    stable = model.check_stability()

# Generate data
print('Generating data...', end='')
data = model.generate_data(T=T)
print('(size={})'.format(len(data[0])))

# Plot the data
model.plot_data(data)

# Sample the posterior (cython extension)
lambda0_, W_, mu_, tau_ = model.sample_ext(data, T=T, size=S, method='cython')

# Plot the sample
plot_sample(lambda0_, model.lamb, 'lambda0')
plot_sample(W_, model.W, 'W')
plot_sample(mu_, model.mu, 'mu')
plot_sample(tau_, model.tau, 'tau')
model.plot_impulse(mu=np.median(mu_[0, 0, :]), tau=np.median(tau_[0, 0, :]))

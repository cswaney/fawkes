from fawkes.models import NetworkPoisson
import numpy as np
import matplotlib.pyplot as plt

def plot_sample(sample, theta, name, burn=0):
    """
        sample: (N,) random sample of parameter.
        theta: true parameter value.
        name: name of the parameter.
        burn: size of burn-in (samples not included).
    """

    assert burn < len(sample), "Burn-in exceeds number of samples."
    cnts, bins, _ = plt.hist(sample, bins=30)
    plt.title("Posterior Distribution: {}".format(name))
    plt.vlines(theta, 0, np.max(cnts) + 10, colors='red')
    plt.ylim([0, np.max(cnts) + 10])
    plt.show()
    plt.clf()

# Set simulation parameters
N = 2
T = 10.0
dt_max = 1.00
S = 1000
burn = 0

# Construct a network
lambda0 = np.array([1.0, 1.0])
W = np.array([[0.5, 0.1], [0.1, 0.5]])
mu = 0.0 * np.ones((N,N))
tau = 1.0 * np.ones((N,N))
params = {'lamb': lambda0, 'weights': W, 'mu': mu, 'tau': tau}
model = NetworkPoisson(N=N, dt_max=dt_max, params=params)

# Generate some data
data = model.generate_data(T=T)
print("N = {}".format(N))
print("T = {}".format(T))
print("M = {}".format(len(data[0])))
print("S = {}".format(S))

# Plot the data
model.plot_data(data)

# Sample the posterior
lambda0_, W_, mu_, tau_ = model.sample_ext(data, T=T, size=S, method='cython')

# Plot the sample
plot_sample(lambda0_[0, :], lambda0[0], 'lambda0_1')
plot_sample(W_[1, 0, :], W[1, 0], 'W_10')
plot_sample(mu_[1, 0, :], mu[1, 0], 'mu_10')
plot_sample(tau_[1, 0, :], tau[1, 0], 'tau_10')
model.plot_impulse(mu=np.median(mu_[1, 0, :]), tau=np.median(tau_[1, 0, :]))

# Repeat on larger sample
N = 2
T = 10000.0
dt_max = 1.00
S = 2000
burn = 500
lambda0 = np.array([1.0, 1.0])
W = np.array([[0.5, 0.1], [0.1, 0.5]])
mu = 0.0 * np.ones((N,N))
tau = 1.0 * np.ones((N,N))
params = {'lamb': lambda0, 'weights': W, 'mu': mu, 'tau': tau}
model = NetworkPoisson(N=N, dt_max=dt_max, params=params)
data = model.generate_data(T=T)
print("N = {}".format(N))
print("T = {}".format(T))
print("M = {}".format(len(data[0])))
print("S = {}".format(S))
# model.plot_data(data)
lambda0_, W_, mu_, tau_ = model.sample_ext(data, T=T, size=S, method='cython')
plot_sample(lambda0_[0, :], lambda0[0], 'lambda0_1')
plot_sample(W_[1, 0, :], W[1, 0], 'W_10')
plot_sample(mu_[1, 0, :], mu[1, 0], 'mu_10')
plot_sample(tau_[1, 0, :], tau[1, 0], 'tau_10')
model.plot_impulse(mu=np.median(mu_[1, 0, :]), tau=np.median(tau_[1, 0, :]))

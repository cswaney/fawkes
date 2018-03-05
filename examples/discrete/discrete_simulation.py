from fawkes.models import DiscreteNetworkPoisson
import matplotlib.pyplot as plt
import numpy as np

"""Simulation of a simple Hawkes network for basic debugging."""

def plot_sample(lambda0_, W_, theta_, burn=0):
    # Biases
    plt.subplot(3,1,1)
    cnts, bins, _ = plt.hist(lambda0_[0, burn:], bins=25)
    plt.xlabel("lamb_1")
    plt.vlines(lambda0[0], 0, np.max(cnts) + 10, colors='red')
    # plt.subplot(2,1,2)
    # cnts, bins, _ = plt.hist(lambda0_[1, burn:], bins=25)
    # plt.xlabel("lamb_1")
    # plt.vlines(lambda0[1], 0, np.max(cnts) + 10, colors='red')
    # plt.show()
    # plt.clf()
    # Weights
    plt.subplot(3,1,2)
    cnts, bins, _ = plt.hist(W_[0, 0, burn:], bins=25)
    plt.xlabel("W_1,1")
    plt.vlines(W[0,0], 0, np.max(cnts) + 10, colors='red')
    # plt.subplot(2,2,2)
    # cnts, bins, _ = plt.hist(W_[0, 1, burn:], bins=25)
    # plt.xlabel("W_1,2")
    # plt.vlines(W[0,1], 0, np.max(cnts) + 10, colors='red')
    # plt.subplot(2,2,3)
    # cnts, bins, _ = plt.hist(W_[1, 0, burn:], bins=25)
    # plt.xlabel("W_2,1")
    # plt.vlines(W[1,0], 0, np.max(cnts) + 10, colors='red')
    # plt.subplot(2,2,4)
    # cnts, bins, _ = plt.hist(W_[1, 1, burn:], bins=25)
    # plt.xlabel("W_2,2")
    # plt.vlines(W[1,1], 0, np.max(cnts) + 10, colors='red')
    # plt.show()
    # plt.clf()
    # Impulse
    plt.subplot(3,1,3)
    cnts, bins, _ = plt.hist(theta_[0, 0, 0, burn:], bins=25)
    plt.xlabel("theta_1,1,1")
    plt.vlines(theta[0,0,0], 0, np.max(cnts) + 10, colors='red')
    # plt.subplot(2,2,2)
    # cnts, bins, _ = plt.hist(theta_[0, 0, 1, burn:], bins=25)
    # plt.xlabel("mu_1,2")
    # plt.vlines(theta[0,0,1], 0, np.max(cnts) + 10, colors='red')
    # plt.subplot(2,2,3)
    # cnts, bins, _ = plt.hist(theta_[0, 1, 0, burn:], bins=25)
    # plt.xlabel("mu_2,1")
    # plt.vlines(theta[0,1,0], 0, np.max(cnts) + 10, colors='red')
    # plt.subplot(2,2,4)
    # cnts, bins, _ = plt.hist(theta_[0, 1, 1, burn:], bins=25)
    # plt.xlabel("mu_2,2")
    # plt.vlines(theta[0,1,1], 0, np.max(cnts) + 10, colors='red')
    plt.show()
    plt.clf()

def simulate(lambda0, W, theta):
    # Network

    params = {'weights': W, 'bias': lambda0, 'impulse': theta}
    net = DiscreteNetworkPoisson(N=N, L=L, B=B, dt=dt, params=params)

    # Synthetic Data
    S = net.generate_data(T)
    print("N={}".format(N))
    print("B={}".format(B))
    print("L={}".format(L))
    print("T={}".format(T))
    print("samples={}".format(samples))
    print("burn={}".format(burn))
    # net.plot_basis()
    Shat = net.convolve(S)
    Lambda = net.calculate_intensity(S, Shat)
    S = S[skip:]
    Shat = Shat[skip:]
    Lambda = Lambda[skip:]
    # net.plot_data(S, Lambda, events=False)

    # Cython Gibbs
    lambda0_, W_, theta_ = net.sample_ext(S, size=samples)
    # lambda0_, W_, theta_ = net.sample(S, size=samples)
    # plot_sample(lambda0_, W_, theta_, burn=burn)
    theta_map = np.median(theta_[:, :, :, burn:], axis=3)
    net.plot_basis(theta=theta_map, mean=True)


# Settings
N = 2
B = 3
L = 60
dt = 1
T = 10000
samples = 100
burn = 100
skip = 0

# Low frequency data
lambda0 = 0.1 * np.ones(N)
W = 0.05 * np.diag(np.ones(N))
theta = (1 / B) * np.ones((B, N, N))
simulate(lambda0, W, theta)

# High frequency data
lambda0 = 1.0 * np.ones(N)
W = 0.5 * np.diag(np.ones(N))
theta = (1 / B) * np.ones((B, N, N))
simulate(lambda0, W, theta)


# Settings
N = 2
B = 10
L = 10
dt = 1
T = 10000
samples = 100
burn = 100
skip = 0

# Low frequency data
lambda0 = 0.1 * np.ones(N)
W = 0.05 * np.diag(np.ones(N))
theta = (1 / B) * np.ones((B, N, N))
simulate(lambda0, W, theta)

# High frequency data
lambda0 = 1.0 * np.ones(N)
W = 0.5 * np.diag(np.ones(N))
theta = (1 / B) * np.ones((B, N, N))
simulate(lambda0, W, theta)

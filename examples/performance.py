from fawkes.models import NetworkPoisson
import numpy as np
import time


# Continuous-Time
N = 2
T = 1000.0
dt_max = 1.00

bias = np.array([1.0, 1.0])
weights = np.array([[0.5, 0.2], [0.2, 0.5]])
mu = 0.0 * np.ones((N,N))
tau = 1.0 * np.ones((N,N))
params = {'lamb': bias, 'weights': weights, 'mu': mu, 'tau': tau}
net = NetworkPoisson(N=N, dt_max=dt_max, params=params)
data = net.generate_data(T=T)


# Class Gibbs sampling
# start = time.time()
# sample = net.sample(data, T, size=10)
# stop = time.time()
# py_time = stop - start
# print('elapsed time: {:.3f}\n'.format(py_time))
#
# start = time.time()
# sample = net.sample_ext(data, T, size=10)
# stop = time.time()
# cy_time = stop - start
# print('elapsed time: {:.3f}'.format(cy_time))
# print('speed-up: {:.2f}\n'.format(py_time / cy_time))


# Parent sampling
start = time.time()
for _ in range(1):
    parents = net.sample_parents(data, bias, weights, mu, tau)
stop = time.time()
py_time = stop - start
print('elapsed time: {:.3f}'.format(py_time))

start = time.time()
for _ in range(1):
    parents = net.sample_parents_ext(data, bias, weights, mu, tau, method='cython')
stop = time.time()
cy_time = stop - start
print('elapsed time: {:.3f}'.format(cy_time))
print('speed-up: {:.2f}\n'.format(py_time / cy_time))


# Discrete-Time

# from fawkes.extensions import sample_parents_discrete
# from fawkes.models import DiscreteNetworkPoisson
#
# N = 2
# B = 1
# dt = 1.0
# dt_max = 10
# T = 16000
# lambda0 = np.array([1.0, 1.0], dtype='float64')
# W = np.array([[0.5, 0.1], [0.1, 0.5]], dtype='float64')
# theta = (1 / B) * np.ones((B, N, N))
# params = {'weights': W, 'bias': lambda0, 'impulse': theta}
# model = DiscreteNetworkPoisson(N=N, L=dt_max, B=B, dt=dt, params=params)
# S = model.generate_data(T)
# Shat = model.convolve(S)
# Lambda = model.calculate_intensity(S, Shat)

# Parent sampling
# start = time.time()
# model.sample_parents(S, Shat, lambda0, W, theta)
# stop = time.time()
# py_time = stop - start
# print('elapsed time: {:.3f}'.format(py_time))
#
# start = time.time()
# model.sample_parents_ext(S, Shat, lambda0, W, theta)
# stop = time.time()
# cy_time = stop - start
# print('elapsed time: {:.3f}'.format(cy_time))
# print('speed-up: {:.2f}'.format(py_time / cy_time))

# Gibbs Sampling
# start = time.time()
# model.sample(S)
# stop = time.time()
# py_time = stop - start
# print('elapsed time: {:.3f}'.format(py_time))
#
# start = time.time()
# model.sample_ext(S)
# stop = time.time()
# cy_time = stop - start
# print('elapsed time: {:.3f}'.format(cy_time))
# print('speed-up: {:.2f}'.format(py_time / cy_time))

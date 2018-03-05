from fawkes.models import DiscreteNetworkPoisson
import matplotlib.pyplot as plt
import numpy as np
import time

# Construct synthetic data
N = 2
B = 3
L = 5
dt = 1
T = 20000
lambda0 = np.array([1.0, 1.0])
W = np.array([[0.5, 0.1], [0.1, 0.5]])
theta = (1 / B) * np.ones((B, N, N))
params = {'weights': W, 'bias': lambda0, 'impulse': theta}
model = DiscreteNetworkPoisson(N=N, L=L, B=B, dt=dt, params=params)
spikes = model.generate_data(T)
spikes_hat = model.convolve(spikes)

# Sample some parents
start = time.time()
parents = model.sample_parents(spikes, spikes_hat, lambda0, W, theta)
python_time = time.time() - start
print("elapsed time: {}".format(python_time))
start = time.time()
parents = model.sample_parents_ext(spikes, spikes_hat, lambda0, W, theta)
cython_time = time.time() - start
print("elapsed time: {}".format(cython_time))
print("speed-up: {}\n".format(python_time / cython_time))

# Compute sufficient statistics without Cython
start = time.time()
stats = model.model.calculate_stats(spikes, parents)
python_time = time.time() - start
print("elapsed time: {}".format(python_time))
# print("alpha={}".format(stats[0]))
# print("beta={}".format(stats[1]))
# print("nu={}".format(stats[2]))
# print("kappa={}".format(stats[3]))
# print("gamma={}\n".format(stats[4]))

# Compute sufficient statistics with cython
start = time.time()
stats_cython = model.model.calculate_stats_ext(spikes, parents, method='cython')
cython_time = time.time() - start
print("elapsed time: {}".format(cython_time))
# print("alpha={}".format(stats_cython[0]))
# print("beta={}".format(stats_cython[1]))
# print("nu={}".format(stats_cython[2]))
# print("kappa={}".format(stats_cython[3]))
# print("gamma={}\n".format(stats_cython[4]))

# Compare the times
print("speed-up: {}".format(python_time / cython_time))

# Check that they agree
print("alpha? {}".format(np.allclose(stats[0], stats_cython[0])))
print("beta? {}".format(np.allclose(stats[1], stats_cython[1])))
print("nu? {}".format(np.allclose(stats[2], stats_cython[2])))
print("kappa? {}".format(np.allclose(stats[3], stats_cython[3])))
print("gamma? {}".format(np.allclose(stats[4], stats_cython[4])))

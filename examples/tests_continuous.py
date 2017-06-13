from fawkes.models import NetworkPoisson
import numpy as np

verbose = False

# Make a network model
N = 2
T = 10.0
dt_max = 1.00
bias = np.array([1.0, 1.0])
weights = np.array([[0.5, 0.1], [0.1, 0.5]])
mu = 0.0 * np.ones((N,N))
tau = 1.0 * np.ones((N,N))
params = {'lamb': bias, 'weights': weights, 'mu': mu, 'tau': tau}
model = NetworkPoisson(N=N, dt_max=dt_max, params=params)
data = model.generate_data(T=T)

# Generate some parents
parents_cython = model.sample_parents_ext(data, bias, weights, mu, tau)
parents = parents_cython + 1

# Compute sufficient statistics without Cython
stats = model.model.calculate_stats(data, parents, T, dt_max)
if verbose:
    print("M_0={}".format(stats[0]))
    print("M_m={}".format(stats[1]))
    print("M_nm={}".format(stats[2]))
    print("xbar_nm={}".format(stats[3]))
    print("nu_nm={}\n".format(stats[4]))

# Compute sufficient statistics with cython
stats_cython = model.model.calculate_stats_ext(data, parents_cython, T, dt_max)
if verbose:
    print("M_0={}".format(stats_cython[0]))
    print("M_m={}".format(stats_cython[1]))
    print("M_nm={}".format(stats_cython[2]))
    print("xbar_nm={}".format(stats_cython[3]))
    print("nu_nm={}".format(stats_cython[4]))

# Check that they agree
print("M_0? {}".format(np.allclose(stats[0], stats_cython[0])))
print("M_m? {}".format(np.allclose(stats[1], stats_cython[1])))
print("M_nm? {}".format(np.allclose(stats[2], stats_cython[2])))
print("xbar_nm? {}".format(np.allclose(stats[3], stats_cython[3])))
print("nu_nm? {}".format(np.allclose(stats[4], stats_cython[4])))

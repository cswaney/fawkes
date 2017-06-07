import numpy as np
# from math import log, exp, sqrt
cimport numpy as np
from libc.math cimport log, exp, sqrt
import cython
from cython.parallel import prange, parallel

cdef double SQRT_2PI = 2.5066282746310002

@cython.boundscheck(False)
def sample_parents_cython(double [::1] times, int [::1] nodes, double [::1] parents,
                          double dt_max, double [::1] lambda0, double [:,::1] W,
                          double [:,::1] mu, double [:,::1] tau):

    cdef int M = times.shape[0]
    cdef int m, n, cm, cn
    cdef double dt
    cdef double p_bkgd, sum
    cdef double Z, L
    cdef double [::1] p = np.zeros(M)
    cdef double [::1] u = np.random.rand(M)
    cdef double u0
    cdef int n_min, n_max

    for m in range(M):

        # print(m)
        cm = int(nodes[m])
        p_bkgd = lambda0[cm]
        sum = p_bkgd
        n_min = m  # or m?
        n_max = m

        # Backwards
        for n in range(m - 1, -1, -1):
            dt = times[m] - times[n]
            if (dt < dt_max) and (dt == 0):  # in window, concurrent
                n_max = n
                n_min = n
            if (dt < dt_max) and (dt > 0):  # in window, behind
                n_min = n
                cn = int(nodes[n])
                Z = dt * (dt_max - dt) / dt_max * SQRT_2PI / sqrt(tau[cn, cm])
                try:
                    L = W[cn, cm] * exp(-tau[cn, cm] / 2. * ( log( (dt / dt_max) / (1 - (dt / dt_max) ) ) - mu[cn, cm]) ** 2) / Z
                except:
                    print('m={}, n={}'.format(m, n))
                    print('dt={}'.format(dt))
                    print('W={}'.format(W[cn, cm]))
                    print('mu={}'.format(mu[cn, cm]))
                    print('tau={}'.format(tau[cn, cm]))
                    print('dt_max={}, Z={}, tau={}'.format(dt_max, Z, tau[cn, cm]))
                    raise("")
                L = W[cn, cm] * exp(-tau[cn, cm] / 2. * ( log( (dt / dt_max) / (1 - (dt / dt_max) ) ) - mu[cn, cm]) ** 2) / Z
                sum = sum + L
                p[n] = L
            else:
                break

        # Forwards
        u0 = p_bkgd / sum
        if u[m] < u0:
            parents[m] = -1
        else:
            for n in range(n_min, n_max):
                u0 = u0 + p[n] / sum
                if u[m] < u0:
                    parents[m] = n
                    break

@cython.boundscheck(False)
def sample_parents_openmp(double [::1] times, int [::1] nodes, double [::1] parents,
                          double dt_max, double [::1] lambda0, double [:,::1] W,
                          double [:,::1] mu, double [:,::1] tau):

    cdef int M = times.shape[0]
    cdef int m, n, cm, cn
    cdef double dt
    cdef double p_bkgd, sum
    cdef double Z, L
    cdef double [:, ::1] p = np.zeros((M, M))
    cdef double [::1] u = np.random.rand(M)
    cdef double u0
    cdef int[::1] n_min = np.arange(M, dtype='int32')
    cdef int[::1] n_max = np.arange(M, dtype='int32')

    with nogil, parallel(num_threads=8):
        for m in prange(M, schedule='dynamic'):

            # print(m)
            cm = int(nodes[m])
            p_bkgd = lambda0[cm]
            sum = p_bkgd
            n_min[m] = m
            n_max[m] = m

            # with gil:
            #     print("m={}, cm={}, n_min={}, n_max={}".format(m, cm, n_min[m], n_max[m]))

            # Backwards
            for n in range(m - 1, -1, -1):
                dt = times[m] - times[n]
                if (dt < dt_max) and (dt == 0):  # concurrent event (not a potential parent)
                    n_min[m] = n
                    n_max[m] = n
                if (dt < dt_max) and (dt > 0):  # event in window (potential parent)
                    n_min[m] = n
                    cn = int(nodes[n])
                    Z = dt * (dt_max - dt) / dt_max * SQRT_2PI / sqrt(tau[cn, cm])
                    L = W[cn, cm] * exp(-tau[cn, cm] / 2. * ( log( (dt / dt_max) / (1 - (dt / dt_max) ) ) - mu[cn, cm]) ** 2) / Z
                    sum = sum + L
                    p[m, n] = L
                else:
                    break

            # with gil:
            #     print("m={}, cm={}, n_min={}, n_max={}".format(m, cm, n_min[m], n_max[m]))

            # Forwards
            u0 = p_bkgd / sum
            if u[m] < u0:
                parents[m] = -1
            else:
                for n in range(n_min[m], n_max[m]):
                    u0 = u0 + p[m, n] / sum
                    if u[m] < u0:
                        parents[m] = n
                        break

            # if parents[m] == 0:
            #     with gil:
            #         print("m={}, u[m]={}, u0={}".format(m, u[m], u0))


@cython.boundscheck(False)
def calculate_statistics_cython(double [::1] times, int [::1] nodes, int[::1] parents, double dt_max,
                                double[::1] M_0,
                                double[::1] M_m,
                                double[:, ::1] M_nm,
                                double[:, ::1] xbar_nm,
                                double[:, ::1] nu_nm):

    cdef int M = int(times.shape[0])
    cdef double dt
    cdef int m, n, cm, cn

    for m in range(M):
        cm = nodes[m]
        n = parents[m]
        M_m[cm] += 1  # any event on this node
        if n > -1:
            cn = nodes[n]
            dt = times[m] - times[n]
            if (dt < dt_max) and (dt > 0):
                M_nm[cn, cm] += 1  # cn -> cm events
                xbar_nm[cn, cm] += log(dt) - log(dt_max - dt)
            else:
                raise("Error: non-background parent outside of window!")
        else:
            M_0[cm] += 1  # background events

    cdef double [:, ::1] mu = np.divide(xbar_nm, M_nm)

    for m in range(M):
        cm = nodes[m]
        n = parents[m]
        if n > -1:
          cn = nodes[n]
          dt = times[m] - times[n]
          if (dt < dt_max) and (dt > 0):
              xbar_nm[cn, cm] = mu[cn, cm]
              nu_nm[cn, cm] += (log(dt) - log(dt_max - dt) - mu[cn, cm]) ** 2

@cython.boundscheck(False)
def calculate_statistics_openmp(double [::1] times, int [::1] nodes, int[::1] parents, double dt_max,
                                double[::1] M_0,
                                double[::1] M_n,
                                double[:, ::1] M_nm,
                                double[:, ::1] xbar_nm,
                                double[:, ::1] nu_nm):

    cdef int M = int(times.shape[0])
    cdef double dt
    cdef int m, n, cm, cn

    with nogil, parallel(num_threads=8):
        for m in prange(M, schedule='dynamic'):
            cm = nodes[m]
            n = parents[m]
            M_n[cm] += 1  # any event on this node
            if n > -1:
                cn = nodes[n]
                dt = times[m] - times[n]
                if (dt < dt_max) and (dt > 0):
                    M_nm[cn, cm] += 1  # cn -> cm events
                    xbar_nm[cn, cm] += log(dt) - log(dt_max - dt)
            else:
                M_0[cm] += 1  # background events

    cdef double [:, ::1] mu = np.divide(xbar_nm, M_nm)

    with nogil, parallel(num_threads=8):
        for m in prange(M, schedule='dynamic'):
            cm = nodes[m]
            n = parents[m]
            if n > -1:
              cn = nodes[n]
              dt = times[m] - times[n]
              if (dt < dt_max) and (dt > 0):
                  xbar_nm[cn, cm] = mu[cn, cm]
                  nu_nm[cn, cm] += (log(dt) - log(dt_max - dt) - mu[cn, cm]) ** 2

@cython.boundscheck(False)
def sample_parents_discrete(int [:, ::1] S,
                            int max_size,
                            double [:, :, ::1] Shat,
                            double [:, ::1] Lambda,
                            double [:, :, ::1] parents,  # T x N x (1 + N * B)
                            double [::1] lambda0,
                            double [:, ::1] W,
                            double [:, :, ::1] theta):

    cdef int T = Shat.shape[0]
    cdef int N = Shat.shape[1]
    cdef int B = Shat.shape[2]
    cdef int t, n, m, b
    cdef double [::1] u = np.zeros(N * B + 1, dtype='float64')
    cdef int [::1] sample = np.zeros(N * B + 1, dtype='int32')
    cdef double [:, :, ::1] unif = np.random.uniform(size=(T, N, max_size))
    cdef double cut = 0
    cdef int i, j

    for t in range(T):
        for n in range(N):
            # Calculate u (1 + N * B)
            u[0] = lambda0[n] / Lambda[t, n]
            sample[0] = 0
            for m in range(N):
                for b in range(B):
                    sample[1 + (m * B) + b] = 0  # reset sample
                    u[1 + (m * B) + b] = (Shat[t, m, b] * W[m, n] * theta[b, m, n]) / Lambda[t, n]

            # Sample w (S[t,n] x (1 + N * B))
            for i in range(S[t, n]):
                cut = u[0]
                if unif[t, n, i] < cut:
                    sample[0] = sample[0] + 1
                    # print("{} / {} = background".format(i + 1, S[t,n]))
                else:
                    for j in range(1, 1 + N * B):
                        cut = cut + u[j]
                        if unif[t, n, i] < cut:
                            sample[j] = sample[j] + 1
                            # print("{} / {} = node {}".format(i + 1, S[t,n], j))
                            break

            parents[t, n, 0] = sample[0]
            for m in range(N):
                for b in range(B):
                    parents[t, n, 1 + (m * B) + b] = sample[1 + (m * B) + b]

# @cython.boundscheck(False)
# def sample_parents_discrete_openmp(int [:, ::1] S,
#                                    int max_size,
#                                    double [:, :, ::1] Shat,
#                                    double [:, ::1] Lambda,
#                                    double [:, :, ::1] parents,  # T x N x (1 + N * B)
#                                    double [::1] lambda0,
#                                    double [:, ::1] W,
#                                    double [:, :, ::1] theta):
#
#     cdef int T = Shat.shape[0]
#     cdef int N = Shat.shape[1]
#     cdef int B = Shat.shape[2]
#     cdef int t, n, m, b
#     cdef double [::1] u = np.zeros(N * B + 1, dtype='float64')
#     cdef int [::1] sample = np.zeros(N * B + 1, dtype='int32')
#     cdef double [:, :, ::1] unif = np.random.uniform(size=(T, N, max_size))
#     # cdef int idx = 0
#     cdef double cut = 0
#     cdef int i, j
#
#
#     # with nogil, parallel(num_threads=8):
#         # for t in prange(T, schedule='dynamic'):
#     for t in range(T):
#         for n in range(N):
#             # Calculate u (1 + N * B)
#             u[0] = lambda0[n] / Lambda[t,n]
#             for m in range(N):
#                 for b in range(B):
#                     sample[1 + (m * B) + b] = 0  # reset sample
#                     u[1 + (m * B) + b] = (Shat[t, m, b] * W[m, n] * theta[b, m, n]) / Lambda[t, n]
#
#             # Sample w (S[t,n] x (1 + N * B))
#             # sample = np.random.multinomial(S[t,n], u).astype('int32')
#             for i in range(S[t,n]):  # i not used, but `for _ in range' not allowed without gil
#                 cut = u[0]
#                 if unif[t, n, i] < cut:
#                     sample[0] = sample[0] + 1
#                     # idx = idx + 1
#                     # print('{}/{} (t={}, n={}, m={}, b={})'.format(idx, size, t, n, m, b))
#                 else:
#                     for j in range(1, 1 + N * B):
#                         cut = cut + u[j]
#                         if unif[t, n, i] < cut:
#                             sample[j] = sample[j] + 1
#                             # idx = idx + 1
#                             # print('{}/{} (t={}, n={}, m={}, b={})'.format(idx, size, t, n, m, b))
#                             break
#
#         parents[t, n, 0] = sample[0]
#         for m in range(N):
#             for b in range(B):
#                 parents[t, n, 1 + (m * B) + b] = sample[1 + (m * B) + b]

@cython.boundscheck(False)
def calculate_stats_discrete(int [:, :, ::] parents,
                             int [:, ::1] spikes,
                             double dt,
                             double [::1] alpha,
                             double [::1] beta,
                             double [:, ::1] nu,
                             double [:, ::1] kappa,
                             double [:, :, ::1] gamma,
                             double alpha0,
                             double beta0,
                             double [:, ::1] nu0,
                             double kappa0,
                             double [::1] gamma0):

    cdef int T = spikes.shape[0]
    cdef int N = spikes.shape[1]
    cdef int B = gamma.shape[2]
    cdef int t, n, m, b

    cdef int alpha_sum = 0
    cdef int kappa_sum = 0
    cdef int nu_sum = 0
    cdef int [::1] gamma_sum = np.zeros(B, dtype='int32')

    # Bias
    for n in range(N):
        for t in range(T):
            alpha_sum = alpha_sum + parents[t, n, 0]
        alpha[n] = alpha0 + alpha_sum
        beta[n] = beta0 + (T * dt)
        alpha_sum = 0

    # weights
    for n in range(N):
        for m in range(N):
            for t in range(T):
                nu_sum = nu_sum + spikes[t, m]
                for b in range(B):
                    kappa_sum = kappa_sum + parents[t, n, 1 + (m * B) + b]
            nu[m,n] = nu0[m,n] + nu_sum
            kappa[m,n] = kappa0 + kappa_sum
            kappa_sum = 0
            nu_sum = 0

    # impulse
    for n in range(N):
        for m in range(N):
            for b in range(B):
                for t in range(T):
                    gamma_sum[b] = gamma_sum[b] + parents[t, n, 1 + (m * B) + b]
                gamma[n, m, b] = gamma0[b] + gamma_sum[b]
                gamma_sum[b] = 0

# @cython.boundscheck(False)
# def calculate_stats_discrete_openmp(int [:, :, ::] parents,
#                                     int [:, ::1] spikes,
#                                     double dt,
#                                     double [::1] alpha,
#                                     double [::1] beta,
#                                     double [:, ::1] nu,
#                                     double [:, ::1] kappa,
#                                     double [:, :, ::1] gamma,
#                                     double alpha0,
#                                     double beta0,
#                                     double [:, ::1] nu0,
#                                     double kappa0,
#                                     double [::1] gamma0):
#
#     cdef int T = spikes.shape[0]
#     cdef int N = spikes.shape[1]
#     cdef int B = gamma.shape[2]
#     cdef int t, n, m, b
#
#     cdef int [::1] alpha_sum = np.zeros(N, dtype='int32')
#     cdef int [:, ::1] kappa_sum = np.zeros((N, N), dtype='int32')
#     cdef int [:, ::1] nu_sum = np.zeros((N, N), dtype='int32')
#     cdef int [:, :, ::1] gamma_sum = np.zeros((B, N, N), dtype='int32')
#
#     # This is wrong!
#     with nogil, parallel(num_threads=8):
#         for n in prange(N, schedule='dynamic'):
#             for m in range(N):
#                 for b in range(B):
#                     for t in range(T):
#                         alpha_sum[n] = alpha_sum[n] + parents[t, n, 0]
#                         kappa_sum[m, n] = kappa_sum[m, n] + parents[t, n, 1 + (m * B) + b]
#                         gamma_sum[n, m, b] = gamma_sum[n, m, b] + parents[t, n, 1 + (m * B) + b]
#                     nu_sum[m, n] = nu_sum[m, n] + spikes[t, m]
#                     gamma[n, m, b] = gamma0[b] + gamma_sum[n, m, b]
#                     # gamma_sum = 0
#                 nu[m,n] = nu0[m,n] + nu_sum[m, n]
#                 kappa[m,n] = kappa0 + kappa_sum[m, n]
#                 # kappa_sum = 0
#                 # nu_sum = 0
#             alpha[n] = alpha0 + alpha_sum[n]
#             beta[n] = beta0 + (T * dt)
#             alpha_sum = 0

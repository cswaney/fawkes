import numpy as np
from numpy.random import gamma
from fawkes.extensions import calculate_statistics_cython, calculate_stats_discrete
import time

FLAGS_VERBOSE = False

# TODO: How to set xbar and nubar when all events are due to background?

def normal_gamma(mu, kappa, alpha, beta, size=1):
    if size == 1:
        T = np.random.gamma(alpha, (1 / beta))  # numpy uses (shape, scale) = (k, theta) = (alpha, 1 / beta) = (shape, rate)
        X = np.random.normal(mu, 1 / (kappa * T))
    else:
        T = np.random.gamma(alpha, (1 / beta), size=size)  # numpy uses (shape, scale) = (k, theta) = (alpha, 1 / beta) = (shape, rate)
        X = np.random.normal(mu, 1 / (kappa * T))
    return X, T

def get_parent_nodes(nodes, parents):
    """Finds the node of the parent of each event.
        nodes: list of integers in [0, ..., N - 1]
        parents: list of integers in [0=background, 1, ..., M - 1].
    """
    parent_nodes = np.empty(len(nodes))
    for i in range(len(nodes)):
        if parents[i] == 0:  # background event means the node caused itself
            parent_nodes[i] = nodes[i]
        else:
            parent_nodes[i] = nodes[parents[i] - 1]
    return parent_nodes.astype('int')

# Continuous-Time Models
class GammaBias():
    """K-dimensional Gamma prior with Gibbs sampling.

    The model is:
        prior = Gamma(alpha, beta)
        likelihood = ...
        posterior (conditional) = Gamma(alpha0, beta0)

    """

    def __init__(self, N, alpha, beta):
        self.N = N
        self.alpha = alpha
        self.beta = beta

    def prior(self, size=1):
        if size == 1:
            size = self.N
        else:
            size = (size, self.N)
        return np.random.gamma(shape=self.alpha, scale=(1 / self.beta), size=size)

    def log_likelihood(self, data):
        pass

    def sample(self, times, nodes, parents, T):
        """Sample from conditional posterior."""
        lamb = np.zeros(self.N)
        for n in range(self.N):
            cp = parents[ nodes == n ]  # parents of events on node n...
            cp = cp[ cp == 0 ]  # ... where parent is background.
            Mn0 = len(cp)
            a = self.alpha + Mn0
            b = self.beta + T  # rate = 1 / scale
            lamb[n] = np.random.gamma(shape=a, scale=(1 / b))
        return lamb

class GammaWeights():
    """ Gamma prior for the weight matrix.

    prior: Gamma(kappa, nu)
    likelihood: ...
    posterior: Gamma(kappa0, nu0)

    """

    def __init__(self, N, adj, kappa, nu):
        self.N = N
        self.adj = adj
        self.kappa = kappa
        self.nu = nu  # N x N

    def prior(self, size=1):
        if size == 1:
            return self.adj * np.random.gamma(shape=self.kappa,
                                              scale=(1 / self.nu))
        else:
            return self.adj * np.random.gamma(shape=self.kappa,
                                              scale=(1 / self.nu),
                                              size=(size, self.N, self.N))

    def log_likelihood(self, data):
        pass

    def sample(self, times, nodes, parents):
        weights = np.zeros((self.N, self.N))
        parent_nodes = get_parent_nodes(nodes, parents)

        # Remove background events!
        nodes_noback = nodes[ parents > 0 ]
        parent_nodes_noback = parent_nodes[ parents > 0 ]

        for n in range(self.N):  # parent node
            for m in range(self.N):  # event node
                cp = parent_nodes_noback[ nodes_noback == m ]  # parent nodes of events on node m...
                cp = cp[ cp == n ]  # ... where parent node is n.
                Mnm = len(cp)
                kappa = self.kappa + Mnm
                cp = nodes[ nodes == n ]
                Mn = len(cp)
                nu = self.nu[n,m] + Mn  # rate = 1 / scale
                weights[n,m] = np.random.gamma(shape=kappa, scale=(1 / nu))
        return weights * self.adj

class NormalGammaImpulse():
    """ Gamma prior for the weight matrix.

    prior: NormalGamma(mu, kappa, alpha, beta)
    likelihood: ...
    posterior: NormalGamma(mu0, kappa0, alpha0, beta0)

    """
    def __init__(self, N, mu, kappa, alpha, beta):
        self.N = N
        self.mu = mu
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

    def prior(self, size=1):
        if size == 1:
            mu, tau = normal_gamma(self.mu,
                                   self.kappa,
                                   self.alpha,
                                   self.beta,
                                   size=(self.N, self.N))
        else:
            mu, tau = normal_gamma(self.mu,
                                   self.kappa,
                                   self.alpha,
                                   self.beta,
                                   size=(size, self.N, self.N))
        return mu, tau

    def log_likelihood(self, data):
        pass

    def sample(self, times, nodes, parents, dt_max):
        """Performs sampling from conditional posterior distribution.

        The prior is (mu, tau) ~ NG(mu, kappa, alpha, beta). The posterior is
        also Normal-Gamma: (mu, tau) | X ~ NG(mu0, kappa0, alpha0, beta0).

        Args:
            times: list of event times in range [0, T].
            nodes: list of event nodes in [0, ..., N - 1].
            parents: list of parent events in [0 = "background", 1, ..., M].

        Returns:

        """

        # Placeholders
        Mu = np.zeros((self.N, self.N))
        Tau = np.zeros((self.N, self.N))

        # Remove background events
        s = times[ parents > 0 ]
        c = nodes[ parents > 0 ]
        w = parents[ parents > 0 ]

        for n in range(self.N):  # parent node
            for m in range(self.N):  # event node

                sm = s[ c == m ]  # node m events
                wm = w[ c == m ]  # parents of node m events
                cm = nodes[ wm - 1 ]  # parent nodes of node m events
                snm = sm[ cm == n ]  # times of node m events with node n parents
                wnm = wm[ cm == n ]  # parents of node m events that are node n events

                s_events = snm  # times of node m events with node n parents
                s_parents = times[ wnm - 1 ]  # times of node n parents of node m events

                # Compute sufficient statistics
                Mn = len(nodes[ nodes == n ])
                Mnm = len(snm)
                x = np.log( (s_events - s_parents) / (dt_max - (s_events - s_parents)))
                if len(x) > 0:
                    xbar = np.mean(x)
                    nubar = np.var(x) * Mnm
                else:
                    print("No events found in impulse parameter sampling step.")
                    xbar = 0.1
                    nubar = 0.1

                # Random sample
                mu = (self.kappa * self.mu + Mnm * xbar) / (self.kappa + Mnm)
                kappa = self.kappa + Mnm
                alpha = self.alpha + (Mnm / 2)
                beta = (nubar / 2) + (Mnm * self.kappa * (xbar - self.mu) ** 2) / (2 * (Mnm + self.kappa))
                mu, tau = normal_gamma(mu, kappa, alpha, beta)
                Mu[n,m] = mu
                Tau[n,m] = tau
        return Mu, Tau

class NetworkPoisson():

    def __init__(self, N, alpha_0, beta_0, A, kappa, nu, mu_mu, kappa_mu,
                 alpha_tau, beta_tau):
        self.N = N
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.A = A
        self.kappa = kappa
        self.nu = nu
        self.mu_mu = mu_mu
        self.kappa_mu = kappa_mu
        self.alpha_tau = alpha_tau
        self.beta_tau = beta_tau

    def prior(self, size=1):
        # sample bias
        if size == 1:
            lamb = np.random.gamma(shape=self.alpha_0, scale=(1 / self.beta_0), size=self.N)
        else:
            lamb = np.random.gamma(shape=self.alpha_0, scale=(1 / self.beta_0), size=(size, self.N))

        # sample weights
        if size == 1:
            W = self.A * np.random.gamma(shape=self.kappa,
                                           scale=(1 / self.nu))
        else:
            W = self.A * np.random.gamma(shape=self.kappa,
                                           scale=(1 / self.nu),
                                           size=(size, self.N, self.N))

        # sample impulse
        if size == 1:
            mu, tau = normal_gamma(self.mu_mu,
                                   self.kappa_mu,
                                   self.alpha_tau,
                                   self.beta_tau,
                                   size=(self.N, self.N))
        else:
            mu, tau = normal_gamma(self.mu_mu,
                                   self.kappa_mu,
                                   self.alpha_tau,
                                   self.beta_tau,
                                   size=(size, self.N, self.N))
        return lamb, W, mu, tau

    def calculate_stats(self, data, parents, T, dt_max):

        start = time.time()
        times, nodes = data

        # sufficient statistics
        M_0 = np.zeros(self.N)
        M_n = np.zeros(self.N)
        M_mn = np.zeros((self.N,self.N))
        xbar_mn = np.zeros((self.N,self.N))
        nu_mn = np.zeros((self.N,self.N))

        # Direct Method
        X = [ [ [] for n in range(self.N)] for n in range(self.N)]
        for t in range(len(times)):
            n = nodes[t]
            # M_n
            M_n[n] += 1
            sn = times[t]
            w = parents[t]  # (0 = background, 1, ..., M - 1 = last possible parent)
            if w > 0:  # if not a background event...
                m = nodes[w - 1]  # parent node
                sm = times[w - 1]  # parent time
                # M_mn
                M_mn[m,n] += 1
                if (sn - sm >= dt_max):
                    print("Error: t={}, dt = {}, dt_max = {}".format(t, sn - sm, dt_max))
                x = np.log((sn - sm) / (dt_max - (sn - sm)))
                X[m][n].append(x)
            else:
                M_0[n] += 1

        # xbar_mn, nu_mn
        for n in range(self.N):  # event node
            for m in range(self.N):  # parent node
                if len(X[m][n]) > 0:
                    xbar_mn[m,n] = np.mean(X[m][n])
                    nu_mn[m,n] = np.var(X[m][n]) * M_mn[m,n]
                else:
                    # print("No {} -> {} events found.".format(m,n))
                    xbar_mn[m,n] = 0.1
                    nu_mn[m,n] = 0.1

        assert M_mn.sum() + M_0.sum() == len(times), "Event count error 1"
        assert M_n.sum() == len(times), "Event count error 2"

        # check for zeros
        # if FLAGS_VERBOSE:
        #     print('M_mn={}'.format(M_mn))
        #     print('xbar_mn={}'.format(xbar_mn))
        #     print('nu_mn={}'.format(nu_mn))
        xbar_mn[ xbar_mn == 0 ] = 0.1
        nu_mn[ nu_mn == 0 ] = 0.1

        stop = time.time()
        if FLAGS_VERBOSE:
            print('Calculateed statistics in {} seconds.'.format(stop - start))
        return M_0, M_n, M_mn, xbar_mn, nu_mn

    # @profile
    def sample(self, data, parents, T, dt_max):

        start = time.time()
        times, nodes = data

        # sufficient statistics
        M_0n = np.zeros(self.N)
        M_n = np.zeros(self.N)
        M_mn = np.zeros((self.N,self.N))
        xbar_mn = np.zeros((self.N,self.N))
        nu_mn = np.zeros((self.N,self.N))

        # Direct Method
        X = [ [ [] for n in range(self.N)] for n in range(self.N)]
        for t in range(len(times)):
            n = nodes[t]
            # M_n
            M_n[n] += 1
            sn = times[t]
            w = parents[t]  # (0 = background, 1, ..., M - 1 = last possible parent)
            if w > 0:  # if not a background event...
                m = nodes[w - 1]  # parent node
                sm = times[w - 1]  # parent time
                # M_mn
                M_mn[m,n] += 1
                x = np.log((sn - sm) / (dt_max - (sn - sm)))
                X[m][n].append(x)
            else:
                M_0n[n] += 1

        # xbar_mn, nu_mn
        for n in range(self.N):  # event node
            for m in range(self.N):  # parent node
                if len(X[m][n]) > 0:
                    xbar_mn[m,n] = np.mean(X[m][n])
                    nu_mn[m,n] = np.var(X[m][n]) * M_mn[m,n]
                else:
                    # print("No {} -> {} events found.".format(m,n))
                    xbar_mn[m,n] = 0.1
                    nu_mn[m,n] = 0.1

        assert M_mn.sum() + M_0n.sum() == len(times), "Event count error 1"
        assert M_n.sum() == len(times), "Event count error 2"

        # check for zeros
        if FLAGS_VERBOSE:
            print('M_mn={}'.format(M_mn))
            print('xbar_mn={}'.format(xbar_mn))
            print('nu_mn={}'.format(nu_mn))
        xbar_mn[ xbar_mn == 0 ] = 0.01
        nu_mn[ nu_mn == 0 ] = 0.01

        # calculate posterior parameters
        alpha_0 = self.alpha_0 + M_0n
        beta_0 = self.beta_0 + T
        kappa = self.kappa + M_mn
        nu = self.nu + M_n.reshape((self.N, 1))
        mu_mu = (self.kappa_mu * self.mu_mu + M_mn * xbar_mn) / (self.kappa_mu + M_mn)
        kappa_mu = self.kappa_mu + M_mn
        alpha_tau = self.alpha_tau + (M_mn / 2)
        beta_tau = (nu_mn / 2) + (M_mn * self.kappa_mu * (xbar_mn - self.mu_mu) ** 2) / (2 * (M_mn + self.kappa_mu))

        # sample posteriors
        lamb = gamma(alpha_0, (1 / beta_0))  # N x 1 parameters
        W = self.A * gamma(kappa, (1 / nu))  # N x N parameters
        mu, tau = normal_gamma(mu_mu, kappa_mu, alpha_tau, beta_tau)  # N x N parameters
        stop = time.time()
        if FLAGS_VERBOSE:
            print('Sampled parameters in {} seconds.'.format(stop - start))
        return lamb, W, mu, tau

    def calculate_stats_ext(self, data, parents, T, dt_max, method='cython'):
        start = time.time()
        times, nodes = data

        # calculate sufficient statistics
        M_0 = np.zeros(self.N)
        M_m = np.zeros(self.N)
        M_nm = np.zeros((self.N,self.N))
        xbar_nm = np.zeros((self.N,self.N))
        nu_nm = np.zeros((self.N,self.N))
        if method == 'cython':
            calculate_statistics_cython(np.array(times, dtype='float64'),
                                        np.array(nodes, dtype='int32'),
                                        np.array(parents, dtype='int32'),
                                        dt_max,
                                        M_0, M_m, M_nm, xbar_nm, nu_nm)
        elif method == 'openmp':
            calculate_statistics_openmp(np.array(times, dtype='float64'),
                                        np.array(nodes, dtype='int32'),
                                        np.array(parents, dtype='int32'),
                                        dt_max,
                                        M_0, M_m, M_nm, xbar_nm, nu_nm)

        # check for zeros
        # if FLAGS_VERBOSE:
            # print('M_nm={}'.format(M_nm))
            # print('xbar_nm={}'.format(xbar_nm))
            # print('nu_nm={}'.format(nu_nm))
        xbar_nm[ xbar_nm == 0 ] = 0.1
        nu_nm[ nu_nm == 0 ] = 0.1

        stop = time.time()
        if FLAGS_VERBOSE:
            print('Calculateed statistics in {} seconds.'.format(stop - start))
        return M_0, M_m, M_nm, xbar_nm, nu_nm

    # @profile
    def sample_ext(self, data, parents, T, dt_max, method='cython'):
        start = time.time()
        times, nodes = data

        # calculate sufficient statistics
        M_0 = np.zeros(self.N)
        M_m = np.zeros(self.N)
        M_nm = np.zeros((self.N,self.N))
        xbar_nm = np.zeros((self.N,self.N))
        nu_nm = np.zeros((self.N,self.N))
        if method == 'cython':
            calculate_statistics_cython(np.array(times, dtype='float64'),
                                        np.array(nodes, dtype='int32'),
                                        np.array(parents, dtype='int32'),
                                        dt_max,
                                        M_0, M_m, M_nm, xbar_nm, nu_nm)
        elif method == 'openmp':
            calculate_statistics_openmp(np.array(times, dtype='float64'),
                                        np.array(nodes, dtype='int32'),
                                        np.array(parents, dtype='int32'),
                                        dt_max,
                                        M_0, M_m, M_nm, xbar_nm, nu_nm)

        # check for zeros
        if FLAGS_VERBOSE:
            print('M_nm={}'.format(M_nm))
            print('xbar_nm={}'.format(xbar_nm))
            print('nu_nm={}'.format(nu_nm))
        # xbar_nm[ xbar_nm == 0 ] = np.random.choice((-0.1, 0.1))
        # nu_nm[ nu_nm == 0 ] = np.random.choice((-0.1, 0.1))
        xbar_nm[ xbar_nm == 0 ] = 0.1
        nu_nm[ nu_nm == 0 ] = 0.1

        # calculate posterior parameters
        alpha_0 = self.alpha_0 + M_0
        beta_0 = self.beta_0 + T
        kappa = self.kappa + M_nm
        nu = self.nu + M_m.reshape((self.N, 1))
        mu_mu = (self.kappa_mu * self.mu_mu + M_nm * xbar_nm) / (self.kappa_mu + M_nm)
        kappa_mu = self.kappa_mu + M_nm
        alpha_tau = self.alpha_tau + (M_nm / 2)
        beta_tau = (nu_nm / 2) + (M_nm * self.kappa_mu * (xbar_nm - self.mu_mu) ** 2) / (2 * (M_nm + self.kappa_mu))

        # sample posteriors
        lamb = gamma(alpha_0, (1 / beta_0))  # N x 1 parameters
        W = self.A * gamma(kappa, (1 / nu))  # N x N parameters
        mu, tau = normal_gamma(mu_mu, kappa_mu, alpha_tau, beta_tau)  # N x N parameters
        stop = time.time()
        if FLAGS_VERBOSE:
            print('Sampled parameters in {} seconds.'.format(stop - start))
        return lamb, W, mu, tau


# Discrete-Time Models
class DiscreteNetworkPoisson():

    def __init__(self, N, B, dt, alpha_0, beta_0, kappa, nu, gamma):
        self.N = N
        self.B = B
        self.dt = dt
        self.alpha_0, self.beta_0 = (alpha_0, beta_0)  # background rate hypers
        self.kappa, self.nu = (kappa, nu)  # connection weight hypers
        self.gamma = gamma  # impulse function hypers

    def prior(self, size=1):

        # sample bias
        if size == 1:
            lamb = np.random.gamma(shape=self.alpha_0, scale=(1 / self.beta_0), size=self.N)
        else:
            lamb = np.random.gamma(shape=self.alpha_0, scale=(1 / self.beta_0), size=(size, self.N))

        # sample weights
        if size == 1:
            W = np.random.gamma(shape=self.kappa, scale=(1 / self.nu))
        else:
            W = np.random.gamma(shape=self.kappa,
                                scale=(1 / self.nu),
                                size=(size, self.N, self.N))

        # sample impulse
        if size == 1:
            theta = np.random.dirichlet(self.gamma, size=(self.N, self.N)).transpose()  # B x N x N
        else:
            theta = np.empty((self.B, self.N, self.N, size))
            for i in range(size):
                theta[:, :, :, i] = normal_gamma(self.gamma, size=(self.N, self.N)).transpose()  # B x N x N x size
        return lamb, W, theta

    def sample_bias(self, parents):
        T, _, _ = parents.shape
        a = self.alpha_0 + parents[:, :, 0].sum(axis=0)
        b = self.beta_0 * np.ones(self.N) + T * self.dt
        return np.random.gamma(a, (1 / b))

    def sample_weights(self, parents, spikes):
        W = np.empty((self.N, self.N))
        for n in range(self.N):  # event
            for m in range(self.N):  # parent
                parents_mb = parents[:, n, (1 + m * self.B):(1 + (m + 1) * self.B)]
                nu_mn = self.nu[m, n] + spikes[:, m].sum(axis=0)
                kappa_mn = self.kappa + parents_mb.sum()
                W[m, n] = np.random.gamma(kappa_mn, (1 / nu_mn))
        return W

    def sample_impulse(self, parents):

        Theta = np.empty((self.B, self.N, self.N))
        for n in range(self.N):  # event node
            for m in range(self.N):  # parent node
                g = self.gamma + parents[:, n, (1 + m * self.B):(1 + (m + 1) * self.B)].sum(axis=0)
                # g = np.zeros(self.B)
                # for b in range(self.B):
                #     g[b] = self.gamma[b] + parents[:, n, 1 + m * self.B + b].sum()
                Theta[:, m, n] = np.random.dirichlet(g)
        return Theta

    # @profile
    def sample(self, spikes, parents):
        lambda0 = self.sample_bias(parents)
        W = self.sample_weights(parents, spikes)
        Theta = self.sample_impulse(parents)
        return lambda0, W, Theta

    def calculate_stats(self, spikes, parents):

        T, _, _ = parents.shape
        alpha = self.alpha_0 + parents[:, :, 0].sum(axis=0)
        beta = self.beta_0 * np.ones(self.N) + T * self.dt

        nu = np.empty((self.N, self.N))
        kappa = np.empty((self.N, self.N))
        for n in range(self.N):  # event
            for m in range(self.N):  # parent
                parents_mb = parents[:, n, (1 + m * self.B):(1 + (m + 1) * self.B)]
                nu[m, n] = self.nu[m, n] + spikes[:, m].sum(axis=0)
                kappa[m,n] = self.kappa + parents_mb.sum()

        gamma = np.empty((self.N, self.N, self.B))
        for n in range(self.N):  # event node
            for m in range(self.N):  # parent node
                # g = self.gamma + parents[:, n, (1 + m * self.B):(1 + (m + 1) * self.B)].sum(axis=0)
                for b in range(self.B):
                    gamma[n, m, b] = self.gamma[b] + parents[:, n, 1 + m * self.B + b].sum()

        return alpha, beta, nu, kappa, gamma

    # @profile
    def sample_ext(self, spikes, parents):

        # Placeholders
        alpha = np.zeros(self.N)
        beta = np.zeros(self.N)
        nu = np.zeros((self.N, self.N))
        kappa = np.zeros((self.N, self.N))
        gamma = np.zeros((self.N, self.N, self.B))

        # Compute statistics
        calculate_stats_discrete(parents.astype('int32'), spikes.astype('int32'),
                                 self.dt, alpha, beta, nu, kappa, gamma,
                                 self.alpha_0, self.beta_0, self.nu, self.kappa, self.gamma)

        # Sample
        lambda0 = np.random.gamma(alpha, (1 / beta))  # N
        W = np.random.gamma(kappa, (1 / nu))  # N x N
        Theta = np.empty((self.B, self.N, self.N))
        for n,m in zip(range(self.N), range(self.N)):
            Theta[:, m, n] = np.random.dirichlet(gamma[n, m, :])  # B x N x N
        return lambda0, W, Theta

    def calculate_stats_ext(self, spikes, parents, method='Cython'):
        # Placeholders
        alpha = np.zeros(self.N)
        beta = np.zeros(self.N)
        nu = np.zeros((self.N, self.N))
        kappa = np.zeros((self.N, self.N))
        gamma = np.zeros((self.N, self.N, self.B))

        # Compute statistics
        if method == 'cython':
            calculate_stats_discrete(parents.astype('int32'), spikes.astype('int32'),
                                     self.dt, alpha, beta, nu, kappa, gamma,
                                     self.alpha_0, self.beta_0, self.nu, self.kappa, self.gamma)
        elif method == 'openmp':
            raise(NotImplementedError)
        else:
            raise(NotImplementedError)
        return alpha, beta, nu, kappa, gamma

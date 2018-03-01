import numpy as np
import fawkes.extensions as ext
import fawkes.priors as priors
import time
import matplotlib.pyplot as plt

FLAGS_VERBOSE = False

def logistic(x, xmax=1, k=1, x0=0):
    return xmax / (1 + np.exp(-k * (x - x0)))

def logit_normal(size, m=0, s=1, xmax=1, k=1, x0=0):
    x = np.random.normal(loc=m, scale=s, size=size)
    return logistic(x, xmax=xmax, k=k, x0=x0)

def normal_gamma(size, mu, kappa, alpha, beta):
    T = np.random.gamma(alpha, beta, size=size)
    X = np.random.normal(mu, 1 / (kappa * T))
    return X, T

class HomogeneousPoisson():
    """N-dimensional homogeneous Poisson model.

    N: number of Poisson processes.
    params: a dict of parameter values:
        - lamb: vector of background rates.
    hypers: a dict of hyperparamter values:
        - alpha_0: background hyper.
        - beta_0: background hyper.
    """

    def __init__(self, N, lambda0=None, hypers=None):

        # Basic
        self.N = N

        # Parameters
        if lambda0 is not None:  # Set parameters from `params`.
            self.lambda0 = params['lambda0']
        else:  # Set parameters to default values.
            self.lambda0 = np.ones(self.N)

        # Hyperparameters
        if hypers is not None:  # Set hyperparamters from `hypers`.
            # assert hypers have the correct shape.
            pass
        else:  # Set hyperparamters to default values.
            self.alpha_0 = 1
            self.beta_0 = 1

        # Priors
        self.bias_model = priors.GammaBias(self.N, self.alpha_0, self.beta_0)
        self.model = priors.HomogeneousPoisson(self.N, self.alpha_0, self.beta_0)

    def init_parameters(self):
        """Sample the model parameters from the prior distribution."""
        print("Sampling model parameters from priors.")
        self.lambda0 = np.random.gamma(self.alpha_0, 1 / self.beta_0, size=(self.N,))
        print("Model parameters have been reset.")

    def generate_data(self, T):
        """Basic data generation based."""

        events = []
        nodes = []
        background_events = [[] for n in range(self.N)]

        for n in range(self.N):
            c = np.random.uniform(low=0, high=T, size=np.random.poisson(T * self.lambda0[n]))
            events.extend(c)
            nodes.extend([n] * len(c))
        idx = np.argsort(events)
        return np.array(events)[idx], np.array(nodes)[idx]

    def plot_data(self, data):
        """Plot intensity with events by node."""
        times, nodes = data
        T = np.ceil(np.max(times))
        grid = np.linspace(0, T, 1000)
        for n in np.unique(nodes):
            plt.plot(grid, self.lambda0[n] * np.ones(len(grid)), linestyle='dashed', alpha=0.5)
            plt.scatter(times[nodes==n], [self.lambda0[n]] * len(times[nodes==n]))
            plt.ylim([0, np.max(self.lambda0) + 0.5])
            plt.xlim([0, T])
        plt.show()

    def compute_likelihood(self, data, T, lambda0=None, log=True, size=None):
        """Compute the likelihood of event-time data, p(s | theta).

        data: list of form [timestamps, classes].
        log: compute log likelihood (default: True)

        """

        if lambda0 is None:
            lambda0 = self.lambda0

        times, nodes = data
        M = len(times)
        if (size is not None) and (M < size):
            print('Insufficient event data (n={}); returning None'.format(nobs))
            return None
        elif (size is not None):
            idx = np.random.randint(low=0, high=M-size)
            times = times[idx:idx + size]
            nodes = nodes[idx:idx + size]
            t0 = times[0]
            tN = times[-1]
            T = tN - t0
            times = times - t0
        # Likelihood computation
        idx, cnts = np.unique(nodes, return_counts=True)
        idx_list = list(idx)
        cnts_list = list(cnts)
        while idx_list[-1] != self.N - 1:
            idx_list.append(idx_list[-1] + 1)
            cnts_list.append(0)
        for i in range(self.N):
            if idx_list[i] != i:
                idx_list.insert(i,i)
                cnts_list.insert(i,0)
        cnts = np.array(cnts_list)
        ll = -T * lambda0.sum() + np.sum(np.log(lambda0) * cnts)
        return ll, T

    def compute_pred_likelihood(self, data, T, size, log=True):
        """Compute approximate predictive likelihood of event-time data, p(s_test | s_train).

        data: list of form [timestamps, classes].
        log: compute log likelihood (default: True)

        """

        lambda0_ = self.sample(data, T, size=size)
        mean = 0.0
        for i in range(size):
            mean += self.compute_likelihood(data, T, lambda0_[i,:])
        return mean / size

    def sample(self, data, T, size=1):
        """Sample the posterior distribution.

            Performs exact sampling if possible. Otherwise, performs Gibbs sampling.

        """

        # print("Sampling posterior...")
        start = time.time()
        bias = self.model.sample(data, T, size=size)
        stop = time.time()
        # print("Performed {:d} sampling steps in {:.3f} seconds.".format(size, stop - start))
        return bias

class NetworkPoisson():
    """N-dimensional Bayesian network model.

    N: number of network nodes.
    dt_max: support of the impulse response.
    params: a dict of parameter values:
        lamb: vector of background rates.
        A: binary network connection matrix.
        W: matrix of network weights.
    hypers: a dict of hyperparamter values:
        alpha_0: background hyper.
        beta_0: background hyper.
        kappa: weight hyper.
        nu: weight hyper.
        mu_mu: impulse hyper.
        kappa_mu: impulse hyper.
        alpha_tau: impulse hyper.
        beta_tau: impulse hyper.
    """

    def __init__(self, N, dt_max, params=None, hypers=None, model="dense"):

        # Basic
        self.N = N
        self.dt_max = dt_max
        self.model = model

        # Parameters
        if params is not None:  # Set parameters from `params`.
            self.lamb = params['lamb']
            self.W = params['weights']
            self.mu = params['mu']
            self.tau = params['tau']
            if 'adj' in params:
                self.A = params['adj']
            else:
                if self.model == 'empty':
                    self.A = np.zeros((self.N, self.N))
                elif self.model == 'diag':
                    self.A = np.diag(np.ones(self.N))
                elif self.model == 'dense':
                    self.A = np.ones((self.N, self.N))
                elif self.model == 'bernoulli':
                    self.A = np.random.bernoulli(1, self.rho, size=(self.N, self.N))
                elif self.model == 'block':
                    pass
                elif self.model == 'distance':
                    pass
        else:  # Set parameters to default values.
            # Bias
            self.lamb = np.ones(self.N)

            # Weights
            self.W = np.zeros((self.N, self.N))

            # Impulse
            self.mu = np.zeros((self.N, self.N))
            self.tau = np.ones((self.N, self.N))

            # Connections
            if self.model == 'empty':
                self.A = np.zeros((self.N, self.N))
            if self.model == 'diag':
                self.A = np.diag(np.ones(self.N))
            if self.model == 'dense':
                self.A = np.ones((self.N, self.N))
            if self.model == 'bernoulli':
                self.A = np.random.bernoulli(1, self.rho, size=(self.N, self.N))
            if self.model == 'block':
                pass
            if self.model == 'distance':
                pass

        # Hyperparameters
        if hypers is not None:  # Set hyperparamters from `hypers`.
            # assert hypers have the correct shape.
            pass
        else:  # Set hyperparamters to default values.
            # Bias
            self.alpha_0 = 1.
            self.beta_0 = 1.

            # Weights
            self.kappa = 1.
            self.nu = 1. * np.ones((self.N, self.N))  # N x N

            # Impulse
            self.mu_mu = 0.
            self.kappa_mu = 1.
            self.alpha_tau = 1.
            self.beta_tau = 1.

        # Impulse
        def logit_normal(dt, mu=self.mu, tau=self.tau):
            """mu and tau can be scalar or matirx/vector; dt is an element from an np.array."""
            # if dt < self.dt_max:
            #     Z = dt * (self.dt_max - dt) / dt_max * (tau / (2 * np.pi)) ** (-0.5)
            #     x = dt / self.dt_max
            #     s = np.log(x / (1 - x))
            #     return (1 / Z) * np.exp( -tau / 2 * (s - mu) ** 2 ) * (dt < self.dt_max)
            # else:
            #     if mu.shape == ():
            #         return 0
            #     else:
            #         # return np.zeros((self.N, self.N))
            #         return np.zeros(mu.shape)

            if dt <= 0:
                print(dt)
            assert (dt < self.dt_max).all(), "Tried to evaluate impulse for dt > dt_max"
            Z = dt * (self.dt_max - dt) / dt_max * (tau / (2 * np.pi)) ** (-0.5)
            x = dt / self.dt_max
            s = np.log(x / (1 - x))
            return (1 / Z) * np.exp( -tau / 2 * (s - mu) ** 2 )
        self.impulse = logit_normal  # (n,m) = (parent node, event node)

        # Priors
        self.bias_model = priors.GammaBias(self.N, self.alpha_0, self.beta_0)
        self.weights_model = priors.GammaWeights(self.N, self.A, self.kappa, self.nu)
        self.impulse_model = priors.NormalGammaImpulse(self.N,
                                                       self.mu_mu,
                                                       self.kappa_mu,
                                                       self.alpha_tau,
                                                       self.beta_tau)
        self.model = priors.NetworkPoisson(self.N,
                                           self.alpha_0,
                                           self.beta_0,
                                           self.A,
                                           self.kappa,
                                           self.nu,
                                           self.mu_mu,
                                           self.kappa_mu,
                                           self.alpha_tau,
                                           self.beta_tau)

    # TODO: use self.model.prior() instead?
    def init_parameters(self):
        """Sample the model parameters from prior distributions."""
        print("Sampling model parameters from priors.")
        self.lamb = np.random.gamma(self.alpha_0, 1 / self.beta_0, size=(self.N,))
        self.W = np.random.gamma(self.kappa, 1 / self.nu)
        self.mu, self.tau = normal_gamma((self.N, self.N),
                                         self.mu_mu,
                                         self.kappa_mu,
                                         self.alpha_tau,
                                         self.beta_tau)
        print("Model parameters have been reset.")

    def check_stability(self, A=None, W=None, return_value=False):
        """Check that the weight matrix is stable; returns the answers.

            return_value: also return the maximum eigenvalue.

        """

        if A is None:
            A = self.A
        if W is None:
            W = self.W

        if self.N < 100:
            eigs = np.linalg.eigvals(A * W)
            maxeig = np.amax(np.real(eigs))
        else:
            from scipy.sparse.linalg import eigs
            maxeig = eigs(A * W, k=1)[0]

        print("Max eigenvalue: {:.2f} (stable={})".format(maxeig, maxeig < 1.0))

        if return_value:
            return maxeig < 1., maxeig
        return maxeig < 1.

    def generate_data(self, T):
        """Data generation based on the superposition principle (Linderman, 2015)."""

        def generate_children(parents, n, T):
            """Children of node n parent events."""
            children = [[] for n in range(self.N)]
            for p in parents:
                for m in range(self.N):
                    size = np.random.poisson(self.A[n, m] * self.W[n, m])
                    c = self.dt_max * logit_normal(size=size, m=self.mu[n, m], s=(1 / self.tau[n, m]))
                    c = c[ p + c < T ]
                    children[m].extend(p + c)
            return children

        events = []
        nodes = []  # node experiencing event
        background_events = [[] for n in range(self.N)]

        # Generate parent events from background rates
        for n in range(self.N):
            c = np.random.uniform(low=0, high=T, size=np.random.poisson(T * self.lamb[n]))
            background_events[n].extend(c)
            events.extend(background_events[n])
            nodes.extend([n] * len(c))

        # Generate children events from parents iteratively
        parents = background_events
        # [print("length of node {} parents: {}".format(n, len(p))) for n,p in enumerate(parents)]
        while max([len(p) for p in parents]) > 0:
            children = [[] for n in range(self.N)]
            for n in range(self.N):
                if len(parents[n]) > 0:
                    c = generate_children(parents[n], n, T)
                    for m in range(self.N):
                        children[m].extend(c[m])
                        events.extend(c[m])
                        nodes.extend([m] * len(c[m]))
                        # print("node {} generated {} type {} events".format(n, len(c[m]), m))
            parents = children
            # [print("length of node {} parents: {}".format(n, len(p))) for n,p in enumerate(parents)]
            idx = np.argsort(events)
        return np.array(events)[idx], np.array(nodes)[idx]

    def generate_data_from_parent(self, T, parent):

        def generate_children(parents, n, T):
            """Children of node n parent events."""
            children = [[] for n in range(self.N)]
            for p in parents:
                for m in range(self.N):
                    size = np.random.poisson(self.A[n, m] * self.W[n, m])
                    c = self.dt_max * logit_normal(size=size, m=self.mu[n, m], s=(1 / self.tau[n, m]))
                    c = c[ p + c < T ]
                    children[m].extend(p + c)
            return children

        parents = [[] for n in range(self.N)]
        t,n = parent
        parents[n].extend([t])

        events = [t]
        nodes = [n]  # node experiencing event

        # Generate children events from parents iteratively
        # [print("length of node {} parents: {}".format(n, len(p))) for n,p in enumerate(parents)]
        while max([len(p) for p in parents]) > 0:
            children = [[] for n in range(self.N)]
            for n in range(self.N):
                if len(parents[n]) > 0:
                    c = generate_children(parents[n], n, T)
                    for m in range(self.N):
                        children[m].extend(c[m])
                        events.extend(c[m])
                        nodes.extend([m] * len(c[m]))
                        # print("node {} generated {} type {} events".format(n, len(c[m]), m))
            parents = children
            # [print("length of node {} parents: {}".format(n, len(p))) for n,p in enumerate(parents)]
            idx = np.argsort(events)
        return np.array(events)[idx], np.array(nodes)[idx]

    def simulate_spike(self, T, parent_node, parent_time, units='s'):

        def generate_children(parents, n, T):
            """Children of node n parent events."""
            children = [[] for n in range(self.N)]
            for p in parents:
                for m in range(self.N):
                    size = np.random.poisson(self.A[n, m] * self.W[n, m])
                    c = self.dt_max * logit_normal(size=size, m=self.mu[n, m], s=(1 / self.tau[n, m]))
                    c = c[ p + c < T ]
                    children[m].extend(p + c)
            return children

        times = []
        nodes = []
        parents = [ [] for n in range(self.N) ]
        parents[parent_node] = [parent_time]
        [print("length of node {} parents: {}".format(n, len(p))) for n,p in enumerate(parents)]

        while max([len(p) for p in parents]) > 0:
            children = [[] for n in range(self.N)]
            for n in range(self.N):
                if len(parents[n]) > 0:
                    c = generate_children(parents[n], n, T)
                    for m in range(self.N):
                        children[m].extend(c[m])
                        if units == 'ms':
                            times.extend([int(1000 * child) for child in c[m]])
                        else:
                            times.extend(c[m])
                        nodes.extend([m] * len(c[m]))
                        print("node {} generated {} type {} events".format(n, len(c[m]), m))
            parents = children
            [print("length of node {} parents: {}".format(n, len(p))) for n,p in enumerate(parents)]
        events = dict(zip(times, nodes))
        return events

    def plot_data(self, data):
        """Plot intensity with events by node."""
        times, nodes = data
        T = np.ceil(np.max(times))
        grid = np.linspace(0, T, 1000)
        I_grid = np.array([self.compute_intensity(data, t) for t in grid]).transpose()  # n x (T/N + 1)
        I_times = np.array([self.compute_intensity(data, t) for t in times]).transpose()  # n x M
        for n in np.unique(nodes):
            # plt.subplot(self.N, 1, n + 1)
            t = grid
            f_grid = I_grid[n,:]
            plt.plot(t, f_grid, alpha=0.2)
            t = times[ nodes == n ]
            f_times = I_times[n,:][ nodes == n ]
            plt.scatter(t, f_times)
            plt.ylim([0, np.max(f_times) + 1])
            plt.xlim([0, T])
        plt.show()
        # plt.clf()

    def plot_impulse(self, mu=None, tau=None):

            """mu and tau can be scalar or matirx/vector"""
            def impulse(dt, mu, tau):
                out = np.zeros(dt.shape)
                dt_ = dt[ (dt > 0) & (dt < self.dt_max) ]
                Z = dt_ * (self.dt_max - dt_) / self.dt_max * (tau / (2 * np.pi)) ** (-0.5)
                x = dt_ / self.dt_max
                s = np.log(x / (1 - x))
                out[ (dt > 0 ) & (dt < self.dt_max) ] = (1 / Z) * np.exp( -tau / 2 * (s - mu) ** 2 )
                return out

            grid = np.linspace(0, self.dt_max, 100)
            plt.plot(grid, impulse(grid, self.mu[0,0], self.tau[0,0]))
            if (mu is not None) and (tau is not None):
                plt.plot(grid, impulse(grid, mu, tau))
            plt.title("Impulse Response: mu={:.2f}, tau={:.2f}".format(mu, tau))
            plt.legend(['True', 'Estimate'])
            plt.show()

    # TODO: vectorize and/or parallelize.
    def compute_intensity(self, data, t, theta=None):
        """Compute the intensity using events up to time t."""
        times, nodes = data
        if theta is None:
            lamb, W, mu, tau = self.lamb, self.W, self.mu, self.tau
        else:
            lamb, W, mu, tau = theta

        # Only use past events.
        nodes = nodes[ times < t ]
        times = times[ times < t ]
        # Only use events in window.
        diffs = t - times
        nodes = nodes[ diffs < self.dt_max ]
        diffs = diffs[ diffs < self.dt_max ]
        # Calculate intensity.
        lamb = self.lamb.copy()
        for i in range(len(diffs)):
            m = nodes[i]
            dt = diffs[i]
            lamb += self.W[m,:] * self.impulse(dt, mu=mu, tau=tau)[m,:]
        return lamb

    # TODO: check
    def compute_intensity_at_event(self, data, m):
        """Compute the aggregate intensity at the m-th event."""

        s, c = data
        dt_max = self.dt_max
        lamb0 = self.lamb
        W = self.W
        mu, tau = self.mu, self.tau
        N = self.N

        lamb = lamb0
        for n in np.arange(m - 1, -1, -1):
            dt = s[m] - s[n]
            if dt > dt_max:
                break
            else:
                cn, cm = c[n], c[m]
                W_nm = W[cn, cm]
                mu_nm = mu[cn, cm]
                lamb += W_nm * self.impulse(dt, mu_nm, tau_nm)
        return lamb

    def compute_likelihood(self, data, T, theta=None, size=None):
        """Compute the log likelihood of event-time data, p(s | theta).

            Note: approximates the integrals in the first term by W_m,n.

        data: list of form [timestamps, classes].

        """

        if theta is None:
            lambda0 = self.lamb
            W = self.W
            mu = self.mu
            tau = self.tau
        else:
            lambda0, W, mu, tau = theta

        times, nodes = data
        M = len(times)
        if (size is not None) and (M < size):
            print('Insufficient event data (n={}); returning None'.format(nobs))
            return None
        elif (size is not None):
            idx = np.random.randint(low=0, high=M-size)
            times = times[idx:idx + size]
            nodes = nodes[idx:idx + size]
            t0 = times[0]
            tN = times[-1]
            T = tN - t0
            times = times - t0
        # Likelihood computation
        ll_a = T * lambda0.sum() + W[nodes,:].sum()
        I = np.array([self.compute_intensity(data, t, theta) for t in times])  # intensity at each (event, node)
        ll_b = np.sum(np.log(np.concatenate([I[nodes == n, n] for n in range(self.N)])))
        # ll_b = np.sum(np.log(I))
        return -ll_a + ll_b, T

    def compute_pred_likelihood(self, data, T, sample):
        """Compute the predictive likelihood of event-time data, p(s_test | s_train).

            Approximated by MCMC as ll = (1/L) * sum( p( s_test | theta^(l) ) )

        data: list of form [timestamps, classes].

        """

        lambda0, W, mu, tau = sample
        mean = 0
        for i in range(len(sample)):
            theta = (lambda0[:,i], W[:,:,i], mu[:,:,i], tau[:,:,i])
            mean += self.compute_likelihood(data, T, theta)
        return mean / len(sample)

    # TODO
    def fit_parents(self, data):
        """Return most likely parent event of each event."""
        raise NotImplementedError

    # @profile
    def sample_parents(self, data, lambda0, W, mu, tau):
        """Sample the parents of events via Poisson thinning.

        This function is intended for use in Gibbs sampling, in which case the
        parameters A, W, lamb, mu and tau are sampled from their marginal
        posterior distributions.

        Args
            times: vector of event times, a list in range [0, T].
            nodes: vector of event nodes, a list in set [0, ..., N - 1].

        Returns
            parents: vector of sampled event parents, a list in set [0, 1, ..., M - 1].

        """

        assert (W >= 0).all(), "Found a negative weight matrix parameter."
        # assert (A >= 0).all(), "Found a negative connection matrix parameter."
        assert (lambda0 >= 0).all(), "Found a negative background intensity."

        start = time.time()
        times, nodes = data
        M = len(times)
        parents = np.zeros(M)
        for m in np.arange(1, M):  # first event is always caused by background
            # print("Event index = {}:".format(m))
            cm = nodes[m]
            sm = times[m]
            P = [lambda0[cm]]  # parent probabilities
            K = [0]  # parent identifiers, 0 = background
            i = 1
            while (sm - times[m - i]) < self.dt_max:
                n = m - i
                # print(" >> index = {}".format(n))
                cn = nodes[n]
                sn = times[n]
                if sn < sm:
                    # lambda_n = A[cn, cm] * W[cn, cm] * self.impulse(sm - sn, mu=mu, tau=tau)[cn, cm]  # impulse of the n-th event
                    mu_nm = mu[cn, cm]
                    tau_nm = tau[cn, cm]
                    W_nm = W[cn, cm]
                    lambda_n = W_nm * self.impulse(sm - sn, mu=mu_nm, tau=tau_nm)
                    P.append(lambda_n)
                    K.append(n + 1)  # events counted from 1 to avoid confusion with 0 as background event
                i += 1
                if m - i < 0:
                    break
            P = np.array(P) / np.sum(P)
            parents[m] = np.random.choice(a=K, p=P)
        stop = time.time()
        if FLAGS_VERBOSE:
            print('Sampled {} parents in {} seconds.'.format(times.shape[0], stop - start))
        return parents.astype('int')

    # @profile
    def sample_parents_ext(self, data, lambda0, W, mu, tau, method='cython'):
        """Sample the parents of events via Poisson thinning.

        This function is intended for use in Gibbs sampling, in which case the
        parameters A, W, lamb, mu and tau are sampled from their marginal
        posterior distributions.

        Args
            times: vector of event times, a list in range [0, T].
            nodes: vector of event nodes, a list in set [0, ..., N - 1].

        Returns
            parents: vector of sampled event parents, a list in set [0, 1, ..., M - 1].

        """

        assert (W >= 0).all(), "Found a negative weight matrix parameter."
        assert (lambda0 >= 0).all(), "Found a negative background intensity."

        start = time.time()
        parents = np.zeros(data[0].shape[0], dtype='float64')
        if method == 'cython':
            ext.sample_parents_cython(np.array(data[0], dtype='float64'),
                                      np.array(data[1], dtype='int32'),
                                      parents,
                                      self.dt_max,
                                      lambda0, W, mu, tau)
        elif method == 'openmp':
            ext.sample_parents_openmp(np.array(data[0], dtype='float64'),
                                      np.array(data[1], dtype='int32'),
                                      parents,
                                      self.dt_max,
                                      lambda0, W, mu, tau)
        stop = time.time()
        if FLAGS_VERBOSE:
            print('Sampled {} parents in {} seconds.'.format(data[0].shape[0], stop - start))
        return parents.astype('int')

    # @profile
    def sample(self, data, T, size=1):
        """Perform Gibbs sampling.

            In practice you may not want to use the entire sample to approximate
            the posterior distribution because the algorithm may take some time
            to approach a steady state. You might discard a number of the
            earliest observations (referred to as "burn-in").

        """

        print("Sampling posterior...")

        times, nodes = data
        M = len(times)
        A = self.A
        dt_max = self.dt_max
        bias = np.zeros((self.N, size))
        weights = np.zeros((self.N, self.N, size))
        mu = np.zeros((self.N, self.N, size))
        tau = np.zeros((self.N, self.N, size))
        b, W, m, t = self.model.prior()  # initialize parameters
        start = time.time()
        sub_start = start
        for i in range(size):
            if i % (size / 20) == 0 and i > 0:
                sub_stop = time.time()
                print("step={}, time={:.2f} s ({:.2f} s)".format(i, sub_stop - start, sub_stop - sub_start))
                sub_start = sub_stop
            parents = self.sample_parents(data, b, W, m, t)
            b, W, m, t = self.model.sample(data, parents, T, dt_max)
            bias[:, i] = b
            weights[:, :, i] = W
            mu[:, :, i] = m
            tau[:, :, i] = t
        stop = time.time()
        print("Performed {} sampling steps in {} seconds.".format(size, stop - start))
        return bias, weights, mu, tau

    def sample_ext(self, data, T, size=1, method='cython'):
        """Perform Gibbs sampling.

            In practice you may not want to use the entire sample to approximate
            the posterior distribution because the algorithm may take some time
            to approach a steady state. You might discard a number of the
            earliest observations (referred to as "burn-in").

        """

        print("Sampling posterior...")
        times, nodes = data
        M = len(times)
        A = self.A
        dt_max = self.dt_max
        bias = np.zeros((self.N, size))
        weights = np.zeros((self.N, self.N, size))
        mu = np.zeros((self.N, self.N, size))
        tau = np.zeros((self.N, self.N, size))
        b, W, m, t = self.model.prior()
        start = time.time()
        sub_start = start
        for i in range(size):
            # print(i)
            if i % (size / 20) == 0 and i > 0:
                sub_stop = time.time()
                print("step={}, time={:.2f} s ({:.2f} s)".format(i, sub_stop - start, sub_stop - sub_start))
                sub_start = sub_stop
            parents = self.sample_parents_ext(data, b, W, m, t, method)
            b, W, m, t = self.model.sample_ext(data, parents, T, dt_max, method=method)
            bias[:, i] = b
            weights[:, :, i] = W
            mu[:, :, i] = m
            tau[:, :, i] = t
        stop = time.time()
        print("Performed {} sampling steps in {} seconds.".format(size, stop - start))
        return bias, weights, mu, tau

class DiscreteNetworkPoisson():

    def __init__(self, N, L, B, dt, params=None, hypers=None):

        self.N = N
        self.L = L
        self.B = B
        self.dt = dt

        # Parameters
        if params is not None:
            self.lambda0 = params['bias']
            self.W = params['weights']
            self.theta = params['impulse']
        else:
            print("No parameters provided. Setting to defaults.")
            self.lamb = np.ones(self.N)
            self.W = np.zeros((self.N, self.N))
            self.theta = (1 / self.B) * np.ones((self.B, self.N, self.N))

        # Hyperparameters
        if hypers is not None:
            # assert hypers have the correct shape.
            pass
        else:
            print("No hyperparameters provided. Setting to defaults.")
            self.alpha_0, self.beta_0 = (1,1)
            self.kappa, self.nu = (1, np.ones((self.N, self.N)))
            self.gamma = np.ones(self.B)

        # Impulse
        def generate_basis(L, B):
            """
                L: number of lags.
                B: number of basis distributions.
            """

            mu = np.linspace(1, L, B + 2)[1:-1]
            phi = np.empty((L + 1, B))
            for b in range(B):
                for l in range(L + 1):
                    if l == 0:
                        phi[l, b] = 0
                    else:
                        if B == 1:
                            phi[l, b] = np.exp( - 1/2 * ((l - mu[b]) / (L / 2))  ** 2)
                        else:
                            phi[l, b] = np.exp( - 1/2 * ((l - mu[b]) / (L / (B - 1))) ** 2)
            return phi / (self.dt * phi.sum(axis=0))
        self.phi = generate_basis(self.L, self.B)

        # Prior
        self.model = priors.DiscreteNetworkPoisson(self.N,
                                                   self.B,
                                                   self.dt,
                                                   self.alpha_0,
                                                   self.beta_0,
                                                   self.kappa,
                                                   self.nu,
                                                   self.gamma)

    # @profile
    def convolve(self, S):
        T = len(S)
        Shat = np.empty((T, self.N, self.B))
        for n in range(self.N):
            for b in range(self.B):
                Shat[:, n, b] = np.convolve(self.phi[:, b], S[:, n])[:T]
        return Shat

    # @profile
    def calculate_intensity(self, S, Shat, lambda0=None, W=None, theta=None):
        """
            S: spike train data (T x N).
            Shat: net.convolve(S) (T x N x B).
        """

        if lambda0 is None:
            lambda0 = self.lambda0
        if W is None:
            W = self.W
        if theta is None:
            theta = self.theta

        T = len(S)
        lamb = np.zeros((T, self.N))
        for n in range(self.N):
            theta_ = theta[:, :, n]  # B x N
            w_ = W[:, n]  # N x 1
            gamma = np.diagonal(np.dot(Shat, theta_), axis1=1, axis2=2)  # T x N
            lamb[:, n] = lambda0[n] + np.dot(gamma, w_)
        return lamb

    def plot_basis(self, parent=0, child=0, theta=None, mean=False):
        if theta is None:
            theta = self.theta
        p, c = parent, child
        L, B = self.phi.shape
        if mean:
            plt.scatter(np.arange(1, L), np.dot(self.phi, theta[:, p, c])[1:])
            plt.plot(np.arange(1, L), np.dot(self.phi, theta[:, p, c])[1:])
        # else:
        for b in np.arange(B):
            plt.scatter(np.arange(1, L), self.phi[1:,b], alpha=0.15)
            plt.plot(np.arange(1, L), self.phi[1:,b], alpha=0.15)
        plt.xlabel("Lag")
        plt.ylabel("Phi")
        if mean:
            plt.legend(['mean'] + np.arange(1, B + 1).tolist(), loc='upper right')
        else:
            plt.legend(np.arange(1, B + 1).tolist(), loc='upper right')
        plt.show()
        # plt.clf()

    def plot_data(self, S, Lambda, events=True, intensity=True):
        """
            S: spike train data (T x N).
            Lambda: intensity matrix (T x N).
        """

        assert (events == True) or (intensity == True), "You need to choose to plot either the events or the intensity, or both."

        T = len(S)
        if events == True:
            for i in range(self.N):
                plt.bar(np.arange(T), S[:, i], alpha=0.20)
        if intensity == True:
            for i in range(self.N):
                plt.plot(np.arange(T), Lambda[:, i])
        plt.show()
        # plt.clf()

    def generate_data(self, T):
        S = np.zeros((T, self.N))
        lamb = np.zeros((T, self.N))
        S[0,:] = np.random.poisson(self.lambda0)
        for t in np.arange(1, T):
            Shat = np.zeros((self.B, self.N))
            if t < self.L:  # partial convolution
                Stemp = np.concatenate((np.zeros((self.L - t, self.N)), S[:t + 1,:]), axis=0)  # padded
                for n in range(self.N):
                    for b in range(self.B):
                        Shat[b, n] = np.convolve(self.phi[:,b], Stemp[:, n], mode='valid')  # B x N
                for n in range(self.N):
                    theta_ = self.theta[:, :, n]  # B x N
                    w_ = self.W[:, n]  # N x 1
                    gamma = np.diag(np.dot(np.transpose(Shat), theta_))  # N x 1
                    lamb[t - 1, n] = self.lambda0[n] + np.dot(gamma, w_)
                    # print("lambda = {}".format(lamb[t - 1, n]))
            else:  # full convolution
                for n in range(self.N):
                    for b in range(self.B):
                        Shat[b, n] = np.convolve(self.phi[:,b], S[t - self.L:t + 1, n], mode='valid')  # B x N
                for n in range(self.N):
                    theta_ = self.theta[:, :, n]  # B x N
                    w_ = self.W[:, n]  # N x 1
                    gamma = np.diag(np.dot(np.transpose(Shat), theta_))  # N x 1
                    lamb[t - 1, n] = self.lambda0[n] + np.dot(gamma, w_)
                    # print("lambda = {}".format(lamb[t - 1, n]))
            S[t,:] = np.random.poisson(lamb[t - 1, :])
        return S

    # @profile
    def sample_parents(self, S, Shat, lambda0=None, W=None, theta=None):
        """Sample a parent node for each event given spike train history.

            lambda0: sampled basis vector (N x 1).
            W: sampled connection matrix (N x N).
            theta: sampled basis weights (B x N x N).
        """

        if lambda0 is None:
            lambda0 = self.lambda0
        if W is None:
            W = self.W
        if theta is None:
            theta = self.theta

        start = time.time()
        T = len(S)
        parents = np.zeros((T, self.N, 1 + self.N * self.B))
        Lambda = self.calculate_intensity(S, Shat, lambda0, W, theta)
        for t in range(T):
            Shat_ = Shat[t, :, :]  # N x B
            for n in range(self.N):
                w_ = W[:,n].reshape((self.N, 1)).repeat(self.B, axis=1)  # N x 1
                theta_ = theta[:, :, n]  # B x N
                lamb = Lambda[t,n]
                u0 = lambda0[n] / lamb
                u_nb = ((Shat_ * w_ * np.transpose(theta_)) / lamb).ravel()  # (N x B, )
                u = np.append(u0, u_nb)
                sample = np.random.multinomial(S[t,n], u)
                parents[t, n, :] = sample
        stop = time.time()

        assert S.sum() == parents.sum(), "Number of spikes not equal to number of parents."

        if FLAGS_VERBOSE:
            print('Sampled {} parents in {} seconds.'.format(S.sum(), stop - start))
        return parents

    # @profile
    def sample_parents_ext(self, S, Shat, lambda0=None, W=None, theta=None):
        """Sample a parent node for each event given spike train history.

            lambda0: sampled basis vector (N x 1).
            W: sampled connection matrix (N x N).
            theta: sampled basis weights (B x N x N).
        """

        if lambda0 is None:
            lambda0 = self.lambda0
        if W is None:
            W = self.W
        if theta is None:
            theta = self.theta

        start = time.time()
        parents = np.zeros((S.shape[0], self.N, 1 + self.N * self.B), dtype='float64')
        Lambda = self.calculate_intensity(S, Shat, lambda0, W, theta)
        ext.sample_parents_discrete(S.astype('int32'),
                                    S.max(),
                                    Shat.astype('float64'),
                                    Lambda,
                                    parents,
                                    lambda0, W, theta)
        stop = time.time()

        assert S.sum() == parents.sum(), "Number of spikes not equal to number of parents."

        if FLAGS_VERBOSE:
            print('Sampled {} parents in {} seconds.'.format(S.sum(), stop - start))
        return parents

    # @profile
    def sample(self, S, size=1):
        """Sample network parameters.

            S: spike train data (T x N).
            size: number of iterations to perform.
        """

        print("Sampling posterior...")

        T = len(S)
        Shat = self.convolve(S)
        bias = np.empty((self.N, size))
        weights = np.empty((self.N, self.N, size))
        impulse = np.empty((self.B, self.N, self.N, size))
        lambda0, W, theta = self.model.prior()
        start = time.time()
        sub_start = start
        for i in range(size):
            if i % (size / 10) == 0 and i > 0:
                sub_stop = time.time()
                print("step={}, time={:.2f} s ({:.2f} s subtime)".format(i, sub_stop - start, sub_stop - sub_start))
                sub_start = sub_stop
            parents = self.sample_parents(S, Shat, lambda0, W, theta)
            lambda0, W, theta = self.model.sample(S, parents)
            bias[:, i] = lambda0
            weights[:, :, i] = W
            impulse[:, :, :, i] = theta
        stop = time.time()
        print("Performed {} sampling steps in {} seconds.".format(size, stop - start))
        return bias, weights, impulse

    def sample_ext(self, S, size=1):
        """Sample network parameters.

            S: spike train data (T x N).
            size: number of iterations to perform.
        """

        print("Sampling posterior...")

        T = len(S)
        Shat = self.convolve(S)

        # Init parameter samples
        bias = np.empty((self.N, size))
        weights = np.empty((self.N, self.N, size))
        impulse = np.empty((self.B, self.N, self.N, size))
        lambda0, W, theta = self.model.prior()

        start = time.time()
        sub_start = start
        for i in range(size):
            if i % (size / 100) == 0 and i > 0:
                sub_stop = time.time()
                print("step={}, time={:.2f} s ({:.2f} s subtime)".format(i, sub_stop - start, sub_stop - sub_start))
                sub_start = sub_stop
            parents = self.sample_parents_ext(S, Shat, lambda0, W, theta.copy(order='C'))
            # lambda0, W, theta = self.model.sample_ext(S, parents)
            lambda0, W, theta = self.model.sample(S, parents)
            bias[:, i] = lambda0
            weights[:, :, i] = W
            impulse[:, :, :, i] = theta
        stop = time.time()
        print("Performed {} sampling steps in {} seconds.".format(size, stop - start))
        return bias, weights, impulse

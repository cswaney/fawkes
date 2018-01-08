from fawkes.models import NetworkPoisson
import numpy as np
import pandas as pd
import h5py as h5

N = 12
dt_max = 60
T = 57600 - 34200 - 3600 * 2

# Load order book from ITCH data
book_columns = []
book_columns += []
with h5.File('', 'r') as hdf:
    book_data = hdf['orderbooks/{}/{}'.format(name, date)][:]
    book_data = pd.DataFrame(book_data, columns=book_columns)

# Select a book / timestamp
N, _ = book_data.shape
t0 = np.random.randint(low=0, high=N)
initial_book = book_data.loc[t0, :]

# Load Gibbs samples
with h5.File('', 'r') as hdf:
    lambda0 = hdf['{}/{}/lambda0'.format(name, date)][:]
    W = hdf['{}/{}/W'.format(name, date)][:]
    mu = hdf['{}/{}/mu'.format(name, date)][:]
    tau = hdf['{}/{}/tau'.format(name, date)][:]

# Compute parameter estimates
lambda0_map = np.median(lambda0)
W_map = np.median(W)
mu_map = np.median(mu)
tau_map = np.median(tau)
params = {'lamb': lambda0_map, 'weights': W_map, 'mu': mu_map, 'tau': tau_map}

# Initialize a model
model = NetworkPoisson(N=N, dt_max=dt_max, params=params)

# Create an order book
book = Book(initial_book)

# Creat a learning agent
agent = Agent()

# Create a simulator
simulator = Simulator(book, model, agent)

# Run the simulator for T sec.
simulator.run(T)

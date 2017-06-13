from fawkes.models import NetworkPoisson
import numpy as np
import pandas as pd
import h5py as h5

"""
This script simulates a network Poisson model using parameter estimates
based on Nasdaq ITCH data. It initializes an order book by choosing a random
snapshot selected from the data, simulates a series of events, and calculates
a number of statistics summarizing the simulated order book data.
"""

class Book():

    __init__(self, book):
        self.prices = book['prices']  # {'bids': [], 'asks': []}
        self.shares = book['shares']  # {'bids': [], 'asks': []}
        self.actions = [(),
                        (),
                        (),
                        (),
                        (),
                        (),
                        (),
                        (),
                        (),
                        (),
                        (),
                        ()]

    def aslist(self):
        prices = (self.book['bids'].keys(), self.book['asks'].keys())
        volumes = (self.book['bids'].values(), self.book['asks'].values())
        listed = [prices, volumes]
        return listed

    def action_to_message(self, actions):
        """Convert an action to a message (price, side, shares) based on current book."""

        return price, side, shares

    def simulate(self, events):
        timestamps, actions = events
        snapshots = [self.book]
        for timestamp, action in zip(timestamps, actions):
            price, side, shares = self.action_to_message(action)
            self.book[side][price] += shares
            if self.book[side][price] == 0:
                self.book[side].pop([price])
                snapshots.append((timestamp, self.aslist()))
        return snapshots

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

# Initialize a model
params = {'lamb': lambda0_map, 'weights': W_map, 'mu': mu_map, 'tau': tau_map}
model = NetworkPoisson(N=N, dt_max=dt_max, params=params)

# Simulate events
events = []
for _ in np.arange():
    events.append(model.generate_data(T=T))

# Construct order book snapshots
samples = []
book = Book(initial_book)
for e in events:
    samples.append(book.simulate(e))

# Compute statistics

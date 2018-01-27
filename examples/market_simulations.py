from fawkes.engine import Agent, Display, Message, Order, OrderBook, Environment
from fawkes.models import NetworkPoisson
import numpy as np

# Basic Simulation
lambda0 = 0.05 * np.ones(N)
W = 0.1 * np.eye(N)
mu = -1.0 * np.ones((N,N))
tau = 1.0 * np.ones((N,N))
model = NetworkPoisson(N=N, dt_max=1.0, params={'lamb': lambda0, 'weights': W, 'mu': mu, 'tau': tau})
agent = Agent(verbose=False)
env = Environment(model, verbose=False)

for i in range(100):
    if (i % 10 == 0) and (i > 0):
        print('i={}'.format(i))
    time, book, events, done = env.reset()
    while not done:
        time, book, events, confirm, messages, done = env.step(None)

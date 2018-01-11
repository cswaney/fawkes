from fawkes.models import NetworkPoisson
import numpy as np
import pandas as pd
import h5py as h5
from pprint import pformat
import matplotlib.pyplot as plt

"""
This script simulates a network Poisson model using parameter estimates
based on Nasdaq ITCH data. It initializes an order book by choosing a random
snapshot selected from the data, simulates a series of events, and calculates
a number of statistics summarizing the simulated order book data.
"""

class Display():

    def __init__(self):
        pass

class Order():

    def __init__(self, label, side, price, shares, refno=None):
        if label == 'delete':
            assert refno is not None, 'delete orders require a refno'
        self.label = label  # order type ('add', 'delete', 'execute')
        self.side = side
        self.price = price
        self.shares = shares
        if label == 'add':
            self.refno = np.random.randint(low=0, high=10 ** 9)
        else:
            self.refno = refno

    def __str__(self):
        items = [self.label, self.side, self.price, self.shares ,self.refno]
        return "<Order(label='{}', side='{}', price={}, shares={}, refno={})>".format(*items)

    def __repr__(self):
        items = [self.label, self.side, self.price, self.shares ,self.refno]
        return "<Order(label='{}', side='{}', price={}, shares={}, refno={})>".format(*items)

class OrderBook():

    def __init__(self):
        self.bids = {}  # {price: Queue, ...}
        self.asks = {}  # {price: Queue, ...}

    def __str__(self):
        bids = [p for p in self.bids]
        bids.sort()
        bids.reverse()
        asks = [p for p in self.asks]
        asks.sort()
        return "bids: {}\nasks: {}".format(bids, asks)

    def __repr__(self):
        bids = [p for p in self.bids]
        bids.sort()
        bids.reverse()
        asks = [p for p in self.asks]
        asks.sort()
        return "bids: {}\nasks: {}".format(bids, asks)

    def reset(self):
        self.bids = {}
        self.asks = {}

    def has_queue(self, side, price):
        if side == "bid":
            if price in self.bids.keys():
                return True
            else:
                return False
        elif side == "ask":
            if price in self.asks.keys():
                return True
            else:
                return False

    def get_queue(self, side, price):
        if side == "bid":
            if price in self.bids.keys():
                return self.bids[price]
            else:
                return False
        elif side == "ask":
            if price in self.asks.keys():
                return self.asks[price]
            else:
                return False

    def update(self, order):
        """Update the status of the order book."""
        if order.label == "add":
            self.add_order(order)
        elif order.label == "delete":
            self.delete_order(order)
        elif order.label == "execute":
            self.execute_order(order)

    def add_order(self, order):
        """Process an add order."""
        assert order.label == "add", "add_order recieved a non-add order."
        assert order.side in ('bid', 'ask'), "add_order side not specified"
        if order.side == "bid":
            if len(self.asks.keys()) > 0:
                if not order.price < min(self.asks.keys()):
                    print("Attempted ADD BID order above best ask; SKIPPING")
                    return
            if order.price in self.bids.keys():
                self.bids[order.price].append(order)
                print("BID order appended to queue at prince {}".format(order.price))
            else:
                self.bids[order.price] = [order]
                print("New BID queue started at price {}".format(order.price))
        elif order.side == "ask":
            if len(self.bids.keys()) > 0:
                if not order.price > max(self.bids.keys()):
                    print("Attempted ADD ASK order below best bid; SKIPPING")
                    return
            if order.price in self.asks.keys():
                self.asks[order.price].append(order)
                print("ASK order appended to queue at prince {}".format(order.price))
            else:
                self.asks[order.price] = [order]
                print("New ASK queue started at price {}".format(order.price))

    def delete_order(self, order):
        """Process a delete order."""
        assert order.label == "delete", "delete_order recieved a non-delete order."
        if order.side == "bid":
            try:
                queue = self.bids[order.price]
                idx = [o.refno for o in queue].index(order.refno)
                queue.pop(idx)
                print("Deleted BID order {}".format(order.refno))
                if len(queue) == 0:
                    self.bids.pop(order.price)
                    print("Removed price {} from BIDS".format(order.price))
            except:
                print("Tried to remove a non-existent BID order.")
        elif order.side == "ask":
            try:
                queue = self.asks[order.price]
                idx = [o.refno for o in queue].index(order.refno)
                queue.pop(idx)
                print("Deleted ASK order {}".format(order.refno))
                if len(queue) == 0:
                    self.asks.pop(order.price)
                    print("Removed price {} from ASKS".format(order.price))
            except:
                print("Tried to remove a non-existent ASK order.")

    def execute_order(self, order):
        """Process a market order."""
        assert order.label == "execute", "execute_order recieved a non-execute order."
        if order.side == 'bid':
            side = 'ask'
        elif order.side == 'ask':
            side = 'bid'
        match = self.get_best(side)
        if match is not None:
            matched_shares = min(order.shares, match.shares)
            order.shares -= matched_shares
            match.shares -= matched_shares
            print("executed {} shares".format(matched_shares))
            while True:
                # Remove completed match
                if match.shares == 0:
                    if side == 'bid':
                        self.bids[match.price].remove(match)
                        if len(self.bids[match.price]) == 0:
                            self.bids.pop(match.price)
                    elif side == 'ask':
                        self.asks[match.price].remove(match)
                        if len(self.asks[match.price]) == 0:
                            self.asks.pop(match.price)
                    # Find the next match
                    match = self.get_best(side)
                    if (match is None) and (order.shares > 0):
                        print("MARKET order was PARTIALLY executed.")
                        break
                    elif (match is None) and (order.shares == 0):
                        print("MARKET order was FULLY executed.")
                        break
                if order.shares == 0:
                    print("MARKET order was FULLY executed.")
                    break
                matched_shares = min(order.shares, match.shares)
                order.shares -= matched_shares
                match.shares -= matched_shares
                print("executed {} shares".format(matched_shares))

    def get_best(self, side):
        """Return the best offer on a given side of the book."""
        if side == "bid":
            prices = list(self.bids.keys())
            if len(prices) > 0:
                return self.bids[max(prices)][0]
            else:
                print("No BID orders available.")
                return None
        elif side == "ask":
            prices = list(self.asks.keys())
            if len(prices) > 0:
                return self.asks[min(prices)][0]
            else:
                print("No ASK orders available.")
                return None

    # TODO: make this a Simulator method
    def random_shares(self, label):
        if label == "add":
            return np.random.choice(self.add_dist.a, self.add_dist.p)
        elif label == "execute":
            return np.random.choice(self.execute_dist.a, self.execute_dist.p)

    def get_price(self, side, level):
        """Find the price at a given (side, level) of the book."""

        # get best bid order
        # get best ask order
        # check that order exists for the side you want
            # if it doesn't, return None and warning
        # if the opposite side exists, spread = ask - bid
        # if the opposite side is None, spread = S (for any S > 0.01)

        best_bid = self.get_best("bid")
        best_ask = self.get_best("ask")

        if side == "bid":
            try:
                bid = best_bid.price
            except:
                print("There are no BID orders. Returning None.")
                return None
            try:
                ask = best_ask.price
                spread = ask - bid
            except:
                spread = 1
        if side == "ask":
            try:
                ask = best_ask.price
            except:
                print("There are no ASK orders. Returning None.")
                return None
            try:
                bid = best_bid.price
                spread = ask - bid
            except:
                spread = 1

        if side == "bid":
            if level == 0:
                if spread > 1:
                    return bid + 1
                else:
                    print("Spread equals the minimum tick.")
                    return None
            elif level == 1:
                return bid
            elif level == 2:
                return bid - 1
        elif side == "ask":
            if level == 0:
                if spread > 1:
                    return ask - 1
                else:
                    print("Spread equals the minimum tick.")
                    return None
            elif level == 1:
                return ask
            elif level == 2:
                return ask + 1

    def random_order(self, side, price):
        """Return the reference number of a random order in (side, price) queue."""
        if side == "bid":
            queue = self.bids[price]
        elif side == "ask":
            queue = self.asks[price]
        N = len(queue)
        return queue[np.random.randint(low=0, high=N)]

    def shares(self):
        bids = {}
        for price in self.bids.keys():
            bids[price] = sum([order.shares for order in self.bids[price]])
        asks = {}
        for price in self.asks.keys():
            asks[price] = sum([order.shares for order in self.asks[price]])
        return {'bid': bids, 'ask': asks}

    def prices(self):
        """The price of each level."""
        bids = [p for p in self.bids]
        bids.sort()
        bids.reverse()
        asks = [p for p in self.asks]
        asks.sort()
        return {'bid': bids, 'ask': asks}

    def show(self, args=None):
        """Plot the order book.

        Args:
            xlim
            ylim
        """

        for price in self.bids.keys():
            # the x of each bar is the price
            x = [price] * len(self.bids[price])
            # the height of each bar is order['shares']
            h = [order.shares for order in self.bids[price]]
            # the bottom of each bar is b += order['shares'], starting from b = 0
            if len(self.bids[price]) == 0:
                b = [0]
                # print('x={}\nh={}\nb={}'.format(x, h, b))
            else:
                b = [sum(h[:i]) for i,_ in enumerate(h)]
                # print('x={}\nh={}\nb={}'.format(x, h, b))
            plt.bar(x, h, bottom=b, width=1, color='C0', edgecolor='white', linewidth=1)
        for price in self.asks.keys():
            x = [price] * len(self.asks[price])
            h = [order.shares for order in self.asks[price]]
            if len(self.asks[price]) == 0:
                b = [0]
                # print('x={}\nh={}\nb={}'.format(x, h, b))
            else:
                b = [sum(h[:i]) for i,_ in enumerate(h)]
                # print('x={}\nh={}\nb={}'.format(x, h, b))
            plt.bar(x, h, bottom=b, width=1, color='C2', edgecolor='white', linewidth=1)
        plt.xlim(2990, 3011)
        plt.xticks(np.arange(2990, 3011, 2))
        plt.ylim(0, 1000)
        plt.show()

class Agent():

    def __init__(self, actions, score=0, inventory=0, orders=[], algorithm=None, time=0):
        self.actions = actions
        self.score = score
        self.inventory = inventory
        self.orders = orders
        self.algorithm = algorithm
        self.time = 0

    def reset(self):
        self.score = 0
        self.inventory = 0
        self.orders = []
        self.time = 0

    def action(self, state):
        """Choose an action using algorithm.

        Valid algorithms must be of the form:

            def algorithm(actions, args):
                ...
                return action

        The arguments passed to algorithm will need to be passed through 'state'. These
        might include: the state of the book, past orders, timestamp, etc...
        """
        return np.random.choice(self.actions, p=[0.999, 0.0005, 0.0005])
        # return self.algorithm(self.actions)

    def update(self, result):
        """Update orders and score based on result."""
        pass
        # ...
        # self.orders += order
        # self.inventory += shares
        # self.score += price * shares

class Simulator():

    def __init__(self, book, model, agent, dt):
        self.book = book
        self.model = model
        self.agent = agent
        self.dt = dt

    def generate_events(self, t_max, event=None):
        """
            Using the model (optionally provide an initial event).

            Args:
                t_max: (int) maximum time of simulation.
                event: type and time of an order book event.
        """
        if event is not None:
            # print('Generating new events...')
            # Find future events generated by the event; no background events!
            # print('before times: {}\nbefore nodes: {}'.format(pformat(self.times), pformat(self.nodes)))
            events = self.model.generate_data_from_parent(t_max, event)
            # print('new events: {}'.format(events))
            return events
            # self.update_events(events)
            # print('after times: {}\nafter nodes: {}'.format(pformat(self.times), pformat(self.nodes)))
        else:
            # Sample model
            times, nodes = self.model.generate_data(t_max)
            self.times, self.nodes = times.tolist(), nodes.tolist()
            # print('initial times: {}\ninitial nodes: {}'.format(self.times, self.nodes))
            self.originals = {'times': self.times.copy(), 'nodes': self.nodes.copy()}
            self.actions = {'times': [], 'nodes': []}
            self.generated = {'times': [], 'nodes': []}

    def run(self, t_max):
        """Apply algorithm to the sequence of order book states."""
        t = self.dt
        i = 0
        while t < t_max:
            # Find events that occurred in (t - dt, t)
            events = []
            while self.times[i] < t:
                if i < len(self.times) - 1:
                    events.append((self.times[i], self.nodes[i]))
                    i += 1
                else:
                    break
            if len(events) > 0:
                print('Found {} events in interval ({}, {}): {}'.format(len(events), round(t - self.dt, 3), round(t, 3), events))
            # Update the book
            for time, node in events:
                order = self.to_order(node)
                print('Updating book for EVENT ({}) at time {}'.format(order, time))
                self.book.update(order)
                self.live_plot(t)
                input()
            # Take an action
            if len(events) > 0:  # at least one event occurred
                time, node = events[-1]
            else:  # no events have occurred
                time, node = (0, None)
            state = time, node
            action = self.agent.action(state)
            if action is not None:
                i += 1  # NEED TO SKIP THE ACTION'S EVENT NEXT LOOP!
                print('Agent performed ACTION {}'.format(action))
                order = self.to_order(action)
                print('Updating book for ACTION ({}) at time {}'.format(order, t))
                self.book.update(order)
                # Generate new events based on the action: doesn't depend on any other past events!
                events = self.generate_events(t_max, (t, action))
                self.update_events(events)
                # print('t={}, events={}'.format(t, events))
                self.live_plot(t)
                input()
            t += self.dt
        print('Reached the end of simulation (t={})'.format(t_max))
        self.live_plot(t)

    # TODO: faster to directly insert the new events?
    def update_events(self, events):
        """
        Args
            events: (list) of (tuples) of the form (float, int) indicating the time and node of events.
        """
        try:
            times, nodes = events
        except:
            return
        self.actions['times'].append(times[0])
        self.actions['nodes'].append(nodes[0])

        if len(times) > 1:
            self.generated['times'].extend(times[1:])
            self.generated['nodes'].append(nodes[1:])

        self.times.extend(times)
        self.nodes.extend(nodes)
        self.nodes = [n for _,n in sorted(zip(self.times, self.nodes))]
        self.times = [t for t,_ in sorted(zip(self.times, self.nodes))]

    def summarize(self):
        """Generate statistics and graphs of simulation."""
        plt.close()
        plt.scatter(self.originals['times'], [1] * len(self.originals['times']))  # originals
        plt.scatter(self.actions['times'], [2] * len(self.actions['times']))  # actions
        plt.scatter(self.generated['times'], [3] * len(self.generated['times']))  # generated
        plt.scatter(self.times, [4] * len(self.times))  # total
        plt.legend(['originals', 'actions', 'generated', 'total'], loc='right')
        plt.show()

    def live_plot(self, t):
        plt.close()
        plt.subplot(211)
        # plot events up to time t
        originals = self.originals['times']
        currents = [s for s in self.originals['times'] if s <= t]
        actions = [s for s in self.actions['times'] if s <= t]
        generated = [s for s in self.generated['times'] if s <= t]
        plt.scatter(originals, [1] * len(originals), alpha=0.25, color='C0')  # originals
        plt.scatter(currents, [1] * len(currents), color='C0')  # originals - current
        plt.scatter(actions, [2] * len(actions), color='C1')  # actions
        plt.scatter(generated, [3] * len(generated), color='C2')  # generated
        # plt.legend(['originals', 'actions', 'generated', 'total'], loc='right')
        plt.ylim(0,4)
        plt.xlim(0,5)
        plt.yticks([1, 2, 3], ['Orig.', 'Act.', 'Gen.'])
        plt.show()
        # plot current order book
        plt.subplot(212)
        self.book.show()
        # plt.waitforbuttonpress()

    def to_order(self, node):
        """Convert an event to an order"""
        nodes = [('add', 'bid', -1),
                 ('add', 'bid', 0),
                 ('add', 'bid', 1),
                 ('add', 'ask', -1),
                 ('add', 'ask', 0),
                 ('add', 'ask', 1),
                 ('delete', 'bid', 0),
                 ('delete', 'bid', 1),
                 ('delete', 'ask', 0),
                 ('delete', 'ask', 1),
                 ('execute', 'bid', None),
                 ('execute', 'ask', None)]
        label, side, level = nodes[node]
        if label == 'add':
            if level == -1:
                if side == 'bid':
                    price = self.book.prices()[side][0] + 1
                elif side == 'ask':
                    price = self.book.prices()[side][0] - 1
            else:
                price = self.book.prices()[side][level]
            shares = np.random.choice([100, 200, 300])
            order = Order(label, side, price, shares, None)
        elif label == 'delete':
            price = self.book.prices()[side][level]
            if side == 'bid':
                order = np.random.choice(self.book.bids[price])
            elif side == 'ask':
                order = np.random.choice(self.book.asks[price])
            order.label = 'delete'
        elif label == 'execute':
            shares = np.random.choice([10, 50, 100])
            order = Order(label, side, None, shares, None)
        return order

# Example
def random_add_order():
    label = 'add'
    side = np.random.choice(['bid', 'ask'])
    if side == 'bid':
        price = np.random.choice(np.arange(2991, 3003, 1, dtype='int'))
    else:
        price = np.random.choice(np.arange(2998, 3011, 1, dtype='int'))
    shares = np.random.choice(np.arange(100, 400, 100))
    refno = None
    return Order(label, side, price, shares, refno)

N = 20
orders = []
n = 0
book = OrderBook()
while n < N:
    order = random_add_order()
    print('{}'.format(order))
    if order.refno not in [order.refno for order in orders] and order.refno is not None:
        orders.append(order)
    if order.label == 'delete':
        for o in orders:
            if o.refno == order.refno:
                orders.remove(o)
    book.update(order)
    n += 1

dt = 0.001
t_max = 5

N = 12
lambda0 = 0.4 * np.ones(N)
W = 0.1 * np.eye(N)
mu = -1.0 * np.ones((N,N))
tau = 1.0 * np.ones((N,N))

# N = 2
# lambda0 = np.array([1.0, 1.0])
# W = np.array([[0.5, 0.1], [0.1, 0.5]])
# mu = -1.0 * np.ones((N,N))
# tau = 1.0 * np.ones((N,N))

model = NetworkPoisson(N=N, dt_max=1.0, params={'lamb': lambda0, 'weights': W, 'mu': mu, 'tau': tau})
agent = Agent([None, 0, 1])
simulator = Simulator(book, model, agent, dt)
simulator.generate_events(t_max)
simulator.live_plot(0)
input()
simulator.run(t_max)

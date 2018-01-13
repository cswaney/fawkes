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

    def __init__(self, **kwargs):

        plt.ion()
        plt.rc('font', size=10, family='Arial')

        self.events_ax = plt.subplot2grid((6,2), (0,0), rowspan=3)
        self.book_ax = plt.subplot2grid((6,2), (3,0), rowspan=3)
        self.score_ax = plt.subplot2grid((6,2), (0,1), rowspan=3)
        self.inventory_ax = plt.subplot2grid((6,2), (3,1), rowspan=3)
        # self.orders_ax = plt.subplot2grid((6,2), (4,1), rowspan=2)
        plt.tight_layout()

        # Default kwargs
        self.events_xlim, self.events_ylim = ([0, 5], [0, 4])
        self.book_xlim, self.book_ylim = ([2990, 3010], [0, 2000])
        self.score_xlim, self.score_ylim = ([0, 5], [-1000000, 1000000])
        self.inv_xlim, self.inv_ylim = ([0, 5], [-1000, 1000])
        # self.orders_xlim, self.orders_ylim = ([2990], [3011])
        self.pause = True

        # Set the xlim and ylim for axes using kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def plot_events(self, time, events):

        originals, actions, generated = events
        o = [s for s in originals['times'] if s <= time]
        a = [s for s in actions['times'] if s <= time]
        g = [s for s in generated['times'] if s <= time]

        self.events_ax.clear()
        self.events_ax.set_title('Events', loc='center')
        # self.events_ax.text(.025, .925, 'Events', horizontalalignment='left', transform=self.events_ax.transAxes)
        self.events_ax.set_xlim(self.events_xlim[0], self.events_xlim[1])
        self.events_ax.set_ylim(0, 4)
        self.events_ax.set_yticks([1, 2, 3], ['Originals', 'Actions', 'Generated'])
        self.events_ax.tick_params(direction='in')

        _ = self.events_ax.scatter(originals['times'], [1] * len(originals['times']), s=30, color='C0', alpha=0.25)
        _ = self.events_ax.scatter(o, [1] * len(o), s=30, color='C0')
        _ = self.events_ax.scatter(a, [2] * len(a), s=30, color='C1')
        _ = self.events_ax.scatter(g, [3] * len(g), s=30, color='C2')

    def plot_book(self, book):

        self.book_ax.clear()
        self.book_ax.set_title('Books', loc='center')
        # self.book_ax.text(.025, .925, 'Book', horizontalalignment='left', transform=self.book_ax.transAxes)
        self.book_ax.set_xlim(self.book_xlim[0], self.book_xlim[1])
        self.book_ax.set_xticks(range(self.book_xlim[0], self.book_xlim[1] + 1, 2))
        self.book_ax.set_ylim(self.book_ylim[0], self.book_ylim[1])
        self.book_ax.tick_params(direction='in')

        bids, asks = book
        for price in bids.keys():
            x = [price] * len(bids[price])
            h = [order.shares for order in bids[price]]
            if len(bids[price]) == 0:
                b = [0]
                # print('x={}\nh={}\nb={}'.format(x, h, b))
            else:
                b = [sum(h[:i]) for i,_ in enumerate(h)]
                # print('x={}\nh={}\nb={}'.format(x, h, b))
            _ = self.book_ax.bar(x, h, bottom=b, width=1, color='C0', edgecolor='white', linewidth=1)
        for price in asks.keys():
            x = [price] * len(asks[price])
            h = [order.shares for order in asks[price]]
            if len(asks[price]) == 0:
                b = [0]
                # print('x={}\nh={}\nb={}'.format(x, h, b))
            else:
                b = [sum(h[:i]) for i,_ in enumerate(h)]
                # print('x={}\nh={}\nb={}'.format(x, h, b))
            _ = self.book_ax.bar(x, h, bottom=b, width=1, color='C1', edgecolor='white', linewidth=1)

    def plot_score(self, hist):
        times = [x for x in hist]
        scores = [x['score'] for x in hist.values()]

        self.score_ax.clear()
        self.score_ax.set_title('Score', loc='center')
        # self.score_ax.text(.025, .925, 'Score', horizontalalignment='left', transform=self.score_ax.transAxes)
        self.score_ax.set_xlim(self.score_xlim[0], self.score_xlim[1])
        # self.score_ax.set_ylim(self.score_ylim[0], self.score_ylim[1])
        self.score_ax.tick_params(right=True, left=False, labelright=True, labelleft=False, direction='in')
        _ = self.score_ax.plot(times, scores, color='C0', linestyle='--', linewidth=1.25)

    def plot_inventory(self, hist):
        times = [x for x in hist]
        inventories = [x['inventory'] for x in hist.values()]

        self.inventory_ax.clear()
        self.inventory_ax.set_title('Inventory', loc='center')
        # self.inventory_ax.text(.025, .925, 'Inventory', horizontalalignment='left', transform=self.inventory_ax.transAxes)
        self.inventory_ax.set_xlim(self.inv_xlim[0], self.inv_xlim[1])
        # self.inventory_ax.set_ylim(self.inv_ylim[0], self.inv_ylim[1])
        self.inventory_ax.tick_params(right=True, left=False, labelright=True, labelleft=False, direction='in')
        _ = self.inventory_ax.plot(times, inventories, color='C0', linestyle='--', linewidth=1.25)

    def plot_orders(self, time, orders):
        self.orders_ax.clear()
        self.orders_ax.set_title('Orders', loc='right')
        # self.orders_ax.set_xlim(0, 9)
        self.orders_ax.set_ylim(self.orders_ylim[0], self.orders_ylim[1])
        self.orders_ax.tick_params(right=True, left=False, labelright=True, labelleft=False, direction='in')

        # _ = self.orders_ax.bar(time, orders, color='C0')

        pass

    def draw(self, pause=True):
        plt.draw()
        plt.tight_layout()
        if pause:
            input('Display paused [Press any button to continue].')

class Message():
    """Used to pass messages back to agent after order book updates."""

    def __init__(self, timestamp, label, side, price, shares, refno):
        assert refno is not None, 'Message did not receive a reference number'
        assert label in ('add', 'delete', 'execute'), 'Message received an invalid label (valid labels are: add, delete, and execute)'
        self.timestamp = timestamp
        self.label = label
        self.side = side
        self.price = price
        self.shares = shares
        self.refno = refno

    def __str__(self):
        items = [round(self.timestamp, 3), self.label, self.side, self.price, self.shares ,self.refno]
        return "<Message(timestamp={}, label='{}', side='{}', price={}, shares={}, refno={})>".format(*items)

    def __repr__(self):
        items = [round(self.timestamp, 3), self.label, self.side, self.price, self.shares ,self.refno]
        return "<Message(timestamp={}, label='{}', side='{}', price={}, shares={}, refno={})>".format(*items)

class Order():
    """Used to update the order book."""

    def __init__(self, timestamp, label, side, price, shares, refno=None):
        if label == 'cancel':
            assert refno is not None, 'Order (label=cancel) did not receive a reference number'
        assert label in ('limit', 'cancel', 'market')
        self.timestamp = timestamp
        self.label = label
        self.side = side
        self.price = price
        self.shares = shares
        if label == 'limit':
            self.refno = np.random.randint(low=0, high=10 ** 9)
        else:
            self.refno = refno

    def __str__(self):
        items = [round(self.timestamp, 3), self.label, self.side, self.price, self.shares ,self.refno]
        return "<Order(timestamp={}, label='{}', side='{}', price={}, shares={}, refno={})>".format(*items)

    def __repr__(self):
        items = [round(self.timestamp, 3), self.label, self.side, self.price, self.shares ,self.refno]
        return "<Order(timestamp={}, label='{}', side='{}', price={}, shares={}, refno={})>".format(*items)

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
        if order is None:
            print('update received None; returning None')
            return None
        elif order.label == 'limit':
            message = self.add_order(order)
            if message is not None:
                return [message]
            else:
                return message
        elif order.label == 'cancel':
            message = self.delete_order(order)
            if message is not None:
                return [message]
            else:
                return message
        elif order.label == 'market':
            return self.execute_order(order)  # possibly multiple messages

    def add_order(self, order):
        """Process an add order."""
        assert order.label == 'limit', "add_order received a {} order.".format(order.label)
        assert order.side in ('bid', 'ask'), "add_order received an order with an invalid side"
        if order.side == 'bid':
            if len(self.asks.keys()) > 0:
                if not order.price < min(self.asks.keys()):
                    print(">> LIMIT BID received above best ask; returning None")
                    return None
            if order.price in self.bids.keys():
                self.bids[order.price].append(order)
                print(">> LIMIT BID appended to queue at price {}".format(order.price))
                return Message(timestamp=order.timestamp,
                               label='add',
                               side=order.side,
                               price=order.price,
                               shares=order.shares,
                               refno=order.refno)
            else:
                self.bids[order.price] = [order]
                print(">> LIMIT BID started a new queue at price {}".format(order.price))
                return Message(timestamp=order.timestamp,
                               label='add',
                               side=order.side,
                               price=order.price,
                               shares=order.shares,
                               refno=order.refno)
        elif order.side == 'ask':
            if len(self.bids.keys()) > 0:
                if not order.price > max(self.bids.keys()):
                    print(">> LIMIT ASK received below best bid; returning None")
                    return None
            if order.price in self.asks.keys():
                self.asks[order.price].append(order)
                print(">> LIMIT ASK appended to queue at price {}".format(order.price))
                return Message(timestamp=order.timestamp,
                               label='add',
                               side=order.side,
                               price=order.price,
                               shares=order.shares,
                               refno=order.refno)
            else:
                self.asks[order.price] = [order]
                print(">> LIMIT ASK started a new queue at price {}".format(order.price))
                return Message(timestamp=order.timestamp,
                               label='add',
                               side=order.side,
                               price=order.price,
                               shares=order.shares,
                               refno=order.refno)

    def delete_order(self, order):
        """Process a delete order."""
        assert order.label == 'cancel', "delete_order received a {} order.".format(order.label)
        if order.side == 'bid':
            try:
                queue = self.bids[order.price]
                idx = [o.refno for o in queue].index(order.refno)
                o = queue.pop(idx)
                print(">> Deleted BID order {}".format(order.refno))
                # Check if the queue if empty now...
                if len(queue) == 0:
                    self.bids.pop(order.price)
                    print(">> Removed queue with price {} from bids".format(order.price))
                return Message(timestamp=order.timestamp,
                               label='delete',
                               side=order.side,
                               price=order.price,
                               shares=o.shares,  # potentially different from original order!
                               refno=order.refno)
            except:
                print(">> BID order with reference number {} and price {} does not exist; returning None".format(order.refno, order.price))
                return None
        elif order.side == 'ask':
            try:
                queue = self.asks[order.price]
                idx = [o.refno for o in queue].index(order.refno)
                o = queue.pop(idx)
                print(">> Deleted ASK order {}".format(order.refno))
               # Check if the queue if empty now...
                if len(queue) == 0:
                    self.asks.pop(order.price)
                    print(">> Removed queue with price {} from asks".format(order.price))
                return Message(timestamp=order.timestamp,
                               label='delete',
                               side=order.side,
                               price=order.price,
                               shares=o.shares,  # potentially different from original order!
                               refno=order.refno)
            except:
                print(">> ASK order with reference number {} and price {} does not exist; returning None".format(order.refno, order.price))
                return None

    def execute_order(self, order):
        """Process a market order."""
        assert order.label == 'market', "execute_order recieved a {} order.".format(order.label)
        if order.side == 'bid':
            side = 'ask'
        elif order.side == 'ask':
            side = 'bid'
        match = self.get_best(side)
        if match is not None:
            matched_shares = min(order.shares, match.shares)
            order.shares -= matched_shares
            match.shares -= matched_shares
            print(">> Executed {} shares @ {}".format(matched_shares, match.price))
            message = Message(timestamp=order.timestamp,
                              label='execute',
                              side=match.side,
                              price=match.price,
                              shares=matched_shares,
                              refno=match.refno)
            messages = [message]
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
                        print(">> MARKET order was PARTIALLY executed.")
                        break
                    elif (match is None) and (order.shares == 0):
                        print(">> MARKET order was FULLY executed.")
                        break
                if order.shares == 0:
                    print(">> MARKET order was FULLY executed.")
                    break
                matched_shares = min(order.shares, match.shares)
                order.shares -= matched_shares
                match.shares -= matched_shares
                print(">> Executed {} shares @ {}".format(matched_shares, match.price))
                message = Message(timestamp=order.timestamp,
                                  label='execute',
                                  side=match.side,
                                  price=match.price,
                                  shares=matched_shares,
                                  refno=match.refno)
                messages.append(message)
            return messages
        else:
            print(">> MARKET order was not executed (no match found); returning None")
            return None

    def get_best(self, side):
        """Return the best offer on a given side of the book."""
        if side == 'bid':
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
        self.history = {0: {'score': 0, 'inventory': 0}}

    def reset(self):
        self.score = 0
        self.inventory = 0
        self.orders = []
        self.time = 0

    def choose_action(self, time, data):
        """Choose an action using algorithm.

        Valid algorithms must be of the form:

            def algorithm(actions, args):
                ...
                return action

        The arguments passed to algorithm will need to be passed through 'state'. These
        might include: the state of the book, past orders, timestamp, etc...
        """
        # events, messages = data
        events = data
        for event in events:
            time, node = event
            print('Agent received EVENT: <Event(time={}, node={})>'.format(round(time, 3), node))
        # for message in messages:
            # print('Agent received MESSAGE: {}'.format(message))
        self.history[time] = {'score': self.score, 'inventory': self.inventory}
        return np.random.choice(self.actions, p=[0.999, 0.0005, 0.0005])
        # return self.algorithm(self.actions)

    def update_orders(self, messages):
        """Update agent based on external events, and update self.

        Args
            message: a message generated by the matching engine.
        """
        if messages is None:
            # print('Agent received update MESSAGE: None; skipping')
            pass
        else:
            for message in messages:
                if message is None:
                    # print('Agent received update MESSAGE: None; skipping')
                    pass
                elif (message.label == 'execute') and (message.refno in [o.refno for o in self.orders]):
                    print('Agent received update MESSAGE: {}'.format(message))
                    match = self.orders[[o.refno for o in self.orders].index(message.refno)]
                    match.shares -= message.shares
                    print('Agent updated ORDER {}'.format(match))
                    if match.shares == 0:
                        self.orders.remove(match)
                        print('(The order was popped)')
                    if message.side == 'ask':
                        self.score += message.shares * message.price
                        self.inventory -= message.shares
                    elif message.side == 'bid':
                        self.score -= message.shares * message.price
                        self.inventory += message.shares
                    self.history[message.timestamp] = {'score': self.score, 'inventory': self.inventory}

    def confirm_order(self, order, messages):
        """Confirm agent order was processed, and update self.

        Args
            order: a copy of the order submitted by the agent.
            messages: responses generated by the matching engine.
        """
        if messages is None:
            print('Agent received confirmation MESSAGE: None; skipping')
            pass
        else:
            for message in messages:
                if message is None:
                    print('Unable to process the order; skipping')
                else:
                    print('Agent received confirmation MESSAGE: {}'.format(message))
                    if message.label == 'add':
                        assert order.refno == message.refno, 'confirm_order receive non-matching ORDER and MESSAGE reference numbers'
                        self.orders.append(order)
                        print('Agent added ORDER {}'.format(order))
                    if (message.label == 'execute') and (message.refno in [o.refno for o in self.orders]):
                        match = self.orders[[o.refno for o in self.orders].index(message.refno)]
                        match.shares -= message.shares
                        print('Agent updated ORDER {}'.format(match))
                        if match.shares == 0:
                            self.orders.remove(match)
                            print('(The order was popped)')
                        if message.side == 'ask':
                            self.score += message.shares * message.price
                            self.inventory -= message.shares
                        elif message.side == 'bid':
                            self.score -= message.shares * message.price
                            self.inventory += message.shares
                        self.history[message.timestamp] = {'score': self.score, 'inventory': self.inventory}
                    if message.label == 'delete':
                        assert order.refno == message.refno, 'confirm_order receive non-matching ORDER and MESSAGE reference numbers'
                        match = self.orders[[o.refno for o in self.orders].index(message.refno)]
                        print('Agent deleted ORDER {}'.format(match))
                        self.orders.remove(match)

class Simulator():

    def __init__(self, book, model, agent, dt):
        self.book = book
        self.model = model
        self.agent = agent
        self.display = Display()
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
            print('Simulator generated {} events'.format(len(self.originals['times'])))

    def run(self, t_max, pause=True):
        """Apply algorithm to the sequence of order book states."""
        t = self.dt
        i = 0
        while t < t_max:
            # Find events that occurred in (t - dt, t), excluding agent's actions
            events = []
            while self.times[i] < t:
                if i < len(self.times) - 1:
                    events.append((self.times[i], self.nodes[i]))
                    i += 1
                else:
                    break
            if len(events) > 0:
                print('Found {} EVENTs in interval ({}, {}): {}'.format(len(events), round(t - self.dt, 3), round(t, 3), events))
            # Update the book
            for time, node in events:
                order = self.to_order(node, time)
                print('>> Updating book for ORDER ({})'.format(order))
                messages = self.book.update(order)
                self.agent.update_orders(messages)
                # self.plot(t, pause)
            # Choose an action
            action = self.agent.choose_action(t, events)  # Agent can decide what to do with data!
            if action is not None:
                i += 1  # Skip this event next iteration
                print('Agent performed ACTION {}'.format(action))
                order = self.to_order(action, t)
                print('>> Updating book for ORDER ({})'.format(order))
                messages = self.book.update(order)
                self.agent.confirm_order(order, messages)
                events = self.generate_events(t_max, (t, action))
                self.update_events(events)
                # self.plot(t, pause)
            t += self.dt
        print('Reached the end of simulation (t={})'.format(t_max))
        self.plot(t, pause=False)

    def update_events(self, events):
        """
        Args
            events: (list) of (tuples) of the form (float, int) indicating the time and node of events.
        """

        # TODO: faster to directly insert the new events?

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

    def to_order(self, node, timestamp):
        """Convert an event to an order"""
        nodes = [('limit', 'bid', -1),
                 ('limit', 'bid', 0),
                 ('limit', 'bid', 1),
                 ('limit', 'ask', -1),
                 ('limit', 'ask', 0),
                 ('limit', 'ask', 1),
                 ('cancel', 'bid', 0),
                 ('cancel', 'bid', 1),
                 ('cancel', 'ask', 0),
                 ('cancel', 'ask', 1),
                 ('market', 'bid', None),
                 ('market', 'ask', None)]
        label, side, level = nodes[node]
        if label == 'limit':
            if level + 1 > len(self.book.prices()[side]):
                print('Failed to generate the order because the specified level is greater than the number of available prices; returning None')
                return None
            elif len(self.book.prices()[side]) == 0:
                print('Failed generate the order because the book is empty; returning None')
                return None
            else:
                if level == -1:
                    if side == 'bid':
                        price = self.book.prices()[side][0] + 1
                    elif side == 'ask':
                        price = self.book.prices()[side][0] - 1
                else:
                    price = self.book.prices()[side][level]
                shares = np.random.choice([100, 200, 300])
                order = Order(timestamp, label, side, price, shares, None)
        elif label == 'cancel':
            if level + 1 > len(self.book.prices()[side]):
                print('Failed to generate the order because the specified level is greater than the number of available prices; returning None')
                return None
            elif len(self.book.prices()[side]) == 0:
                print('Failed generate the order because the book is empty; returning None')
                return None
            price = self.book.prices()[side][level]
            if side == 'bid':
                order = np.random.choice(self.book.bids[price])
            elif side == 'ask':
                order = np.random.choice(self.book.asks[price])
            order.label = 'cancel'
            order.timestamp = timestamp
        elif label == 'market':
            shares = np.random.choice([10, 50, 100])
            order = Order(timestamp, label, side, None, shares, None)
        return order

    def plot(self, time, pause):
        # plt.close()
        # plt.subplot(211)
        # # plot events up to time t
        # originals = self.originals['times']
        # currents = [s for s in self.originals['times'] if s <= t]
        # actions = [s for s in self.actions['times'] if s <= t]
        # generated = [s for s in self.generated['times'] if s <= t]
        # plt.scatter(originals, [1] * len(originals), alpha=0.25, color='C0')  # originals
        # plt.scatter(currents, [1] * len(currents), color='C0')  # originals - current
        # plt.scatter(actions, [2] * len(actions), color='C1')  # actions
        # plt.scatter(generated, [3] * len(generated), color='C2')  # generated
        # # plt.legend(['originals', 'actions', 'generated', 'total'], loc='right')
        # plt.ylim(0,4)
        # plt.xlim(0,5)
        # plt.yticks([1, 2, 3], ['Orig.', 'Act.', 'Gen.'])
        # plt.show()
        # # plot current order book
        # plt.subplot(212)
        # self.book.show()

        self.display.plot_events(time, (self.originals, self.actions, self.generated))
        self.display.plot_book((self.book.bids, self.book.asks))
        self.display.plot_score(self.agent.history)
        self.display.plot_inventory(self.agent.history)
        # self.display.plot_orders(t, self.agent.orders)
        self.display.draw(pause)

# Example
def random_limit_order():
    timestamp = -1
    label = 'limit'
    side = np.random.choice(['bid', 'ask'])
    if side == 'bid':
        price = np.random.choice(np.arange(2991, 3003, 1, dtype='int'))
    else:
        price = np.random.choice(np.arange(2998, 3011, 1, dtype='int'))
    shares = np.random.choice(np.arange(100, 400, 100))
    refno = None
    return Order(timestamp, label, side, price, shares, refno)

N = 20
orders = []
n = 0
book = OrderBook()
while n < N:
    order = random_limit_order()
    print('{}'.format(order))
    if order.refno not in [order.refno for order in orders] and order.refno is not None:
        orders.append(order)
    if order.label == 'cancel':
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
simulator.plot(0, pause=False)
simulator.run(t_max, pause=False)

import numpy as np
import matplotlib.pyplot as plt
# from models import NetworkPoisson

plt.ion()

# Add: Choose a random number of shares based on empirical distribution.
# Delete: Choose an order at random from specified queue.
# Execute: Choose number of shares randomly, with maximum set to level 0 liquidity.

# books = pd.read_csv('GOOG_010213.csv')
# dists = {}
# for node in range(6):
#     tmp = book[ books.node == node ]
#     cnts, bins = np.histogram(tmp.shares.values, bins=np.arange(0, 310, 10))
#     dists[i] = (bins[1:], cnts / np.sum(cnts))

class Order():

    def __init__(self, label, side, price, shares, refno=None):
        if label == 'delete':
            assert refno is not None, 'delete orders require a refno'
        self.label = label  # order type (A, E, D)
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

    # TODO: Don't allow adds for prices that show up on opposite side!
    def add_order(self, order):
        """Process an add order."""
        assert order.label == "add", "add_order recieved a non-add order."
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
        if side == "bid":
            prices = list(self.bids.keys())
            if len(prices) > 0:
                return self.bids[max(prices)][0]
            else:
                print("No more BID orders remaining.")
                return None
        elif side == "ask":
            prices = list(self.asks.keys())
            if len(prices) > 0:
                return self.asks[min(prices)][0]
            else:
                print("No more ASK orders remaining.")
                return None

    # TODO: make this a Simulator method
    def random_shares(self, label):
        if label == "add":
            return np.random.choice(self.add_dist.a, self.add_dist.p)
        elif label == "execute":
            return np.random.choice(self.execute_dist.a, self.execute_dist.p)

    def get_price(self, side, level):
        """Find the price at the given (side, level) of the book."""

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
                if spread > 0.01:
                    return bid + 0.01
                else:
                    print("Spread equals the minimum tick.")
                    return None
            elif level == 1:
                return bid
            elif level == 2:
                return bid - 0.01
        elif side == "ask":
            if level == 0:
                if spread > 0.01:
                    return ask - 0.01
                else:
                    print("Spread equals the minimum tick.")
                    return None
            elif level == 1:
                return ask
            elif level == 2:
                return ask + 0.01

    # TODO
    def get_order(self, side, price, refno):
        pass

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
        return bids, asks

    def prices(self):
        """The price of each level."""
        bids = [p for p in self.bids]
        bids.sort()
        bids.reverse()
        asks = [p for p in self.asks]
        asks.sort()
        return {'bids': bids, 'asks': asks}

    def show(self):
        """Plot the order book."""

        plt.close()
        for price in self.bids.keys():
            # the x of each bar is the price
            x = [price] * len(self.bids[price])
            # the height of each bar is order['shares']
            h = [order.shares for order in self.bids[price]]
            # the bottom of each bar is b += order['shares'], starting from b = 0
            if len(self.bids[price]) == 0:
                b = [0]
                # print('{}\n{}\n{}'.format(x, h, b))
            else:
                b = [0]
                for order in self.bids[price][:-1]:
                    b.append(sum(b) + order.shares)
                # print('{}\n{}\n{}'.format(x, h, b))
            plt.bar(x, h, bottom=b, width=1, align='center', alpha=0.5, color='blue', edgecolor='white', linewidth=1)
        for price in self.asks.keys():
            # the x of each bar is the price
            x = [price] * len(self.asks[price])
            # the height of each bar is order['shares']
            h = [order.shares for order in self.asks[price]]
            # the bottom of each bar is b += order['shares'], starting from b = 0
            if len(self.asks[price]) == 0:
                b = [0]
                # print('{}\n{}\n{}'.format(x, h, b))
            else:
                b = [0]
                for order in self.asks[price][:-1]:
                    b.append(sum(b) + order.shares)
                # print('{}\n{}\n{}'.format(x, h, b))
            plt.bar(x, h, bottom=b, width=1, align='center', alpha=0.5, color='green', edgecolor='white', linewidth=1)
        plt.xlim(2990, 3011)
        plt.xticks(np.arange(2990, 3011, 2))
        plt.ylim(0, 1000)
        plt.show()

def random_order(orders):
    if len(orders) < 10:
        label = 'add'
        side = np.random.choice(['bid', 'ask'])
        if side == 'bid':
            price = np.random.choice(np.arange(2991, 3003, 1, dtype='int'))
        else:
            price = np.random.choice(np.arange(2998, 3011, 1, dtype='int'))
        shares = np.random.choice(np.arange(100, 400, 100))
        refno = None
    else:
        label = np.random.choice(['add', 'delete', 'execute'])
        if label == 'add':
            side = np.random.choice(['bid', 'ask'])
            if side == 'bid':
                price = np.random.choice(np.arange(2991, 3003, 1, dtype='int'))
            else:
                price = np.random.choice(np.arange(2998, 3011, 1, dtype='int'))
            shares = np.random.choice(np.arange(100, 400, 100))
            refno = None
        elif label == 'delete':
            o = np.random.choice(orders, p=[1 / len(orders)] * len(orders))
            side = o.side
            price = o.price
            shares = o.shares
            refno = o.refno
        elif label == 'execute':
            side = np.random.choice(['bid', 'ask'])
            price = None
            shares = np.random.choice(np.arange(10, 110, 10))
            refno = None
    return Order(label, side, price, shares, refno)

book = OrderBook()
orders = []
while True:
    order = random_order(orders)
    print('{}'.format(order))
    if order.refno not in [order.refno for order in orders] and order.refno is not None:
        orders.append(order)
    if order.label == 'delete':
        for o in orders:
            if o.refno == order.refno:
                orders.remove(o)
    book.update(order)
    book.show()
    print(book.shares())
    input()

# simulation
# env = Simulator()
# state = env.reset()
# while not env.done:
#     action = learner.choose(state)
#     state, done = env.step(action)
# score = env.evaluate()

# class Simulator():
#
#     def __init__(self, network, shares, T, dt=1):
#         self.book = OrderBook()
#         self.network = network
#         self.N = shares
#         self.shares = 0
#         self.T = T
#         self.clock = 0  # milliseconds
#         self.dt = dt
#         self.done = False
#         times, nodes = self.network.generate_data(self.T)
#         times = np.array(1000 * np.round(times, decimals=3), dtype='int')
#         nodes = np.array(nodes, dtype='int')
#         self.events = dict(zip(times, nodes))
#
#     def reset(self):
#         self.book = OrderBook()
#         self.shares = 0
#         self.clock = 0
#         self.done = False
#         self.score = 0
#         times, nodes = self.network.generate_data(self.T)
#         times = np.array(1000 * np.round(times, decimals=3), dtype='int')
#         nodes = np.array(nodes, dtype='int')
#         self.events = dict(zip(times, nodes))
#
#         return (self.book, self.T - self.clock, self.N - self.shares)
#
#     def step(self, action):
#         if not self.done:
#             # Handle simulated orders
#             if self.clock == self.times[self.ix]:
#                 label, side, level = node_message(self.nodes[self.ix])
#                 if label == "add":
#                     price = OrderBook.get_price(side, level)
#                     shares = self.random_shares('add')
#                     order = Order(label=label, side=side, price=price, shares=shares)
#                 elif label == "delete":
#                     price = OrderBook.get_price(side, level)
#                     rand = OrderBook.random_order(price)
#                     order = Order(label=label, side=side, price=rand.price,
#                                   shares=rand.shares, refno=rand.refno)
#                 elif label == "execute":
#                     price = OrderBook.get_price(side, level)
#                     shares = self.random_shares('execute')
#                     order = Order(label=label, side=side, price=price, shares=shares)
#                 self.book.update(order)
#             # Handle user action
#             if action is not None:
#                 order = self.convert_action(action)
#                 self.book.update(order)
#                 # Record feedback for user
#                 info = [order]
#                 # Simulate child events of event
#                 node, _ = action
#                 children = net.simulate_spike(self.T, node, self.clock, units='ms')
#                 # Add events to simulated orders
#                 self.events.update(children)
#                 for time, node in children.items():
#                     if time not in self.events:
#                         self.events.update({time: node})
#                     else:
#                         wiggled = time + np.random.choice([-1, 1])
#                         if (wiggled not in self.events) and (wiggled > self.clock):
#                             self.events.update({wiggled: node})
#                         else:
#                             print("Unable to add child {}, {} to events.".format(time, node))
#             # Update the clock
#             if self.clock == self.T - dt:
#                 self.clock = T
#                 self.done = True
#             else:
#                 self.clock += self.dt
#         else:
#             print("Simulation terminated.")
#         return self.book, self.done, info
#
#     nodedefs = [('add', 'bid', -1),
#                 ('add', 'bid', 0),
#                 ('add', 'bid', 1),
#                 ('add', 'ask', -1),
#                 ('add', 'ask', 0),
#                 ('add', 'ask', 1),
#                 ('delete', 'bid', 0),
#                 ('delete', 'bid', 1),
#                 ('delete', 'ask', 0),
#                 ('delete', 'ask', 1),
#                 ('execute', 'bid', None),
#                 ('execute', 'ask', None)]
#
#     def convert_action(self, action):
#         """Convert an action to an order."""
#         node, details = action
#         label, side, level = nodedefs[node]
#         if label in ('add', 'execute'):
#             price = self.book.get_price(side, level)
#             shares = details['shares']
#             # Don't need to provide refno
#         elif label == 'delete':
#             # Don't need to provide shares
#             price = self.book.get_price(side, level)
#             refno = details['refno']
#         elif label == 'execute':
#             # Don't need to provide price
#             shares = details['shares']
#             # Don't need to provide refno
#         return Order(label, side, price, shares, refno)
#
#     def random_shares(node):
#         bins, p = dists[node]
#         return np.random.choice(bins, p)

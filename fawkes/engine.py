from fawkes.models import NetworkPoisson
import numpy as np
import pandas as pd
import h5py as h5
from pprint import pformat
import matplotlib.pyplot as plt

def random_limit_order(bmin=2991, bmax=3003, amin=2998, amax=3011):
    side = np.random.choice(['bid', 'ask'])
    shares = np.random.choice(np.arange(100, 400, 100))
    if side == 'bid':
        price = np.random.choice(np.arange(bmin, bmax, dtype='int'))
    else:
        price = np.random.choice(np.arange(amin, amax, dtype='int'))
    return Order(-1, 'limit', side, price, shares, None)

class Display():

    # TODO: Have axes setup on creation.

    def __init__(self, **kwargs):

        """
        Keywords:
            t_max: the maximum time to display.

        """

        plt.ion()
        plt.rc('font', size=10, family='Arial')

        self.events_ax = plt.subplot2grid((3,2), (0,0), rowspan=1)
        self.book_ax = plt.subplot2grid((3,2), (1,0), rowspan=1)
        self.score_ax = plt.subplot2grid((3,2), (0,1), rowspan=1)
        self.inventory_ax = plt.subplot2grid((3,2), (1,1), rowspan=1)
        self.orders_ax = plt.subplot2grid((3,2), (2,0), colspan=2)
        plt.tight_layout()

        # Default kwargs
        self.T = 5
        self.book_xlim, self.book_ylim = ([2990, 3011], [0, 2000])
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
        self.events_ax.set_xlim(0, self.T)
        self.events_ax.set_ylim(0, 4)
        # self.events_ax.set_yticks([1, 2, 3], ['Originals', 'Actions', 'Generated'])
        self.events_ax.set_yticks([])
        self.events_ax.tick_params(direction='in')

        _ = self.events_ax.scatter(originals['times'], [1] * len(originals['times']), s=30, color='C0', alpha=0.25)
        _ = self.events_ax.scatter(o, [1] * len(o), s=30, color='C0', label='Originals')
        _ = self.events_ax.scatter(a, [2] * len(a), s=30, color='C1', label='Actions')
        _ = self.events_ax.scatter(g, [3] * len(g), s=30, color='C2', label='Responses')
        self.events_ax.legend()

    def plot_book(self, book):

        self.book_ax.clear()
        self.book_ax.set_title('Books', loc='center')
        # self.book_ax.text(.025, .925, 'Book', horizontalalignment='left', transform=self.book_ax.transAxes)
        self.book_ax.set_xlim(self.book_xlim[0], self.book_xlim[1])
        self.book_ax.set_xticks(range(self.book_xlim[0] + 1, self.book_xlim[1], 2))
        self.book_ax.set_ylim(self.book_ylim[0], self.book_ylim[1])
        self.book_ax.tick_params(direction='in')

        bids, asks = book
        bmax = 0
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
            if np.sum(h) > bmax:
                bmax = np.sum(h)
        amax = 0
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
            if np.sum(h) > amax:
                amax = np.sum(h)
        ymax = max(bmax, amax)
        self.book_ax.set_ylim(0, np.ceil(ymax / 1000) * 1000)

    def plot_score(self, hist):
        times = [x for x in hist]
        scores = [x['score'] for x in hist.values()]

        self.score_ax.clear()
        self.score_ax.set_title('Score', loc='center')
        # self.score_ax.text(.025, .925, 'Score', horizontalalignment='left', transform=self.score_ax.transAxes)
        self.score_ax.set_xlim(0, self.T)
        # self.score_ax.set_ylim(self.score_ylim[0], self.score_ylim[1])
        self.score_ax.tick_params(right=True, left=False, labelright=True, labelleft=False, direction='in')
        _ = self.score_ax.plot(times, scores, color='C0', linestyle='--', linewidth=1.25)

    def plot_inventory(self, hist):
        times = [x for x in hist]
        inventories = [x['inventory'] for x in hist.values()]

        self.inventory_ax.clear()
        self.inventory_ax.set_title('Inventory', loc='center')
        # self.inventory_ax.text(.025, .925, 'Inventory', horizontalalignment='left', transform=self.inventory_ax.transAxes)
        self.inventory_ax.set_xlim(0, self.T)
        # self.inventory_ax.set_ylim(self.inv_ylim[0], self.inv_ylim[1])
        self.inventory_ax.tick_params(right=True, left=False, labelright=True, labelleft=False, direction='in')
        _ = self.inventory_ax.plot(times, inventories, color='C0', linestyle='--', linewidth=1.25)

    def plot_orders(self, orders):
        """orders is a list of orders maintained by Agent"""
        colors = {'bid': 'C0', 'ask': 'C1'}
        self.orders_ax.clear()
        self.orders_ax.set_title('Orders', loc='center')
        if orders['bid'] is not None:
            xmin = orders['bid'].price - 2
            _ = self.orders_ax.bar(orders['bid'].price,
                                   orders['bid'].shares,
                                   width=1,
                                   color=colors['bid'],
                                   edgecolor='white',
                                   linewidth=1)
        if orders['ask'] is not None:
            xmax = orders['ask'].price + 2
            _ = self.orders_ax.bar(orders['ask'].price,
                                   orders['ask'].shares,
                                   width=1,
                                   color=colors['ask'],
                                   edgecolor='white',
                                   linewidth=1)
        self.orders_ax.set_xlim(2990, 3011)
        self.orders_ax.set_ylim(0, 200)
        self.orders_ax.set_xticks(np.arange(2991, 3011, 2))

        # if len(orders) > 0:
        #     orders = sorted(orders, key=lambda x: x.price)
        #     xmax = max(20, len(orders)) + 1
        #     self.orders_ax.set_xlim(0, xmax)
        #     self.orders_ax.set_ylim(0, max([o.shares for o in orders]) + 100)
        #     prices = [o.price for o in orders]
        #     self.orders_ax.set_xticks(range(0, xmax + 1))
        #     self.orders_ax.set_xticklabels([''] + prices)
        #     for i, order in enumerate(orders):
        #         if order.side == 'bid':
        #             color = 'C0'
        #         elif order.side == 'ask':
        #             color = 'C1'
        #         _ = self.orders_ax.bar(i + 1,
        #                                order.shares,
        #                                width=1,
        #                                color=color,
        #                                edgecolor='white',
        #                                linewidth=1)
        # else:
        #     self.orders_ax.set_xlim(0, 10)
        #     self.orders_ax.set_xticks([])
        #     self.orders_ax.set_xticklabels([])
        #     self.orders_ax.set_ylim(0, 200)

    def draw(self, pause=True):
        plt.draw()
        plt.tight_layout()
        if pause:
            input('Display paused [Press any button to continue].')

class Message():
    """Used to pass messages back to agent after order book updates."""

    def __init__(self, timestamp, label, side, price, shares, refno, uid=None):
        assert refno is not None, 'Message did not receive a reference number'
        assert label in ('add', 'delete', 'execute'), 'Message received an invalid label (valid labels are: add, delete, and execute)'
        self.timestamp = timestamp
        self.label = label
        self.side = side
        self.price = price
        self.shares = shares
        self.refno = refno
        self.uid = uid

    def __str__(self):
        items = [round(self.timestamp, 3), self.label, self.side, self.price, self.shares ,self.refno, self.uid]
        return "<Message(timestamp={}, label='{}', side='{}', price={}, shares={}, refno={}, uid={})>".format(*items)

    def __repr__(self):
        items = [round(self.timestamp, 3), self.label, self.side, self.price, self.shares ,self.refno]
        return "<Message(timestamp={}, label='{}', side='{}', price={}, shares={}, refno={}, uid={})>".format(*items)

class Order():
    """Used to update the order book."""

    def __init__(self, timestamp, label, side, price, shares, newPrice=None, refno=None, uid=None):
        if label == 'cancel':
            assert refno is not None, 'Order (label=cancel) did not receive a reference number'
        assert label in ('limit', 'cancel', 'market', 'replace'), "Order received invvalid argument label={}".format(label)
        assert side in ('bid', 'ask'), "Order received invvalid argument side={}".format(side)
        if price is not None:
            assert price > 0, "Order received invvalid argument price={}".format(price)
        if shares is not None:
            assert shares > 0, "Order received invvalid argument shares={}".format(shares)
        self.timestamp = timestamp
        self.label = label
        self.side = side
        self.price = price
        self.shares = shares
        self.newPrice = newPrice
        if label == 'limit' and refno is None:
            self.refno = np.random.randint(low=0, high=10 ** 9)
        else:
            self.refno = refno
        self.uid = uid

    def __str__(self):
        items = [round(self.timestamp, 3),
                 self.label,
                 self.side,
                 self.price,
                 self.shares,
                 self.newPrice,
                 self.refno,
                 self.uid]
        return "<Order(timestamp={}, label='{}', side='{}', price={}, shares={}, newPrice={}, refno={}, uid={})>".format(*items)

    def __repr__(self):
        items = [round(self.timestamp, 3),
                 self.label,
                 self.side,
                 self.price,
                 self.shares,
                 self.newPrice,
                 self.refno,
                 self.uid]
        return "<Order(timestamp={}, label='{}', side='{}', price={}, shares={}, newPrice={}, refno={}, uid={})>".format(*items)

    def copy(self):
        order = Order(timestamp=self.timestamp,
                      label=self.label,
                      side=self.side,
                      price=self.price,
                      shares=self.shares,
                      newPrice=self.newPrice,
                      refno=self.refno,
                      uid=self.uid)
        return order

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
            return self.execute_order(order)
        elif order.label == 'replace':
            return self.replace_order(order)

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
                self.bids[order.price].append(order.copy())
                print(">> LIMIT BID appended to queue at price {}".format(order.price))
                return Message(timestamp=order.timestamp,
                               label='add',
                               side=order.side,
                               price=order.price,
                               shares=order.shares,
                               refno=order.refno,
                               uid=order.uid)
            else:
                self.bids[order.price] = [order.copy()]
                print(">> LIMIT BID started a new queue at price {}".format(order.price))
                return Message(timestamp=order.timestamp,
                               label='add',
                               side=order.side,
                               price=order.price,
                               shares=order.shares,
                               refno=order.refno,
                               uid=order.uid)
        elif order.side == 'ask':
            if len(self.bids.keys()) > 0:
                if not order.price > max(self.bids.keys()):
                    print(">> LIMIT ASK received below best bid; returning None")
                    return None
            if order.price in self.asks.keys():
                self.asks[order.price].append(order.copy())
                print(">> LIMIT ASK appended to queue at price {}".format(order.price))
                return Message(timestamp=order.timestamp,
                               label='add',
                               side=order.side,
                               price=order.price,
                               shares=order.shares,
                               refno=order.refno,
                               uid=order.uid)
            else:
                self.asks[order.price] = [order.copy()]
                print(">> LIMIT ASK started a new queue at price {}".format(order.price))
                return Message(timestamp=order.timestamp,
                               label='add',
                               side=order.side,
                               price=order.price,
                               shares=order.shares,
                               refno=order.refno,
                               uid=order.uid)

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
                               refno=order.refno,
                               uid=order.uid)
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
                               refno=order.refno,
                               uid=order.uid)
            except:
                print(">> ASK order with reference number {} and price {} does not exist; returning None".format(order.refno, order.price))
                return None

    def execute_order(self, order):
        """Process a market order."""
        assert order.label == 'market', "execute_order received a {} order.".format(order.label)
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
                              refno=match.refno,
                              uid=match.uid)
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
                                  refno=match.refno,
                                  uid=match.uid)
                messages.append(message)
            return messages
        else:
            print(">> MARKET order was not executed (no match found); returning None")
            return None

    def replace_order(self, order):
        assert order.label == 'replace', "replace_order received a {} order.".format(order.label)
        cancel_order = Order(timestamp=order.timestamp,
                             label='cancel',
                             side=order.side,
                             price=order.price,
                             shares=order.shares,
                             refno=order.refno,
                             uid=order.uid)
        limit_order = Order(timestamp=order.timestamp,
                            label='limit',
                            side=order.side,
                            price=order.newPrice,
                            shares=order.shares,
                            uid=order.uid)
        del_message = self.delete_order(cancel_order)
        if del_message is None:
            print("Unable to process delete part of replace order; returning None")
            return None
        else:
            add_message = self.add_order(limit_order)
            if add_message is None:
                print("Unable to process add part of replace order;")
                return None  # NOTE: this should produce an error because the order was deleted!
            else:
                return [del_message, add_message]

    def split_order(self, order):
        assert order.label == 'replace', "replace_order received a {} order.".format(order.label)
        cancel_order = Order(timestamp=order.timestamp,
                             label='cancel',
                             side=order.side,
                             price=order.price,
                             shares=order.shares,
                             refno=order.refno,
                             uid=order.uid)
        limit_order = Order(timestamp=order.timestamp,
                            label='limit',
                            side=order.side,
                            price=order.newPrice,
                            shares=order.shares,
                            uid=order.uid)
        return cancel_order, limit_order

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

    def get_best_bid_price(self):
        try:
            return self.get_best('bid').price
        except:
            return None

    def get_best_ask_price(self):
        try:
            return self.get_best('ask').price
        except:
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

    def __init__(self, **kwargs):
        self.actions = [None,
                        {'label': 'limit', 'side': 'bid'},
                        {'label': 'limit', 'side': 'ask'},
                        {'label': 'cancel', 'side': 'bid'},
                        {'label': 'cancel', 'side': 'ask'},
                        {'label': 'market', 'side': 'bid'},
                        {'label': 'market', 'side': 'ask'}]
        self.score = 0
        self.inventory = 0
        self.orders = []
        self.algorithm = None
        self.verbose = True
        self.uid = np.random.randint(low=0, high=100)
        self.make_fee = 0
        self.take_fee = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.data = {0: {'score': self.score, 'inventory': self.inventory}}

    def reset(self):
        self.score = 0
        self.inventory = 0
        self.orders = []
        self.data = {0: {'score': self.score, 'inventory': self.inventory}}

    def choose_action(self, time, book, events):
        """Choose an action using algorithm.

        Valid algorithms must be of the form:

            def algorithm(actions, args):
                ...
                return Order

        The arguments passed to algorithm will need to be passed through 'state'. These
        might include: the state of the book, past orders, timestamp, etc...
        """

        for event in events:
            t, n = event
            if self.verbose:
                print('(t={}) Agent received EVENT: <Event(time={}, node={})>'.format(round(time, 3), round(t, 3), n))
        if self.algorithm is None:
            return self.default_algo(time, book, events)
        if self.algorithm is 'market':
            return self.market_algo(time, 'bid', 10)
        if self.algorithm is 'limit':
            return self.limit_algo(time, 'bid')
        else:
            return self.algorithm(time, book, events)

    def update(self, time, action, confirm, feed, done):
        if done:  # record the final score and inventory
            self.data[time] = {'score': self.score, 'inventory': self.inventory}
        self.confirm_order(time, action, confirm)
        self.update_orders(time, feed)

    def update_orders(self, time, feed):
        """Update agent based on external events, and update self. The agent only needs to
           update when in receives an execute message (add and delete messages are
           handled by `confirm_order`).

        Args
           messages: a list of messages generated by the matching engine.
        """

        if feed == []:
            if self.verbose:
                print('(t={}) Agent received empty message feed; skipping'.format(round(time, 3)))
        for message in feed:
            if message is None:
                if self.verbose:
                    print('(t={}) Agent received update MESSAGE: None; skipping'.format(round(time, 3)))
                pass
            elif message.label == 'add' or message.label == 'delete':
                print('(t={}) Agent received update MESSAGE: {}'.format(round(time, 3), message))
            elif message.label == 'execute' and message.uid == self.uid:  # agents limit order is hit
                print('(t={}) Agent received update MESSAGE: {}'.format(round(time, 3), message))
                match = self.orders[[o.refno for o in self.orders].index(message.refno)]
                # print('About to delete {} shares from order {}'.format(message.shares, match))
                match.shares -= message.shares
                # print('Deleted {} shares from order {}'.format(message.shares, match))
                assert match.shares >= 0, "Match shares less than zero!"
                print('>> Updated ORDER {}'.format(match))
                if match.shares == 0:
                    self.orders.remove(match)
                    print('>> Removed the order')
                if message.side == 'ask':
                    self.score += (message.shares * message.price) + (message.shares * self.make_fee)
                    self.inventory -= message.shares
                elif message.side == 'bid':
                    self.score -= (message.shares * message.price) + (message.shares * self.make_fee)
                    self.inventory += message.shares
                self.data[message.timestamp] = {'score': self.score, 'inventory': self.inventory}

    # TODO: only pass the message reference number?
    def confirm_order(self, time, order, confirm):
        """Confirm agent order was processed, and update self.

        Args
            order: copy of the order submitted by the agent.
            success: bool indicator. True if the order was processed by matching engine.
            messages: list of messages generated by the matching engine.
        """
        if confirm == []:
            if self.verbose:
                print('(t={}) Agent received empty confirmation feed; skipping'.format(round(time, 3)))
        for message in confirm:
            if message is None:
                print('(t={}) Agent received confirmation MESSAGE: None; skipping'.format(round(time, 3)))
            else:
                print('(t={}) Agent received confirmation MESSAGE: {}'.format(round(time, 3), message))
                if message.label == 'add':
                    assert order.refno == message.refno, '>> confirm_order receive non-matching ORDER and MESSAGE reference numbers'
                    self.orders.append(order.copy())
                    print('>> Added ORDER {}'.format(order))
                elif message.label == 'execute' and message.uid == self.uid:  # own trade
                    match = self.orders[[o.refno for o in self.orders].index(message.refno)]
                    # print('About to delete {} shares from order {}'.format(message.shares, match))
                    match.shares -= message.shares
                    # print('Deleted {} shares from order {}'.format(message.shares, match))
                    assert match.shares >= 0, "Match shares less than zero!"
                    print('>> Updated ORDER {}'.format(match))
                    if match.shares == 0:
                        self.orders.remove(match)
                        print('>> Removed the order')
                elif message.label == 'execute':  # trade (following market order)
                    if order.side == 'bid':
                        self.score -= (message.shares * message.price) - (message.shares * self.take_fee)
                        self.inventory += message.shares
                    elif order.side == 'ask':
                        self.score += (message.shares * message.price) - (message.shares * self.take_fee)
                        self.inventory -= message.shares
                    self.data[message.timestamp] = {'score': self.score, 'inventory': self.inventory}
                elif message.label == 'delete':
                    assert order.refno == message.refno, '>> confirm_order receive non-matching ORDER and MESSAGE reference numbers'
                    match = self.orders[[o.refno for o in self.orders].index(message.refno)]
                    print('>> Deleted ORDER {}'.format(match))
                    self.orders.remove(match)

    def default_algo(self, time, book, events):
        """

            book: object returned by OrderBook.shares()
        """

        action = np.random.choice(self.actions, p=[0.999] + [0.001 / 6] * 6)
        if action is not None:
            if action['label'] == 'limit':
                shares = 100
                if action['side'] == 'bid':
                    if len(book['bid']) == 0 and len(book['ask']) == 0:
                        print('The book is empty; returning None')
                        return None
                    elif len(book['bid']) == 0:
                        print('The bid side of book is empty')
                        price = min(book['ask']) - np.random.choice((1, 2))
                    else:
                        price = max(book['bid']) + np.random.choice((-1, 0, 1))
                elif action['side'] == 'ask':
                    if len(book['ask']) == 0 and len(book['bid']) == 0:
                        print('The book is empty; returning None')
                        return None
                    elif len(book['ask']) == 0:
                        print('The ask side of book is empty')
                        price = min(book['bid']) + np.random.choice((1, 2))
                    else:
                        price = min(book['ask']) + np.random.choice((-1, 0, 1))
                return Order(time, action['label'], action['side'], price, shares, uid=self.uid)
            if action['label'] == 'cancel':
                if action['side'] == 'bid':
                    if len(book['bid']) == 0:
                        print('No bids to cancel; returning None')
                        return None
                    else:
                        price = max(book['bid']) + np.random.choice((-1, 0))
                elif action['side'] == 'ask':
                    if len(book['ask']) == 0:
                        print('Non asks to cancel; returning None')
                        return None
                    else:
                        price = min(book['ask']) + np.random.choice((0, 1))
                orders = [o for o in self.orders if o.price == price]
                if len(orders) > 0:
                    order = orders[np.random.choice([0, -1])]
                    return Order(time, action['label'], order.side, order.price, order.shares, order.refno, uid=self.uid)
                else:
                    print('No orders to cancel at selected price; returning None')
                    return None
            if action['label'] == 'market':
                if action['side'] == 'bid':
                    if len(book['ask']) == 0:
                        print('No ask orders to execute against; returning None')
                        return None
                    else:
                        price = min(book['ask'])
                        shares = min(book['ask'][price], 100)
                elif action['side'] == 'ask':
                    if len(book['bid']) == 0:
                        print('No bid orders to execute against; returning None')
                        return None
                    else:
                        price = max(book['bid'])
                        shares = min(book['bid'][price], 100)
                return Order(time, action['label'], action['side'], price, shares, uid=self.uid)  # TODO: don't need to give a price here?
        else:
            return None

    def market_algo(self, time, side, period):
        """Place a market order every `period` seconds."""
        if round(time, 3) % period == 0:
            return Order(time, 'market', side, None, 100, uid=self.uid)
        else:
            return None

    def limit_algo(self, time, side):
        """Place a limit and wait for fill."""
        if round(time, 3) == 0:
            if len(book['bid']) == 0 and len(book['ask']) == 0:
                print('The book is empty; returning None')
                return None
            elif len(book[side]) == 0:
                print('The {} side of book is empty'.format(side))
                price = min(book['ask']) - 1
            else:
                price = max(book['bid'])
            return Order(time, 'limit', side, price, 600, uid=self.uid)
        else:
            return None

class MarketMaker(Agent):

    def __init__(self, **kwargs):
        Agent.__init__(self)
        self.orders = {'bid': None, 'ask': None}
        self.actions = [None,
                        {'label': 'limit', 'side': 'bid'},
                        {'label': 'limit', 'side': 'ask'},
                        {'label': 'cancel', 'side': 'bid'},
                        {'label': 'cancel', 'side': 'ask'},
                        {'label': 'market', 'side': 'bid'},
                        {'label': 'market', 'side': 'ask'},
                        {'label': 'replace', 'side': 'bid'},
                        {'label': 'replace', 'side': 'ask'}]

    def default_algo(self, time, book, events):
        action = np.random.choice(self.actions, p=[0.999] + [0.001 / 8] * 8)
        if action is not None:
            label = action['label']
            side = action['side']
            if label == 'limit':
                if self.orders[side] is not None:
                    return None
                if len(book['bid']) == 0 and len(book['ask']) == 0:
                    print('The book is empty; returning None')
                    return None
                else:
                    shares = 100
                    if side == 'bid':
                        if len(book['bid']) == 0:
                            print('The bid side of book is empty')
                            price = min(book['ask']) - np.random.choice((1, 2))
                        else:
                            price = max(book['bid']) + np.random.choice((0, 1))
                    elif side == 'ask':
                        if len(book['ask']) == 0:
                            print('The ask side of book is empty')
                            price = max(book['bid']) + np.random.choice((1, 2))
                        else:
                            price = min(book['ask']) + np.random.choice((-1, 0))
                    return Order(timestamp=time,
                                 label=label,
                                 side=side,
                                 price=price,
                                 shares=shares,
                                 uid=self.uid)
            if label == 'cancel':
                order = self.orders[side]
                if order is not None:
                    return Order(timestamp=time,
                                 label=label,
                                 side=side,
                                 price=order.price,
                                 shares=order.shares,
                                 refno=order.refno,
                                 uid=order.uid)
                else:
                    print('No {} order to cancel; returning None'.format(side))
                    return None
            if label == 'market':
                opposite = {'bid': 'ask', 'ask': 'bid'}
                side = opposite[side]
                if len(book[side]) == 0:
                    print('No {} orders to execute against; returning None'.format(side))
                    return None
                else:
                    if side == 'bid':
                        price = max(book[side])
                    elif side == 'ask':
                        price = min(book[side])
                    shares = min(book[side][price], 100)
                return Order(time, label, side, price, shares, uid=self.uid)
            if label == 'replace':
                order = self.orders[side]
                if order is not None:
                    return Order(timestamp=time,
                                 label='replace',
                                 side=side,
                                 price=order.price,
                                 shares=order.shares,
                                 newPrice=order.price + np.random.choice((-1,1)),
                                 refno=order.refno,
                                 uid=order.uid)
                else:
                    print('No {} order to replace; returning None'.format(side))
                    return None

                # order = self.orders[side]
                # if order is not None:
                #     cancel_order = Order(time,
                #                          'cancel',
                #                          side,
                #                          order.price,
                #                          order.shares,
                #                          order.refno,
                #                          order.uid)
                #     limit_order = Order(time,
                #                         'limit',
                #                         side,
                #                         order.price + np.random.choice((-1,1)),
                #                         order.shares,
                #                         uid=order.uid)
                #     return (cancel_order, limit_order)
                # else:
                #     print('No {} order to replace; returning None'.format(side))
                #     return None

    def update_orders(self, time, feed):
        """Update agent based on external events, and update self. The agent only needs to
           update when in receives an execute message (add and delete messages are
           handled by `confirm_order`).

        Args
           messages: a list of messages generated by the matching engine.
        """

        if feed == []:
            pass
            # if self.verbose:
                # print('(t={}) Agent received empty message feed; skipping'.format(round(time, 3)))
        for message in feed:
            if message is None:
                if self.verbose:
                    print('(t={}) Agent received update MESSAGE: None; skipping'.format(round(time, 3)))
                pass
            elif message.label == 'add' or message.label == 'delete':
                print('(t={}) Agent received update MESSAGE: {}'.format(round(time, 3), message))
            elif message.label == 'execute' and message.uid == self.uid:
                print('(t={}) Agent received update MESSAGE: {}'.format(round(time, 3), message))
                match = self.orders[message.side]
                assert match.refno == message.refno, "update received execute with non-matching refno"
                match.shares -= message.shares
                assert match.shares >= 0, "Match shares less than zero!"
                print('>> Updated ORDER {}'.format(match))
                if match.shares == 0:
                    self.cancel_order(match)
                    print('>> Deleted the order')
                if message.side == 'ask':
                    self.score += (message.shares * message.price) + (message.shares * self.make_fee)
                    self.inventory -= message.shares
                elif message.side == 'bid':
                    self.score -= (message.shares * message.price) + (message.shares * self.make_fee)
                    self.inventory += message.shares
                self.data[message.timestamp] = {'score': self.score, 'inventory': self.inventory}

    def confirm_order(self, time, order, confirm):
        """Confirm agent order was processed, and update self.

        Args
            order: copy of the order submitted by the agent.
            success: bool indicator. True if the order was processed by matching engine.
            messages: list of messages generated by the matching engine.
        """
        if confirm == []:
            pass
            # if self.verbose:
                # print('(t={}) Agent received empty confirmation feed; skipping'.format(round(time, 3)))
        for message in confirm:
            if message is None:
                print('(t={}) Agent received confirmation MESSAGE: None; skipping'.format(round(time, 3)))
            else:
                print('(t={}) Agent received confirmation MESSAGE: {}'.format(round(time, 3), message))
                if message.label == 'add':
                    if order.label != 'replace':
                        assert order.refno == message.refno, '>> confirm_order receive non-matching reference numbers'
                        self.add_order(order)
                        print('>> Added ORDER {}'.format(order))
                    else:
                        order = Order(timestamp=message.timestamp,
                                      label='limit',
                                      side=message.side,
                                      price=message.price,
                                      shares=message.shares,
                                      uid=message.uid)
                        order.refno = message.refno
                        self.add_order(order)
                        print('>> Added ORDER {}'.format(order))
                elif message.label == 'execute':
                    if message.uid == self.uid:
                        match = self.orders[message.side]
                        assert message.refno == match.refno, "confirm_order received own-execute with non-matching reno"
                        match.shares -= message.shares
                        assert match.shares >= 0, "Match shares less than zero!"
                        print('>> Updated ORDER {}'.format(match))
                        if match.shares == 0:
                            self.cancel_order(match)
                            print('>> Deleted the order')
                    else:
                        if order.side == 'bid':
                            self.score -= (message.shares * message.price) - (message.shares * self.take_fee)
                            self.inventory += message.shares
                        elif order.side == 'ask':
                            self.score += (message.shares * message.price) - (message.shares * self.take_fee)
                            self.inventory -= message.shares
                        self.data[message.timestamp] = {'score': self.score, 'inventory': self.inventory}
                elif message.label == 'delete':
                    assert order.refno == message.refno, ">> confirm_order receive non-matching refno"
                    match = self.orders[message.side]
                    assert match.refno == message.refno, ">> confirm_order received delete with non-matching refno"
                    self.cancel_order(match)
                    print('>> Deleted ORDER {}'.format(match))

    def cancel_order(self, order):
        assert self.orders[order.side].refno == order.refno, "cancel_order received non-matching refno"
        self.orders[order.side] = None

    def add_order(self, order):
        assert self.orders[order.side] is None, "Agent already has a {} order".format(order.side)
        self.orders[order.side] = order.copy()

class Environment():

    def __init__(self, model, **kwargs):
        self.model = model
        self.T = 60
        self.t = 0
        self.dt = 0.001
        self.N = 30
        self.cursor = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.book = OrderBook()
        self.display = Display(T=self.T)

    def reset(self):

        self.t = 0
        self.cursor = 0
        self.book = OrderBook()

        n = 0
        while n < self.N:
            order = random_limit_order()
            if self.verbose:
                print('{}'.format(order))
            self.book.update(order)
            n += 1

        _, nodes = self.random_events()
        print('Initializing book from {} events'.format(len(nodes)))
        for node in nodes:
            order = self.to_order(node)
            print('>> Updating book for ORDER ({})'.format(order))
            messages = self.book.update(order)

        self.generate_events()
        return self.t, self.book.shares(), [], False

    def random_events(self):
        times, nodes = self.model.generate_data(self.T)
        return times, nodes

    def generate_events(self):
        times, nodes = self.model.generate_data(self.T)
        self.times, self.nodes = times.tolist(), nodes.tolist()
        self.originals = {'times': self.times.copy(), 'nodes': self.nodes.copy()}
        self.actions = {'times': [], 'nodes': []}
        self.generated = {'times': [], 'nodes': []}
        print('Simulator generated {} events'.format(len(self.originals['times'])))

    def generate_children(self, event):
        """
            Using the model (optionally provide an initial event).

            Args:
                t_max: (int) maximum time of simulation.
                event: type and time of an order book event.
        """

        assert event is not None, "Parent event is None"
        return self.model.generate_data_from_parent(self.T, event)

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

    def step(self, action):
        """
            The environment responds to the agents order/action and moves forward one unit
            of time.

            action: Order from the agent.
        """

        # Find events that occurred in the last dt seconds (NOTE: excludes agent's actions)
        events = []
        while self.times[self.cursor] < self.t:
            if self.cursor < len(self.times) - 1:
                events.append((self.times[self.cursor], self.nodes[self.cursor]))
                self.cursor += 1
            else:
                break
        if len(events) > 0:
            print('Found {} EVENTs in interval ({}, {}): {}'.format(len(events), round(self.t - self.dt, 3), round(self.t, 3), events))

        # Update book
        feed = []
        for time, node in events:
            order = self.to_order(node)
            print('>> Updating book for ORDER ({})'.format(order))
            messages = self.book.update(order)
            if messages is not None:
                feed.extend(messages)

        # Execute action; generate new events
        confirm = []
        if action is not None:
            if action.label == 'replace':
                print('(t={}) Agent submitted ORDER {}'.format(round(self.t, 3), action))
                cancel_order, limit_order = self.book.split_order(action)
                event = self.to_event(cancel_order)
                if event is not None:
                    print('>> Updating book ...'.format(cancel_order))
                    messages = self.book.update(cancel_order)
                    if messages is not None:
                        confirm.extend(messages)
                        self.cursor += 1  # skip this event next iteration
                        self.update_events(self.generate_children((self.t, event)))
                    event = self.to_event(limit_order)
                    if event is not None:
                        print('>> Updating book ...'.format(limit_order))
                        messages = self.book.update(limit_order)
                        if messages is not None:
                            confirm.extend(messages)
                            self.cursor += 1  # skip this event next iteration
                            self.update_events(self.generate_children((self.t, event)))
            else:
                print('(t={}) Agent submitted ORDER {}'.format(round(self.t, 3), action))
                event = self.to_event(action)
                if event is not None:
                    # print('(t={}) Agent chose EVENT {}'.format(round(self.t, 3), event))
                    print('>> Updating book ...'.format(action))
                    messages = self.book.update(action)
                    if messages is not None:
                        confirm.extend(messages)
                        self.cursor += 1  # skip this event next iteration
                        self.update_events(self.generate_children((self.t, event)))

        # Update time; check for end of simulation
        self.t += self.dt
        if round(self.t, 3) == self.T:
            done = True
        else:
            done = False
        return self.t, self.book.shares(), events, confirm, feed, done

    def to_order(self, node):
        """Convert an event to an order."""

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
                elif level == 0:
                    if side == 'bid':
                        price = self.book.prices()[side][0]
                    elif side == 'ask':
                        price = self.book.prices()[side][0]
                if level == 1:
                    if side == 'bid':
                        price = self.book.prices()[side][0] - 1
                    elif side == 'ask':
                        price = self.book.prices()[side][0] + 1
                shares = np.random.choice([100, 200, 300])
                order = Order(self.t, label, side, price, shares)
        elif label == 'cancel':
            if level + 1 > len(self.book.prices()[side]):
                print('Failed to generate the order because the specified level is greater than the number of available prices; returning None')
                return None
            elif len(self.book.prices()[side]) == 0:
                print('Failed generate the order because the book is empty; returning None')
                return None
            else:
                price = self.book.prices()[side][level]
            if side == 'bid':
                orders = [o for o in self.book.bids[price] if o.uid is None]
            elif side == 'ask':
                orders = [o for o in self.book.asks[price] if o.uid is None]
            if orders != []:
                order = np.random.choice(orders)
                order.label = 'cancel'
                order.timestamp = self.t
            else:
                print('No orders available to cancel at level {}; returning None'.format(level))
                return None
        elif label == 'market':
            if side == 'bid':
                price = self.book.get_best_ask_price()
                if price is not None:
                    shares = min(self.book.shares()['ask'][price], 100)
                else:
                    shares = 100
            elif side == 'ask':
                price = self.book.get_best_bid_price()
                if price is not None:
                    shares = min(self.book.shares()['bid'][price], 100)
                else:
                    shares = 100
            order = Order(self.t, label, side, None, shares, None)
        return order

    def to_event(self, order):
        """Convert an order into an event."""

        if order.label == 'limit':
            if order.side == 'bid':
                bid = self.book.get_best_bid_price()
                if bid is not None:
                    if order.price == bid + 1:
                        return 0
                    elif order.price == bid:
                        return 1
                    elif order.price == bid - 1:
                        return 2
            elif order.side == 'ask':
                ask = self.book.get_best_ask_price()
                if ask is not None:
                    if order.price == ask - 1:
                        return 3
                    elif order.price == ask:
                        return 4
                    elif order.price == ask + 1:
                        return 5
        elif order.label == 'cancel':
            if order.side == 'bid':
                bid = self.book.get_best_bid_price()
                if bid is not None:
                    if order.price == bid:
                        return 6
                    elif order.price == bid - 1:
                        return 7
            elif order.side == 'ask':
                ask = self.book.get_best_ask_price()
                if ask is not None:
                    if order.price == ask:
                        return 8
                    elif order.price == ask + 1:
                        return 9
        elif order.label == 'market':
            if order.side == 'bid':
                return 10
            elif order.side == 'ask':
                return 11
        elif order.label == 'replace':
            return self.to_event(Order(order.timestamp,
                                       'add',
                                       order.side,
                                       order.newPrice,
                                       order.shares,
                                       order.uid))  # NOTE: model doesn't include replace orders yet
        print("Order did not match any event; returning None")
        return None

def plot(env, agent, pause=False):
    env.display.plot_events(env.t, (env.originals, env.actions, env.generated))
    env.display.plot_book((env.book.bids, env.book.asks))
    env.display.plot_score(agent.data)
    env.display.plot_inventory(agent.data)
    env.display.plot_orders(agent.orders)
    env.display.draw(pause)

# Test (Agent)
# N = 12
# lambda0 = 0.05 * np.ones(N)
# W = 0.1 * np.eye(N)
# mu = -1.0 * np.ones((N,N))
# tau = 1.0 * np.ones((N,N))
# model = NetworkPoisson(N=N, dt_max=1.0, params={'lamb': lambda0, 'weights': W, 'mu': mu, 'tau': tau})
# agent = Agent(verbose=False)
# env = Environment(model, verbose=False)
# time, book, events, done = env.reset()
# plot(env, agent, pause=True)
# while not done:
#     action = agent.choose_action(time, book, events)
#     time, book, events, confirm, messages, done = env.step(action)
#     result = agent.update(time, action, confirm, messages, done)
#     # if (action is not None) or (events != []):
#         # plot(env, agent, pause=True)
# plot(env, agent)

# Test (Market Maker)
N = 12
lambda0 = 0.05 * np.ones(N)
W = 0.1 * np.eye(N)
mu = -1.0 * np.ones((N,N))
tau = 1.0 * np.ones((N,N))
model = NetworkPoisson(N=N, dt_max=1.0, params={'lamb': lambda0, 'weights': W, 'mu': mu, 'tau': tau})
agent = MarketMaker(verbose=False)
env = Environment(model, verbose=False)
time, book, events, done = env.reset()
# plot(env, agent, pause=True)
while not done:
    action = agent.choose_action(time, book, events)
    time, book, events, confirm, messages, done = env.step(action)
    result = agent.update(time, action, confirm, messages, done)
    # if (action is not None) or (events != []):
    #     plot(env, agent, pause=True)
plot(env, agent)

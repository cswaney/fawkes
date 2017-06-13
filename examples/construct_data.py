import h5py as h5
import pandas as pd
import numpy as np
import time
import os

"""
This script constructs event data that we can feed to the continuous-time Gibbs sampler.
"""

import_path = "/Volumes/datasets/ITCH/hdf5/"
export_path = "/Volumes/datasets/ITCH/events/"
dates = []
for d in [d.lstrip('itch-').rstrip('.hdf5') for d in os.listdir(import_path)]:
    if d[:2] == '07':
        dates.append(d)
names = ['AAPL', 'GOOG', 'PFE', 'MMM', 'CR', 'LSTR', 'NSR']  # Apple, Google, Pfizer, 3M, Crane, Landstar, Neustar

def import_data(name, date, mkt_open=34200, mkt_close=57600):
    msg_columns = ['sec', 'nano', 'type', 'event', 'side', 'price', 'shares', 'refno', 'newrefno']
    book_columns = ['sec', 'nano']
    num_cols = 10
    book_columns += ['bidprc_{}'.format(i + 1) for i in range(num_cols)]
    book_columns += ['askprc_{}'.format(i + 1) for i in range(num_cols)]
    book_columns += ['bidvol_{}'.format(i + 1) for i in range(num_cols)]
    book_columns += ['askvol_{}'.format(i + 1) for i in range(num_cols)]
    with h5.File(import_path + 'itch-{}.hdf5'.format(date), 'r') as hdf:
        try:
            messages = hdf['/messages/{}'.format(name)][:]
            books = hdf['/orderbooks/{}'.format(name)][:]
            messages = pd.DataFrame(messages, columns=msg_columns)
            books = pd.DataFrame(books, columns=book_columns)
            books = books.loc[:, ['sec', 'nano', 'bidprc_1', 'askprc_1', 'bidvol_1', 'askvol_1']]
            messages = messages[ (messages['sec'] >= mkt_open) & (messages['sec'] < mkt_close)]
            books = books[ (books['sec'] >= mkt_open) & (books['sec'] < mkt_close)]
            return messages, books
        except:
            print("Unable to find data. Skipping.")
            return None

def get_events(messages, books):

    messages = messages.reset_index(drop=True)
    books = books.reset_index(drop=True)
    assert len(messages) == len(books), "ERROR: number of message and book observations differ."

    def find_level(price, best, side):
        if side == 1:  # bid
            if price > best:
                return -1
            elif price == best:
                return 0
            elif price == best - 100:
                return 1
            else:
                return None
        elif side == -1:  # ask
            if price < best:
                return -1
            elif price == best:
                return 0
            elif price == best + 100:
                return 1
            else:
                return None

    def find_node(mtype, level, side):
        if mtype == 2:  # add
            if side == 1:
                if level == -1:
                    return 0
                elif level == 0:
                    return 1
                elif level == 1:
                    return 2
            if side == -1:
                if level == -1:
                    return 3
                elif level == 0:
                    return 4
                elif level == 1:
                    return 5
        elif mtype == 4:  # delete
            if side == 1:
                if level == 0:
                    return 6
                elif level == 1:
                    return 7
            elif side == -1:
                if level == 0:
                    return 8
                elif level == 1:
                    return 9
        elif mtype == 5:  # execute
            if level == 0:
                if side == 1:
                    return 10
                if side == -1:
                    return 11
            elif level == 1:
                if side == 1:
                    return 12
                if side == -1:
                    return 13

            if side == 1:
                if level == 0:
                    return 10
                elif level == 1:
                    return 11
            elif side == -1:
                if level == 0:
                    return 12
                elif level == 1:
                    return 13

    def node_properties(node):
        properties = [('add', 'bid', -1),
                      ('add', 'bid', 0),
                      ('add', 'bid', 1),
                      ('add', 'ask', -1),
                      ('add', 'ask', 0),
                      ('add', 'ask', 1),

                      ('del', 'bid', 0),
                      ('del', 'bid', 1),
                      ('del', 'ask', 0),
                      ('del', 'ask', 1),

                      ('exe', 'bid', 0),
                      ('exe', 'ask', 0),
                      ('exe', 'bid', 1),
                      ('exe', 'ask', 1)]
        return properties[node]

    data = []
    start = time.time()
    for i in range(1, len(messages)):
        if i % int(len(messages) / 50) == 0:
            print("+", end='', flush=True)
        message = messages.loc[i, :]
        timestamp = message.sec + message.nano / 10 ** 9
        price = message.price
        shares = message.shares
        side = message.side
        mtype = message.type
        if mtype == 5:  # cancellation
            if side == 1:
                bid = books.loc[i - 1, 'bidprc_1']
                vol = books.loc[i - 1, 'bidvol_1']
                if price == bid and shares == vol:
                    level = 1
                elif price == bid:
                    level = 0
                else:
                    level = None
            elif side == -1:
                ask = books.loc[i - 1, 'askprc_1']
                vol = books.loc[i - 1, 'askvol_1']
                if price == ask and shares == vol:
                    level = 1
                elif price == ask:
                    level = 0
                else:
                    level = None
            if level is not None:
                node = find_node(mtype, level, side)
                data.append([timestamp, node])
        if mtype in (2, 4):  # add, execute
            if side == 1:
                bid = books.loc[i - 1, 'bidprc_1']
                level = find_level(price, bid, side)
            elif side == -1:
                ask = books.loc[i - 1, 'askprc_1']
                level = find_level(price, ask, side)
            if level is not None:
                node = find_node(mtype, level, side)
                data.append([timestamp, node])
    print(" (messages: {}, elapsed time: {:.2f} s)".format(len(messages), time.time() - start))
    return pd.DataFrame(np.array(data), columns=['timestamp', 'type'])

def export_data(events, date, name):
    """
        Overwrite the datasets if they already exists.
    """
    with h5.File(export_path + 'events.hdf5', 'a') as hdf:
        try:
            hdf.create_dataset(name='{}/{}'.format(name, date), data=events)
        except:
            print("Dataset already exists: deleting and creating a new dataset.")
            del hdf['{}/{}'.format(name, date)]
            hdf.create_dataset(name='{}/{}'.format(name, date), data=events)

for n in names:
    for d in dates:
        print('{}, {}: '.format(n, d), end='')
        data = import_data(name=n, date=d)
        if data is not None:
            M, B = data
            E = get_events(messages=M, books=B)
            _ = export_data(events=E, date=d, name=n)

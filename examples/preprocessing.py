import h5py as h5
import pandas as pd
import numpy as np
import time
import os

"""This script constructs event data that we can feed to the continuous-time Gibbs sampler."""

root = '/Volumes/datasets/ITCH/'
dates = [date for date in os.listdir('{}/csv/'.format(root)) if date != '.DS_Store']
names = [name.lstrip(' ') for name in pd.read_csv('{}/SP500.txt'.format(root))['Symbol']]
names.sort()

def import_data(name, date, mkt_open=34200, mkt_close=57600, num_cols=5):
    msg_columns = ['sec', 'nano', 'type', 'event', 'side', 'price', 'shares', 'refno', 'newrefno']
    book_columns = ['sec', 'nano']
    book_columns += ['bidprc{}'.format(i) for i in range(num_cols)]
    book_columns += ['askprc{}'.format(i) for i in range(num_cols)]
    book_columns += ['bidvol{}'.format(i) for i in range(num_cols)]
    book_columns += ['askvol{}'.format(i) for i in range(num_cols)]
    try:
        messages = pd.read_csv('{}/csv/{}/messages/messages_{}.txt'.format(root, date, name))
        books = pd.read_csv('{}/csv/{}/books/books_{}.txt'.format(root, date, name))
    except:
        print('Unable to find data; skipping.')
        return None
    messages = messages[ (messages['sec'] >= mkt_open) & (messages['sec'] < mkt_close)].iloc[1:,:].reset_index(drop=True)
    books = books[ (books['sec'] >= mkt_open) & (books['sec'] < mkt_close)].iloc[:-1,:].reset_index(drop=True)
    return messages.reset_index(drop=True), books.reset_index(drop=True)

def get_events(messages, books):
    start = time.time()
    messages['bidprc'] = books['bidprc0']
    messages['askprc'] = books['askprc0']
    messages['bidvol'] = books['bidvol0']
    messages['askvol'] = books['askvol0']
    messages['time'] = messages['sec'] + messages['nano'] / 10**9
    messages['event'] = -1
    messages.loc[(messages['type'] == 'A') & (messages['price'] == messages['bidprc']) & (messages['side'] == 'B'), 'event'] = 0
    messages.loc[(messages['type'] == 'A') & (messages['price'] > messages['bidprc']) & (messages['side'] == 'B'), 'event'] = 1
    messages.loc[(messages['type'] == 'A') & (messages['price'] == messages['askprc']) & (messages['side'] == 'S'), 'event'] = 2
    messages.loc[(messages['type'] == 'A') & (messages['price'] < messages['askprc']) & (messages['side'] == 'S'), 'event'] = 3
    messages.loc[(messages['type'] == 'E') & (messages['price'] == messages['bidprc']) & (messages['side'] == 'B') & (-messages['shares'] < messages['bidvol']), 'event'] = 4
    messages.loc[(messages['type'] == 'E') & (messages['price'] == messages['bidprc']) & (messages['side'] == 'B') & (-messages['shares'] == messages['bidvol']), 'event'] = 5
    messages.loc[(messages['type'] == 'E') & (messages['price'] == messages['askprc']) & (messages['side'] == 'S') & (-messages['shares'] < messages['askvol']), 'event'] = 6
    messages.loc[(messages['type'] == 'E') & (messages['price'] == messages['askprc']) & (messages['side'] == 'S') & (-messages['shares'] == messages['askvol']), 'event'] = 7
    messages.loc[(messages['type'] == 'D') & (messages['price'] == messages['bidprc']) & (messages['side'] == 'B') & (-messages['shares'] < messages['bidvol']), 'event'] = 8
    messages.loc[(messages['type'] == 'D') & (messages['price'] == messages['bidprc']) & (messages['side'] == 'B') & (-messages['shares'] == messages['bidvol']), 'event'] = 9
    messages.loc[(messages['type'] == 'D') & (messages['price'] == messages['askprc']) & (messages['side'] == 'S') & (-messages['shares'] < messages['askvol']), 'event'] = 10
    messages.loc[(messages['type'] == 'D') & (messages['price'] == messages['askprc']) & (messages['side'] == 'S') & (-messages['shares'] == messages['askvol']), 'event'] = 11
    events = messages.loc[messages['event'] > -1, ('time', 'event')].reset_index(drop=True)
    if len(events) > 0:
        return events
    else:
        return None

def export_data(events, date, name, path):
    """Overwrite the datasets if they already exists."""
    with h5.File('{}/events/{}'.format(root, path), 'a') as hdf:
        try:
            hdf.create_dataset(name='{}/{}'.format(name, date), data=events)
        except:
            # print("Dataset already exists: deleting and creating a new dataset.")
            del hdf['{}/{}'.format(name, date)]
            hdf.create_dataset(name='{}/{}'.format(name, date), data=events)

path = 'large2007.hdf5'
for n in names:
    for d in dates:
        start = time.time()
        print('{}, {}:'.format(n, d), end='')
        data = import_data(name=n, date=d)
        if data is not None:
            M, B = data
            E = get_events(messages=M, books=B)
            if E is not None:
                _ = export_data(events=E, date=d, name=n, path=path)
                print(" (events: {}, elapsed time: {:.2f} s)".format(E.shape[0], time.time() - start))
            else:
                print(" (events: {}, elapsed time: {:.2f} s)".format(0, time.time() - start))

import h5py as h5
import pandas as pd
import numpy as np
import time
import os

"""This script constructs a table of daily statistics from ITCH data."""

root = '/Volumes/datasets/ITCH/'
trades = pd.read_csv('/Volumes/datasets/ITCH/stats/trades.txt', index_col=0)
hidden = pd.read_csv('/Volumes/datasets/ITCH/stats/hidden_trades.txt', index_col=0)
dates = [date for date in os.listdir('{}/csv/'.format(root)) if date != '.DS_Store']
names = [name.lstrip(' ') for name in pd.read_csv('{}/SP500.txt'.format(root))['Symbol']]
names.sort()

def import_data(name, date, mkt_open=34200, mkt_close=57600):
    try:
        df = pd.read_csv('{}/csv/{}/books/books_{}.txt'.format(root, date, name))
    except:
        print('Unable to find data; skipping.')
        return None
    df = df[ (df['sec'] >= mkt_open) & (df['sec'] < mkt_close)].iloc[:-1,:].reset_index(drop=True)
    return df.reset_index(drop=True)

date = '072413'
spread = []
prices = []
volumes = []
for name in names:
    print('name={}, date={}'.format(name, date))
    df = import_data(name, date, mkt_open=34200 + 3600, mkt_close=57600 - 3600)
    if df is not None:
        # Spread
        df['spread'] = df['askprc0'] - df['bidprc0']
        sprd_median = np.median(df['spread'])
        sprd_mean = np.mean(df['spread'])
        sprd_min = round(np.min(df['spread']), 2)
        sprd_max = round(np.max(df['spread']), 2)
        spreads.append([name, sprd_median, sprd_mean, sprd_min, sprd_max])

        # Prices & Volatility
        df['volume'] = df['askvol0'] + df['bidvol0']
        df['midprc'] = (df['askprc0'] + df['bidprc0']) / 2
        df['vwaprc'] = (df['askvol0'] * df['askprc0'] + df['bidvol0'] * df['bidprc0']) / df['volume']
        prc_median = np.median(df['midprc'])
        prc_mean = np.mean(df['midprc'])
        prc_min = round(np.min(df['midprc']), 2)
        prc_max = round(np.max(df['midprc']), 2)
        prc_vol = np.std(df['midprc'])
        prices.append([name, prc_median, prc_mean, prc_min, prc_max, prc_vol])

        # Trade Volume
        t_volume = -trades[(trades['name'] == name) & (trades['date'] == int(date))]['shares'].sum()
        h_volume = hidden[(hidden['name'] == name) & (hidden['date'] == int(date))]['shares'].sum()
        volumes.append([name, t_volume, h_volume])

spreads = pd.DataFrame(spreads, columns=('name', 'median', 'mean', 'min', 'max'))
prices = pd.DataFrame(prices, columns=('name', 'median', 'mean', 'min', 'max', 'vol'))
volumes = pd.DataFrame(volumes, columns=['name', 'trade', 'hidden'])
df = pd.merge(spreads, prices, on='name', suffixes=['_sprd', '_prc'])
df = pd.merge(df, volumes, on='name')
df['date'] = date
df.to_csv('/Users/colinswaney/Desktop/itch_{}.txt'.format(date))

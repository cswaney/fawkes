import h5py as h5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mkt_open = 34200
mkt_close = 57600
t0 = mkt_open + 3600
tN = mkt_close - 3600
msg_types = {'add': 2, 'cancel': 3, 'delete': 4, 'execute': 5, 'execute_w/price': 6}
msg_sides = {'bid': 1, 'ask': -1}
data_path = '/Volumes/datasets/ITCH/hdf5/'  # itch-MMDDYY.hdf5
img_path = '/Users/colinswaney/Desktop/Figures/basic/{}'
sample_path = '/Users/colinswaney/Desktop/continuous_dt_max=5.hdf5'
name = 'GOOG'
with h5.File(sample_path, 'r') as hdf:
    dates = [date for date in hdf[name].keys()]

def import_data(path, name, date):

    with h5.File(path + 'itch-{}.hdf5'.format(date), 'r') as hdf:
        msg_data = hdf['messages/{}'.format(name)][:]
        book_data = hdf['orderbooks/{}'.format(name)][:]

    columns = ['sec', 'nano', 'type', 'event', 'side', 'price', 'shares', 'refno', 'newrefno']
    messages = pd.DataFrame(msg_data, columns=columns)

    columns = ['sec', 'nano']
    _, num_cols = book_data.shape
    columns += ['bidprc_{}'.format(i + 1) for i in range(int((num_cols - 2) / 4))]
    columns += ['askprc_{}'.format(i + 1) for i in range(int((num_cols - 2) / 4))]
    columns += ['bidvol_{}'.format(i + 1) for i in range(int((num_cols - 2) / 4))]
    columns += ['askvol_{}'.format(i + 1) for i in range(int((num_cols - 2) / 4))]
    books = pd.DataFrame(book_data, columns=columns)

    return messages, books

def import_all(path, name, dates):
    messages = []
    books = []
    for date in dates:
        msgs, bks = import_data(data_path, name, date)
        print('date={}, messages={}'.format(date, msgs.shape[0]))
        msgs = msgs[ (msgs['sec'] > t0) & (msgs['sec'] < tN)]
        bks = bks[ (bks['sec'] > t0) & (bks['sec'] < tN)]
        messages.append(msgs)
        books.append(bks)
    messages = pd.concat(messages, axis=0)
    books = pd.concat(books, axis=0)
    return messages, books

def create_summary(messages, books, decimals=2):

    def acc(d):
        return "{:.{}f}".format(d, decimals)

    """Creates Table 4.1 for a single name."""
    columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    index=['Price', 'Spread', 'Imbalance', 'Adds', 'Deletes', 'Executes']
    table = pd.DataFrame(np.zeros((6, 8)), columns=columns, index=index)
    books['midprice'] = (books['bidprc_1'] + books['askprc_1']) / 20000
    table.loc['Price'] = books['midprice'].describe()
    books['spread'] = (books['askprc_1'] - books['bidprc_1']) / 10000
    table.loc['Spread'] = books['spread'].describe()
    books['bidvol'] = books.loc[:, 'bidvol_1':'bidvol_10'].sum(axis=1)
    books['askvol'] = books.loc[:, 'askvol_1':'askvol_10'].sum(axis=1)
    books['imbalance'] = books['bidvol'] - books['askvol']
    table.loc['Imbalance'] = books['imbalance'].describe()
    adds = messages[ messages['type'] == 2]
    deletes = messages[ messages['type'] == 4]
    executes = messages[ messages['type'] == 5]
    table.loc['Adds'] = adds['shares'].describe()
    table.loc['Deletes'] = -deletes['shares'].describe()
    table.loc['Executes'] = -executes['shares'].describe()
    columns=['N (1000\'s)', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    return table.to_latex(formatters=[acc] * table.shape[1])

messages, books = import_all(data_path, name, dates)
create_summary(message, books)

def plot_series(messages, books):
    """Create figure summarizing daily order book activity."""

    messages = messages[ (messages['sec'] > mkt_open) & (messages['sec'] < mkt_close) ]
    books = books[ (books['sec'] >= mkt_open) & (books['sec'] < mkt_close) ]
    messages = messages.reset_index(drop=True)
    books = books.reset_index(drop=True)
    books['price'] = (books['bidprc_1'] + books['askprc_1']) / 2 / 10000
    books['min'] = np.floor(books['sec'] / 60).astype('int')
    messages['min'] = np.floor(messages['sec'] / 60).astype('int')
    adds = messages[ messages['type'] == msg_types['add'] ]
    executes = messages[ messages['type'] == msg_types['execute'] ]
    means = books.groupby('min').mean()
    nulls = pd.DataFrame(np.zeros(len(means)), columns=['null'], index=means.index)
    grouped = executes.groupby(by=['min', 'side'])
    trades = grouped['shares'].apply(lambda x: np.abs(x).sum()).reset_index(level=1)
    buys = trades[ trades['side'] == 1 ]
    sells = trades[ trades['side'] == -1 ]
    buys = pd.merge(left=pd.DataFrame(nulls),
                    right=pd.DataFrame(buys),
                    left_index=True,
                    right_index=True,
                    how='outer').drop('null', axis=1)
    buys[ buys['shares'].isnull() ] = 0
    sells = pd.merge(left=pd.DataFrame(nulls),
                     right=pd.DataFrame(sells),
                     left_index=True,
                     right_index=True,
                     how='outer')
    sells[ sells['shares'].isnull() ] = 0
    hours = pd.date_range(start='01-02-2013 09:30:00', end='01-02-2013 16:00:00', freq='H')

    def plot_price():
        ax = plt.subplot(3, 1, 1)
        price, = plt.plot(means['price'], color='k', linewidth=1, linestyle='-')
        ax.yaxis.set_ticks_position('right')
        plt.yticks(fontsize=8)
        plt.ylabel('Price', fontsize=8)
        plt.ylim([means['price'].min() - 0.50, means['price'].max() + 0.50])
        plt.xticks(np.arange(mkt_open / 60, mkt_close / 60, 3600 / 60), [h.strftime('%H:%M:%S') for h in hours])
        plt.xticks(rotation=45)
        plt.xlim([mkt_open / 60, mkt_close / 60])
        plt.grid(linestyle='--', linewidth=0.25)
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            left='off',
            right='off',
            labelbottom='off')

    def plot_liquidity():
        ax = plt.subplot(3, 1, 2)
        bids, = plt.plot(means['bidvol_1'], linewidth=1, color='C0')
        plt.fill_between(means.index, 0, means['bidvol_1'], color='C0')
        asks, = plt.plot(-means['askvol_1'], linewidth=1, color='C3')
        plt.fill_between(means.index, 0, -means['askvol_1'], color='C3')
        plt.ylabel('Liquidity', fontsize=8)
        ax.yaxis.set_ticks_position('right')
        plt.yticks(fontsize=8)
        plt.xticks(np.arange(mkt_open / 60, mkt_close / 60, 3600 / 60), [h.strftime('%H:%M:%S') for h in hours])
        plt.xticks(rotation=45)
        plt.xlim([mkt_open / 60, mkt_close / 60])
        plt.grid(linestyle='--', linewidth=0.25)
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            left='off',
            right='off',
            labelbottom='off')

    def plot_trades():
        ax = plt.subplot(3, 1, 3)
        b, = plt.plot(buys['shares'], linewidth=1, color='C0')
        plt.fill_between(buys.index, 0, buys['shares'], color='C0')
        s, = plt.plot(-sells['shares'], linewidth=1, color='C3')
        plt.fill_between(sells.index, 0, -sells['shares'], color='C3')
        plt.ylabel('Trades', fontsize=8)
        ax.yaxis.set_ticks_position('right')
        plt.yticks(fontsize=8)
        plt.xticks(np.arange(mkt_open / 60, mkt_close / 60, 3600 / 60), [h.strftime('%H:%M:%S') for h in hours])
        plt.xticks(rotation=45, fontsize=8)
        plt.xlim([mkt_open / 60, mkt_close / 60])
        plt.grid(linestyle='--', linewidth=0.25)

    plot_price()
    plot_liquidity()
    plot_trades()

    plt.tight_layout()
    plt.show()
    plt.clf()

messages, books = import_data(data_path, name, date)
time_series(messages, books)

# Message Frequency (Redux)
# def frequency_plot():
#     ax = plt.subplot(3,1,1)
#     counts =  adds.groupby(['min', 'side']).size().reset_index(level=1)
#     bids = counts[ counts['side'] == 1].rename(columns={0: 'total'})
#     asks = counts[ counts['side'] == -1].rename(columns={0: 'total'})
#     plt.fill_between(bids.index, 0 , bids['total'])
#     plt.fill_between(asks.index, 0, -asks['total'])
#     plt.xticks(np.arange(mkt_open / 60, mkt_close / 60, 3600 / 60), [])
#     plt.xlim([mkt_open / 60, mkt_close / 60])
#     ax.yaxis.set_ticks_position("right")
#     # ax.yaxis.set_label_position("right")
#     plt.yticks(np.arange(-1000, 1500, 500), np.abs(np.arange(-1000, 1500, 500)), fontsize=8)
#     plt.tick_params(
#         axis='both',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom='off',      # ticks along the bottom edge are off
#         top='off',         # ticks along the top edge are off
#         left='off',
#         right='off',
#         labelbottom='off') # labels along the bottom edge are off
#     plt.ylim([-1000, 1000])
#     plt.ylabel('Add messages', fontsize=8)
#     plt.grid(linestyle='--', linewidth=0.25)
#
#     ax = plt.subplot(3,1,3)
#     counts =  executes.groupby(['min', 'side']).size().reset_index(level=1)
#     bids = counts[ counts['side'] == 1].rename(columns={0: 'total'})
#     asks = counts[ counts['side'] == -1].rename(columns={0: 'total'})
#     plt.fill_between(bids.index, 0 , bids['total'])
#     plt.fill_between(asks.index, 0, -asks['total'])
#     plt.xticks(np.arange(mkt_open / 60, mkt_close / 60, 3600 / 60),
#                [h.strftime('%H:%M') for h in hours], rotation=45, fontsize=8)
#     plt.xlim([mkt_open / 60, mkt_close / 60])
#     ax.yaxis.set_ticks_position("right")
#     # ax.yaxis.set_label_position("right")
#     plt.yticks(np.arange(-100, 150, 50), np.abs(np.arange(-100, 150, 50)), fontsize=8)
#     plt.tick_params(
#         axis='both',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom='off',      # ticks along the bottom edge are off
#         top='off',         # ticks along the top edge are off
#         left='off',
#         right='off')
#     plt.ylim([-100, 100])
#     plt.ylabel('Execute messages', fontsize=8)
#     plt.grid(linestyle='--', linewidth=0.25)
#     plt.tight_layout()
#     plt.savefig('{}/messages_frequency.pdf'.format(img_path))
#     plt.clf()

def plot_stems(messages, books, xmax):
    """Stem plots of order volume by order type."""

    messages = messages[ (messages['sec'] > mkt_open) & (messages['sec'] < mkt_close) ]
    books = books[ (books['sec'] >= mkt_open) & (books['sec'] < mkt_close) ]
    messages = messages.reset_index(drop=True)
    books = books.reset_index(drop=True)
    books['price'] = (books['bidprc_1'] + books['askprc_1']) / 2 / 10000
    books['min'] = np.floor(books['sec'] / 60).astype('int')
    messages['min'] = np.floor(messages['sec'] / 60).astype('int')
    adds = messages[ messages['type'] == msg_types['add'] ]
    deletes = messages[ messages['type'] == msg_types['delete'] ]
    executes = messages[ messages['type'] == msg_types['execute'] ]

    def plot_adds():
        ax = plt.subplot(3,1,1)
        counts = pd.value_counts(adds['shares'])
        N = counts.sum()
        counts = counts[ counts > adds.shape[0] * 0.01 ]  # at least 1 % of observations
        plt.stem(counts.index, counts / N, markerfmt='C0.', basefmt='None', linefmt='C0-')
        plt.xlim(xmin=-20, xmax=xmax)
        plt.ylim([0, 0.5])
        ax.yaxis.set_ticks_position('right')
        plt.yticks([0, 0.25, 0.50], fontsize=8)
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            left='off',
            right='off',
            labelbottom='off')
        plt.grid(linestyle='--', linewidth=0.25)
        plt.ylabel('Add messages', fontsize=8)
        print(counts / adds.shape[0])

    def plot_deletes():
        ax = plt.subplot(3,1,2)
        counts = pd.value_counts(-deletes['shares'])
        N = counts.sum()
        counts = counts[ counts > deletes.shape[0] * 0.01 ]
        plt.stem(counts.index, counts / N, markerfmt='C0.', basefmt='None', linefmt='C0-')
        plt.xlim(xmin=-20, xmax=xmax)
        plt.ylim([0, 0.5])
        ax.yaxis.set_ticks_position('right')
        plt.yticks([0, 0.25, 0.50], fontsize=8)
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            left='off',
            right='off',
            labelbottom='off')
        plt.grid(linestyle='--', linewidth=0.25)
        plt.ylabel('Delete messages', fontsize=8)
        print(counts / deletes.shape[0])

    def plot_executes():
        ax = plt.subplot(3,1,3)
        counts = pd.value_counts(-executes['shares'])
        N = counts.sum()
        counts = counts[ counts > executes.shape[0] * 0.01 ]
        plt.stem(counts.index, counts / N, markerfmt='C0.', basefmt='None', linefmt='C0-')
        plt.xlim(xmin=-20, xmax=xmax)
        plt.ylim([0, 0.5])
        ax.yaxis.set_ticks_position('right')
        plt.yticks([0, 0.25, 0.50], fontsize=8)
        plt.xticks(fontsize=8)
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            left='off',
            right='off',
            labelbottom='on')
        plt.grid(linestyle='--', linewidth=0.25)
        plt.ylabel('Execute messages', fontsize=8)
        print(counts / executes.shape[0])

    plot_adds()
    plot_deletes()
    plot_executes()

    plt.tight_layout()
    plt.savefig('/Users/colinswaney/Desktop/stem_plot_{}'.format(name))
    plt.show()
    plt.clf()

messages, books = import_all(data_path, name, dates)
plot_stems(messages, books, xmax=520)

# Time Series
def time_series():
    means = books.groupby('min').mean()
    nulls = pd.DataFrame(np.zeros(len(means)), columns=['null'], index=means.index)
    grouped = executes.groupby(by=['min', 'side'])
    trades = grouped['shares'].apply(lambda x: np.abs(x).sum()).reset_index(level=1)
    buys = trades[ trades['side'] == 1 ]
    sells = trades[ trades['side'] == -1 ]
    buys = pd.merge(left=pd.DataFrame(nulls),
                    right=pd.DataFrame(buys),
                    left_index=True,
                    right_index=True,
                    how='outer').drop('null', axis=1)
    buys[ buys['shares'].isnull() ] = 0
    sells = pd.merge(left=pd.DataFrame(nulls),
                     right=pd.DataFrame(sells),
                     left_index=True,
                     right_index=True,
                     how='outer')
    sells[ sells['shares'].isnull() ] = 0
    hours = pd.date_range(start='01-02-2013 09:30:00', end='01-02-2013 16:00:00', freq='H')

    ax = plt.subplot(3, 1, 1)
    price, = plt.plot(means['price'], color='k', linewidth=1, linestyle='-')
    ax.yaxis.set_ticks_position('right')
    plt.yticks(fontsize=8)
    plt.ylabel('Price', fontsize=8)
    plt.ylim([means['price'].min() - 0.50, means['price'].max() + 0.50])
    plt.xticks(np.arange(mkt_open / 60, mkt_close / 60, 3600 / 60), [h.strftime('%H:%M:%S') for h in hours])
    plt.xticks(rotation=45)
    plt.xlim([mkt_open / 60, mkt_close / 60])
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off')

    ax = plt.subplot(3, 1, 2)
    bids, = plt.plot(means['bidvol_1'], linewidth=1, color='C0')
    plt.fill_between(means.index, 0, means['bidvol_1'], color='C0')
    asks, = plt.plot(-means['askvol_1'], linewidth=1, color='C3')
    plt.fill_between(means.index, 0, -means['askvol_1'], color='C3')
    plt.ylabel('Liquidity', fontsize=8)
    ax.yaxis.set_ticks_position('right')
    plt.yticks(fontsize=8)
    plt.xticks(np.arange(mkt_open / 60, mkt_close / 60, 3600 / 60), [h.strftime('%H:%M:%S') for h in hours])
    plt.xticks(rotation=45)
    plt.xlim([mkt_open / 60, mkt_close / 60])
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off')

    ax = plt.subplot(3, 1, 3)
    b, = plt.plot(buys['shares'], linewidth=1, color='C0')
    plt.fill_between(buys.index, 0, buys['shares'], color='C0')
    s, = plt.plot(-sells['shares'], linewidth=1, color='C3')
    plt.fill_between(sells.index, 0, -sells['shares'], color='C3')
    plt.ylabel('Trades', fontsize=8)
    ax.yaxis.set_ticks_position('right')
    plt.yticks(fontsize=8)
    plt.xticks(np.arange(mkt_open / 60, mkt_close / 60, 3600 / 60), [h.strftime('%H:%M:%S') for h in hours])
    plt.xticks(rotation=45, fontsize=8)
    plt.xlim([mkt_open / 60, mkt_close / 60])
    plt.grid(linestyle='--', linewidth=0.25)

    plt.tight_layout()
    plt.savefig('/Users/colinswaney/Desktop/time_series.pdf')
    plt.show()
    plt.clf()

# Book Shape
def shape_plot():

    messages = []
    books = []
    for date in dates:
        msgs, bks = import_data(data_path + 'itch-{}.hdf5'.format(date), 'GOOG')
        print('date={}, messages={}'.format(date, msgs.shape[0]))
        msgs = msgs[ (msgs['sec'] > t0) & (msgs['sec'] < tN)]
        bks = bks[ (bks['sec'] > t0) & (bks['sec'] < tN)]
        messages.append(msgs)
        books.append(bks)
    messages = pd.concat(messages, axis=0)
    books = pd.concat(books, axis=0)

    columns = ['bidvol_{}'.format(i) for i in range(10, 0, -1)]
    columns += ['askvol_{}'.format(i) for i in range(1, 11, 1)]
    volume = books.loc[:, columns].mean(axis=0)

    columns = ['bidprc_{}'.format(i) for i in range(10, 0, -1)]
    columns += ['askprc_{}'.format(i) for i in range(1, 11, 1)]
    price = books.loc[:, columns].mean(axis=0) / 100

    plt.subplot(2, 1, 1)
    plt.bar(price[:10], volume[:10], color='C0', width=3.2)
    plt.bar(price[10:], volume[10:], color='C3', width=3.2)
    columns = ['Bid {}'.format(i) for i in range(10, 0, -1)]
    columns += ['Ask {}'.format(i) for i in range(1, 11, 1)]
    plt.xticks(price, ['{:.2f}'.format(p / 100) for p in price], rotation=90)
    plt.tick_params(
        axis='both',
        which='both',
        top='off',
        left='off',
        right='off')
    plt.ylabel('', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25, axis='y')

    messages = []
    books = []
    for date in dates:
        msgs, bks = import_data(data_path + 'itch-{}.hdf5'.format(date), 'PFE')
        print('date={}, messages={}'.format(date, msgs.shape[0]))
        msgs = msgs[ (msgs['sec'] > t0) & (msgs['sec'] < tN)]
        bks = bks[ (bks['sec'] > t0) & (bks['sec'] < tN)]
        messages.append(msgs)
        books.append(bks)
    messages = pd.concat(messages, axis=0)
    books = pd.concat(books, axis=0)

    columns = ['bidvol_{}'.format(i) for i in range(10, 0, -1)]
    columns += ['askvol_{}'.format(i) for i in range(1, 11, 1)]
    volume = books.loc[:, columns].mean(axis=0)

    columns = ['bidprc_{}'.format(i) for i in range(10, 0, -1)]
    columns += ['askprc_{}'.format(i) for i in range(1, 11, 1)]
    price = books.loc[:, columns].mean(axis=0) / 100

    plt.subplot(2, 1, 2)
    plt.bar(price[:10], volume[:10], color='C0', width=0.4)
    plt.bar(price[10:], volume[10:], color='C3', width=0.4)
    columns = ['Bid {}'.format(i) for i in range(10, 0, -1)]
    columns += ['Ask {}'.format(i) for i in range(1, 11, 1)]
    plt.xticks(price, ['{:.2f}'.format(p / 100) for p in price], rotation=90)
    plt.tick_params(
        axis='both',
        which='both',
        top='off',
        left='off',
        right='off')
    plt.ylabel('', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25, axis='y')

    plt.tight_layout()
    plt.savefig('/Users/colinswaney/Desktop/book_shape.pdf')
    plt.show()
    plt.clf()

# Import Data
def get_level(price, best, side):
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

def get_node(mtype, level, side):
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

def extract_events(messages, books):
    data = []
    for i in range(1, len(messages)):
        if i % 10000 == 0:
            print("message # {}".format(i))
        message = messages.ix[i,:]
        time = message.sec + message.nano / 10 ** 9
        price = message.price
        shares = message.shares
        side = message.side
        mtype = message.type
        if mtype == 5:
            if side == 1:
                bid = books.ix[i - 1, 'bidprc_1']
                vol = books.ix[i - 1, 'bidvol_1']
                if price == bid and shares == vol:
                    level = 1
                elif price == bid:
                    level = 0
                else:
                    level = None
            elif side == -1:
                ask = books.ix[i - 1, 'askprc_1']
                vol = books.ix[i - 1, 'askvol_1']
                if price == ask and shares == vol:
                    level = 1
                elif price == ask:
                    level = 0
                else:
                    level = None
            if level is not None:
                node = get_node(mtype, level, side)
                if node is None:
                    print(message)
                    print(books.ix[i-1, :])
                    input("level = {}\n".format(level))
                data.append([time, node])
        if mtype in (2, 4):
            if side == 1:
                bid = books.ix[i - 1, 'bidprc_1']
                level = get_level(price, bid, side)
            elif side == -1:
                ask = books.ix[i - 1, 'askprc_1']
                level = get_level(price, ask, side)
            if level is not None:
                node = get_node(mtype, level, side)
                if node is None:
                    print(message)
                    print(books.ix[i-1, :])
                    input("level = {}\n".format(level))
                data.append([time, node])
    events = pd.DataFrame(np.array(data), columns=['time', 'type'])
    events['type'] = events['type'].astype('int')
    return events

def import_events(path, name):

    with h5.File(path, 'r') as hdf:
        messages = hdf['/messages/{}'.format(name)][:]
        books = hdf['/orderbooks/{}'.format(name)][:]

    columns = ['sec', 'nano', 'type', 'event', 'side', 'price', 'shares', 'refno', 'newrefno']
    messages = pd.DataFrame(messages, columns=columns)
    columns = ['sec', 'nano']
    _, num_cols = books.shape
    columns += ['bidprc_{}'.format(i + 1) for i in range(int((num_cols - 2) / 4))]
    columns += ['askprc_{}'.format(i + 1) for i in range(int((num_cols - 2) / 4))]
    columns += ['bidvol_{}'.format(i + 1) for i in range(int((num_cols - 2) / 4))]
    columns += ['askvol_{}'.format(i + 1) for i in range(int((num_cols - 2) / 4))]
    books = pd.DataFrame(books, columns=columns)
    books = books.ix[:, ['sec', 'nano', 'bidprc_1', 'askprc_1', 'bidvol_1', 'askvol_1']]
    events = extract_events(messages, books)
    return events
events = import_events('/Users/colinswaney/Desktop/itch-010213.hdf5', 'GOOG')
events = events[ events['time'] > mkt_open ]


# Event Counts
N, _ = events.shape
grouped = events.groupby('type')
counts = grouped.size()
counts.index = [node_properties(i) for i in range(12)]
counts / N  # Table
counts / (mkt_close - mkt_open)  # Table


# Time Series (by node)
def event_series(events):
    for i in range(0,6):
        plt.subplot(6,1,i+1)
        e = events[ events.type == i ].reset_index(drop=True)
        plt.hist(e.time, bins=np.arange(34200, 57000, 300), histtype='step', color='k')
        # plt.xlabel(node_properties(i))
        plt.xticks([], [])
    hours = pd.date_range(start='01-02-2013 09:30:00', end='01-02-2013 16:00:00', freq='H')
    plt.xticks(np.arange(mkt_open, mkt_close, 3600), [h.strftime('%H:%M:%S') for h in hours], rotation=45)
    plt.tight_layout()
    plt.show()
    plt.clf()
    for i in range(0,4):
        plt.subplot(4,1,i+1)
        e = events[ events.type == i + 6].reset_index(drop=True)
        plt.hist(e.time, bins=np.arange(34200, 57000, 300), histtype='step', color='k')
        # plt.xlabel(node_properties(i + 6))
        plt.xticks([], [])
    hours = pd.date_range(start='01-02-2013 09:30:00', end='01-02-2013 16:00:00', freq='H')
    plt.xticks(np.arange(mkt_open, mkt_close, 3600), [h.strftime('%H:%M:%S') for h in hours], rotation=45)
    plt.tight_layout()
    plt.show()
    plt.clf()
    for i in range(0,2):  # 4 to include walking book
        plt.subplot(2,1,i+1)  # 4 to include walking book
        e = events[ events.type == i + 10].reset_index(drop=True)
        plt.hist(e.time, bins=np.arange(34200, 57000, 300), histtype='step', color='k')
        # plt.xlabel(node_properties(i + 10))
        plt.xticks([], [])
    hours = pd.date_range(start='01-02-2013 09:30:00', end='01-02-2013 16:00:00', freq='H')
    plt.xticks(np.arange(mkt_open, mkt_close, 3600), [h.strftime('%H:%M:%S') for h in hours], rotation=45)
    plt.tight_layout()
    plt.show()
    plt.clf()
event_series(events)






# Volume Histograms
# def volume_plot(messages, title=""):
#     plt.hist(np.abs(messages['shares']), bins=np.arange(0, 1010, 10))
#     plt.title(title)
#     plt.xticks(np.arange(0, 1100, 100))
#     plt.show()
#     plt.clf()
# volume_hist(adds, 'Histogram of ADDS shares')
# volume_hist(deletes, 'Histogram of DELETES shares')
# volume_hist(executes, 'Histogram of EXECUTES shares')














name = 'GOOG'
date = '070113'
dt_max = 60
N = 12
burn = 2000
with h5.File('/Volumes/datasets/ITCH/samples/samples_dt_max={}.hdf5'.format(dt_max), 'r') as hdf:
    dates = [date for date in hdf[name].keys()]

def import_data(name, date, dt_max, burn):
    print("Import data for name {} and date {}.".format(name, date))
    with h5.File('/Volumes/datasets/ITCH/samples/samples_dt_max={}.hdf5'.format(dt_max), 'r') as hdf:
        lambda0 = hdf['/{}/{}/lambda0'.format(name, date)][:]
        W = hdf['/{}/{}/W'.format(name, date)][:]
        mu = hdf['/{}/{}/mu'.format(name, date)][:]
        tau = hdf['/{}/{}/tau'.format(name, date)][:]
    return lambda0[:, burn:], W[:, :, burn:], mu[:, :, burn:], tau[:, :, burn:]
lambda0, W, mu, tau = import_data(name, date, dt_max, burn)

"""Plot the posterior distribution of all of the bias vector elements."""

def bias_plot(lambda0, burn=0):

    means = np.mean(lambda0[:, burn:], axis=1)

    # Level 0, BID
    ax = plt.subplot(3, 2, 1)
    cnts, bins, _ = plt.hist(lambda0[0, burn:], bins=30, alpha=0.50)
    plt.vlines(means[0], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 0', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 0, ASK
    plt.subplot(3, 2, 1)
    cnts, bins, _ = plt.hist(lambda0[3, burn:], bins=30, alpha=0.50)
    plt.vlines(means[3], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, BID
    plt.subplot(3, 2, 2)
    cnts, bins, _ = plt.hist(lambda0[1, burn:], bins=30, alpha=0.50)
    plt.vlines(means[1], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3, 2, 2)
    cnts, bins, _ = plt.hist(lambda0[4, burn:], bins=30, alpha=0.50)
    plt.vlines(means[4], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, BID
    plt.subplot(3, 2, 3)
    cnts, bins, _ = plt.hist(lambda0[2, burn:], bins=30, alpha=0.50)
    plt.vlines(means[2], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 2', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, ASK
    plt.subplot(3, 2, 3)
    cnts, bins, _ = plt.hist(lambda0[5, burn:], bins=30, alpha=0.50)
    plt.vlines(means[5], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_lambda0_add.pdf')

    #### DELETES ####

    # Level 1, BID
    plt.subplot(3, 2, 4)
    cnts, bins, _ = plt.hist(lambda0[6, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[6], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 5])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3, 2, 4)
    cnts, bins, _ = plt.hist(lambda0[8, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[8], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    # plt.ylim([0, np.max(cnts) + 1])
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        # labelbottom='off'
        labelbottom='on')
    plt.xticks(fontsize=8)
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, BID
    plt.subplot(3, 2, 5)
    cnts, bins, _ = plt.hist(lambda0[7, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[7], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 2', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 5])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, ASK
    plt.subplot(3, 2, 5)
    cnts, bins, _ = plt.hist(lambda0[9, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[9], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    # plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_lambda0_delete.pdf')


    #### EXECUTES ####

    # Level 1, BID
    plt.subplot(3, 2, 6)
    cnts, bins, _ = plt.hist(lambda0[10, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[10], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3, 2, 6)
    cnts, bins, _ = plt.hist(lambda0[11, burn:], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[11], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    # plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.225, 0.025), np.arange(0.0, 0.225, 0.025), fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    plt.show()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_lambda0_execute.pdf')

    plt.show()
    plt.clf()
bias_plot(lambda0)

def bias_series(name, date, dt_max, burn=0):
    medians = []
    lower = []
    upper = []
    for date in dates:
        lambda0, W, mu, tau = import_data(name, date, dt_max, burn)
        medians.append(np.percentile(lambda0, 0.50, axis=1).reshape((N, 1)))
        lower.append(np.percentile(lambda0, 0.05, axis=1).reshape((N, 1)))
        upper.append(np.percentile(lambda0, 0.95, axis=1).reshape((N, 1)))
    medians = np.concatenate(medians, axis=1).transpose()
    lower = np.concatenate(lower, axis=1).transpose()
    upper = np.concatenate(upper, axis=1).transpose()
    L, _ = medians.shape
    plt.plot(medians[:, 0], linewidth=0.5)  # add 0
    plt.fill_between(x=np.arange(L), y1=lower[:, 0].reshape(L), y2=upper[:, 0].reshape(L), alpha=0.20)
    plt.plot(medians[:, 1], linewidth=0.5)  # add 0
    plt.fill_between(x=np.arange(L), y1=lower[:, 1].reshape(L), y2=upper[:, 1].reshape(L), alpha=0.20)
    plt.show()
    # plt.plot(medians[2:4, :].transpose())  # add 1
    # plt.plot(medians[4:6, :].transpose())  # add 2
    # plt.plot(medians[6:8, :].transpose())  # cancel 1
    # plt.plot(medians[8:10, :].transpose())  # cancel 2
    # plt.plot(medians[10:12, :].transpose())  # execute
    plt.clf()
bias_series(name, date, dt_max, burn)


"""
First, plot MAP estimates of the connection matrix. Next, plot the posterior
distribution of the diagonal elements of the connection matrix (i.e. the
self-connections).

Third, for each date, plot the MAP estimates of trade self-connections.
"""

def weights_plot(W, name, date, dt_max, burn=0):
    W = W[:, :, burn:]
    plt.imshow(W.mean(axis=2), cmap='Blues', vmin=0, vmax=0.5)
    cbar = plt.colorbar(pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel("Child nodes", labelpad=20, fontsize=8)
    tick_labels = ['Add 0 (Bid)',
                   'Add 1 (Bid)',
                   'Add 2 (Bid)',
                   'Add 0 (Ask)',
                   'Add 1 (Ask)',
                   'Add 2 (Ask)',
                   'Delete 1 (Bid)',
                   'Delete 2 (Bid)',
                   'Delete 1 (Ask)',
                   'Delete 2 (Ask)',
                   'Execute (Bid)',
                   'Execute (Ask)']
    plt.xticks(np.arange(0, 12), tick_labels, rotation=90, fontsize=8)
    plt.ylabel("Parent nodes", labelpad=20, fontsize=8)
    plt.yticks(np.arange(0, 12), tick_labels, fontsize=8)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/Figures/posterior_W_{}_{}_dt_max={}.pdf'.format(name, date, dt_max))
    plt.show()
    plt.clf()
weights_plot(W, name, date, dt_max)

def self_connections_plot(W, name, date, dt_max, burn=0):

    diagonal = np.diagonal(W)[burn:, :]
    means = np.mean(diagonal, axis=0)

    # Level 0, BID
    plt.subplot(3, 2, 1)
    cnts, bins, _ = plt.hist(diagonal[:, 0], bins=30, alpha=0.50)
    plt.vlines(means[0], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 0', fontsize=8)
    # plt.xlim([0.10, 0.70])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 0, ASK
    plt.subplot(3, 2, 1)
    cnts, bins, _ = plt.hist(diagonal[:, 3], bins=30, alpha=0.50)
    plt.vlines(means[3], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0.10, 0.70])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, BID
    plt.subplot(3, 2, 2)
    cnts, bins, _ = plt.hist(diagonal[:, 1], bins=30, alpha=0.50)
    plt.vlines(means[1], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0.10, 0.70])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3, 2, 2)
    cnts, bins, _ = plt.hist(diagonal[:, 4], bins=30, alpha=0.50)
    plt.vlines(means[4], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0.10, 0.70])
    # plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, BID
    plt.subplot(3, 2, 3)
    cnts, bins, _ = plt.hist(diagonal[:, 2], bins=30, alpha=0.50)
    plt.vlines(means[2], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 2', fontsize=8)
    # plt.xlim([0.10, 0.70])
    plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, ASK
    plt.subplot(3, 2, 3)
    cnts, bins, _ = plt.hist(diagonal[:, 5], bins=30, alpha=0.50)
    plt.vlines(means[5], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0.10, 0.70])
    # plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.15, 0.75, 0.1), np.arange(0.15, 0.75, 0.1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_W_add.pdf')
    # plt.clf()

    #### DELETES ####

    # Level 1, BID
    plt.subplot(3, 2, 4)
    cnts, bins, _ = plt.hist(diagonal[:, 6], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[6], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0.1, 0.6])
    plt.ylim([0, np.max(cnts) + 5])
    # plt.xticks(np.arange(0.1, 0.7, .1), np.arange(0.1, 0.7, .1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3, 2, 4)
    cnts, bins, _ = plt.hist(diagonal[:, 8], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[8], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0.1, 0.6])
    # plt.xticks(np.arange(0.1, 0.7, .1), np.arange(0.1, 0.7, .1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off',
        labelbottom='on')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, BID
    plt.subplot(3, 2, 5)
    cnts, bins, _ = plt.hist(diagonal[:, 7], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[7], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 2', fontsize=8)
    # plt.xlim([0.1, 0.6])
    plt.ylim([0, np.max(cnts) + 5])
    # plt.xticks(np.arange(0.1, 0.7, .1), np.arange(0.1, 0.7, .1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 2, ASK
    plt.subplot(3, 2, 5)
    cnts, bins, _ = plt.hist(diagonal[:, 9], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[9], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0.1, 0.6])
    # plt.xticks(np.arange(0.1, 0.7, .1), np.arange(0.1, 0.7, .1), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_W_delete.pdf')
    # plt.clf()

    #### EXECUTES ####

    # Level 1, BID
    plt.subplot(3,2,6)
    cnts, bins, _ = plt.hist(diagonal[:, 10], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[10], 0, np.max(cnts), linewidth=1.00, color='C0')
    plt.ylabel('Level 1', fontsize=8)
    # plt.xlim([0, 0.20])
    plt.ylim([0, np.max(cnts) + 5])
    # plt.xticks(np.arange(0.0, 0.35, 0.05), np.arange(0.0, 0.35, 0.05), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)

    # Level 1, ASK
    plt.subplot(3,2,6)
    cnts, bins, _ = plt.hist(diagonal[:, 11], bins=30, alpha=0.50, normed=True)
    plt.vlines(means[11], 0, np.max(cnts), linewidth=1.00, color='C1')
    # plt.xlim([0, 0.20])
    # plt.ylim([0, np.max(cnts) + 100])
    # plt.xticks(np.arange(0.0, 0.35, 0.05), np.arange(0.0, 0.35, 0.05), fontsize=8)
    plt.yticks([])
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        left='off')
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    # plt.savefig('/Users/colinswaney/Desktop/posterior_W_execute.pdf')
    plt.show()
    plt.clf()
self_connections_plot(W, name, date, dt_max)

def weights_series(name, date, dt_max, burn=0):
    medians = []
    lower = []
    upper = []
    for date in dates:
        lambda0, W, mu, tau = import_data(name, date, dt_max, burn)
        # weights_plot(W, name, date, dt_max)
        medians.append(np.percentile(np.diagonal(W), 0.50, axis=0).reshape((N, 1)))
        lower.append(np.percentile(np.diagonal(W), 0.05, axis=0).reshape((N, 1)))
        upper.append(np.percentile(np.diagonal(W), 0.95, axis=0).reshape((N, 1)))
    medians = np.concatenate(medians, axis=1).transpose()
    lower = np.concatenate(lower, axis=1).transpose()
    upper = np.concatenate(upper, axis=1).transpose()
    L, _ = medians.shape
    plt.plot(medians[:, 0])  # add 0
    plt.fill_between(x=np.arange(L), y1=lower[:, 0].reshape(L), y2=upper[:, 0].reshape(L), alpha=0.10)
    plt.plot(medians[:, 1])  # add 0
    plt.fill_between(x=np.arange(L), y1=lower[:, 1].reshape(L), y2=upper[:, 1].reshape(L), alpha=0.10)
    plt.show()
    # plt.plot(medians[2:4, :].transpose())  # add 1
    # plt.plot(medians[4:6, :].transpose())  # add 2
    # plt.plot(medians[6:8, :].transpose())  # cancel 1
    # plt.plot(medians[8:10, :].transpose())  # cancel 2
    # plt.plot(medians[10:12, :].transpose())  # execute
    plt.clf()
weights_series(name, date, dt_max, burn)

def plot_posterior(sample, title, burn=0):
    cnts, bins, _ = plt.hist(sample[burn:], bins=30, alpha=0.50)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.show()
    plt.clf()
plot_posterior(lambda0[0, :], 'Add to Bid Level 0')


"""Calculate and plot the impulse responses based on MAP estimates of mu and tau."""

def logit_normal(dt, mu, tau, dt_max):
    assert (dt < dt_max).all(), "dt must be less than dt_max."
    assert (dt > 0).all(), "dt must be greater than zero."
    Z = dt * (dt_max - dt) / dt_max * (tau / (2 * np.pi)) ** (-0.5)
    x = dt / dt_max
    s = np.log(x / (1 - x))
    return (1 / Z) * np.exp( -tau / 2 * (s - mu) ** 2 )

def plot_impulse(parent, child, W, mu, tau, dt_max, xmax, ymax=None, burn=0):
    eps = 0.001
    dt = np.linspace(0 + eps, xmax - eps, 100)
    mu_map = np.median(mu[parent, child, burn:])
    tau_map = np.median(tau[parent, child, burn:])
    values = logit_normal(dt, mu_map, tau_map, dt_max)
    weights = np.median(W[parent, child, burn:]) / np.max(np.median(W, axis=2))
    plt.plot(dt, values, alpha=weights, linewidth=0.5)
    if ymax is not None:
        plt.ylim([0, ymax])
    else:
        plt.ylim([0, np.max(values) * 1.1])
    plt.xlim([0, xmax])
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')
    return np.max(values)

# Impulse response matrix for different timescales
M = np.zeros((12, 12))
for xmax in [60, 10, 1, 0.1, 0.01]:
    for i in range(12):
        for j in range(12):
            plt.subplot(12, 12, (i * 12)+ j + 1)
            if xmax == 60:
                m = plot_impulse(i, j, W, mu, tau, dt_max, xmax)
                M[i, j] = m * 1.1
            else:
                _ = plot_impulse(i, j, W, mu, tau, dt_max, xmax, M[i, j])
    print('Saving figure for xmax={}...'.format(xmax))
    plt.savefig('/Users/colinswaney/Dropbox/Research/hft-hawkes/results/{}/impulse_xmax={}.pdf'.format(name, xmax))
    plt.clf()

# Same thing, but averaging estimates over the month
W_medians = []
mu_medians = []
tau_medians = []
for date in dates:
    _, W, mu, tau = import_data(name, date, dt_max, burn)
    W_medians.append(np.median(W, axis=2).reshape((12, 12, 1)))
    mu_medians.append(np.median(mu, axis=2).reshape((12, 12, 1)))
    tau_medians.append(np.median(tau, axis=2).reshape((12, 12, 1)))
W_avg = np.mean(np.concatenate(W_medians, axis=2), axis=2)
mu_avg = np.mean(np.concatenate(mu_medians, axis=2), axis=2)
tau_avg = np.mean(np.concatenate(tau_medians, axis=2), axis=2)

def plot_impulse(parent, child, W, mu, tau, dt_max, xmax, ymax=None):

    """
        W: N x N
        mu: N x N
        tau: N x N
    """

    eps = 0.001
    dt = np.linspace(0 + eps, xmax - eps, 100)
    values = logit_normal(dt, mu[parent, child], tau[parent, child], dt_max)
    weights = np.median(W[parent, child]) / np.max(W)
    plt.plot(dt, values, linewidth=0.5)
    plt.xlim([0, xmax])
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')
    return np.max(values)

# the impulse matrix
xmax = 60
for i in range(12):
    for j in range(12):
        plt.subplot(12, 12, (i * 12)+ j + 1)
        _ = plot_impulse(i, j, W_avg, mu_avg, tau_avg, dt_max, xmax)
plt.savefig('/Users/colinswaney/Dropbox/Research/hft-hawkes/results/{}/impulse_xmax={}.pdf'.format(name, xmax))
plt.clf()

# the weight matrix
def weights_plot(W, name=None, date=None, dt_max=None, burn=0):
    plt.imshow(W, cmap='Blues', vmin=0, vmax=0.5)
    cbar = plt.colorbar(pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel("Child nodes", labelpad=20, fontsize=8)
    tick_labels = ['Add 0 (Bid)',
                   'Add 1 (Bid)',
                   'Add 2 (Bid)',
                   'Add 0 (Ask)',
                   'Add 1 (Ask)',
                   'Add 2 (Ask)',
                   'Delete 1 (Bid)',
                   'Delete 2 (Bid)',
                   'Delete 1 (Ask)',
                   'Delete 2 (Ask)',
                   'Execute (Bid)',
                   'Execute (Ask)']
    plt.xticks(np.arange(0, 12), tick_labels, rotation=90, fontsize=8)
    plt.ylabel("Parent nodes", labelpad=20, fontsize=8)
    plt.yticks(np.arange(0, 12), tick_labels, fontsize=8)
    plt.tight_layout()
    plt.savefig('/Users/colinswaney/Desktop/connections_avg.pdf')
    plt.clf()
weights_plot(W_avg)


"""Plot the intensity based on MAP estimates of the parameters and the observed event data."""

def estimate(lambda0, W, mu, tau, est='median'):
    if est == 'median':
        lambda0Hat = np.median(lambda0, axis=-1)
        WHat = np.median(W, axis=-1)
        muHat = np.median(mu, axis=-1)
        tauHat = np.median(tau, axis=1)
    return lambda0Hat, WHat, muHat, tauHat

# Make a network with estimated parameters
lambda0_map, W_map, mu_map, tau_map = estimate(lambda0, W, mu, tau, 'median')
params = {'bias': lambda0_map, 'weights': W_map, 'mu': mu_map, 'tau': tau_map}
model = NetworkPoisson(N=N, dt_max=dt_max, params=params)


"""Perform kernel density estimation on Gibbs sample and plot."""
from scipy.stats import gaussian_kde
def plot_density(x):
    # x = lambda0[0, :]
    grid = np.linspace(x.min(), x.max(), 1000)
    kde = gaussian_kde(x).evaluate(grid)
    plt.plot(grid, kde)
    plt.fill_between(grid, kde)
    plt.show()
    plt.clf()

"""
For each date, compute median of Gibbs sample, then calculate the time series
average of the medians.
"""






# Import event data
with h5.File() as hdf:
    events = hdf['{}/{}'.format(name, date)][:]

# Calculate the intensity
Lambda = net.calculate_intensity(events)

# Timestamps for  xlabels
seconds = pd.date_range(start='01-02-2013 10:30:00', end='01-02-2013 14:59:59', freq='S')
hours = pd.date_range(start='01-02-2013 10:30:00', end='01-02-2013 14:59:59', freq='H')

def add_intensity_plot():
    plt.subplot(3, 1, 1)
    plt.fill_between(seconds, 0, Lambda[:, 0])
    plt.fill_between(seconds, 0, -Lambda[:, 3])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off')
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 0', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)

    plt.subplot(3, 1, 2)
    plt.fill_between(seconds, 0, Lambda[:, 1])
    plt.fill_between(seconds, 0, -Lambda[:, 4])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off')
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 1', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)

    plt.subplot(3, 1, 3)
    plt.fill_between(seconds, 0, Lambda[:, 2])
    plt.fill_between(seconds, 0, -Lambda[:, 5])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 2', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    plt.savefig('/Users/colinswaney/Desktop/intensity_fit_add.pdf')
    plt.clf()

def delete_intensity_plot():
    ax = plt.subplot(2, 1, 1)
    plt.fill_between(seconds, 0, Lambda[:, 6])
    plt.fill_between(seconds, 0, -Lambda[:, 8])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off')
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 1', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)

    plt.subplot(2, 1, 2)
    plt.fill_between(seconds, 0, Lambda[:, 7])
    plt.fill_between(seconds, 0, -Lambda[:, 9])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 2', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    plt.savefig('/Users/colinswaney/Desktop/intensity_fit_delete.pdf')
    plt.clf()

def execute_intensity_plot():
    plt.fill_between(seconds, 0, Lambda[:, 10])
    plt.fill_between(seconds, 0, -Lambda[:, 11])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylabel('Level 2', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()
    plt.savefig('/Users/colinswaney/Desktop/intensity_fit_execute.pdf')
    plt.clf()

def intensity_subplot():

    fig, ax = plt.subplots()
    plt.fill_between(seconds, 0, Lambda[:, 0])
    plt.fill_between(seconds, 0, -Lambda[:, 3])
    plt.xticks(hours, [h.strftime('%H:%M:%S') for h in hours], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    left='off',
                    right='off',
                    labelbottom='off')
    plt.xlim([seconds[0], seconds[-1]])
    plt.ylim([-1.5, 1.5])
    plt.ylabel('Level 0', fontsize=8, labelpad=10)
    plt.grid(linestyle='--', linewidth=0.25)
    plt.tight_layout()

    # left, bottom, width, height = (0.70, 0.70, .20, .20)
    # ax_inner = fig.add_axes([left, bottom, width, height])
    # ax_inner.fill_between(np.arange(0, 60), 0, Lambda[1000:1060, 0])
    # ax_inner.fill_between(np.arange(0, 60), 0, -Lambda[1000:1060, 3])
    # ax_inner.tick_params(axis='both',
    #                      which='both',
    #                      bottom='off',
    #                      top='off',
    #                      left='off',
    #                      right='off',
    #                      labelbottom='off',
    #                      labelleft='off')

    ax2 = ax.twinx()
    trades = pd.DataFrame(spikes[:, -2:], index=seconds, columns=['buys', 'sells']).reset_index(drop=True)
    trades['min'] = (trades.index.values / 60).astype(int)
    grouped = trades.groupby('min').mean()
    minutes = pd.date_range(start='01-02-2013 10:30:00', end='01-02-2013 14:59:59', freq='T')
    b, = ax2.plot(minutes, grouped['buys'], linewidth=1, color='k')
    # plt.fill_between(buys.index, 0, buys['shares'])
    s, = ax2.plot(minutes, -grouped['sells'], linewidth=1, color='k')
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_xlim([seconds[0], seconds[-1]])
    # plt.fill_between(sells.index, 0, -sells['shares'])
    # plt.ylabel('Trades', fontsize=8)
    # ax.yaxis.set_ticks_position('right')
    # plt.yticks(fontsize=8)
    # plt.xticks(np.arange(mkt_open / 60, mkt_close / 60, 3600 / 60), [h.strftime('%H:%M:%S') for h in hours])
    # plt.xticks(rotation=45, fontsize=8)
    # plt.xlim([mkt_open / 60, mkt_close / 60])
    # plt.grid(linestyle='--', linewidth=0.25)

    plt.show()
    plt.clf()

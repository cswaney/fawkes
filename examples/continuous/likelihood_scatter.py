import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""Create scatterplots of likelihood versus microstructure characteristics."""


df = pd.read_csv('/Users/colinswaney/Desktop/net/data/likelihood_M=180.txt', index_col=0)

M = 180
df['bits'] = (df['ll_net'] - df['ll_hom']) / M
df.median(axis=0)
df.mean(axis=0)
grouped = df.groupby('name')
medians = grouped.median().reset_index()

# 1. Likelihood vs. CRSP
crsp = pd.read_csv('/Users/colinswaney/Desktop/net/data/crsp.txt')
crsp = crsp[crsp['date'] == '07/24/2013']
crsp=crsp.drop(['PERMNO', 'NUMTRD', 'SHROUT', 'COMNAM', 'date'], axis=1)
crsp.columns = ['name', 'exchange', 'price', 'volume']
combined = pd.merge(medians, crsp)
# ... vs. price (close) *
plt.subplot(121)
plt.scatter(np.log(combined['price']), combined['bits'])
# ... vs. volume *
plt.subplot(122)
plt.scatter(np.log(combined['volume']), combined['bits'])
plt.tight_layout()


# 2. Likelihood vs. ITCH
stability = pd.read_csv('/Users/colinswaney/Desktop/net/data/stability.txt', index_col=0)
volumes = pd.read_csv('/Users/colinswaney/Desktop/net/data/volumes.txt', index_col=0)
prices = pd.read_csv('/Users/colinswaney/Desktop/net/data/prices.txt', index_col=0)
spreads = pd.read_csv('/Users/colinswaney/Desktop/net/data/spreads.txt')
# ... vs. volume *
combined = pd.merge(medians, volumes)
plt.subplot(221)
plt.title('log(volume)')
plt.scatter(np.log(combined['total']), combined['bits'])
# ... vs. price *
combined = pd.merge(medians, prices)
plt.subplot(222)
plt.title('log(price)')
plt.scatter(np.log(combined['median']), combined['bits'] / M)
# ... vs. volatility *
plt.subplot(223)
plt.title('log(volatility)')
plt.scatter(np.log(combined['vol']), combined['bits'] / M)
# ... vs. spread *
combined = pd.merge(medians, spreads)
combined = combined[combined['date'] == 72413]
plt.subplot(224)
plt.title('log(spread)')
plt.scatter(np.log(combined['mean']), combined['bits'] / M)
plt.tight_layout()

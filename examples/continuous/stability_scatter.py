import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""Create scatterplots of stability versus microstructure characteristics."""

df = pd.read_csv('/Users/colinswaney/Desktop/net/data/stability.txt', index_col=0)

# 1. Stability vs. CRSP
crsp = pd.read_csv('/Users/colinswaney/Desktop/net/data/crsp.txt')
crsp = crsp[crsp['date'] == '07/24/2013']
crsp=crsp.drop(['PERMNO', 'NUMTRD', 'SHROUT', 'COMNAM', 'date'], axis=1)
crsp.columns = ['name', 'exchange', 'price', 'volume']
combined = pd.merge(df, crsp)

# ... vs. price
plt.subplot(121)
plt.scatter(np.log(combined['price']), combined['eig'])

# ... vs. volume **
plt.scatter(np.log(combined['volume']), combined['eig'])
plt.subplot(122)
plt.tight_layout()

# 2. Stability vs. ITCH
volumes = pd.read_csv('/Users/colinswaney/Desktop/net/data/volumes.txt', index_col=0)
prices = pd.read_csv('/Users/colinswaney/Desktop/net/data/prices.txt', index_col=0)
spreads = pd.read_csv('/Users/colinswaney/Desktop/net/data/spreads.txt')

# ... vs. volume **
combined = pd.merge(df, volumes)
plt.subplot(221)
plt.title('log(volume)')
plt.scatter(np.log(combined['total']), combined['eig'])

# ... vs. price
combined = pd.merge(df, prices)
plt.subplot(222)
plt.title('log(price)')
plt.scatter(np.log(combined['median']), combined['eig'])

# ... vs. volatility
plt.subplot(223)
plt.title('log(volatility)')
plt.scatter(np.log(combined['vol']), combined['eig'])

# ... vs. spread
combined = pd.merge(df, spreads)
combined = combined[combined['date'] == 72413]
plt.subplot(224)
plt.title('log(spread)')
plt.scatter(np.log(combined['mean']), combined['eig'])
plt.tight_layout()

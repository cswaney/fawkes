import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()
plt.rc('font', size=8, family='Arial')

N = 11
x = np.random.randn(N)
y = np.random.randn(N)
z = np.random.randn(N)
w = np.random.randn(N)
v = np.random.randn(N)
t = np.arange(N)

events_ax = plt.subplot2grid((6,2), (0,0), rowspan=3)
book_ax = plt.subplot2grid((6,2), (3,0), rowspan=3)
score_ax = plt.subplot2grid((6,2), (0,1), rowspan=2)
inventory_ax = plt.subplot2grid((6,2), (2,1), rowspan=2)
orders_ax = plt.subplot2grid((6,2), (4,1), rowspan=2)
for i in np.arange(1,N+1):

    # Draw x
    events_ax.clear()
    events_ax.set_xlim(-1,N)
    events_ax.set_ylim(-5,5)
    # events_ax.xaxis.tick_top()
    events_ax.set_title('Events', loc='right')
    # events_ax.text(x=0.85, y=0.92, s='Events', transform=events_ax.transAxes)
    events_ax.tick_params(direction='in')
    events_dots = events_ax.scatter(t[:i], x[:i], color='C1')

    # Draw y
    book_ax.clear()
    book_ax.set_xlim(-1,N)
    book_ax.set_ylim(-5,5)
    book_ax.set_title('Books', loc='right')
    # book_ax.text(x=0.85, y=0.93, s='Books', transform=book_ax.transAxes)
    book_ax.tick_params(direction='in')
    book_bars = book_ax.bar(t[:i], y[:i], color='C2')

    # Draw score
    score_ax.clear()
    score_ax.set_xlim(0,N-1)
    score_ax.set_ylim(-5,5)
    score_ax.set_title('Score', loc='right')
    # score_ax.xaxis.tick_top()
    score_ax.tick_params(right=True, left=False, labelright=True, labelleft=False, direction='in')
    # score_ax.text(x=0.85, y=0.9, s='Score', transform=score_ax.transAxes)
    score_line, = score_ax.plot(t[:i], z[:i], color='C3', linestyle='--')

    # Draw inventory
    inventory_ax.clear()
    inventory_ax.set_xlim(0,N-1)
    inventory_ax.set_ylim(-5,5)
    # inventory_ax.set_xticks([])
    inventory_ax.set_title('Inventory', loc='right')
    inventory_ax.tick_params(right=True, left=False, labelright=True, labelleft=False, direction='in')
    # inventory_ax.text(x=0.80, y=0.9, s='Inventory', transform=inventory_ax.transAxes)
    inventory_line, = inventory_ax.plot(t[:i], w[:i], color='C4')

    # Draw orders
    orders_ax.clear()
    orders_ax.set_xlim(-1,N)
    orders_ax.set_ylim(-5,5)
    orders_ax.set_title('Orders', loc='right')
    # orders_ax.set_xticks([])
    orders_ax.tick_params(right=True, left=False, labelright=True, labelleft=False, direction='in')
    # orders_ax.text(x=0.85, y=0.9, s='Orders', transform=orders_ax.transAxes)
    orders_bars = orders_ax.bar(t[:i], v[:i], color='C0')

    plt.tight_layout()
    plt.draw()
    input('paused')

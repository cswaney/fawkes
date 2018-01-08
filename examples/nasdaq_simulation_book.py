from engine import Order, OrderBook

add_1 = Order('add', 'bid', 30.00, 100)
add_2 = Order('add', 'ask', 30.05, 100)
add_3 = Order('add', 'bid', 30.01, 50)
add_4 = Order('add', 'bid', 30.01, 100)
add_5 = Order('add', 'ask', 30.03, 50)
add_6 = Order('add', 'ask', 30.02, 100)
delete_1 = Order('delete', add_1.side, add_1.price, None, add_1.refno)
delete_2 = Order('delete', add_2.side, add_2.price, None, add_2.refno)
delete_3 = Order('delete', 'bid', None, None, None)
execute_1 = Order('execute', 'ask', price=None, shares=10)
execute_2 = Order('execute', 'bid', price=None, shares=50)
execute_3 = Order('execute', 'bid', price=None, shares=500)
book = OrderBook()
book.update(add_1)
book.update(add_2)
book.update(add_3)
book.update(add_4)
book.update(add_5)
book.update(add_6)
book.update(delete_1)
book.update(delete_2)
book.update(delete_3)
book.update(execute_1)
book.update(execute_2)
book.update(execute_3)

# simulation
env = Simulator()
state = env.reset()
while not env.done:
    action = learner.choose(state)
    state, done = env.step(action)
score = env.evaluate()

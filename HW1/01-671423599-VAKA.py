import matplotlib as m
import numpy as np
import random as rand
import time
import matplotlib.pyplot as plt

rand.seed(time.time())
coords = np.empty([2, 1000])

for i in range(1000):
  coords[0][i] = rand.uniform(2, -2)
  coords[1][i] = rand.uniform(2, -2)

def step(input):
  if input<0:
    return 0
  else:
    return 1

def network(x, y):
  return step(-1.5 + step(1 + x - y) + step(1 - x - y) - step(-x))

for i in range(1000):
  if network(coords[0][i], coords[1][i]) == 0:
    plt.plot(coords[0][i], coords[1][i], 'bo')
  else:
    plt.plot(coords[0][i], coords[1][i], 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
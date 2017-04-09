"""
This code demonstrates the relative speed of implementations of the Hungarian algorithm for a
stock assignment problem. The three implementations that are compared are
1) scipy.optimize.linear_sum_assignment: this can be found in the standard scipy install.
2) munkres 1.0.9: older implementation, can be installed using pip.
   ```sh
   sudo pip install munkres
   ```
3) hungarian 0.2.3: uses C++ wrapper, can be installed using pip.
   ```sh
   sudo pip install hungarian
   ```
   The API here is as follows:
   ```python
   import hungarian
   ret_tup = hungarian.lap(C)
   ```
   ret_tup[0] is the row assignments
   ret_tup[1] is the column assignments
The code below assumes all three are installed.
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_checkerboard
import matplotlib.pyplot as plt

from munkres import Munkres
import hungarian

debug_cost = True

if debug_cost:
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
else:
    n_clusters = (4, 3)
    data, rows, columns = make_checkerboard(
        shape=(300, 300), n_clusters=n_clusters, noise=10,
        shuffle=False, random_state=0)

    plt.matshow(data, cmap=plt.cm.Blues)
    plt.title("Original dataset")
    cost = data


# scipy implementation
print "Running scipy linear_sum_assignment on cost matrix of form ({}, {})".format(cost.shape[0], cost.shape[1])
s_t = time.time()
row_ind, col_ind = linear_sum_assignment(cost)
print "Elapsed time: {} seconds.".format(time.time()-s_t)
print "Row assignments:\n{}, \nColumn assignments:\n{}".format(row_ind, col_ind)

# Munkres implementation
print "\nRunning Munkres on cost matrix of form ({}, {})".format(cost.shape[0], cost.shape[1])
s_t = time.time()
m = Munkres()
indexes = m.compute(cost)
row_ind = [val[0] for val in indexes]
col_ind = [val[0] for val in indexes]
print "Elapsed time: {} seconds.".format(time.time()-s_t)
print "Row assignments:\n{}, \nColumn assignments:\n{}".format(row_ind, col_ind)


# Hungarian implementation
print "\nRunning hungarian on cost matrix of form ({}, {})".format(cost.shape[0], cost.shape[1])
s_t = time.time()
vals = hungarian.lap(cost)
print "Elapsed time: {} seconds.".format(time.time()-s_t)
print "Row assignments:\n{}, \nColumn assignments:\n{}".format(vals[0], vals[1])


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
import copy
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_checkerboard
import matplotlib.pyplot as plt

from munkres import Munkres
import hungarian

debug_cost = False

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
cost1 = copy.deepcopy(cost)
cost2 = copy.deepcopy(cost)
cost3 = copy.deepcopy(cost)

# scipy implementation
print "Running scipy linear_sum_assignment on cost matrix of form ({}, {})".format(cost.shape[0], cost.shape[1])
s_t = time.time()
row_ind, col_ind = linear_sum_assignment(cost1)
sp_time = time.time() - s_t
print "Elapsed time: {} seconds.".format(time.time()-s_t)
print "Row assignments:\n{}, \nColumn assignments:\n{}".format(row_ind, col_ind)
# compute profit
total = 0
for row in row_ind:
    for col in col_ind:
        value = cost[row][col]
        total += value
print "Total profit: {}".format(total)

# Munkres implementation
print "\nRunning Munkres on cost matrix of form ({}, {})".format(cost.shape[0], cost.shape[1])
s_t = time.time()
m = Munkres()
indexes = m.compute(cost2)
row_ind = np.array([val[0] for val in indexes])
col_ind = np.array([val[1] for val in indexes])
m_time = time.time() - s_t
print "Elapsed time: {} seconds.".format(m_time)
print "Row assignments:\n{}, \nColumn assignments:\n{}".format(row_ind, col_ind)
# compute profit
total = 0
for row in row_ind:
    for col in col_ind:
        value = cost[row][col]
        total += value
print "Total profit: {}".format(total)


# Hungarian implementation
print "\nRunning hungarian on cost matrix of form ({}, {})".format(cost.shape[0], cost.shape[1])
s_t = time.time()
vals = hungarian.lap(cost3)
row_ind = vals[0]
col_ind = vals[1]
h_time = time.time()-s_t
print "Elapsed time: {} seconds.".format(h_time)
print "Row assignments:\n{}, \nColumn assignments:\n{}".format(row_ind, col_ind)
# compute profit
total = 0
for row in row_ind:
    for col in col_ind:
        value = cost[row][col]
        total += value
print "Total profit: {}".format(total)

times_d = {"Scipy": sp_time, "Munkres": m_time, "Hungarian": h_time}
print "Speedup of Munkres over baseline (Scipy): {}".format(times_d["Scipy"]/times_d["Munkres"])
print "Speedup of HUngarian over baseline (Scipy): {}".format(times_d["Scipy"]/times_d["Hungarian"])

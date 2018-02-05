import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from learn.poly_overlap import metric_overlap


class ShapeFeatureSet():
    def __init__(self, feature_set, feature_set_scale, num_vertex_per_feat):
        self.feature_set = feature_set
        self.feature_set_scale = feature_set_scale
        self.num_vertex_per_feat = num_vertex_per_feat


def feature_2_polygon(feature_vec):
    return np.reshape(feature_vec, (-1, 2))


def plot_coords(ax, ob, color):
    x, y = ob.xy
    ax.plot(x, y, 'o', c=color, zorder=1)


# each 202-dimensional feature vector represents a 2-dimensional polygon object
with open('./data/shape_feature_set.pickle', 'rb') as f:
    features = pickle.load(f)

# compute 3 nearest neighbors for a polygon in metric overlap space: need to use brute force
features = features.T  # sklearn's expecting row-major data
nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute', metric=metric_overlap).fit(features)

query_ind = 0
s_t = time.time()
dists, inds = nbrs.kneighbors(features[query_ind, :])
print "Time take to search {} polygons for 3 closest " \
      "neighbors: {:.2f} seconds.".format(features.shape[0], time.time()-s_t)
dists = dists.flatten()
inds = inds.flatten()
print "Distances of closest neigbors: {}".format(dists)

ex1 = feature_2_polygon(features[query_ind, :])
n1 = feature_2_polygon(features[inds[0], :])
n2 = feature_2_polygon(features[inds[1], :])
n3 = feature_2_polygon(features[inds[2], :])

fig = plt.figure()
ax1 = fig.add_subplot(141)
ax1.plot(ex1[:, 0], ex1[:, 1], 'bo', label='query')
ax1.legend()
ax2 = fig.add_subplot(142)
ax2.plot(n1[:, 0], n1[:, 1], 'ro', label='n1')
ax2.legend()
ax3 = fig.add_subplot(143)
ax3.plot(n2[:, 0], n2[:, 1], 'go', label='n2')
ax3.legend()
ax4 = fig.add_subplot(144)
ax4.plot(n3[:, 0], n3[:, 1], 'mo', label='n3')
ax4.legend()
plt.suptitle('3 nearest neighbors of a given polygon.')
plt.show()

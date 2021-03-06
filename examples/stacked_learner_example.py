"""
Show example of stacked learner.
"""

import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from learn.learners import StackedLearner


train_feats_file = "./data/training_feat.pickle"
train_labels_file = "./data/training_label.pickle"
test_feats_file = "./data/testing_feat.pickle"
test_labels_file = "./data/testing_label.pickle"
train_feats = pickle.load(open(train_feats_file, "rb"))
train_labels = pickle.load(open(train_labels_file, "rb"))
test_feats = pickle.load(open(test_feats_file, "rb"))
test_labels = pickle.load(open(test_labels_file, "rb"))

print "Shapes: ", train_feats.shape, train_labels.shape, test_feats.shape, test_labels.shape
# fix issues
temp = train_labels
train_labels = test_labels
test_labels = temp
print "Shapes: ", train_feats.shape, train_labels.shape, test_feats.shape, test_labels.shape

print np.nonzero(test_labels == 0)[0].shape, np.nonzero(test_labels == 1)[0].shape

param_dict = {"C": [10, 100, 1000]}
label_dict = {"none": 0, "corner": 1}
sl = StackedLearner(stacked_learner="linear_svm", param_dict=param_dict)
sl.fit_from_arrays(train_feats, train_labels)
sl.evaluate(test_feats, test_labels, op_point=0.15, label_dict=label_dict)

plt.show()

"""
Code for machine learning.
"""
import copy
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC


class StackedLearner(object):
    def __init__(self, base_learners_dict=None, stacked_learner="logistic", param_dict={'C': 10},
                 nfolds=5):
        """
        Pass in list of base learner objects, along with type of stacked
        learner, as well as its parameters.

        :param base_learners_dict: dictionary where each key points to a unique learner. In the ideal
                                   case, you would want these learners to generate train-test sets
                                   directly, but we will allow the user to pass in already existing
                                   arrays for these. In that case, this defaults to None.
        :param stacked_learner: pick from ["logistic", "linear_svm", "rbf_svm"]
        :param param_dict: index by exact parameter name, and a list of parameters. For example,
                           if you're using the RBF SVM, you might want to pass in
                           param_dict = {'C': [10, 100], 'gamma': [0.01, 0.1]}. The search will
                           then be conducted over all possible parameters.
        :param nfolds: number of folds for classification
        """
        self.base_learners_dict = base_learners_dict
        self.s_type = stacked_learner
        self.nfolds = nfolds

        if stacked_learner == "logistic":
            self.stacked_learner = LogisticRegression()
        elif stacked_learner == "linear_svm":
            self.stacked_learner = LinearSVC()
        elif stacked_learner == "rbf_svm":
            self.stacked_learner = SVC()
        else:
            raise RuntimeError("Incorrect choice of stacked learner.")
        self.param_dict = param_dict
        self.tr_scores = []
        self.models = []

    def fit_from_arrays(self, train_outputs, train_labels):
        """

        :param train_outputs: n x l numpy array, where l is number of learners, and each attribute is
                              a probability from a learner on its respective sample.
        :param train_labels: labels for the samples.
        :return:
        """
        if self.s_type == "logistic" or self.s_type == "linear_svm":
            C_list = self.param_dict['C']
            for curr_C in C_list:
                curr_learner = copy.copy(self.stacked_learner)
                curr_learner.C = curr_C
                scores = cross_val_score(curr_learner, train_outputs, train_labels, cv=self.nfolds,
                                         scoring=metrics.make_scorer(metrics.auc))
                self.tr_scores.append(scores)
                curr_learner.fit(train_outputs, train_labels)
                self.models.append(curr_learner)
                print "Mean AUC: " + str(scores.mean())

    def evaluate(self, test_outputs, test_labels):
        """
        Evaluate results.

        :param test_outputs:
        :param test_labels:
        :return:
        """
        for i, curr_model in enumerate(self.models):
            print "Evaluating model " + self.s_type + "\n" + str(curr_model)
            scores = cross_val_score(curr_model, test_outputs, test_labels, cv=self.nfolds,
                                     scoring=metrics.make_scorer(metrics.auc))
            print "Mean AUC: " + str(scores.mean())



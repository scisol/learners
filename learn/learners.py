"""
Code for machine learning.
"""
import os
import copy
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb


class StackedLearner(object):
    def __init__(self, base_learners_dict=None, stacked_learner="logistic", param_dict={'C': 10},
                 nfolds=5, results_dir=None):
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
            self.stacked_learner = SVC(kernel="linear", probability=True)
        elif stacked_learner == "rbf_svm":
            self.stacked_learner = SVC(kernel="rbf", probability=True)
        else:
            raise RuntimeError("Incorrect choice of stacked learner.")

        if results_dir is None:
            results_dir = "."
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.param_dict = param_dict
        self.tr_preds = []
        self.tr_probs = []
        self.tr_aucs = []
        self.models = []
        self.model_name = []
        self.labels = []
        self.results_dir = []
        self.classes_ = []
        self.splitter = StratifiedKFold(n_splits=self.nfolds)

    def fit_from_arrays(self, train_outputs, train_labels):
        """

        :param train_outputs: n x l numpy array, where l is number of learners, and each attribute is
                              a probability from a learner on its respective sample.
        :param train_labels: labels for the samples.
        :return:
        """
        # assert set(np.unique(train_labels).tolist()) == set([label_dict[v] for v in label_dict.keys()])
        classes = np.unique(train_labels)

        if self.s_type == "logistic" or self.s_type == "linear_svm":
            C_list = self.param_dict['C']
            for curr_C in C_list:
                # get train test split
                auc_list = []
                for train_index, test_index in self.splitter.split(train_outputs, train_labels):
                    curr_learner = copy.copy(self.stacked_learner)
                    curr_learner.C = curr_C
                    X_train, X_test = train_outputs[train_index], train_outputs[test_index]
                    y_train, y_test = train_labels[train_index], train_labels[test_index]
                    curr_learner.fit(X_train, y_train)
                    self.tr_preds.append(curr_learner.predict(X_test))
                    curr_probs = curr_learner.predict_proba(X_test)
                    self.tr_probs.append(curr_probs)

                    # compute ROC curve for each label
                    for i in xrange(classes.shape[0]):
                        fpr, tpr, thresholds = metrics.roc_curve(y_test, curr_probs[:, i], pos_label=classes[i])
                        auc_list.append(metrics.auc(fpr, tpr))

                self.models.append(curr_learner)
                self.model_name.append(self.s_type + "_C_" + str(curr_C))
                print "Mean AUC: " + str(np.mean(np.array(auc_list)))

    def evaluate(self, test_outputs, test_labels, op_point=1.0, label_dict=None, save_roc=True):
        """
        Evaluate results. Generate ROC curves based on an FPR operating point.

        :param test_outputs:
        :param test_labels:
        :param op_point:
        :param label_dict:
        :param save_roc:
        :return:
        """
        classes = np.unique(test_labels)
        if label_dict is None:
            label_dict = {}
            for i in classes:
                label_dict[str(i)] = i

        nclasses = classes.shape[0]
        plot_colors = gencolorarray(nclasses)

        for i, curr_model in enumerate(self.models):
            print "Evaluating model " + self.s_type + "\n" + str(curr_model)
            curr_probs = curr_model.predict_proba(test_outputs)
            plt.figure()
            aucs = []
            # compute ROC curve for each label
            for j in xrange(classes.shape[0]):
                fpr, tpr, thresholds = metrics.roc_curve(test_labels, curr_probs[:, j], pos_label=classes[j])
                label_name = get_label(label_dict, classes[j])
                plt.plot(fpr, tpr, label=label_name, c=plot_colors[j], linewidth=3.0)
                plt.xlim([0, op_point])
                plt.xlabel('FPR')
                plt.ylabel('TDR')
                aucs.append(metrics.auc(fpr, tpr).mean())
            print "AUCs: " + str(aucs)
            plt.legend()
            plt.title("ROC curves for " + self.model_name[i])


def get_label(label_dict, curr_label):
    for curr_key in label_dict.keys():
        if label_dict[curr_key] == curr_label:
            return curr_key
    print "Key not found."


def gencolorarray(numcolors):
    # ensure numcolors is an integer by using exception
    color_list = []
    try:
        for i in xrange(1, numcolors + 1):
            p_color = float(i) / numcolors
            color_val = hsv_to_rgb(p_color, 1, 1)
            color_list.append(color_val)
    except:
        print "numcolors must be an integer\n"

    return color_list


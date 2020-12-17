import numpy as np
import pandas as pd
import graphviz
from IPython.display import Image
import matplotlib.pyplot as plt

from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
import abc

def plot(xs, ys, labels, title='', x_label='', y_label=''):
    for x, y, label in zip(xs, ys, labels):
        plt.ylim((0.5, 0.55))
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()

def train_valid_test_split(all_x, all_y, train_size, valid_size, test_size):
    """
    Generate the train, validation, and test dataset.

    Parameters
    ----------
    all_x : DataFrame
        All the input samples
    all_y : Pandas Series
        All the target values
    train_size : float
        The proportion of the data used for the training dataset
    valid_size : float
        The proportion of the data used for the validation dataset
    test_size : float
        The proportion of the data used for the test dataset

    Returns
    -------
    x_train : DataFrame
        The train input samples
    x_valid : DataFrame
        The validation input samples
    x_test : DataFrame
        The test input samples
    y_train : Pandas Series
        The train target values
    y_valid : Pandas Series
        The validation target values
    y_test : Pandas Series
        The test target values
    """
    assert train_size >= 0 and train_size <= 1.0
    assert valid_size >= 0 and valid_size <= 1.0
    assert test_size >= 0 and test_size <= 1.0
    assert train_size + valid_size + test_size >= 0.9999
    assert train_size + valid_size + test_size <= 1.0001
    
    train_size_loc = int(all_x.shape[0]*train_size)
    valid_size_loc = int(all_x.shape[0]*valid_size) + train_size_loc
    
    x_train = all_x[:train_size_loc]
    x_valid = all_x[train_size_loc:valid_size_loc]
    x_test = all_x[valid_size_loc:]
    y_train = all_y[:train_size_loc]
    y_valid = all_y[train_size_loc:valid_size_loc]
    y_test = all_y[valid_size_loc:]
        
    return x_train, x_valid, x_test, y_train, y_valid, y_test

class NoOverlapVoterAbstract(VotingClassifier):
    @abc.abstractmethod
    def _calculate_oob_score(self, classifiers):
        raise NotImplementedError
        
    @abc.abstractmethod
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        raise NotImplementedError
    
    def __init__(self, estimator, voting='soft', n_skip_samples=4):
        # List of estimators for all the subsets of data
        estimators = [('clf'+str(i), estimator) for i in range(n_skip_samples + 1)]
        
        self.n_skip_samples = n_skip_samples
        super().__init__(estimators, voting)
    
    def fit(self, X, y, sample_weight=None):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        
        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(X, y, clone_clfs, self.n_skip_samples)
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))
        self.oob_score_ = self._calculate_oob_score(self.estimators_)
        
        return self

def calculate_oob_score(classifiers):
    '''
    Calculate the mean out-of-bag score from the classifiers.
    '''
    oob_score = 0
    for clf in classifiers:
        oob_score += clf.oob_score_ 
    return oob_score / len(classifiers)

def non_overlapping_estimators(x, y, classifiers, n_skip_samples):
    '''
    Fit the classifiers to non overlapping data.

    Parameters
    ----------
    x : [DataFrame] The input samples
    y : [Pandas Series] The target values
    '''
    fit_classifiers = []
    
    for i in range(n_skip_samples):
        fit_classifiers.append(
            classifiers[i].fit(x[i::n_skip_samples], y[i::n_skip_samples])
        )

    return fit_classifiers

class NoOverlapVoter(NoOverlapVoterAbstract):
    def _calculate_oob_score(self, classifiers):
        return calculate_oob_score(classifiers)
        
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        return non_overlapping_estimators(x, y, classifiers, n_skip_samples)

def train_model(alpha_factors, features, target_label, clf_parameters, train):
    if train:
        temp = alpha_factors.dropna().copy()
        X = temp[features]
        y = temp[target_label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        clf = RandomForestClassifier(**clf_parameters)
        clf_nov = NoOverlapVoter(clf)
        clf_nov.fit(X_train, y_train)

        train_score = clf_nov.score(X_train, y_train.values)
        test_score = clf_nov.score(X_test, y_test.values)
        oob_score = clf_nov.oob_score_

        # Re-training
        clf_nov.fit(X, y)
        train_score_rt = clf_nov.score(X, y.values)
        oob_score_rt = clf_nov.oob_score_

        return [clf_nov, train_score, test_score, oob_score, train_score_rt, oob_score_rt]
    else:
        return None

def plot_tree_classifier(clf, feature_names=None):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        rotate=True)

    return Image(graphviz.Source(dot_data).pipe(format='png'))

def rank_features_by_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    max_feature_name_length = max([len(feature) for feature in feature_names])

    print('      Feature{space: <{padding}}      Importance'.format(padding=max_feature_name_length - 8, space=' '))

    for x_train_i in range(len(importances)):
        print('{number:>2}. {feature: <{padding}} ({importance})'.format(
            number=x_train_i + 1,
            padding=max_feature_name_length,
            feature=feature_names[indices[x_train_i]],
            importance=importances[indices[x_train_i]]))
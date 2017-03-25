#!/usr/bin/python
# coding=utf8
# author=zhtsh

import sys
import logging
from os import path
import numpy as np

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../libsvm')))
from svmutil import *


class ClassifierStrategy(object):

    """
    base classifier strategy class
    """

    def train(self, context):
        pass

    def predict(self, test_x):
        return np.zeros(test_x.shape[0])

    def evaluate(self, y, p_label):
        yy = [value for value in y]
        label = [value for value in p_label]
        acc, mse, scc = evaluations(yy, label)
        return (acc, mse, scc)

    def save_model(self, model_path):
        pass

    def load_model(self, model_path):
        pass

class LRClassifierStrategy(ClassifierStrategy):

    """
    logistic regression classifier strategy class
    """

    def __init__(self,
                 alpha=0.03,
                 iterations=100,
                 threshold=0.5):
        self._alpha = alpha
        self._iterations = iterations
        self._theta = None
        self._threshold = threshold

    def train(self, context):
        (x, y) = context.get_samples()
        (m, n) = x.shape
        self._theta = np.zeros(n)
        theta = np.zeros(n)
        for i in range(self._iterations):
            cost, gradient = self._cost_function(self._theta, x, y)
            logging.info('iteration: %d, cost: %f' % (i+1, cost))
            for j in range(n):
                theta[j] = self._theta[j] - self._alpha*gradient[j]
            self._theta[:] = theta[:]

    def predict(self, test_x):
        probability = self._hypothesis(x)
        return 1 if probability>=0.5 else 0

    def _cost_function(self, theta, x, y):
        m, n = x.shape
        cost = 0.0
        gradient = np.zeros(n)
        for i in range(m):
            h = self._hypothesis(theta, x[i])
            cost += (-y[i]*np.log(h) -
                      (1-y[i])*np.log(1-h))
            for j in range(n):
                gradient[j] += (h - y[i]) * x[i][j]
        cost = cost / m
        return (cost, gradient)

    def _hypothesis(self, x):
        return 1.0 / (1.0 + np.exp(-np.inner(self._theta, x)))


class NNClassifierStrategy(ClassifierStrategy):

    """
    neural network classifier strategy class
    """

    def train(self, context):
        pass

    def predict(self, test_x):
        pass


class SVMClassifierStrategy(ClassifierStrategy):

    """
    support vector machine classifier strategy class
    """

    # svm type definition
    C_SVC = 0   # multi-class classification
    NU_SVC = 1  # multi-class classification
    ONE_CLASS = 2   # binary class classification
    EPSILON_SVR = 3 # regression
    NU_SVR = 4  # regression

    # kernel type definition
    LINEAR = 0  # linear: u'*v
    POLYNOMIAL = 1  # polynomial: (gamma*u'*v + coef0)^degree
    GAUSSIAN = 2    #  radial basis function: exp(-gamma*|u-v|^2)
    SIGMOID = 3 # sigmoid: tanh(gamma*u'*v + coef0)
    PRECOMPUTED_KERNEL = 4  # precomputed kernel (kernel values in training_set_file)

    def __init__(self,
                 svm_type=C_SVC,
                 kernel_type=GAUSSIAN,
                 degree=3,
                 gamma=None,
                 coef0=0,
                 cost=1,
                 nu=0.5,
                 epsilon_svr=0.1,
                 cachesize=100,
                 epsilon=0.001,
                 shrinking=0,
                 probability_estimates=0,
                 weight=None,
                 n_fold=None):
        self._prob = None
        param = ''
        if svm_type != SVMClassifierStrategy.C_SVC:
            param = '-s %d' % svm_type
        if kernel_type != SVMClassifierStrategy.GAUSSIAN:
            param += (' -t %d' % kernel_type)
        if degree != 3:
            param += (' -d %d' % degree)
        if gamma:
            param += (' -g %f' % gamma)
        if coef0 != 0:
            param += (' -r %d' % coef0)
        if cost != 1:
            param += (' -c %d' % cost)
        param += (' -n %f -p %f' % (nu, epsilon))
        if cachesize != 100:
            param += (' -m %d' % cachesize)
        param += (' -e %f -h %d -b %d' % (epsilon, shrinking, probability_estimates))
        if weight:
            for key, value in weight:
                param += (' -w%d %d' % (key, value))
        if n_fold:
            param += (' -v %d' % n_fold)
        self._param = svm_parameter(param)
        self._model = None

    def train(self, context):
        (features, labels) = context.get_samples()
        x = [[value for value in feature] for feature in features]
        y = [label for label in labels]
        self._prob = svm_problem(y, x)
        logging.info('training svm model...')
        self._model = svm_train(self._prob, self._param)

    def predict(self, test_x):
        y = [0 for i in range(test_x.shape[0])]
        x = [[value for value in feature] for feature in test_x]
        p_label, _, _ = svm_predict(y, x, self._model)
        return np.array(p_label)

    def save_model(self, model_path):
        logging.info('save model to path: %s' % model_path)
        svm_save_model(model_path, self._model)

    def load_model(self, model_path):
        logging.info('load model from path: %s' % model_path)
        self._model = svm_load_model(model_path)
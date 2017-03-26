#!/usr/bin/python
# coding=utf8
# author=zhtsh

import sys
import json
import logging
import numpy as np
from os import path
from random import uniform

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../libsvm')))
from svmutil import *


class ClassifierStrategy(object):

    """
    base classifier strategy class
    """

    BGD = 0 # Batch gradient descent
    SGD = 1 # Stochastic gradient descent

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
                 epsilon=0.01,
                 regularization=False,
                 optimization=ClassifierStrategy.BGD,
                 threshold=0.5):
        self._alpha = alpha
        self._iterations = iterations
        self._epsilon = epsilon
        self._regularization = regularization
        self._optimization = optimization
        self._theta = None
        self._threshold = threshold

    def train(self, context):
        (x, y) = context.get_samples()
        if self._optimization == ClassifierStrategy.SGD:
            self._train_sgd(x, y)
        else:
            self._train_bgd(x, y)

    def _train_bgd(self, x, y):
        (m, n) = x.shape
        self._theta = np.ones(n)
        for i in range(self._iterations):
            cost, gradient = self._cost_function_bgd(x, y)
            if cost <= self._epsilon:
                logging.info('cost < %f, stop iteration' % self._epsilon)
            for j in range(n):
                self._theta[j] = self._theta[j] - self._alpha*gradient[j]
            logging.info('iteration: %d, cost: %f' % (i+1, cost))
            logging.info('gradient: %s' % str(gradient))
            logging.info('theta: %s' % str(self._theta))

    def _train_sgd(self, x, y):
        (m, n) = x.shape
        self._theta = np.zeros(n)
        for i in range(self._iterations):
            for k in range(m):
                self._alpha = 10.0 / (100.0 + k)
                index = int(uniform(0, m))
                gradient = self._cost_function_sgd(x[index], y[index])
                # gradient = self._cost_function_sgd(x[k], y[k])
                for j in range(n):
                    self._theta[j] = self._theta[j] - self._alpha*gradient[j]
            logging.info('theta: %s' % str(self._theta))

    def predict(self, test_x):
        probability = [self._hypothesis(x) for x in test_x]
        return np.array([1 if prob>=self._threshold else 0 for prob in probability])

    def _cost_function_bgd(self, x, y):
        m, n = x.shape
        cost = 0.0
        gradient = np.zeros(n)
        for i in range(m):
            h = self._hypothesis(x[i])
            cost += (-y[i]*np.log(h) - (1-y[i])*np.log(1-h))
            for j in range(n):
                gradient[j] += (h - y[i]) * x[i][j]
        cost = cost / m
        gradient = gradient / m
        return (cost, gradient)

    def _cost_function_sgd(self, x, y):
        n = x.size
        gradient = np.zeros(n)
        h = self._hypothesis(x)
        for j in range(n):
            gradient[j] = (h - y) * x[j]
        return gradient

    def _hypothesis(self, x):
        return 1.0 / (1.0 + np.exp(-np.inner(self._theta, x)))

    def save_model(self, model_path):
        logging.info('saving mode to path: %s' % model_path)
        with open(model_path, 'wb') as model_file:
            model_data = {}
            model_data['alpha'] = self._alpha
            model_data['iterations'] = self._iterations
            model_data['epsilon'] = self._epsilon
            model_data['regularization'] = self._regularization
            model_data['optimization'] = self._optimization
            model_data['theta'] = [value for value in self._theta]
            model_data['threshold'] = self._threshold
            json_data = json.dumps(model_data)
            model_file.write(json_data)

    def load_model(self, model_path):
        logging.info('loading model from path: %s' % model_path)
        with open(model_path, 'rb') as model_file:
            model_data = json.load(model_file)
            self._alpha = model_data['alpha']
            self._iterations = model_data['iterations']
            self._epsilon = model_data['epsilon']
            self._regularization = model_data['regularization']
            self._optimization = model_data['optimization']
            self._theta = np.array(model_data['theta'])
            self._threshold = model_data['threshold']


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
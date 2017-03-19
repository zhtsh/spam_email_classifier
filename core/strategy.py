#!/usr/bin/python
# coding=utf8
# author=zhtsh

import logging
import numpy as np

class ClassifierStrategy(object):

    """
    base classifier strategy class
    """

    def train(self, context):
        pass

    def predict(self, x):
        pass


class LRClassifierStrategy(ClassifierStrategy):

    """
    logistic regression classifier strategy class
    """

    def __init__(self, alpha=0.03, iterations=100):
        self._alpha = alpha
        self._iterations = iterations
        self._theta = None

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

    def predict(self, x):
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

    def predict(self, x):
        pass


class SVMClassifierStrategy(ClassifierStrategy):

    """
    support vector machine classifier strategy class
    """

    def train(self, context):
        pass

    def predict(self, x):
        pass
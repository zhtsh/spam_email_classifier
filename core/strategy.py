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
                 lambda_factor=100,
                 optimization=ClassifierStrategy.BGD,
                 threshold=0.5):
        self._alpha = alpha
        self._iterations = iterations
        self._epsilon = epsilon
        self._regularization = regularization
        self._lambda_factor = lambda_factor if regularization else 0
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
        self._theta = np.zeros(n)
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
        cost = 0
        gradient = np.zeros(n)
        for i in range(m):
            h = self._hypothesis(x[i])
            cost += (-y[i]*np.log(h) - (1-y[i])*np.log(1-h))
            for j in range(n):
                gradient[j] += (h - y[i]) * x[i][j]
                if j != 0:
                    gradient[j] += self._lambda_factor * self._theta[j]
        theta = np.array(self._theta)
        theta[0] = 0
        cost = cost / m + self._lambda_factor / (2.0 * m) * theta.dot(theta)
        gradient = (gradient + self._lambda_factor * theta) / m
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
            model_data['lambda'] = self._lambda_factor
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
            self._lambda_factor = model_data['lambda']
            self._optimization = model_data['optimization']
            self._theta = np.array(model_data['theta'])
            self._threshold = model_data['threshold']


class NNClassifierStrategy(ClassifierStrategy):

    """
    neural network classifier strategy class
    simple neural network, only one hidden layer
    """

    def __init__(self,
                 hidden_layer_units=100,
                 alpha=0.03,
                 iterations=10,
                 epsilon=0.01,
                 regularization=False,
                 lambda_factor=100,
                 threshold=0.5):
        self._hidden_layer_units = hidden_layer_units
        self._alpha = alpha
        self._iterations = iterations
        self._epsilon = epsilon
        self._regularization = regularization
        self._lambda_factor = lambda_factor if regularization else 0
        self._threshold = threshold
        self._features_count = 0
        self._theta = None
        self._theta_size = 0
        self._theta1 = None
        self._theta2 = None
        self._gradient = None
        self._gradient1 = None
        self._gradient2 = None
        self._a1 = None
        self._a2 = None
        self._a3 = None
        self._delta2 = None
        self._delta3 = None
        self._Delta1 = None
        self._Delta2 = None

    def train(self, context):
        features, labels = context.get_samples()
        self._features_count = features.shape[1]
        self._theta_size = self._features_count * self._hidden_layer_units + \
                            (self._hidden_layer_units + 1) * 1
        self._initialize_all()
        # checking whether gradient is correct,
        # compare it with numerical gradient
        self._back_propagation(features, labels)
        if not self._checking_gradient():
            logging.info('gradient is not equal to numerical value approximatly')
            return
        # use gradient descent to compute theta
        for i in range(self._iterations):
            self._back_propagation(features, labels)

    def predict(self, test_x):
        m, n = test_x.shape
        test_y = np.zeros(m)
        self._roll_theta(self._theta)
        for i in range(m):
            self._forward_propagation(test_x[i])
            test_y[i] = 1 if self._a3>=0.5 else 0
        return test_y

    def save_model(self, model_path):
        json_data = {}
        json_data['hidden_layer_units'] = self._hidden_layer_units
        json_data['alpha'] = self._alpha
        json_data['iterations'] = self._iterations
        json_data['epsilon'] = self._epsilon
        json_data['regularization'] = self._regularization
        json_data['lambda_factor'] = self._lambda_factor
        json_data['features_count'] = self._features_count
        theta = [value for value in self._theta]
        json_data['theta'] = theta
        gradient = [value for value in self._gradient]
        json_data['gradient'] = gradient
        json_data['threshold'] = self._threshold
        with open(model_path, 'wb') as model_file:
            model_file.write(json.dumps(json_data))

    def load_model(self, model_path):
        json_data = json.load(open(model_path, 'rb'))
        self._hidden_layer_units = json_data['hidden_layer_units']
        self._alpha = json_data['alpha']
        self._iterations = json_data['iterations']
        self._epsilon = json_data['epsilon']
        self._regularization = json_data['regularization']
        self._lambda_factor = json_data['lambda_factor']
        self._features_count = json_data['features_count']
        self._threshold = json_data['threshold']
        self._theta = np.array(json_data['theta'])
        self._theta_size = self._theta.size
        self._gradient = np.array(json_data['gradient'])
        self._initialize_matrixs()

    def _initialize_all(self):
        """
        initialize all algorithm parameters
        """
        # initialize theta in [-epsilon, epsilon]
        self._theta = np.random.random(self._theta_size) * 2 * self._epsilon - self._epsilon
        self._gradient = np.zeros(self._theta_size)
        self._initialize_matrixs()

    def _initialize_matrixs(self):
        self._theta1 = np.zeros((self._hidden_layer_units, self._features_count))
        self._theta2 = np.zeros((1, self._hidden_layer_units + 1))
        self._Delta1 = np.zeros((self._hidden_layer_units, self._features_count))
        self._Delta2 = np.zeros((1, self._hidden_layer_units + 1))
        self._gradient1 = np.zeros((self._hidden_layer_units, self._features_count))
        self._gradient2 = np.zeros((1, self._hidden_layer_units + 1))

    def _roll_theta(self, theta):
        """
        split theta vector into matrix
        """
        for i in range(self._hidden_layer_units):
            for j in range(self._features_count):
                index = i * self._hidden_layer_units + j * self._features_count
                self._theta1[i][j] = theta[index]
        for i in range(self._hidden_layer_units + 1):
            index = self._hidden_layer_units * self._features_count + i
            self._theta2[0][i] = theta[index]

    def _unroll_gradient(self):
        """
        unroll gradient matrix into vector
        """
        for i in range(self._hidden_layer_units):
            for j in range(self._features_count):
                index = i * self._hidden_layer_units + j * self._features_count
                self._theta1[i][j] = self._theta[index]
                self._gradient[index] = self._gradient1[i][j]
        for i in range(self._hidden_layer_units + 1):
            index = self._hidden_layer_units * self._features_count + i
            self._gradient[index] = self._gradient2[0][i]

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(z))

    def _cost_function(self, x, y, theta):
        m, n = x.shape
        cost = 0
        self._roll_theta(theta)
        for i in range(m):
            self._forward_propagation(x[i])
            cost += (-y[i]*np.log(self._a3) - (1-y[i])*np.log(1-self._a3))
        cost = cost / m
        theta1 = self._theta1 * self._theta1
        theta1[:][0] = 0
        cost += self._lambda_factor / (2.0 * m) * theta1.sum()
        theta2 = self._theta2 * self._theta2
        theta2[:][0] = 0
        cost += self._lambda_factor / (2.0 * m) * theta2.sum()
        return cost

    def _forward_propagation(self, x):
        self._a1 = x
        z2 = self._theta1.dot(self._a1)
        self._a2 = np.ones(self._hidden_layer_units+1)
        self._a2[1:] = self._sigmoid(z2)
        z3 = self._theta2.dot(self._a2)
        self._a3 = self._sigmoid(z3)

    def _back_propagation(self, features, labels):
        """
        back propagation algorithm, iterate one time to compute gradient
        :param features: samples features matrix
        :param labels: samples labels vector
        :return:
        """
        m, n = features.shape
        self._Delta1 = 0
        self._Delta2 = 0
        self._gradient1 = 0
        self._gradient2 = 0
        self._roll_theta(self._theta)
        for i in range(m):
            self._forward_propagation(features[i])
            self._delta3 = self._a3 - labels[i]
            self._delta2 = (self._theta2.T.dot(self._delta3)) * self._a2 * (1 - self._a2)
            self._Delta1 = self._Delta1 + self._delta2.dot(self._a1.T)
            self._Delta2 = self._Delta2 + self._delta3.dot(self._a2.T)
        for i in range(self._hidden_layer_units):
            for j in range(n):
                self._gradient1[i][j] = self._Delta1[i][j] / m
                if j != 0 and self._regularization:
                    self._gradient1[i][j] += (self._lambda_factor / m * self._theta1[i][j])
        for j in range(self._hidden_layer_units+1):
            self._gradient2[0][j] = self._Delta2[0][j] / m
            if j != 0 and self._regularization:
                self._gradient2[0][j] += (self._lambda_factor / m * self._theta2[0][j])
        self._unroll_gradient(n)

    def _checking_gradient(self, x, y):
        gradient_approx = np.zeros(self._theta_size)
        for i in range(self._theta_size):
            theta_plus = np.array(self._theta)
            theta_plus[i] += self._epsilon
            theta_minus = np.array(self._theta)
            theta_minus[i] -= self._epsilon
            gradient_approx[i] = (self._cost_function(x, y, theta_plus) - \
                self._cost_function(x, y, theta_minus)) / (2 * self._epsilon)
            if abs(self._gradient[i] - gradient_approx[i]) > 0.01:
                return False
        return True


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
#!/usr/bin/python
# coding=utf8
# author=zhtsh


class SpamClassifierContext(object):

    """
    spam email classifier context class
    """

    def __init__(self, classifier_strategy):
        self._classifier_strategy = classifier_strategy
        self.x = None
        self.y = None

    def train(self):
        self._classifier_strategy.train(self)

    def predict(self, x):
        return self._classifier_strategy.predict(x)

    def load_samples(self, postive_samples_dir, negative_samples_dir):
        self.x = None
        self.y = None

    def get_samples(self):
        return (self.x, self.y)
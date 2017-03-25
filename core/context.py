#!/usr/bin/python
# coding=utf8
# author=zhtsh

import sys
import logging
import numpy as np
from os import path, listdir

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../core')))
from utils import EmailETLHelper

class SpamClassifierContext(object):

    """
    spam email classifier context class
    """

    def __init__(self, classifier_strategy):
        self._classifier_strategy = classifier_strategy
        self._dictionary = {}
        self._features = None
        self._labels = None

    def train(self):
        self._classifier_strategy.train(self)

    def predict(self, test_x):
        return self._classifier_strategy.predict(test_x)

    def evaluate(self, y, p_label):
        return self._classifier_strategy.evaluate(y, p_label)

    def load_samples(self, postive_samples_dir, negative_samples_dir):
        spam_files = [path.join(postive_samples_dir, f) for f in listdir(postive_samples_dir)
                      if path.isfile(path.join(postive_samples_dir, f))]
        nonspam_files = [path.join(negative_samples_dir, f) for f in listdir(negative_samples_dir)
                         if path.isfile(path.join(negative_samples_dir, f))]
        m = len(spam_files) + len(nonspam_files)
        etl_helper = EmailETLHelper.instance()
        n = etl_helper.get_feature_count()
        self._features = np.zeros((m, n))
        self._labels = np.zeros(m)
        logging.info('loading spam samples...')
        index = self._set_samples_values(spam_files, 0, 1)
        logging.info('loading non spam samples...')
        index = self._set_samples_values(nonspam_files, index, 0)

    def save_model(self, model_path):
        self._classifier_strategy.save_model(model_path)

    def load_model(self, model_path):
        self._classifier_strategy.load_model(model_path)

    def _set_samples_values(self, files, index, label):
        etl_helper = EmailETLHelper.instance()
        for file_path in files:
            self._features[index] = etl_helper.get_feature_from_body_file(file_path)
            self._labels[index] = label
            index += 1
        return index

    def get_samples(self):
        return (self._features, self._labels)
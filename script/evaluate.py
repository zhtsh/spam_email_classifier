#!/usr/bin/python
# coding=utf8
# author=zhtsh

import sys
import logging
from os import path

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../core')))
from context import SpamClassifierContext
from strategy import SVMClassifierStrategy

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    postive_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/spam_test'))
    negative_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/nonspam_test'))
    model_path = path.abspath(path.join(path.dirname(__file__), '../data/svm.model'))
    classifier_strategy = SVMClassifierStrategy()
    classifier_context = SpamClassifierContext(classifier_strategy)
    classifier_context.load_samples(postive_samples_dir, negative_samples_dir)
    classifier_context.load_model(model_path)
    test_x, test_y = classifier_context.get_samples()
    acc, mse, scc = classifier_context.evaluate(test_x, test_y)
    logging.info('acc: %f, mse: %f, scc: %f' % (acc, mse, scc))
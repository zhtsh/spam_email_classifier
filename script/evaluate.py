#!/usr/bin/python
# coding=utf8
# author=zhtsh

import sys
import logging
from os import path
from optparse import OptionParser

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../core')))
from context import SpamClassifierContext
from strategy import SVMClassifierStrategy
from strategy import LRClassifierStrategy
from strategy import NNClassifierStrategy

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = OptionParser()
    parser.add_option("-t", "--type", dest="model_type", default="lr",
                      help="model type: lr, svm, nn")
    parser.add_option("-e", "--evaluate", dest="evaluate", default="test",
                      help="evaluate data set: train or test")
    (options, args) = parser.parse_args()
    if options.model_type == "lr":
        model_path = path.abspath(path.join(path.dirname(__file__), '../data/lr.model'))
        classifier_strategy = LRClassifierStrategy()
    elif options.model_type == "svm":
        model_path = path.abspath(path.join(path.dirname(__file__), '../data/svm.model'))
        classifier_strategy = SVMClassifierStrategy()
    elif options.model_type == "nn":
        model_path = path.abspath(path.join(path.dirname(__file__), '../data/nn.model'))
        classifier_strategy = NNClassifierStrategy()
    else:
        parser.print_help()
        sys.exit(1)
    if options.evaluate == "train":
        postive_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/spam_train'))
        negative_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/nonspam_train'))
    else:
        postive_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/spam_test'))
        negative_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/nonspam_test'))
    classifier_context = SpamClassifierContext(classifier_strategy)
    classifier_context.load_samples(postive_samples_dir, negative_samples_dir)
    classifier_context.load_model(model_path)
    test_x, test_y = classifier_context.get_samples()
    acc, mse, scc = classifier_context.evaluate(test_x, test_y)
    logging.info('acc: %f, mse: %f, scc: %f' % (acc, mse, scc))
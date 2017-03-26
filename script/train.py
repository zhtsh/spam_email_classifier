#!/usr/bin/python
# coding=utf8
# author=zhtsh

import sys
import logging
from os import path
from optparse import OptionParser

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../core')))
from context import SpamClassifierContext
from strategy import ClassifierStrategy
from strategy import SVMClassifierStrategy
from strategy import LRClassifierStrategy
from strategy import NNClassifierStrategy

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = OptionParser()
    parser.add_option("-t", "--type", dest="model_type", default="lr",
                      help="model type: lr, svm, nn")
    parser.add_option("-r", "--regularization", dest="regularization", default="0",
                      help="value: 0 or 1, whether to use regularization items, svm ignore this option")
    parser.add_option("-o", "--optimization", dest="optimization", default="bgd",
                      help="gradient descent type: bgd, sgd")
    (options, args) = parser.parse_args()
    regularization = True if options.regularization == "1" else False
    optimization = ClassifierStrategy.SGD if options.optimization=="sgd" else ClassifierStrategy.BGD
    if options.model_type == "lr":
        model_path = path.abspath(path.join(path.dirname(__file__), '../data/lr.model'))
        if optimization == ClassifierStrategy.SGD:
            classifier_strategy = LRClassifierStrategy(iterations=30,
                                                       regularization=regularization,
                                                       optimization=optimization)
        else:
            classifier_strategy = LRClassifierStrategy(regularization=regularization,
                                                       optimization=optimization)
    elif options.model_type == "svm":
        model_path = path.abspath(path.join(path.dirname(__file__), '../data/svm.model'))
        classifier_strategy = SVMClassifierStrategy(svm_type=SVMClassifierStrategy.C_SVC,
                                                    kernel_type=SVMClassifierStrategy.LINEAR,
                                                    cost=100,
                                                    cachesize=1024)
    elif options.model_type == "nn":
        model_path = path.abspath(path.join(path.dirname(__file__), '../data/nn.model'))
        classifier_strategy = NNClassifierStrategy()
    else:
        parser.print_help()
        sys.exit(1)
    postive_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/spam_train'))
    negative_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/nonspam_train'))
    classifier_context = SpamClassifierContext(classifier_strategy)
    classifier_context.load_samples(postive_samples_dir, negative_samples_dir)
    classifier_context.train()
    classifier_context.save_model(model_path)
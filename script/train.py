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
                      help="gradient descent type: bgd, sgd, nn and svm ignore this option")
    parser.add_option("-i", "--iterations", dest="iterations", default=20,
                      help="iteration count")
    parser.add_option("-k", "--tfidf", dest="tfidf", default="0",
                      help="whether to use tfidf model")
    parser.add_option("-g", "--kernel", dest="kernel", default="0",
                      help="svm kernel type: 0:linear, 1:polynomial, 2:gaussian, 3:sigmoid, 4:precomputed")
    (options, args) = parser.parse_args()
    regularization = True if options.regularization == "1" else False
    optimization = ClassifierStrategy.SGD if options.optimization=="sgd" else ClassifierStrategy.BGD
    tfidf = True if options.tfidf == "1" else False
    try:
        iterations = int(options.iterations)
    except:
        iterations = 20
    if options.model_type == "lr":
        model_path = path.abspath(path.join(path.dirname(__file__), '../data/lr.model'))
        classifier_strategy = LRClassifierStrategy(iterations=iterations,
                                                   regularization=regularization,
                                                   optimization=optimization)
    elif options.model_type == "svm":
        try:
            kernel_type = int(options.kernel)
        except:
            parser.print_help()
            sys.exit(1)
        if kernel_type < 0 or kernel_type > 4:
            parser.print_help()
            sys.exit(1)
        model_path = path.abspath(path.join(path.dirname(__file__), '../data/svm.model'))
        classifier_strategy = SVMClassifierStrategy(svm_type=SVMClassifierStrategy.C_SVC,
                                                    kernel_type=kernel_type,
                                                    cost=100,
                                                    cachesize=1024)
    elif options.model_type == "nn":
        model_path = path.abspath(path.join(path.dirname(__file__), '../data/nn.model'))
        classifier_strategy = NNClassifierStrategy(hidden_layer_units=100,
                                                   iterations=iterations)
    else:
        parser.print_help()
        sys.exit(1)
    postive_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/spam_train'))
    negative_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/nonspam_train'))
    classifier_context = SpamClassifierContext(classifier_strategy, tfidf)
    classifier_context.load_samples(postive_samples_dir, negative_samples_dir)
    classifier_context.train()
    classifier_context.save_model(model_path)
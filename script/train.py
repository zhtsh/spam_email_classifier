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
    postive_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/preprocess_spam'))
    negative_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/preprocess_nonspam'))
    model_path = path.abspath(path.join(path.dirname(__file__), '../data/svm.model'))
    classifier_strategy = SVMClassifierStrategy(svm_type=SVMClassifierStrategy.ONE_CLASS,
                                                kernel_type=SVMClassifierStrategy.LINEAR,
                                                cost=100,
                                                cachesize=1024)
    classifier_context = SpamClassifierContext(classifier_strategy)
    classifier_context.load_samples(postive_samples_dir, negative_samples_dir)
    classifier_context.train()
    classifier_context.save_model(model_path)
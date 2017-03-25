#!/usr/bin/python
# coding=utf8
# author=zhtsh

import sys
import logging
import numpy as np
from os import path

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../core')))
from context import SpamClassifierContext
from strategy import SVMClassifierStrategy
from utils import EmailETLHelper

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python predict.py mime_email_file')
        sys.exit(1)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model_path = path.abspath(path.join(path.dirname(__file__), '../data/svm.model'))
    classifier_strategy = SVMClassifierStrategy()
    classifier_context = SpamClassifierContext(classifier_strategy)
    classifier_context.load_model(model_path)
    etl_helper = EmailETLHelper.instance()
    feature = etl_helper.get_feature_from_email(sys.argv[1])
    test_x = np.array([feature])
    test_lable = classifier_context.predict(test_x)
    if test_lable[0]:
        print('This is a spam email')
    else:
        print('This is a noraml email')
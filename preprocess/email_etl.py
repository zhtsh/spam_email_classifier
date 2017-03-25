#!/usr/bin/python
# coding=utf8
# author=zhtsh

import os
import sys
import logging
from os import path
from random import shuffle
from shutil import copyfile

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../core')))
from utils import EmailETLHelper

def split_samples(files, train_samples_dir, test_samples_dir):
    """
    random split samples into train set and test set
    train set: 80%, test set: 20%
    :param files: all email samples files
    :param train_samples_dir: train samples output dir
    :param test_samples_dir: test samples output dir
    """

    shuffle(files)
    index = int(len(files) * 0.8)
    for i, file in enumerate(files):
        file_name = path.basename(file)
        if i < index:
            file_path = path.join(train_samples_dir, file_name)
        else:
            file_path = path.join(test_samples_dir, file_name)
        logging.info('copying file: %s' % file_name)
        copyfile(file, file_path)

def temp_process():
    root_dir = path.abspath(path.join(path.dirname(__file__), '../'))
    preprocess_spam_dir = path.join(root_dir, 'data/preprocess_spam')
    preprocess_nonspam_dir = path.join(root_dir, 'data/preprocess_nonspam')
    spam_files = []
    non_spam_files = []
    for child_item in os.walk(preprocess_spam_dir):
        for file_name in child_item[2]:
            file_path = path.join(child_item[0], file_name)
            spam_files.append(file_path)
    for child_item in os.walk(preprocess_nonspam_dir):
        for file_name in child_item[2]:
            file_path = path.join(child_item[0], file_name)
            non_spam_files.append(file_path)
    spam_train_dir = path.join(root_dir, 'data/spam_train')
    spam_test_dir = path.join(root_dir, 'data/spam_test')
    nonspam_train_dir = path.join(root_dir, 'data/nonspam_train')
    nonspam_test_dir = path.join(root_dir, 'data/nonspam_test')
    split_samples(spam_files, spam_train_dir, spam_test_dir)
    split_samples(non_spam_files, nonspam_train_dir, nonspam_test_dir)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    root_dir = path.abspath(path.join(path.dirname(__file__), '../'))
    corpus_dir = path.join(root_dir, 'corpus')

    #==================
    # temp process
    temp_process()
    sys.exit(0)
    #==================

    spam_files = []
    non_spam_files = []
    # preprocess spam and non spam email
    for child_item in os.walk(corpus_dir):
        child_dir = child_item[0]
        if child_dir != corpus_dir:
            if path.basename(child_dir).find('spam') != -1:
                preprocess_dir = path.join(root_dir, 'data/preprocess_spam')
                is_spam = True
            else:
                preprocess_dir = path.join(root_dir, 'data/preprocess_nonspam')
                is_spam = False
            for file_name in child_item[2]:
                logging.info('preprocessing file: %s' % file_name)
                file_path = path.join(child_dir, file_name)
                email_body_text = EmailETLHelper.get_body_from_email(file_path)
                if email_body_text:
                    preprocess_path = path.join(preprocess_dir, file_name)
                    preprocess_file = open(preprocess_path, 'wb')
                    preprocess_file.write(email_body_text)
                    preprocess_file.close()
                    if is_spam:
                        spam_files.append(preprocess_path)
                    else:
                        non_spam_files.append(preprocess_path)

    spam_train_dir = path.join(root_dir, 'data/spam_train')
    spam_test_dir = path.join(root_dir, 'data/spam_test')
    nonspam_train_dir = path.join(root_dir, 'data/nonspam_train')
    nonspam_test_dir = path.join(root_dir, 'data/nonspam_test')
    # random split samples into train set and test set
    # train set: 80%, test set: 20%
    split_samples(spam_files, spam_train_dir, spam_test_dir)
    split_samples(non_spam_files, nonspam_train_dir, nonspam_test_dir)
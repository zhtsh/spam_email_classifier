#!/usr/bin/python
# coding=utf8
# author=zhtsh

import logging
from os import path, listdir

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    postive_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/preprocess_spam'))
    negative_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/preprocess_nonspam'))
    spam_files = [path.join(postive_samples_dir, f) for f in listdir(postive_samples_dir)
                  if path.isfile(path.join(postive_samples_dir, f))]
    nonspam_files = [path.join(negative_samples_dir, f) for f in listdir(negative_samples_dir)
                     if path.isfile(path.join(negative_samples_dir, f))]
    files = spam_files + nonspam_files
    dictionary = set()
    for file_path in files:
        logging.info('processing file: %s' % file_path)
        with open(file_path, 'rb') as file:
            terms = file.read().strip().split()
            for term in terms:
                dictionary.add(term)
    dictionary_path = path.abspath(path.join(path.dirname(__file__), '../data/dictionary.txt'))
    with open(dictionary_path, 'wb') as dictionary_file:
        for term in dictionary:
            dictionary_file.write(term + '\n')

#!/usr/bin/python
# coding=utf8
# author=zhtsh

import logging
from os import path, listdir
from gensim import corpora

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    postive_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data_1/preprocess_spam'))
    negative_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data_1/preprocess_nonspam'))
    spam_files = [path.join(postive_samples_dir, f) for f in listdir(postive_samples_dir)
                  if path.isfile(path.join(postive_samples_dir, f))]
    nonspam_files = [path.join(negative_samples_dir, f) for f in listdir(negative_samples_dir)
                     if path.isfile(path.join(negative_samples_dir, f))]
    files = spam_files + nonspam_files
    dictionary = corpora.Dictionary()
    documents = []
    for file_path in files:
        with open(file_path, 'rb') as file:
            terms = file.read().strip().split()
            documents.append(terms)
        if len(documents) == 1000:
            logging.info('add %d documents to dictionary' % len(documents))
            dictionary.add_documents(documents)
            documents = []
    if documents:
        logging.info('add %d documents to dictionary' % len(documents))
        dictionary.add_documents(documents)
    dictionary.filter_extremes(no_below=100, no_above=0.5)
    dictionary_path = path.abspath(path.join(path.dirname(__file__), '../data_1/corpus.dictionary'))
    dictionary.save(dictionary_path)
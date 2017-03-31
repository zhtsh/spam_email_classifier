#!/usr/bin/python
# coding=utf8
# author=zhtsh

import logging
from os import path, listdir
from gensim import corpora

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../core')))
from utils import MyCorpus

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    corpus = MyCorpus()
    dictionary = corpora.Dictionary()
    documents = []
    for terms in corpus:
        documents.append(terms)
        if len(documents) == 1000:
            logging.info('add %d documents to dictionary' % len(documents))
            dictionary.add_documents(documents)
            documents = []
    if documents:
        logging.info('add %d documents to dictionary' % len(documents))
        dictionary.add_documents(documents)
    dictionary.filter_extremes(no_below=100, no_above=0.5)
    dictionary_path = path.abspath(path.join(path.dirname(__file__), '../data/corpus.dictionary'))
    dictionary.save(dictionary_path)
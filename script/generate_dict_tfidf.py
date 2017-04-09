#!/usr/bin/python
# coding=utf8
# author=zhtsh

import sys
import logging
from os import path, listdir
from gensim import corpora, models

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../core')))
from utils import MyCorpus

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('generating corpus dictionary...')
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
    dictionary.filter_extremes(no_below=20, no_above=0.7, keep_n=500)
    # dictionary.filter_extremes(no_below=20, no_above=0.7)
    dictionary_path = path.abspath(path.join(path.dirname(__file__), '../data/corpus.dict'))
    dictionary.save(dictionary_path)

    logging.info('generating tfidf model...')
    documents = []
    for terms in corpus:
        documents.append(dictionary.doc2bow(terms))
    tfidf_model = models.TfidfModel(documents)
    tfidf_path = path.abspath(path.join(path.dirname(__file__), '../data/corpus.tfidf_model'))
    tfidf_model.save(tfidf_path)
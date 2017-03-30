#!/usr/bin/python
# coding=utf8
# author=zhtsh

import re
import nltk
import email
import logging
import base64
import numpy as np
from os import path
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from HTMLParser import HTMLParser


# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):

    _contents = []

    def handle_data(self, data):
        data = data.strip()
        data = data.replace('\r', '')
        data = data.replace('\n', '')
        if data:
            self._contents.append(data)

    def get_body(self):
        return ' '.join(self._contents)

class EmailETLHelper(object):

    """
    mime email message etl helper class, it is a singleton class
    """

    @classmethod
    def instance(cls, use_tfidf=False):
        if not hasattr(cls, '_instance'):
            cls._instance = EmailETLHelper(use_tfidf)
        return cls._instance

    def __init__(self, use_tfidf):
        """
        instance initializtion, loading dictionary file
        """

        dictionary_path = path.abspath(path.join(path.dirname(__file__), '../data_1/corpus.dictionary'))
        logging.info('loading dictionary...')
        self._dictionary = corpora.Dictionary.load(dictionary_path)
        # TODO: for tfidf model
        self._use_tfidf = use_tfidf

    def get_feature_count(self):
        """
        get feature count n, including bias 1
        :return: n
        """

        return len(self._dictionary.token2id) + 1

    @classmethod
    def get_body_from_email(cls, email_path):
        """
        preprocess mime email message, return clean email body string
        :param email_path: mime email path
        :return: email body string
        """

        english_stopwords = stopwords.words('english')
        wnl = WordNetLemmatizer()
        words = set()
        for word in brown.words():
            words.add(word.lower())
        email_message = email.message_from_file(open(email_path, 'rb'))
        email_bodies = []
        for part in email_message.walk():
            content_encoding = part.get('Content-Transfer-Encoding')
            content_type = part.get_content_type().lower()
            if content_type == 'text/plain':
                part_content = part.get_payload()
                if content_encoding and content_encoding.lower() == 'base64':
                    part_content = base64.decodestring(part_content)
                email_bodies.append(part_content)
            elif content_type == 'text/html':
                try:
                    part_content = part.get_payload()
                    if content_encoding and content_encoding.lower() == 'base64':
                        part_content = base64.decodestring(part_content)
                    parser = MyHTMLParser()
                    parser.feed(part_content)
                    email_bodies.append(parser.get_body())
                except Exception as e:
                    logging.info(str(e))
            else:
                continue
        email_body_text = ''
        if email_bodies:
            email_body_text = ' '.join(email_bodies)
            email_body_text = re.sub(r'[^a-zA-Z]+', ' ', email_body_text)
            tokens = nltk.word_tokenize(email_body_text)
            filtered_tokens = [wnl.lemmatize(word.lower()) for word in tokens
                               if len(word)>1 and word not in english_stopwords and word in words]
            email_body_text = ' '.join(filtered_tokens)
        return email_body_text

    def get_feature_from_email(self, email_path):
        """
        get feature from mime email message
        :param email_path: mime email path
        :return: np.array feature
        """

        email_body = self.get_body_from_email(email_path)
        return self.get_feature_from_body_string(email_body)

    def get_feature_from_body_file(self, body_path):
        """
        get feature from clean email body file
        :param body_path: email body path
        :return: np.array feature
        """

        email_body = open(body_path, 'rb').read().strip()
        return self.get_feature_from_body_string(email_body)

    def get_feature_from_body_string(self, email_body):
        """
        get feature from clean email body string
        :param body_path: email body path
        :return: np.array feature
        """

        tokens = email_body.strip().split()
        doc_tf = self._dictionary.doc2bow(tokens)
        feature = np.zeros(self.get_feature_count())
        feature[0] = 1
        for token, _ in doc_tf:
            feature[token+1] = 1
        return feature
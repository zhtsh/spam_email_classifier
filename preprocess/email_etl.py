#!/usr/bin/python
# coding=utf8
# author=zhtsh

import os
import sys
import re
import nltk
import email
import logging
import base64
from os import path
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
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


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    root_dir = path.abspath(path.join(path.dirname(__file__), '../'))
    corpus_dir = path.join(root_dir, 'corpus')
    english_stopwords = stopwords.words('english')
    wnl = WordNetLemmatizer()
    words = set()
    for word in brown.words():
        words.add(word.lower())
    for child_item in os.walk(corpus_dir):
        child_dir = child_item[0]
        if child_dir != corpus_dir:
            if path.basename(child_dir).find('spam') != -1:
                preprocess_dir = path.join(root_dir, 'data/preprocess_spam')
            else:
                preprocess_dir = path.join(root_dir, 'data/preprocess_nonspam')
            for file_name in child_item[2]:
                logging.info('preprocessing file: %s' % file_name)
                file_path = path.join(child_dir, file_name)
                email_message = email.message_from_file(open(file_path, 'rb'))
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
                if email_bodies:
                    email_body_text = ' '.join(email_bodies)
                    email_body_text = re.sub(r'[^a-zA-Z]+', ' ', email_body_text)
                    tokens = nltk.word_tokenize(email_body_text)
                    filtered_tokens = [wnl.lemmatize(word.lower()) for word in tokens
                                       if len(word)>1 and word not in english_stopwords and word in words]
                    email_body_text = ' '.join(filtered_tokens)
                    preprocess_path = path.join(preprocess_dir, file_name)
                    preprocess_file = open(preprocess_path, 'wb')
                    preprocess_file.write(email_body_text)
                    preprocess_file.close()
#!/usr/bin/python
# coding=utf8
# author=zhtsh

import os
import sys
import re
import nltk
import email
from os import path
from nltk.corpus import stopwords
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
    if len(sys.argv) < 2:
        print('Usage: python email_etl.py corpus_dir')
        sys.exit(1)
    corpus_dir = sys.argv[1]
    english_stopwords = stopwords.words('english')
    for child_item in os.walk(corpus_dir):
        child_dir = child_item[0]
        if child_dir != corpus_dir:
            if path.basename(child_dir).find('spam') != -1:
                preprocess_dir = path.join(corpus_dir, 'preprocess_spam')
            else:
                preprocess_dir = path.join(corpus_dir, 'preprocess_nonspam')
            for file_name in child_item[2]:
                file_path = path.join(child_dir, file_name)
                email_message = email.message_from_file(open(file_path, 'rb'))
                email_bodies = []
                for part in email_message.walk():
                    content_type = part.get_content_type().lower()
                    if content_type == 'text/plain':
                        email_bodies.append(part.get_payload())
                    elif content_type == 'text/html':
                        parser = MyHTMLParser()
                        parser.feed(part.get_payload())
                        email_bodies.append(parser.get_body())
                    else:
                        continue
                if email_bodies:
                    email_body_text = ' '.join(email_bodies)
                    email_body_text = re.sub(r'[^a-zA-Z]+', ' ', email_body_text)
                    tokens = nltk.word_tokenize(email_body_text)
                    filtered_tokens = [word for word in tokens if len(word)>1 and word not in english_stopwords]
                    email_body_text = ' '.join(filtered_tokens)
                    preprocess_path = path.join(preprocess_dir, file_name)
                    preprocess_file = open(preprocess_path, 'wb')
                    preprocess_file.write(email_body_text)
                    preprocess_file.close()
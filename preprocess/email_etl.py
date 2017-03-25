#!/usr/bin/python
# coding=utf8
# author=zhtsh

import os
import sys
import logging
from os import path

sys.path.append(path.abspath(path.join(path.dirname(__file__), '../core')))
from utils import EmailETLHelper

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    root_dir = path.abspath(path.join(path.dirname(__file__), '../'))
    corpus_dir = path.join(root_dir, 'corpus')
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
                email_body_text = EmailETLHelper.get_body_from_email(file_path)
                if email_body_text:
                    preprocess_path = path.join(preprocess_dir, file_name)
                    preprocess_file = open(preprocess_path, 'wb')
                    preprocess_file.write(email_body_text)
                    preprocess_file.close()
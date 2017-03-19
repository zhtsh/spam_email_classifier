#!/usr/bin/python
# coding=utf8
# author=zhtsh

from os import path, listdir

if __name__ == '__main__':
    postive_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/preprocess_spam'))
    negative_samples_dir = path.abspath(path.join(path.dirname(__file__), '../data/preprocess_nonspam'))
    spam_files = [path.join(postive_samples_dir, f) for f in listdir(postive_samples_dir)
                  if path.isfile(path.join(postive_samples_dir, f))]
    nonspam_files = [path.join(negative_samples_dir, f) for f in listdir(negative_samples_dir)
                     if path.isfile(path.join(negative_samples_dir, f))]
    files = spam_files + nonspam_files
    dictionary = set()
    for file_path in files:
        with open(file_path, 'rb') as file:
            terms = file.readall().strip().split()
            for term in terms:
                dictionary.add(term)
    dictionary_path = path.abspath(path.join(path.dirname(__file__), '../data/dictionary.txt'))
    with open('', 'wb') as dictionary_file:
        for term in dictionary:
            dictionary_file.write(term + '\n')

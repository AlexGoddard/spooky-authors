#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import nltk
from nltk.stem import PorterStemmer

__name__ = '__main__'


class Sentence:
    def __init__(self, sid, text):
        self.sid = sid
        self.text = text

    def __str__(self):
        return str(self.text)


class Author:
    def __init__(self, tag, name):
        self.tag = tag
        self.name = name
        self.sentences = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    @property
    def book(self):
        book = []
        for sentence in self.sentences:
            book.extend(sentence.text)
        return book


def main():
    authors = {
        'EAP': Author('EAP', 'Edgar Allen Poe'),
        'HPL': Author('HPL', 'HP Lovecraft'),
        'MWS': Author('MWS', 'Mary Shelley')
    }

    stemmer = PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend([',', '.', ';', "'", ':'])

    with open('data/train.csv', encoding='utf8') as train_file:
        reader = csv.reader(train_file, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            sid = row[0]
            sentence = row[1]
            author = row[2]
            words = [word for word in nltk.word_tokenize(sentence) if word.lower() not in stopwords]
            stemmed_sentence = [stemmer.stem(word) for word in words]
            authors[author].add_sentence(Sentence(sid, stemmed_sentence))


if __name__ == '__main__':
    main()

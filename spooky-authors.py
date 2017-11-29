#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import nltk
from nltk.stem import PorterStemmer
from random import randint

__name__ = '__main__'

# Data files
TRAINING_FILE = 'data/train.csv'
TESTING_FILE = 'data/test.csv'


class Sentence:
    def __init__(self, sid, raw_text, cleaned_list):
        self.sid = sid
        self.raw_text = raw_text
        self.cleaned_list = cleaned_list

    def __str__(self):
        return self.raw_text


class Author:
    def __init__(self, initials, name):
        self.initials = initials
        self.name = name
        self.sentences = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    @property
    def book(self):
        return [word for sentence in self.sentences for word in sentence.cleaned_list]


class TextProcessor:
    def __init__(self):
        self.authors = {'EAP': Author('EAP', 'Edgar Allen Poe'),
                        'HPL': Author('HPL', 'HP Lovecraft'),
                        'MWS': Author('MWS', 'Mary Shelley')}

    def process_training_data(self):
        stemmer = PorterStemmer()
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend([',', '.', ';', "'", ':', '``', "''", "'s"])

        with open(TRAINING_FILE, encoding='utf8') as training_file:
            reader = csv.reader(training_file, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0:  # Skip the title row
                    continue
                sentence_id, raw_text, initials = row
                words = nltk.word_tokenize(raw_text)
                cleaned_words = [word for word in words if word.lower() not in stopwords]
                stemmed_words = [stemmer.stem(word) for word in cleaned_words]
                self.authors[initials].add_sentence(Sentence(sentence_id, raw_text, stemmed_words))

    def print_sample_stats(self):
        for initials in self.authors:
            author = self.authors[initials]
            print("Author: " + author.name)
            print("Number of sentences: " + str(len(author.sentences)))
            print("Total words: " + str(len(author.book)))
            print("Sample sentence:")
            sentence = author.sentences[randint(0, len(author.sentences) - 1)]
            print("\tSentence Id: " + sentence.sid)
            print("\tRaw text: " + sentence.raw_text)
            print("\tCleaned words: " + str(sentence.cleaned_list))
            print()


def main():
    processor = TextProcessor()
    processor.process_training_data()
    processor.print_sample_stats()


if __name__ == '__main__':
    main()

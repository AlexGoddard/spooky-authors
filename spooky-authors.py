#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import nltk
import operator
import numpy
from nltk.stem import PorterStemmer
from random import randint
from collections import Counter
from notSpacy import countPosTagsIndividually, pos_tags

__name__ = '__main__'

# Data files
TRAINING_FILE = 'data/train.csv'
TESTING_FILE = 'data/test.csv'

VECTOR_LENGTH = 50


class Sentence:
    def __init__(self, sid, raw_text, cleaned_list, noun_count, adjective_count, verb_count):
        self.sid = sid
        self.raw_text = raw_text
        self.cleaned_list = cleaned_list
        self.noun_count = noun_count
        self.adjective_count = adjective_count
        self.verb_count = verb_count

    def __str__(self):
        return self.raw_text


class Author:
    def __init__(self, initials, name):
        self.initials = initials
        self.name = name
        self.sentences = []

    @property
    def book(self):
        return [word for sentence in self.sentences for word in sentence.cleaned_list]

    @property
    def average_nouns(self):
        return numpy.mean([sentence.noun_count for sentence in self.sentences])

    @property
    def average_adjectives(self):
        return numpy.mean([sentence.adjective_count for sentence in self.sentences])

    @property
    def average_verbs(self):
        return numpy.mean([sentence.verb_count for sentence in self.sentences])

    @property
    def most_common_words(self):
        c = Counter(self.book)
        return [common[0] for common in c.most_common(VECTOR_LENGTH)]

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def vectorized_count(self, stemmed_words):
        vectorized_list = [0] * VECTOR_LENGTH
        for idx, word in enumerate(stemmed_words):
            try:
                index = self.most_common_words.index(word)
                vectorized_list[index] += 1
            except ValueError:
                continue
        return sum(vectorized_list)

    def pos_tag_similarity(self, nouns, adjectives, verbs):
        diff = 0
        diff += abs(self.average_nouns - nouns)
        diff += abs(self.average_adjectives - adjectives)
        diff += abs(self.average_verbs - verbs)
        return diff


class TextProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.authors = {'EAP': Author('EAP', 'Edgar Allen Poe'),
                        'HPL': Author('HPL', 'HP Lovecraft'),
                        'MWS': Author('MWS', 'Mary Shelley')}

    def process_training_data(self):
        stemmer = PorterStemmer()
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend([',', '.', ';', "'", ':', '``', "''", "'s", "?"])

        with open(TRAINING_FILE, encoding='utf8') as training_file:
            reader = csv.reader(training_file, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0:  # Skip the title row
                    continue
                if idx > 10:
                    break
                sentence_id, raw_text, initials = row
                annotated_sentence = pos_tags(raw_text)
                nouns, adjectives, verbs = countPosTagsIndividually(annotated_sentence)
                words = nltk.word_tokenize(raw_text)
                cleaned_words = [word for word in words if word.lower() not in stopwords]
                stemmed_words = [stemmer.stem(word) for word in cleaned_words]
                self.authors[initials].add_sentence(Sentence(sentence_id, raw_text, stemmed_words,
                                                             nouns, adjectives, verbs))

    def process_sentence(self, raw_text):
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend([',', '.', ';', "'", ':', '``', "''", "'s", "?"])
        words = nltk.word_tokenize(raw_text)
        cleaned_words = [word for word in words if word.lower() not in stopwords]
        stemmed_words = [self.stemmer.stem(word) for word in cleaned_words]
        return stemmed_words

    def determine_author(self, test_sentence):
        vector_sum = 0
        vector_values = {'EAP': [0, 0], 'HPL': [0, 0], 'MWS': [0, 0]}
        processed_words = self.process_sentence(test_sentence)
        for initials in self.authors:
            author = self.authors[initials]
            vector_count = author.vectorized_count(processed_words)
            vector_values[initials][0] = vector_count

            annoted_sentence = pos_tags(test_sentence)
            nouns, adjectives, verbs = countPosTagsIndividually(annoted_sentence)
            vector_values[initials][1] = author.pos_tag_similarity(nouns, adjectives, verbs)

            vector_sum += vector_count
        best_guess_word_bag = max(vector_values.items(), key=operator.itemgetter(1))
        best_guess_pos = min(vector_values.items(), key=operator.itemgetter(1))
        if best_guess_word_bag[1][0] == 0:
            probability = 33
        else:
            probability = round((best_guess_word_bag[1][0] / vector_sum) * 100, 2)
        return best_guess_word_bag[0], probability

    def print_sample_stats(self):
        for initials in self.authors:
            author = self.authors[initials]
            print("Author: " + author.name)
            print("Number of sentences: " + str(len(author.sentences)))
            print("Total words: " + str(len(author.book)))
            print("Most common words:" + str(author.most_common_words))
            print("Sample sentence:")
            sentence = author.sentences[randint(0, len(author.sentences) - 1)]
            print("\tSentence Id: " + sentence.sid)
            print("\tRaw text: " + sentence.raw_text)
            print("\tCleaned words: " + str(sentence.cleaned_list))
            print()


def main():
    processor = TextProcessor()
    processor.process_training_data()
    correct = 0
    count = 0
    sentence = "This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall."
    guess = processor.determine_author(sentence)
    # with open(TESTING_FILE, encoding='utf8') as testing_file:
    #     reader = csv.reader(testing_file, delimiter=',')
    #     for idx, row in enumerate(reader):
    #         if idx == 0:  # Skip the title row
    #             continue
    #         count += 1
    #         sentence_id, raw_text, initials = row
    #         guess = processor.determine_author(raw_text)
    #         if guess[0] == initials:
    #             correct += 1
    # print(str(correct) + " / " + str(count))


if __name__ == '__main__':
    main()

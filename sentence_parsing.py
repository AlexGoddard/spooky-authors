import spacy
import re

# loads once for the entire program scope
nlp = spacy.load('en')

# defining regular expressions for extracting syntactic dependencies
noun_phrase_regex = re.compile(r'NN.?')
verb_phrase_regex = re.compile(r'VB.?')
adj_phrase_regex = re.compile(r'JJ')


def annotate_sentence(sentence):
    '''
    Returns a list of word and its token given a sentence.
    '''

    annotated_sentence = nlp(sentence)
    tokens = list()
    for token in annotated_sentence:
        tokens.append((token, token.tag_))
    return tokens



def countPosTagsIndividually(raw_sentence):
    '''
        Returns nouns, verbs, adjective count in a sentence
    '''
    tokens = annotate_sentence(raw_sentence)
    nouns, adjectives, verbs = 0, 0, 0
    for entry in tokens:
        # print(entry[1])
        if re.search(noun_phrase_regex, entry[1]):
            nouns += 1
        if re.search(verb_phrase_regex, entry[1]):
            verbs += 1
        if re.search(adj_phrase_regex, entry[1]):
            adjectives += 1
    # print(nouns, adjectives, verbs)
    return (nouns, adjectives, verbs)

'''
Usage example.
'''
# print(countPosTagsIndividually('How do we go to the mount hood?'))
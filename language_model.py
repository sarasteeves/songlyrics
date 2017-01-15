# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 18:16:17 2017

@author: Sara
"""

from collections import defaultdict
from nltk import word_tokenize
from numpy import cumsum, sum, searchsorted
from numpy.random import rand

# Create class which will have a dictionary of character occurences and list of possible characters
class charModel(object):
    def __init__(self, order=1):
        self._characters = []
        self._occurrences = defaultdict(int)
        self._order = order
    
    # TRAINING
    def train(self, text):
        # update list of characters with new ones seen in training data
        for i in list(set(text)):
            if i not in self._characters:
                self._characters.append(i)
        # record moving window of characters (size order) and the character that follows, together with frequency in _occurrences
        for i in range(len(text)-self._order):
            self._occurrences[text[i:i+self._order], text[i+self._order]] += 1
    
    # Get occurences of character set from _occurrences and find weighted probabilities
    def predict(self, character):
        # Create list of counts of each 'character set, next character' 
        probs = [self._occurrences[(character, c)] for c in self._characters]
        # pick a character based on the weighted probabilities of each character following the character set
        return self._characters[self._weighted_pick(probs)]

    # GENERATING NEW TEXT
    def generate(self, start_chars, n):
        result = start_chars # initialise output with seed text
        # add new predicted characters to output using the predict function
        for i in range(n):
            new_char = self.predict(start_chars)
            result += new_char
            start_chars = start_chars[1:] + new_char
        return result

    @staticmethod
    def _weighted_pick(weights):
        countSum = sum(weights) # sum of all 'character set, next character' counts
        cumulativeSum = cumsum(weights) # cumulative sum of 'character set, next character' counts
        return searchsorted(cumulativeSum, rand()*countSum) # generate index number for next character using weighted probability

# Create class equivalent to character class, but for words
class wordModel(object):
    def __init__(self):
        self._words = []
        self._occurences = defaultdict(int)
    
    # TRAINING
    def train(self, text):
        # update list of words with new ones seen in training data
        vocabulary = list(word_tokenize(text)) # create list of words from text
        for i in list(set(vocabulary)):
            if i not in self._words:
                self._words.append(i)
        # record moving window of bigrams and the word that follows, together with frequency in _occurences
        for i in range(len(vocabulary)-2):
            self._occurences[vocabulary[i]+ ' ' + vocabulary[i+1], vocabulary[i+2]] += 1
    
    # Get occurences of bigrams from _occurrences and find weighted probabilities for next word
    def predict(self, bigram):
        probs = [self._occurences[(bigram, b)] for b in self._words]
        return self._words[self._weighted_pick(probs)]

    # GENERATING NEW TEXT
    def generate(self, start_words, n):
        result = start_words
        for i in range(n):
            new = self.predict(start_words)
            result += ' ' + new
            words = list(word_tokenize(result))
            start_words = words[-2] + ' ' + new
        return result

    @staticmethod
    def _weighted_pick(weights):
        countSum = sum(weights)
        cumulativeSum = cumsum(weights)
        return searchsorted(cumulativeSum, rand()*countSum)
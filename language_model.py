# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 18:16:17 2017

@author: Sara
"""

from collections import defaultdict
from numpy import cumsum, sum, searchsorted
from numpy.random import rand

# Create class which will have a dictionary of character occurences and list of possible characters
class languageModel(object):
    def __init__(self, order=1):
        self._characters = []
        self._occurences = defaultdict(int)
        self._order = order
    
    # TRAINING
    def train(self, text):
        # update list of characters with new ones seen in training data
        for i in list(set(text)):
            if i not in self._characters:
                self._characters.append(i)
        # record moving window of characters (size order) and the character that follows, together with frequency in _occurences
        for i in range(len(text)-self._order):
            self._occurences[text[i:i+self._order], text[i+self._order]] += 1
    
    # Get occurences of character set from _occurrences and find weighted probabilities
    def predict(self, character):
        # WORK OUT WHAT THIS LAST ONE IS
        probs = [self._occurences[(character, c)] for c in self._characters]
        return self._characters[self._weighted_pick(probs)]

    # 
    def generate(self, start_chars, n):
        result = start_chars
        for i in range(n):
            new_char = self.predict(start_chars)
            result += new_char
            start_chars = start_chars[1:] + new_char
        return result

    @staticmethod
    def _weighted_pick(weights):
        # BREAK DOWN TO MORE LINES FOR CLARITY
        return searchsorted(cumsum(weights), rand()*sum(weights))
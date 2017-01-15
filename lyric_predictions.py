# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 18:31:47 2017

@author: Sara
"""
import pandas as pd
import language_model as lm

def generate_lyrics_char(dataset, artist, seed, n):
    lyrics = dataset.loc[dataset['artist']==artist, 'text'] # get all text for entries with specified artist
    model = lm.charModel(order=len(seed))
    for song in lyrics:
        song = song.lower()
        model.train(song)
            
    return model.generate(seed, n)

def generate_lyrics_word(dataset, artist, seed, n):
    lyrics = dataset.loc[dataset['artist']==artist, 'text'] # get all text for entries with specified artist
    model = lm.wordModel()
    for song in lyrics:
        song = song.lower()
        model.train(song)
            
    return model.generate(seed, n)

if __name__=='__main__':
    
    # Load dataset
    # using 55000+ Song Lyrics dataset from Kaggle (https://www.kaggle.com/mousehead/songlyrics)
    data = pd.read_csv('songdata.csv')
    
    print 'Column names:'
    print data.columns.values
    print data.describe()
    print
    print 'Generating lyrics for Coldplay (by character)...'
    print generate_lyrics_char(data, 'Coldplay', "i don'", 200) 
    print
    print
    print 'Generating lyrics for Coldplay (by word)...'
    print generate_lyrics_word(data, 'Coldplay', "i don", 100) 
    print
    print
    print 'Generating lyrics for Robbie Williams (by character)...'
    print generate_lyrics_char(data, 'Robbie Williams', "i don'", 200)
    print
    print
    print 'Generating lyrics for Robbie Williams (by word)...'
    print generate_lyrics_word(data, 'Robbie Williams', "i don", 100)     
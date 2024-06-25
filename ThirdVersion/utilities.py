import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# stemmer = LancasterStemmer() # used for stemming words

# tokenize
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# find the root 
def stem(word):
    stemmer.stem(word.lower())

# determine each word in the sentence 
def bag_of_words(tokenized, words):
    # break down the sentence
    sentence = [stem(word) for word in tokenized]

    # initialize
    bag = np.zeros(len(words), dtype=np.float32)
    # mark 1 if exist
    for i, w in enumerate(words):
        if w in sentence:
            bag[i] = 1

    return bag

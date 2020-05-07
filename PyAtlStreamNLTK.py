import nltk
from nltk.book import *
from nltk.corpus import treebank
t = treebank.parsed_sents(text1)[0]
t.draw()
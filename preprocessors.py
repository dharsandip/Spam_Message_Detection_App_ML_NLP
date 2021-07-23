
import numpy as np
import pandas as pd
import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import gensim
from gensim.utils import simple_preprocess
nltk.download('wordnet')

stop_words = stopwords.words('english')

# Remove punctuations
def remove_punc(text):
    Text_punc_removed = [char for char in text if not char in string.punctuation]
    Text_punc_removed = ''.join(Text_punc_removed)
    return Text_punc_removed


# Remove stopwords
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stop_words:
            result.append(token)
    return result


# Stemming of words
ps = PorterStemmer()
def stem(text):
    Text_stemmed = [ps.stem(char) for char in text]
    return Text_stemmed


# Lemmatization of words
lemmatizer = WordNetLemmatizer()
def lemma(text):
    Text_lemmatized = [lemmatizer.lemmatize(char) for char in text]
    return Text_lemmatized











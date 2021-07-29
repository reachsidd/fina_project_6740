import re
import string

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nltk
import sklearn.model_selection

nltk.download('punkt')
nltk.download('stopwords')
import bs4
import wordcloud

import keras.preprocessing


# get raw text from htmls
def clean_html(text):
    soup = bs4.BeautifulSoup(text, 'html.parser')
    return soup.get_text()


# remove stopwords, punctuations and lower the case
def remove_stopwords_case_normalization(text):
    stopwords = nltk.corpus.stopwords.words('english')
    print(stopwords)
    text_new = text.lower()
    text_new = "".join([i for i in text_new if i not in string.punctuation])
    print(text_new)
    words = text_new.split()
    print(words)
    text_new = " ".join([i for i in words if i not in stopwords])
    print(text_new)

    return text_new


def clean_text(text):
    # Remove urls
    text_new = re.sub(r'http\S+', '', text)
    print('1=',text_new)
    text_new = clean_html(text_new)
    print('2=',text_new)
    text_new = remove_stopwords_case_normalization(text_new)
    print('3=',text_new)
    # Remove extra white space
    #text_new = re.sub("s+", " ", text_new)
    print('4=',text_new)
    return text_new


text="House Dem Aide: We Didn’t Even See Comey’s Letter Until"
print(text)
text_new = clean_text(text)
print(text_new)
text_new=text.replace("’","'")
print(text_new)
print(text_new.replace("'",'%'))



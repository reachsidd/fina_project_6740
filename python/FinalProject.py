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
import tensorflow as tf


# get raw text from htmls
def clean_html(text):
    soup = bs4.BeautifulSoup(text, 'html.parser')
    return soup.get_text()


# remove stopwords, punctuations and lower the case
def remove_stopwords_case_normalization(text):
    stopwords = nltk.corpus.stopwords.words('english')
    #print(stopwords)
    text_new = text.lower()
    text_new = re.sub('[^a-z]', ' ', text_new)
    # print(text_new)
    text_new = "".join([i for i in text_new if i not in string.punctuation])
    #print(text_new)
    words = text_new.split()
    #print(words)
    text_new = " ".join([i for i in words if i not in stopwords])
    #print(text_new)
    return text_new


def clean_text(text):
    # Remove urls
    text_new = re.sub(r'http\S+', '', text)
    text_new = clean_html(text_new)
    text_new = remove_stopwords_case_normalization(text_new)
    return text_new


def main():
    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 10)

    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    submit = pd.read_csv("../data/submit.csv")
    test = test.merge(submit, on='id')

    # print(train.columns)
    # print(train.head())
    # print(test.columns)
    # print(test.head())
    # print(len(test.index))
    # print(submit.columns)
    # print(submit.head())
    # print(len(submit.index))

    # null check
    print(train.isna().sum())
    # null check
    print(test.isna().sum())

    # filling NULL values with empty string
    train = train.fillna('')
    test = test.fillna('')

    #FOR TESTING ONLY
    train = train.head(100)
    test = test.head(100)
    # plt.figure(figsize=(12, 8))
    # sns.set(style="whitegrid", font_scale=1.2)
    # chart = sns.countplot(x="title", hue="label", data=train)
    # chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    # plt.show()

    train['content'] = train['text'] + ' ' + train['title']
    train.drop(columns=['title', 'text', 'author', 'id'], inplace=True)

    test['content'] = test['text'] + ' ' + test['title']
    test.drop(columns=['title', 'text', 'author', 'id'], inplace=True)

    train['content'] = train['content'].apply(clean_text)
    print(train.head())

    test['content'] = test['content'].apply(clean_text)
    print(test.head())

    # # WORDCLOUD FOR  CLEAN TEXT(LABEL - 1 - True)
    # plt.figure(figsize=(20, 20))  # Text that is not Fake
    # wc = wordcloud.WordCloud(max_words=2000, width=1600, height=800, stopwords=wordcloud.STOPWORDS).generate(
    #     " ".join(train[train.label == 1].content))
    # plt.imshow(wc, interpolation='bilinear')
    # plt.show()
    #
    # # WORDCLOUD FOR  CLEAN TEXT(LABEL - 0 - fake)
    # plt.figure(figsize=(20, 20))  # Text that is Fake
    # wc = wordcloud.WordCloud(max_words=2000, width=1600, height=800, stopwords=wordcloud.STOPWORDS).generate(
    #     " ".join(train[train.label == 0].content))
    # plt.imshow(wc, interpolation='bilinear')
    # plt.show()



    # NLP - Tokenize and apply Porter’s Stemmer algorithm
    ps = nltk.stem.porter.PorterStemmer()
    train['content'] = train['content'].apply(lambda x:' '.join([ps.stem(word) for word in nltk.tokenize.word_tokenize(x)]))
    print(train.head())

    test['content'] = test['content'].apply(lambda x:' '.join([ps.stem(word) for word in nltk.tokenize.word_tokenize(x)]))
    print(test.head())

    max_features = 300
    maxlen = 300
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train['content'])
    voc_size=len(tokenizer.word_index)
    print("voc_size=",voc_size)
    x_train = tokenizer.texts_to_matrix(train['content'])
    print(x_train.shape)
    x_test = tokenizer.texts_to_matrix(test['content'])
    print(x_test.shape)
    y_train = train['label']
    y_test = test['label']
    print(y_train.shape)
    print(y_test.shape)

    #Creating and training  model
    # We have used embedding layers with LSTM
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(voc_size, 40, input_length=maxlen))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())



    #x_train, x_test, y_train, y_test = train['content'], train['label'], test['content'], test['label']


if __name__ == '__main__':
    main()
    print("done")

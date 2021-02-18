# -*- coding: utf-8 -*-
"""Fake news detector

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MjXEFWdfUmo7XCkMMIF_zl649l3yDPO1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

train_df = pd.read_csv('/content/drive/MyDrive/Colab Datasets/fake-news/train.csv')

train_df.head()

sns.countplot(data = train_df, x = train_df['label'])

train_df.dropna(inplace = True)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text'])
X_seq = tokenizer.texts_to_sequences(train_df['text'])
X_pad = pad_sequences(X_seq, maxlen = 500)

vocab_size = len(tokenizer.word_index) + 1

Y = train_df['label']

model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, 40, input_length = 500),
        keras.layers.LSTM(100),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

model.fit(X_pad, Y, validation_split = 0.2, batch_size = 16, epochs = 10)

model.save('fake_news_detector.hd5')


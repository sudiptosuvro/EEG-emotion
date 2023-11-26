# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv("emotions.csv")
data

sample = data.loc[0, 'fft_0_b':'fft_749_b']
plt.figure(figsize=(24, 10))
plt.plot(range(len(sample)), sample)
plt.title('Features from first column through last')
plt.show()

data['label'].value_counts()

label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

def input_data(df):
    df = df.copy()
    df['label'] = df['label'].replace(label_mapping)
    
    y = df['label'].copy()
    x = df.drop('label', axis=1).copy()
    
    x_train, y_train, x_test, y_test = train_test_split(x, y, train_size=0.7, random_state=123)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = input_data(data)

if len(x_train) != len(y_train):
    min_len = min(len(x_train), len(y_train))
    x = x_train[:min_len]
    y = y_train[:min_len]

x = np.array(x)
y = np.array(y)

x_train.shape, x_test.shape

# Build the RNN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, SimpleRNN, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

inputs = tf.keras.Input(shape=(x_train.shape[1],))

expand_dims = tf.expand_dims(inputs, axis=2)
rnn = tf.keras.layers.SimpleRNN(256, return_sequences=True)(expand_dims)
flatten = tf.keras.layers.Flatten()(rnn)
outputs = tf.keras.layers.Dense(3, activation='softmax')(flatten)

RNN_MODEL = tf.keras.Model(inputs=inputs, outputs=outputs)
RNN_MODEL.summary()

# Compile the model
RNN_MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history_RNN = RNN_MODEL.fit(
    x_train, 
    y_train, 
    validation_split=0.2, 
    batch_size=32, 
    epochs=50, 
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
    ]
)

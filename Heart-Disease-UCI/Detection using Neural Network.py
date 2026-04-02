import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.layers import Dropout
from keras import regularizers
from sklearn import model_selection
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint


data_raw = pd.read_csv('heart.csv')

data = data_raw[~data_raw.isin(['?'])]
data = data.dropna(axis=0)

data = data.apply(pd.to_numeric)

X = np.array(data.drop(['target'], 1))
y = np.array(data['target'])

mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    stratify=y, 
                                                                    random_state=42, 
                                                                    test_size = 0.2)


Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)

Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

# Hyper-parameters

defaults=dict(
    dropout = 0.3,
    hidden_layer_size_1 = 16,
    hidden_layer_size_2 = 8,
    learn_rate = 0.001,
    epochs = 50,
)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(defaults['hidden_layer_size_1'], input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(defaults['dropout']))
    model.add(Dense(defaults['hidden_layer_size_2'], kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(defaults['dropout']))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    adam = Adam(learning_rate= defaults['learn_rate'])
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()

filepath = "UCI.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(X_train, Y_train_binary, 
                    validation_data=(X_test, Y_test_binary), 
                    epochs=defaults['epochs'],
                    callbacks=[checkpoint])
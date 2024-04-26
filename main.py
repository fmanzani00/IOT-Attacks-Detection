import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Reemplace con el ID de su GPU si es necesario
import tensorflow as tf

if tf.config.experimental.list_physical_devices('GPU'):
    print('GPU(s) detected!')
else:
    print('No GPU found.')

# Ahora puede usar TensorFlow con la GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from pandas import concat


def timeseries_to_supervised(data, lag=1):
    columns = [data.shift(i) for i in range(1, lag + 1)]
    columns.append(data)
    data = concat(columns, axis=1)
    data = data.iloc[1:]
    return data


df = pd.read_csv("S1_balanced.csv")

df["TCP count"] = df["Count tcp W=15.0s"]
df["Syn flag"] = df["Count tcp.flags.syn == 1 W=15.0s In"]
df["Reset flag"] = df["Count tcp.flags.reset == 1 W=15.0s"]
df["Avg frame len"] = df["Average frame.len W=15.0s"]
df["Distinct port"] = df["Distinct tcp.srcport W=15.0s In"]
df["Percent ssh"] = df["Percent ssh W=15.0s"]
df.drop(["Count tcp.flags.syn == 1 W=15.0s In", "Count tcp.flags.reset == 1 W=15.0s",
         "Average frame.len W=15.0s", "Distinct tcp.srcport W=15.0s In", "Percent ssh W=15.0s", "Count tcp W=15.0s",
         "Index", "Time", "Reset flag"], axis=1, inplace=True)

df.info()

df["Attacks"] = df["Attacks"].fillna(0)
df["Attacks"] = df["Attacks"].apply(lambda x: 1 if x > 0 else x)
df = df.dropna()

df = timeseries_to_supervised(df)

df = df.head(10000)

X = df
y = X["Attacks"]
X.drop(["Attacks"], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=1)

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
y_pred_bin = np.round(y_pred)

accuracy = accuracy_score(y_test, y_pred_bin)
#recall = recall_score(y_test, y_pred_bin)
#precision = precision_score(y_test, y_pred_bin)

print("Acierto:", accuracy)
#print("Recuerdo:", recall)
#print("Precisión:", precision)

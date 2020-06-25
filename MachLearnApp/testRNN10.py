import numpy as np
import pandas as pd
import os
import sys
import numpy 
import matplotlib.pyplot as plt
import pandas
import math
import sklearn
import tensorflow as tf
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error


def setUpData(x, y, time_steps):
    xs, ys = [], []
    for i in range(len(x) - time_steps):
        v = x.iloc[i:(i + time_steps)].values
        xs.append(v)
    ys.append(y.iloc[i + time_steps])
    return np.array(xs), np.array(ys)


print(sys.path)

df = pd.read_csv("./train.csv")
print(df.shape)

df_test = pd.read_csv("./test.csv")
print(df_test.shape)

df = df[df["CellName"] == "Cell_000111"]
df_test = df_test[df_test["CellName"] == "Cell_000111"]

print(df.shape, df_test.shape)

a = [df, df_test]
dataset = pd.concat(a)

print(dataset.head())

dataset["Date"] = pd.to_datetime(dataset.Date.astype(str))
dataset["Hour"] = pd.to_timedelta(dataset.Hour, unit="h")
dataset["DateTime"] = pd.to_datetime(dataset.Date + dataset.Hour)
dataset = dataset.drop(["Hour", "Date"], axis=1)

print(dataset.head())

dataset = dataset.set_index("DateTime")
dataset = dataset.sort_index()
print(dataset.head())

dataset = dataset.drop(["CellName"], axis=1)
print(dataset.head())

print(dataset.shape)

dr_train = dataset[:len(df)]
dr_test = dataset[len(df):]
print(dr_train.shape,dr_test.shape)

dr_train.to_csv("modifiedTrain.csv")
dr_test.to_csv("modifiedTest.csv")


sc = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
#dataset_scaled = tf.keras.utils.normalize(dataset, axis=0, order=2)
dataset_scaled = sc.fit_transform(dataset)
#print(dataset_scaled.head())

train, test = dataset_scaled[:len(df)],dataset_scaled[len(df):]

print(train.shape,test.shape)

print(train.size)

X_train = []
y_train = []
for i in range(24, train.size):
    X_train.append(train[i-24:i, 0])
    y_train.append(train[i ,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#X_train, y_train = setUpData(X_train, y_train, 24)

print(y_train.shape,X_train.shape)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units = 24, activation='sigmoid', return_sequences = True, input_shape = (X_train.shape[1], 1)))
#regressor.add(LSTM(units = 24, return_sequences = True))
#regressor.add(Dropout(0.2))
#regressor.add(LSTM(units = 24, return_sequences = True))
#regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 24, activation='sigmoid'))
regressor.add(Dropout(0.4))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

regressor.fit(X_train, y_train, epochs=100, batch_size = 400)

test,predict = test[:-24],test[-24:]

data_test = dataset_scaled[len(dataset_scaled) - len(test) - 24 :]

print(data_test)

X_test = []
y_test = []
for i in range(24, len(data_test)):
    X_test.append(data_test[i-24:i,:])
    y_test.append(data_test[i])
X_test, y_test = np.array(X_test), np.array(y_test)

#X_test, y_test = setUpData(X_test, y_test, 24)


y_test = sc.inverse_transform(y_test)

predicts = regressor.predict(X_test)
predicts = sc.inverse_transform(predicts)
print(predicts)

plt.plot(y_test, color = 'blue', label = 'Actual')
plt.plot(predicts, color = 'orange', label = 'Prediction')
plt.title('LSTM Predictions for Cell 000111 - 100 epochs / sigmoid / 40% dropout')
plt.xlabel('Test Input')
plt.ylabel('Traffic')
plt.legend()
plt.show()



#model = Sequential(
#        tf.keras.layers.LSTM(units = 4, input_shape = (x_train.shape[1], 1)),
#        tf.keras.layers.Dense(units = 1))

#model = Sequential()
#model.add(LSTM(units = 4, return_sequences = True, input_shape = (x_train.shape[1], 1)))
#model.add(Dropout(0.2))
#model.add(Dense(units = 1))

#regressor.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#regressor.fit(x_train, y_train, epochs=10, batch_size=200)

#train_predict = model.predict(x_train)
#test_predict = model.predict(x_test)


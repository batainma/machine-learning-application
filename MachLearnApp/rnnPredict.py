import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error

# RNN Prediction model using Long Short-Term Memory functions from Tensorflow
# to predict the x amount of future timesteps from cell tower traffic
# Mohammed Bataineh
def main():
        train_data = pd.read_csv("./modifiedTrain.csv")
        test_data = pd.read_csv("./modifiedTest.csv")

        # Don't need this since they are all the same anyway
        train_data = train_data.drop(['CellName'], axis=1)
        test_data = test_data.drop(['CellName'], axis=1)
        train_data = train_data.drop(['Hour'], axis=1)
        test_data = test_data.drop(['Hour'], axis=1)

        # Oops, should merge 'formatData.py' with this so I don't need to do this again
        train_data = train_data.set_index('Date')
        train_data = train_data.sort_index()
        test_data = test_data.set_index('Date')
        test_data = test_data.sort_index()

        # Normalize
        train_data = tf.keras.utils.normalize(
            train_data, axis=0, order=2
        )
        print(train_data.head())

        print(train_data.shape)

        x_train, y_train = setUpData(train_data, train_data.Traffic, 24)
        x_test, y_test = setUpData(test_data, test_data.Traffic, 24)

        print(x_train.shape, y_train.shape)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        #train_data['Traffic'].plot()
        regressor = Sequential()
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        regressor.fit(x_train, y_train, epochs=10, batch_size=400)


        plt.plot(y_test, color = 'blue', label = 'Actual')
        plt.plot(predicts, color = 'orange', label = 'Prediction')
        plt.title('LSTM Predictions for Cell 000111 - 25 epochs')
        plt.xlabel('Test Input')
        plt.ylabel('Traffic')
        plt.legend()
        plt.show()

        
def setUpData(x, y, time_steps):
    xs, ys = [], []
    for i in range(len(x) - time_steps):
        v = x.iloc[i:(i + time_steps)].values
        xs.append(v)
    ys.append(y.iloc[i + time_steps])
    return np.array(xs), np.array(ys)	



if __name__ == "__main__":
        main()


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
dataset = pd.read_csv('Google_Stock_Price_Train.csv',index_col="Date",parse_dates=True)
training_set=pd.DataFrame(dataset['Open'])

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
model=Sequential()
model.add(Bidirectional(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units = 50, return_sequences = True)))
model.add(Dropout(0.15))
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 1, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 100, batch_size = 32)
model.save('stocks.h5')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

df = pd.read_parquet('PFR_reactor.parquet')

x = df[['TOTAL_MOLAR_FLOW', 'T_MOLAR_FRACTION', 'H_MOLAR_FRACTION', 'B_MOLAR_FRACTION', 'D_MOLAR_FRACTION', 'M_MOLAR_FRACTION',
        'TEMPERATURE_0', 'PRESSURE']]
y = df[['TEMPERATURE', 'T_MOLAR_FLOW', 'H_MOLAR_FLOW','B_MOLAR_FLOW','D_MOLAR_FLOW','M_MOLAR_FLOW']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

# Neural Network architecture
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=y_train.shape[1], activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# train the model
model.fit(x_train, y_train, epochs=200, batch_size=64, validation_split=0.05, callbacks=[early_stopping])

# testing
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)


# Check for bias/variance
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred)
print(f'MSE train: {mse_train}')
print(f'MSE test: {mse_test}')

# saves model
model.save("pfr_multi_neural_net_relu64_relu32_linear.h5")
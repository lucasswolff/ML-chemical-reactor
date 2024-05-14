import pandas as pd
import numpy as np
from create_database import Eletrolytic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

voltage, enthalpy, efficiency = 18, -574.178, 0.95

eletrolytic_db = Eletrolytic(voltage, enthalpy, efficiency)

df, _ = eletrolytic_db.generate_db()

# selects only initial variables, with will be our optimization input
df = df[['ENERGY', 'NUM_CELLS', 'MOLAR_FLOW_NACL_0', 
        'MOLAR_FLOW_TOTAL_0', 'PROFIT']]

x, y = df.drop('PROFIT', axis=1).values, df['PROFIT'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Neural Network architecture
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# train the model
model.fit(x_train, y_train, epochs=200, batch_size=64, validation_split=0.05, callbacks=[early_stopping])

# testing
y_pred = model.predict(x_test_scaled)
y_pred_train = model.predict(x_train_scaled)

# Check for bias/variance
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred)
print(f'MSE train: {mse_train}')
print(f'MSE test: {mse_test}')

# saves model and scaler
model.save("neural_net_relu64_relu62_linear.h5")
#dump(scaler, 'MinMaxScaler.joblib')

# Enhancements
# add l2 regularization
# normalize features
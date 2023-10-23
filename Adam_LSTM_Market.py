import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
import keras 
from tensorflow.keras.layers import LSTM, Dense
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import CryptoHistoricalDataClient
import os 
import talib

'''
Challenges of LSTM
Gradient vanishing/exploding: LSTMs can suffer from vanishing or exploding gradients, making it difficult to train them effectively over long sequences.
Proper architecture design: Selecting appropriate LSTM architecture, such as the number of layers and hidden units, is crucial for achieving optimal performance.

Analysis criteria: 
step 0: LSTM calc is done for "Column 3" Finding the integer difference between price points. This is used as input into the main LSTM to find up or down probabilities. 
step 1: Understand the meaning of the models success, Using the drawn future flow model. 
step 2: Compare (Loop) different layers and number of nodes to find optimal tuning for the Daily chart. 


Understanding the Problems:

Vanishing –
As the backpropagation algorithm advances downwards(or backward) from the output layer towards the input layer, the gradients often get smaller and smaller and approach zero which eventually leaves the weights of the initial or lower layers nearly unchanged. As a result, the gradient descent never converges to the optimum. This is known as the vanishing gradients problem.

Exploding –
On the contrary, in some cases, the gradients keep on getting larger and larger as the backpropagation algorithm progresses. This, in turn, causes very large weight updates and causes the gradient descent to diverge. This is known as the exploding gradients problem.

Another popular technique to mitigate the exploding gradients problem is to clip the gradients during backpropagation so that they never exceed some threshold. This is called Gradient Clipping.

This optimizer will clip every component of the gradient vector to a value between –1.0 and 1.0.
Meaning, all the partial derivatives of the loss w.r.t each  trainable parameter will be clipped between –1.0 and 1.0
optimizer = keras.optimizers.SGD(clipvalue = 1.0)
The threshold is a hyperparameter we can tune.
The orientation of the gradient vector may change due to this: for eg, let the original gradient vector be [0.9, 100.0] pointing mostly in the direction of the second axis, but once we clip it by some value, we get [0.9, 1.0] which now points somewhere around the diagonal between the two axes.
To ensure that the orientation remains intact even after clipping, we should clip by norm rather than by value.
optimizer = keras.optimizers.SGD(clipnorm = 1.0)
Now the whole gradient will be clipped if the threshold we picked is less than its ℓ2 norm. For eg: if clipnorm=1 then the vector [0.9, 100.0] will be clipped to [0.00899, 0.999995] , thus preserving its orientation.
'''

def load_financial_data(file_path="btc_bars_hourly.csv"):
    # Check timeframe already exists in folder
    if os.path.exists(file_path):
        print("File exists. Loading the existing DataFrame from file.")
        return pd.read_csv(file_path)
    
    try:
        client = CryptoHistoricalDataClient()
        request_params = CryptoBarsRequest(
                            symbol_or_symbols=["BTC/USD"],
                            timeframe=TimeFrame.Hour,
                            start="2016-09-01T00:00:00",  
                            end="2022-09-07T00:00:00" 
                            )
        btc_bars = client.get_crypto_bars(request_params)
        if hasattr(btc_bars, 'df'):
            btc_bars.df.to_csv(file_path, index=False)
            return btc_bars.df
        
        else:
            print("btc_bars does not have a 'df' attribute.")
            return None
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def calculate_indicators(df):
    df['RSI'] = talib.RSI(df['close'].values, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACDSignal'] = macdsignal

    df.dropna(inplace=True)

    return df

def preprocess_data(df, time_steps=10):
    scaler_multi = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_multi.fit_transform(df)
        
    scaler_single = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler_single.fit_transform(df[['close']])
    
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_close[i])
        
    X, y = np.array(X), np.array(y)

    #reshape X to be 3D 
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    return X, y, scaler_multi, scaler_single

def plot_results(true_data, predicted_data, scaler):
    plt.plot(true_data, label='True Values')
    plt.plot(scaler.inverse_transform(predicted_data), label='Predictions')
    plt.legend()
    plt.show()

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(75, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(53))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future(model, last_sequence, future_steps, scaler):
    future_predictions = []
    last_sequence_close = last_sequence[:, 3].reshape(-1, 1)
    for i in range(future_steps):
        new_prediction = model.predict(last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1]))[0]
        last_sequence_close = np.append(last_sequence_close[1:], new_prediction)
        last_sequence[:, 3] = last_sequence_close.flatten()
        future_predictions.append(new_prediction)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return future_predictions



def save_data_to_csv(true_data, predicted_data, file_name):
    df = pd.DataFrame({
        'True_Values': true_data,
        'Predicted_Values': predicted_data.ravel()
    })
    
    df.to_csv(file_name, index=False)


if __name__ == "__main__":


    df = load_financial_data()

# Add RSI and MACD
#   df = calculate_indicators(df)

    X, y, scaler_multi, scaler_single = preprocess_data(df)
    
    #train the model
    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=500, batch_size=32)
    
    model.save("ADAM_model.h5")

    predicted_values = model.predict(X)
    
    plot_results(df['close'].values[10:], predicted_values, scaler_single)
    
    last_sequence = X[-1, :, :]
    
    # Number of future points to predict
    future_steps = 10
    
    future_predictions = predict_future(model, last_sequence, future_steps, scaler_single)
    save_data_to_csv 
    all_data = np.append(df['close'].values[10:], future_predictions)
    plt.plot(df['close'].values[10:], label='True Values')
    plt.plot(all_data, label='Model + Future Predictions')
    plt.legend()
    plt.show()
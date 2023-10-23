import pandas as pd
import matplotlib.pyplot as plt
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import CryptoHistoricalDataClient
import numpy as np
#Load finance data from Alpaca. 

def load_financial_data(file_path="btc_bars_hourly.csv"):
    client = CryptoHistoricalDataClient()
    request_params = CryptoBarsRequest(
                        symbol_or_symbols=["BTC/USD"],
                        timeframe=TimeFrame.Day,
                        start="2019-09-01T00:00:00",
                        end="2022-09-07T00:00:00"
                    )
    try:
        btc_bars = client.get_crypto_bars(request_params)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    if not hasattr(btc_bars, 'df'):
        print("btc_bars does not have a 'df' attribute.")
        return None
    btc_bars.df.to_csv(file_path, index=False)
    return btc_bars.df


# TA Calculations. 
#Not using TA lib due to errors. 
def calculate_ema(df, column, period):
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_sma(df, column, period):
    return df[column].rolling(window=period).mean()

def calculate_macd(df, fast_length, slow_length, signal_length):
    df['fast_ma'] = calculate_ema(df, 'close', fast_length)
    df['slow_ma'] = calculate_ema(df, 'close', slow_length)
    df['macd'] = df['fast_ma'] - df['slow_ma']
    df['signal'] = calculate_sma(df, 'macd', signal_length)
    df['hist'] = df['macd'] - df['signal']
    return df

#Color set. 
def determine_colors(df):
    df['macd_above'] = df['macd'] >= df['signal']
    df['hist_a_up'] = (df['hist'] > df['hist'].shift(1)) & (df['hist'] > 0)
    df['hist_a_down'] = (df['hist'] < df['hist'].shift(1)) & (df['hist'] > 0)
    df['hist_b_down'] = (df['hist'] < df['hist'].shift(1)) & (df['hist'] <= 0)
    df['hist_b_up'] = (df['hist'] > df['hist'].shift(1)) & (df['hist'] <= 0)
    return df


#Plot Data 2
def plot_price_movements(df, ax):
    ax.plot(range(len(df)), df['close'], label='Price', color='black')
    ax.set_title('Price Movements')
    ax.legend(loc='upper left')

def plot_macd_and_histogram(df, ax1, ax2):
    color_choice = 'lime' if df.iloc[-1]['macd_above'] else 'red'
    ax1.plot(range(len(df)), df['macd'].values, label='MACD', color=color_choice)
    ax1.plot(range(len(df)), df['signal'], label='Signal Line', color='blue')
    ax1.set_title('MACD and Signal Line')
    ax1.legend(loc='upper left')
    
    colors = df.apply(lambda x: 'aqua' if x['hist_a_up'] else ('blue' if x['hist_a_down'] else ('red' if x['hist_b_down'] else 'maroon')), axis=1)
    ax2.bar(range(len(df)), df['hist'], color=colors)
    ax2.axhline(0, color='white')
    ax2.set_title('Histogram')

def plot_scatter(df, ax):
    df.reset_index(drop=True, inplace=True)  # Reset index to be numeric
    
    crossover_above = ((df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))).to_numpy()
    crossover_below = ((df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1))).to_numpy()
    
    indices_above = np.where(crossover_above)[0]
    indices_below = np.where(crossover_below)[0]
    
    # This ensures that the data is numeric and can be plotted
    ax.scatter(indices_above, df['macd'].iloc[indices_above].astype(float), color='g')
    ax.scatter(indices_below, df['macd'].iloc[indices_below].astype(float), color='r')

def plot_data(df, results):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 9))
    
    plot_price_movements(df, ax1)
    plot_macd_and_histogram(df, ax2, ax3)
    plot_scatter(df, ax2)
    
    annotation_str = f"Final Capital: ${results['Final capital']}\n" + \
                     f"Total Return: ${results['Total return']}\n" + \
                     f"Percentage Return: {results['Percentage return']}%"
                     
    ax1.annotate(annotation_str, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                 bbox=dict(boxstyle="round", fc="w"))
    
    plt.tight_layout()
    plt.show()
    
    

def stochastic_rsi(df, column='close', period=14, k_period=3, d_period=3):
    delta = df[column].diff(1)
    gains = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    losses = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gains / losses
    rsi = 100 - (100 / (1 + rs))
    
    lowest_rsi = rsi.rolling(window=k_period).min()
    highest_rsi = rsi.rolling(window=k_period).max()
    stoch_rsi = 100 * (rsi - lowest_rsi) / (highest_rsi - lowest_rsi)
    
    return stoch_rsi
 
def backtest_strategy(df, initial_capital=10000):
    capital = initial_capital
    stock_quantity = 0
    martingale_factor = 1.8
    base_investment = initial_capital / 10
    current_investment = base_investment
    previous_trade_win = None

    df['long_term_ma'] = df['close'].rolling(window=200).mean()
    df['srsi'] = stochastic_rsi(df)
    df.reset_index(drop=True, inplace=True)

    # For comparison metrics
    initial_price = df.iloc[0]['close']
    final_price = df.iloc[-1]['close']
    value_of_doing_nothing = (final_price / initial_price - 1) * 100
    final_value_all_in = (initial_capital / initial_price) * final_price
    percentage_return_all_in = (final_value_all_in / initial_capital - 1) * 100

    for i, row in df.iterrows():
        if i < max(200, 14 + 3):  # Skip the first x data points where indicators are not available
            continue

        if row['crossover_above']:
            if capital > 0 and row['close'] > row['long_term_ma'] and row['srsi'] < 80:
                if previous_trade_win is False:
                    current_investment *= martingale_factor

                current_investment = min(current_investment, capital)
                buy_price = row['close']
                stock_quantity = current_investment / buy_price
                capital -= current_investment

        elif row['crossover_below']:
            if stock_quantity > 0 and row['close'] < row['long_term_ma'] and row['srsi'] > 20:
                sell_price = row['close']
                capital += stock_quantity * sell_price
                previous_trade_win = sell_price > buy_price

    final_capital = capital + stock_quantity * df.iloc[-1]['close']
    total_return = final_capital - initial_capital
    percentage_return = (final_capital / initial_capital - 1) * 100

    return {
        "Final capital": final_capital,
        "Total return": total_return,
        "Percentage return": percentage_return,
        "Value of doing nothing (percentage)": value_of_doing_nothing,
        "Final value of all in": final_value_all_in,
        "Percentage return all in": percentage_return_all_in
    }


if __name__ == "__main__":
    df = load_financial_data()
    print(df.columns)

    if df is not None:
        fast_length = 12
        slow_length = 26
        signal_length = 9
        df = calculate_macd(df, fast_length, slow_length, signal_length)
        df = determine_colors(df)

        df['crossover_above'] = (df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))
        df['crossover_below'] = (df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1))
        
        results = backtest_strategy(df)
        plot_data(df, results)
        
        print(f"Final capital: ${results['Final capital']}")
        print(f"Total return: ${results['Total return']}")
        print(f"Percentage return: {results['Percentage return']}%")


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def compute_technical_indicators(data, indicators):
    for indicator, windows in indicators.items():
        if indicator == 'SMA':
            for window in windows:
                data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
        elif indicator == 'EMA':
            for window in windows:
                data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
        elif indicator == 'RSI':
            for window in windows:
                data[f'RSI_{window}'] = compute_rsi(data['Close'], window)
        elif indicator == 'Momentum':
            for window in windows:
                data[f'Momentum_{window}'] = data['Close'].diff(window)
        elif indicator == 'Volatility':
            for window in windows:
                data[f'Volatility_{window}'] = data['Close'].rolling(window=window).std()
        elif indicator == 'ROC':
            for window in windows:
                data[f'ROC_{window}'] = data['Close'].pct_change(periods=window)
        elif indicator == 'BB':
            for window in windows:
                data[f'BB_Middle_{window}'] = data['Close'].rolling(window=window).mean()
                data[f'BB_Upper_{window}'] = data[f'BB_Middle_{window}'] + 2 * data['Close'].rolling(window=window).std()
                data[f'BB_Lower_{window}'] = data[f'BB_Middle_{window}'] - 2 * data['Close'].rolling(window=window).std()
        elif indicator == 'MACD':
            for short_window, long_window, signal_window in windows:
                data[f'EMA_Short_{short_window}'] = data['Close'].ewm(span=short_window, adjust=False).mean()
                data[f'EMA_Long_{long_window}'] = data['Close'].ewm(span=long_window, adjust=False).mean()
                data[f'MACD_{short_window}_{long_window}'] = data[f'EMA_Short_{short_window}'] - data[f'EMA_Long_{long_window}']
                data[f'MACD_Signal_{short_window}_{long_window}_{signal_window}'] = data[f'MACD_{short_window}_{long_window}'].ewm(span=signal_window, adjust=False).mean()
        elif indicator == 'ATR':
            for window in windows:
                high_low = data['High'] - data['Low']
                high_close = (data['High'] - data['Close'].shift()).abs()
                low_close = (data['Low'] - data['Close'].shift()).abs()
                tr = high_low.combine(high_close, max).combine(low_close, max)
                data[f'ATR_{window}'] = tr.rolling(window=window).mean()
    return data

def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Local time'], dayfirst=True)

    data['Hour'] = data['Local time'].dt.hour
    data['DayOfWeek'] = data['Local time'].dt.dayofweek

    indicators = {
        'SMA': [20, 50],
        'EMA': [20, 50],
        'RSI': [14, 30],
        'BB': [20, 50],
        'MACD': [(12, 26, 9), (26, 52, 9)],
        'ATR': [14, 50],
        'Momentum': [14, 30],
        'Volatility': [14, 30],
        'ROC': [14, 30],
    }

    data = compute_technical_indicators(data, indicators)
    data.dropna(inplace=True)
    data['Future Close'] = data['Close'].shift(-24)
    data.dropna(inplace=True)

    features = data.drop(columns=['Local time', 'Future Close'])
    target = data['Future Close']

    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val, scaler

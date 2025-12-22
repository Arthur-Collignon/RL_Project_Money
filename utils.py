import pandas as pd
import numpy as np

def calculate_RSI(dataframe):
    data_RSI=dataframe.copy()
    gain_loss_df = pd.DataFrame()
    number_of_periods = 24
    gain_mean = []
    loss_mean = []
    gain_loss_df['change'] = data_RSI['open'] - data_RSI['close']
    gain_loss_df['gain'] = np.where(gain_loss_df['change'] > 0, gain_loss_df['change'], 0)
    gain_loss_df['loss'] = np.where(gain_loss_df['change'] < 0, -gain_loss_df['change'], 0)

    # Calcul des moyennes mobiles des changements
    for i in range(len(gain_loss_df['change'])):
        if i < number_of_periods:
            gain_mean.append(0.01)
            loss_mean.append(0.01)
            continue
        elif i == number_of_periods:
            gain_mean.append(gain_loss_df['gain'][:number_of_periods].mean())
            loss_mean.append(gain_loss_df['loss'][:number_of_periods].mean())
            continue
        else:
            gain_mean.append(gain_loss_df['gain'][i - number_of_periods:i].mean())
            loss_mean.append(gain_loss_df['loss'][i - number_of_periods:i].mean())

    # Calcul du RSI
    dataframe['feature_RSI'] = 100 - (100 / (1 + (np.array(gain_mean) / np.array(loss_mean))))

    # Calcul des 0 et 1 quand le RSI est en surachat (>70) ou en survente (<30)
    dataframe['feature_RSI_overbought'] = np.where(dataframe['feature_RSI'] > 70, 1, 0)
    dataframe['feature_RSI_oversold'] = np.where(dataframe['feature_RSI'] < 30, 1, 0)

    return dataframe


def calculate_MACD(df):
    data_MACD = df.copy()
    signal_period = 9

    # Calcul des MME
    MME_12 = data_MACD['close'].ewm(span=12, adjust=False).mean()
    MME_26 = data_MACD['close'].ewm(span=26, adjust=False).mean()

    # Calcul de la ligne de signal (slow line)
    slow_MACD = MME_12 - MME_26
    fast_MACD = slow_MACD.ewm(span=signal_period, adjust=False).mean()

    histogram = slow_MACD - fast_MACD

    df['feature_fast_line'] = fast_MACD
    df['feature_slow_line'] = slow_MACD
    df['feature_histogram'] = histogram

    return df

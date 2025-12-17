import pandas as pd
import numpy as np

def calculate_RSI(dataframe):
    data_RSI=dataframe.copy()
    gain_loss_df = pd.DataFrame()
    data_to_export = pd.DataFrame()
    number_of_periods = 24
    gain_mean = []
    loss_mean = []
    gain_loss_df['change'] = data_RSI['open'] - data_RSI['close']
    gain_loss_df['gain'] = np.where(gain_loss_df['change'] > 0, gain_loss_df['change'], 0)
    gain_loss_df['loss'] = np.where(gain_loss_df['change'] < 0, -gain_loss_df['change'], 0)

    # Calcul des moyennes mobiles des changements
    for i in range(len(gain_loss_df['change'])):
        if i < number_of_periods:
            gain_mean.append(np.nan)
            loss_mean.append(np.nan)
            continue
        elif i == number_of_periods:
            gain_mean.append(gain_loss_df['gain'][:number_of_periods].mean())
            loss_mean.append(gain_loss_df['loss'][:number_of_periods].mean())
            continue
        else:
            gain_mean.append(gain_loss_df['gain'][i - number_of_periods:i].mean())
            loss_mean.append(gain_loss_df['loss'][i - number_of_periods:i].mean())

    # Calcul du RSI
    data_to_export['RSI'] = 100 - (100 / (1 + (np.array(gain_mean) / np.array(loss_mean))))

    # Calcul des 0 et 1 quand le RSI est en surachat (>70) ou en survente (<30)
    data_to_export['RSI_overbought'] = np.where(data_to_export['RSI'] > 70, 1, 0)
    data_to_export['RSI_oversold'] = np.where(data_to_export['RSI'] < 30, 1, 0)

    return data_to_export


def calculate_MACD(dataframe):
    data_MACD = dataframe.copy()
    data_to_export = pd.DataFrame()
    MME_12 = 12
    MME_26 = 26
    signal_period = 9

    # Calcul des MME
    fast_MME = data_MACD['close'].ewm(span=MME_12, adjust=False).mean()
    slow_MME = data_MACD['close'].ewm(span=MME_26, adjust=False).mean()

    # Calcul de la ligne de signal (slow line)
    slow_MACD = MME_12 - MME_26
    fast_MACD = slow_MACD.ewm(span=signal_period, adjust=False).mean()

    histogram = slow_MACD - fast_MACD

    data_to_export['fast_line'] = fast_MACD
    data_to_export['slow_line'] = slow_MACD
    data_to_export['histogram'] = histogram

    return data_to_export

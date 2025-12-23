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
            gain_mean.append(0.0001)
            loss_mean.append(0.0001)
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

    # join data_to_export to dataframe
    dataframe = dataframe.join(data_to_export, how='left')

    return dataframe


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
    slow_MACD = fast_MME - slow_MME
    fast_MACD = slow_MACD.ewm(span=signal_period, adjust=False).mean()

    histogram = slow_MACD - fast_MACD

    data_to_export['feature_fast_line'] = fast_MACD
    data_to_export['feature_slow_line'] = slow_MACD
    data_to_export['feature_histogram'] = histogram

    # join data_to_export to dataframe
    dataframe = dataframe.join(data_to_export, how='left')

    return dataframe


def reward_function_updated(history):

    current_val = history['portfolio_valuation', -1]
    prev_val = history['portfolio_valuation', -2]
    portfolio_ret = (current_val / prev_val) - 1

    position = history['position', -1]  # La position prise à l'étape précédente
    position_precedente = history['position', -2]  # La position prise à l'étape d'avant

    current_price = history['data_close', -1]
    prev_price = history['data_open', -1]
    market_ret = (current_price / prev_price) - 1

    # Récompense de base : Alpha
    alpha_reward = (portfolio_ret - market_ret) * 100


    # Récompenses de directions
    direction_reward = 0

    # Si le marché baisse significativement
    if market_ret < -0.005:
        if position <= position_precedente*0.8: # et que l'agent a réduit sa position LONG d'au moins 20%
            direction_reward += 0.5  # Bonus pour avoir évité la chute
        if position > position_precedente*1.2: # et que l'agent a augmenté sa position LONG d'au moins 20%
            direction_reward -= 1  # Punition pour avoir aggravé la chute
        else:
            direction_reward -= 0.2  # Petite punition pour avoir maintenu une position LONG

    # Si le marché monte significativement
    if market_ret > 0.005:
        if position <= position_precedente*0.8: # et que l'agent a réduit sa position LONG d'au moins 20%
            direction_reward -= 0.5  # Punition pour avoir manqué la hausse
        if position > position_precedente*1.2: # et que l'agent a augmenté sa position LONG d'au moins 20%
            direction_reward += 1  # Bonus pour avoir profité de la hausse
        else:
            direction_reward += 0.2  # Petit bonus pour avoir maintenu une position LONG


    reward = alpha_reward*0.75 + direction_reward*0.25
    total_reward = np.clip(reward, -2, 2)

    return total_reward

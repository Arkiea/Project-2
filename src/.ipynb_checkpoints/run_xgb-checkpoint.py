# in-built
import itertools
import os.path
import datetime as dt
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finta import TA
from binance import Client

# machine learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ML models
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


client = Client()
fiat = 'USDT'
root_dir = os.path.join(os.path.dirname(__file__), '..')
# name of the y_test entry/exit signals column
target_col = 'test'
# number of weeks of test data
num_weeks_test_data = 12


def main():
    # Set up master dataframe
    crypto = 'BTC'
    today = (dt.datetime.today()).strftime('%Y-%m-%d')
    try:
        # load cached ohlcv_df from csv file
        ohlcv_df = pd.read_csv(
            os.path.join(root_dir, f"ohlcv_df_{today}.csv"),
            index_col='date',
            infer_datetime_format=True,
            parse_dates=True,
        )
    except FileNotFoundError:
        # download ohlcv data from binance
        ohlcv_df = get_historical_data(crypto)
        # save the data to a csv
        ohlcv_df.to_csv(os.path.join(root_dir, f"ohlcv_df_{today}.csv"))

    # add technical indicators
    master_df = add_TA_and_signal(ohlcv_df)
    # get all 3-column combinations from master_df.columns
    ta_columns = set(master_df.columns) - set([target_col])
    ta_combinations = list(itertools.combinations(ta_columns, 3))
    for combination in ta_combinations:
        name = '_'.join(sorted(combination))
        file_path_result = os.path.join(root_dir, 'result', f"{name}.png")
        # skip if the result is already generated
        if os.path.isfile(file_path_result):
            continue
        # slice only the combination columns and drop all null values
        df = master_df[list(combination) + [target_col]].dropna()
        # Segment the features from the target
        y = df[target_col]
        X = df.drop(columns=target_col)

        # Split training and test data
        end_training = (dt.datetime.today() - dt.timedelta(weeks=num_weeks_test_data)).strftime('%Y-%m-%d')
        X_train = X.loc[: end_training]
        X_test = X.loc[end_training:]
        y_train = y.loc[: end_training]
        y_test = y.loc[end_training:]

        # scale the data using StandardScaler
        X_scaler = StandardScaler()
        X_scaler.fit(X_train)
        X_train_scaled = X_scaler.transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        # create the Xgboost base model
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=5,
            objective='reg:logistic',
            learning_rate=0.1,
            random_state=1,
        )
        file_path_model = os.path.join(root_dir, 'model', f"{name}.json")
        try:
            # load model if possible
            model.load_model(file_path_model)
        except (xgb.core.XGBoostError, AttributeError):
            # fit the model using the training data
            model.fit(X_train_scaled, y_train)
            # save the model to model folder
            model.save_model(file_path_model)

        # make predictions
        predictions = model.predict(X_test_scaled)

        # Use a classification report to evaluate the model using the predictions and testing data
        report = classification_report(y_test, predictions, output_dict=True)

        # save classification report
        with open(os.path.join(root_dir, 'result', f"{name}.json"), 'w') as f:
            f.write(json.dumps(report, indent=4))

        # Calculate the model's returns
        predictions_df = X_test.copy()
        # Add y_test to the DataFrame
        predictions_df['y_test'] = y_test
        # Add the model predictions to the DataFrame
        predictions_df['predictions'] = predictions
        # Add the actual returns to the DataFrame
        predictions_df['daily returns'] = master_df['returns']
        # Add the strategy returns to the DataFrame
        predictions_df['model returns'] = predictions_df['daily returns'] * predictions_df['predictions'].shift()
        predictions_df.to_csv(os.path.join(root_dir, 'result', f"{name}.csv"))


def add_TA_and_signal(ohlcv_df):
    """ Add technical indicators and buy/sell signal """
    # Add bollinger bands
    bbands_df = TA.BBANDS(ohlcv_df)
    # Add custom TA entry/exit signals using bollinger bands
    bbands_df['close_vs_BB'] = np.select(
        [
            bbands_df['BB_UPPER'] < ohlcv_df['close'],
            bbands_df['BB_LOWER'] > ohlcv_df['close'],
        ],
        [-1, 1],
        default=0
    )
    # Add EMA
    ema_df = pd.DataFrame(
        [
            TA.EMA(ohlcv_df, 5),
            TA.EMA(ohlcv_df, 12),
        ]
    ).T
    # ema_df['EMA_DIFFERENCE'] = np.where(ema_df.iloc[:,1] > ema_df.iloc[:,0], 1 , -1)
    ema_df['EMA_DIFFERENCE'] = ema_df.iloc[:,1] - ema_df.iloc[:,0]
    # calculate returns
    returns_df = pd.DataFrame(ohlcv_df['close'].pct_change())
    returns_df.columns = ['returns']
    # set up entry/exit signals
    returns_df[target_col] = np.where(returns_df['returns'] > 0, 1, -1)
    returns_df[target_col] = returns_df[target_col].shift(-1)
    # Add custom indicator that counts the consecutive number of green/red days
    returns_df['consecutive'] = (
        (
            returns_df[target_col].groupby(
                # true if the previous value is different from the the current
                (returns_df[target_col] != returns_df[target_col].shift())
                # cumulatively sum them up to categorise them into groups of the same values
                .cumsum()
            )
            # count each value in the group starting from 1
            .cumcount() + 1
        # multiply each value by +/- 1 if the original was +ve or -ve
        ) * np.where(returns_df[target_col] > 0, 1, -1)
    # shift it back to normal because target_col is shifted -1
    ).shift()

    return pd.concat(
        [
            ohlcv_df,
            TA.SMA(ohlcv_df, 4),
            TA.SMA(ohlcv_df, 100),
            bbands_df,
            ema_df,
            returns_df,
            TA.RSI(ohlcv_df, 14),
            TA.DMI(ohlcv_df),
            TA.VWAP(ohlcv_df),
            TA.PIVOT_FIB(ohlcv_df),
        ],
        axis='columns',
    )



def get_historical_data(currency):
    """ Download kline candlestick data from Binance """
    klines = client.get_historical_klines(
        currency + fiat,
        Client.KLINE_INTERVAL_1DAY,
        "5 year ago UTC"
    )
    # klines columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    cols_ohlcv = ('open', 'high', 'low', 'close', 'volume')
    df = pd.DataFrame((x[:6] for x in klines), columns=['timestamp', *cols_ohlcv])
    df[[*cols_ohlcv]] = df[[*cols_ohlcv]].astype(float)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop(columns='timestamp', inplace=True)

    return df


if __name__ == '__main__':
    main()

# in-built
import itertools
import os.path
import datetime as dt
import json
import sys
import subprocess 
import find_best_model as fbm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finta import TA
from binance import Client

#Streamlit
import streamlit as st
from PIL import Image

# machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ML models
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

#warnings
import warnings
warnings.filterwarnings("ignore")

client = Client()
fiat = 'USDT'
root_dir = os.path.join(os.path.dirname(__file__), '..')
# name of the y_test entry/exit signals column
target_col = 'test'
# number of weeks of test data
num_weeks_test_data = 12


   
#Initialise streamlit variables
title = st.empty()
st_list = st.empty()
header = st.empty()
subheader = st.empty()
dataframe_data = st.empty()
image = st.empty()
subheader_two = st.empty()
dataframe_data_two = st.empty()
process = st.empty()
get_data = st.empty()
process = st.empty()
my_bar = st.empty()
line_chart = st.empty()
image1 = st.empty()
image2 = st.empty()
image3 = st.empty()
image4 = st.empty()
image5 = st.empty()
image6 = st.empty()
image7 = st.empty()
image8 = st.empty()
image9 = st.empty()
image10 = st.empty()



def main():
    title.title('Crypto Trading Bot')
    stage = st_list.selectbox("Stage", ("Get Data", "Process Data", "Best Model"))
    file = open("crypto.csv", "r")
    crypto_df =  pd.read_csv(file)
    
    crypto = crypto_df['crypto'].iloc[0]
    interval = crypto_df["inter"].iloc[0]
    start_str = crypto_df["start"].iloc[0]
        
    # Set up master dataframe
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
        ohlcv_df = get_historical_data(crypto, interval)
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
        
        count = ta_combinations.index(combination)
        count+=1
        loop = True
        check(crypto, today, ohlcv_df, count, stage, loop)
        
    check(crypto, today, ohlcv_df, len(ta_combinations), stage, False)
    
        
        
def check(crypto, today, ohlcv_df, count, stage, loop):
        if stage == "Get Data":
            header.subheader(f"Ticker: {crypto}")
            subheader.subheader(f"Prices from 5 years ago to {today}")
            dataframe_data.dataframe(ohlcv_df)
            line_chart.line_chart(ohlcv_df["close"])
                
        if stage == "Process Data":
            if loop == True:
                header.subheader("Incomplete...")
            else:
                header.subheader("All models loaded! Finally :) ")
            process.write(f"{count} models loaded")
            my_bar.progress(count/(len(ta_combinations)))
            
        if stage == "Best Model":
            subprocess.run([f"{sys.executable}", "src/find_best_model.py"])
            
            fbm = open("result.csv", "r")
            result_df =  pd.read_csv(fbm)
            
            
             
            subheader.subheader("Best Model")
            dataframe_data.dataframe(result_df.head(1))
            name = str(result_df['ta_indicators'].iloc[0]) + '.png'
            name = Image.open(f"{root_dir}/result/{name}")
            image.image(name, caption= 'Best Model' )
                     
            subheader_two.subheader("Top 10 Models with their returns and accuracy")
            dataframe_data_two.dataframe(result_df.head(10))
            
            process.write("Top 10 models")
            #Kind of spoon fed... difficult as images sometimes weren't visible with streamlit
            image1.image(name, caption= 'Best Model' )
            #Image 2
            name = str(result_df['ta_indicators'].iloc[1]) + '.png'
            name = Image.open(f"{root_dir}/result/{name}")
            image2.image(name, caption= '2' )
            #Image3
            name = str(result_df['ta_indicators'].iloc[2]) + '.png'
            name = Image.open(f"{root_dir}/result/{name}")
            image3.image(name, caption= '3' )
            #Image4
            name = str(result_df['ta_indicators'].iloc[3]) + '.png'
            name = Image.open(f"{root_dir}/result/{name}")
            image4.image(name, caption= '4' )
                        #Image5
            name = str(result_df['ta_indicators'].iloc[4]) + '.png'
            name = Image.open(f"{root_dir}/result/{name}")
            image5.image(name, caption= '5' )
                        #Image6
            name = str(result_df['ta_indicators'].iloc[5]) + '.png'
            name = Image.open(f"{root_dir}/result/{name}")
            image6.image(name, caption= '6' )
            #Image7
            name = str(result_df['ta_indicators'].iloc[6]) + '.png'
            name = Image.open(f"{root_dir}/result/{name}")
            image7.image(name, caption= '7' )
            #Image8
            name = str(result_df['ta_indicators'].iloc[7]) + '.png'
            name = Image.open(f"{root_dir}/result/{name}")
            image8.image(name, caption= '8' )
            #image9
            name = str(result_df['ta_indicators'].iloc[8]) + '.png'
            name = Image.open(f"{root_dir}/result/{name}")
            image9.image(name, caption= '9' )
            #image10
            name = str(result_df['ta_indicators'].iloc[9]) + '.png'
            name = Image.open(f"{root_dir}/result/{name}")
            image10.image(name, caption= '10' )

                
                

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



def get_historical_data(currency, interval): 
    klines = client.get_historical_klines(
        symbol = currency + fiat,
        interval = interval,
        start_str = start_str #"5 year ago UTC",
    )
    # klines columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    cols_ohlcv = ('open', 'high', 'low', 'close', 'volume')
    df = pd.DataFrame((x[:6] for x in klines), columns=['timestamp', *cols_ohlcv])
    df[[*cols_ohlcv]] = df[[*cols_ohlcv]].astype(float)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop(columns='timestamp', inplace=True)

    return df


#if __name__ == '__main__':
if  os.path.exists('crypto.csv'):
    main()
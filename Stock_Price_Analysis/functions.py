import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import validators
import itertools
import datetime
import time
import json
import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

from prophet import Prophet

import xgboost as xgb
from xgboost import plot_importance

from secrets_api import AV_API_TOKEN


pd.set_option('display.max_columns', None)


symbol = 'TSLA'
api_url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={AV_API_TOKEN}'



# Variables

current_columns = ['Ticker', 'Price', 'Company', 'Market Capitalization', 
              'Sector', 'Industry', 'Beta', 'EBITDA', 'PE_Ratio', 'PEG_Ratio', 
              'Diluted_EPS_TTM', 'Return_On_Assets', 'Return_On_Equity', 
              'Quarterly_Earnings_Growth', 'Quarterly_Revenue_Growth', 
              'Analysts_Target_Price', '52_Week_High', '52_Week_Low', 
              '50_Day_Average', '200_Day_Average', 'Trailing_PE', 'Forward_PE']


indicators = ['EMA', 'MACD', 'WMA', 'DEMA', 'PPO', 'VWAP', 'MACDEXT']




# Functions

def user_input_attempt2():
    
    user_input = input('Enter a company ticker symbol.\n')
    
    try:
        url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={user_input}&apikey={AV_API_TOKEN}'
        r = requests.get(url)
        data = r.json()
        name = data['bestMatches'][0]['2. name']
        symbol = data['bestMatches'][0]['1. symbol']
    except:
        print(f'Error loading data for {user_input}')

    try:
        choice = input(f'Confirming your choice: {name} - Ticker: {symbol}' + '\n' + 
                       'Please confirm YES or NO.' + '\n')
        if 'y' in choice.lower():
            print(f'Thank you. Gathering information on {name}.')
            return(symbol)
            pass
        else:
            print('Apologies for issues searching for the company. Please try again later.')
            pass
        
    except:
        ('Please try again later.')
        
        

def user_input_func():
    
    user_input = input('Enter a company name or ticker symbol.\n')
    
    try:
        url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={user_input}&apikey={AV_API_TOKEN}'
        r = requests.get(url)
        data = r.json()
        name = data['bestMatches'][0]['2. name']
        symbol = data['bestMatches'][0]['1. symbol']
    except:
        print(f'Could not find infomation on {user_input}. Please try again.')
        user_input_attempt2()

    try:
        choice = input(f'Confirming your choice: {name} - Ticker: {symbol}' + '\n' + 
                       'Please confirm YES or NO.' + '\n')
        if 'y' in choice.lower():
            print(f'Thank you. Gathering information on {name}. \n')
            return(symbol)
        else:
            print('Please retry with the company ticker symbol.')
            symbol = user_input_attempt2()
            return(symbol)
    except:
        print('Unforeseen error.')




def data_sourced(api_type, option):
    sourced_url = f'https://www.alphavantage.co/query?function={api_type}&apikey={AV_API_TOKEN}'
    r_sourced = requests.get(sourced_url)
    ds_sourced = r_sourced.json()
    ds_dataframe = pd.DataFrame.from_dict(ds_sourced[option])
    
    return(ds_dataframe)

def news_sourced(api_type, option, stock):
    sourced_url = f'https://www.alphavantage.co/query?function={api_type}&ticker={stock}&apikey={AV_API_TOKEN}'
    r_sourced = requests.get(sourced_url)
    ds_sourced = r_sourced.json()
    ds_dataframe = pd.DataFrame.from_dict(ds_sourced[option])
    
    return(ds_dataframe)



# API Pull Request for basic company info

def stock_overview_pull_request(stock):
    current_stock_df = pd.DataFrame(columns = current_columns)
    overview_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={stock}&apikey={AV_API_TOKEN}'
    price_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock}&outputsize=full&apikey={AV_API_TOKEN}'
    overview_data = requests.get(overview_url).json()
    price_data = requests.get(price_url).json()
    if len(overview_data) > 0:
            try:
                current_stock_df = current_stock_df.append(
                    pd.Series(
                    [
                        stock,
                        round(float(list(price_data['Time Series (Daily)'].values())[0]['4. close']), 2),
                        overview_data['Name'],
                        round(int(overview_data['MarketCapitalization']) / 1000000000, 2),
                        overview_data['Sector'],
                        overview_data['Industry'],
                        overview_data['Beta'],
                        overview_data['EBITDA'],
                        overview_data['PERatio'],
                        overview_data['PEGRatio'],
                        overview_data['DilutedEPSTTM'],
                        overview_data['ReturnOnAssetsTTM'],
                        overview_data['ReturnOnEquityTTM'],
                        overview_data['QuarterlyEarningsGrowthYOY'],
                        overview_data['QuarterlyRevenueGrowthYOY'],
                        overview_data['AnalystTargetPrice'],
                        overview_data['52WeekHigh'],
                        overview_data['52WeekLow'],
                        overview_data['50DayMovingAverage'],
                        overview_data['200DayMovingAverage'],
                        overview_data['TrailingPE'],
                        overview_data['ForwardPE']
                    ],
                    index = current_columns),
                ignore_index = True
                )
                return(current_stock_df)
        
            except Exception as e:
                print(e)
    else:
        print(f'Issue loading details for {stock}')
        
        


# API Pull Requestion for Historical company data, i.e. price, indicators

def stock_history_pull_request(stock):
    current_indicators_df = pd.DataFrame()
    
    price_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={stock}&outputsize=full&apikey={AV_API_TOKEN}'
    ema_url = f'https://www.alphavantage.co/query?function=EMA&symbol={stock}&interval=weekly&time_period=10&series_type=open&apikey={AV_API_TOKEN}'
    macd_url = f'https://www.alphavantage.co/query?function=MACD&symbol={stock}&interval=daily&series_type=open&apikey={AV_API_TOKEN}'
    wma_url = f'https://www.alphavantage.co/query?function=WMA&symbol={stock}&interval=weekly&time_period=10&series_type=open&apikey={AV_API_TOKEN}'
    dema_url = f'https://www.alphavantage.co/query?function=DEMA&symbol={stock}&interval=weekly&time_period=10&series_type=open&apikey={AV_API_TOKEN}'
    ppo_url = f'https://www.alphavantage.co/query?function=PPO&symbol={stock}&interval=daily&series_type=close&fastperiod=10&matype=1&apikey={AV_API_TOKEN}'

    price_data = requests.get(price_url).json()
    ema_data = requests.get(ema_url).json()
    macd_data = requests.get(macd_url).json()
    wma_data = requests.get(wma_url).json()
    dema_data = requests.get(dema_url).json()
    ppo_data = requests.get(ppo_url).json()

    price_df = pd.DataFrame(price_data['Time Series (Daily)'].values(), index=price_data['Time Series (Daily)'].keys())
    ema_df = pd.DataFrame(ema_data['Technical Analysis: EMA'].values(), index=ema_data['Technical Analysis: EMA'].keys())
    macd_df = pd.DataFrame(macd_data['Technical Analysis: MACD'].values(), index=macd_data['Technical Analysis: MACD'].keys())
    wma_df = pd.DataFrame(wma_data['Technical Analysis: WMA'].values(), index=wma_data['Technical Analysis: WMA'].keys())
    dema_df = pd.DataFrame(dema_data['Technical Analysis: DEMA'].values(), index=dema_data['Technical Analysis: DEMA'].keys())
    ppo_df = pd.DataFrame(ppo_data['Technical Analysis: PPO'].values(), index=ppo_data['Technical Analysis: PPO'].keys())


    current_indicators_df = pd.concat([price_df, ema_df, macd_df, wma_df, 
                                       dema_df, ppo_df], axis=1)
    
    current_indicators_df.reset_index(inplace=True)
    current_indicators_df = current_indicators_df.drop(['1. open', '2. high', '3. low', '4. close'], axis=1)
    current_indicators_df = current_indicators_df.rename(columns={'index':'ds', '5. adjusted close':'y',
                                                                  '6. volume':'Volume', '7. dividend amount':'Dividend', 
                                                                  '8. split coefficient':'Split'})
    current_indicators_df = current_indicators_df[['ds', 'y', 'Volume', 'Dividend', 'Split', 'EMA', 'MACD', 'MACD_Signal', 'MACD_Hist', 'WMA', 'DEMA', 'PPO']]
    
    
    current_indicators_df = current_indicators_df.dropna(subset=['y'])
    current_indicators_df = current_indicators_df.dropna(thresh=6)
    current_indicators_df = current_indicators_df.fillna(method='ffill')
    
    return(current_indicators_df)




def combining_dataframes(df1, df2):
    df1['Adjusted_Time'] = pd.to_datetime(df1.ds).dt.strftime('%Y-%m')
    df1['Date'] = df1.index
    df1 = pd.merge(df1, df2, on='Adjusted_Time', how='left')
    df1 = df1.fillna(method='ffill')
    df1 = df1.fillna(method='bfill')
    df1 = df1.drop(columns=(['Adjusted_Time', 'Date']))
    
    return(df1)



def corr_matrix_plot(df):
    corr_df = df.copy()
    
    corr_df = corr_df.drop(columns=['ds'])
    corr_df = corr_df.astype(float)
    
    corr_df = corr_df.corr()
    
    sns.heatmap(corr_df)
    #plt.title('Correlation Chart for Time-Series Variables')
    plt.show()
    
    return(corr_df)



def model_lstm(df, stock):
    
    model_df = df.set_index('ds')
    
    model_df = model_df.astype(float)
    
    model_test = model_df[:365]
    
    scaler = StandardScaler()
    scaler = scaler.fit(model_df)
    df_for_training = scaler.transform(model_df)
    
    trainX = []
    trainY = []
    
    n_future, n_past = 1, 21
    
    for i in range(n_past, len(df_for_training) - n_future + 1):
        trainX.append(df_for_training[i - n_past:i, 0:model_df.shape[1]])
        trainY.append(df_for_training[i + n_future - 1:i + n_future, 0])
    
    modelX, modelY = np.array(trainX), np.array(trainY)
    
    trainX, trainY = modelX[365:], modelY[365:]
    testX, testY = modelX[:365], modelY[:365]
    
    lstm_modeling = Sequential()
    lstm_modeling.add(LSTM(256, activation='relu', 
                            input_shape=(trainX.shape[1], trainX.shape[2]), 
                            return_sequences=True))
    lstm_modeling.add(LSTM(32, activation='relu', return_sequences=False))
    lstm_modeling.add(Dense(trainY.shape[1]))
    
    lstm_modeling.compile(optimizer='adam', loss='mse')
    print(lstm_modeling.summary())
    
    history = lstm_modeling.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.10, verbose=0)
    
    forecast = lstm_modeling.predict(testX)
    
    forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
    
    y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]
    
    df_forecast = pd.DataFrame({'Date':np.array(model_test.index), 'y':y_pred_future})
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
    df_forecast = df_forecast.set_index('Date')
    
    original = model_df
    original = original[pd.to_datetime(original.index) >= pd.to_datetime('2020-01-01')]
    
    sns.lineplot(x = pd.to_datetime(original.index), y = original['y'], label='Actual Data', alpha=0.6)
    sns.lineplot(x = pd.to_datetime(df_forecast.index), y = df_forecast['y'], label='Prediction', color='red')
    plt.legend()
    plt.suptitle(f'{stock}')
    plt.title('Actual vs Forecast using LSTM')
    plt.tick_params(axis='x', rotation=60)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    
    lstm_rmse = mean_squared_error(y_true=model_test['y'], y_pred=df_forecast['y'], squared=False)
    
    last_value = model_test['y'][0] - df_forecast['y'][0]
 
    
    return(round(lstm_rmse, 3), round(last_value, 3))
    
    
    
def model_xgboost(df, stock):
    
    df = df.set_index('ds')
    
    df = df.astype(float)
    
    train = df[365:]
    test = df[:365]
    
    xg_model = xgb.XGBRegressor()
    trainX = train.drop(columns=['y'])
    trainY = train['y']
    testX = test.drop(columns=['y'])
    testY = test['y']
    
    xg_model.fit(trainX, trainY)
    
    _ = plot_importance(xg_model, height=0.9)
    plt.title('Feature Importance for XGBoost')
    plt.suptitle(f'{stock}')
    plt.show()
    plt.clf()
    
    testY = testY.to_frame()
    
    testY['Predictions'] = xg_model.predict(testX)
    
    sns.lineplot(x = pd.to_datetime(testY.index), y = testY['y'], label='Actual', alpha=0.6)
    sns.lineplot(x = pd.to_datetime(testY.index), y = testY['Predictions'], label='Predictions', color='orange')
    plt.title('Actual vs Forecast using XGBoost')
    plt.suptitle(f'{stock}')
    plt.tick_params(axis='x', rotation=60)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    plt.show()
    
    xgb_rmse = mean_squared_error(y_true=testY['y'], y_pred=testY['Predictions'], squared=False)
    
    last_value = testY['y'][0] - testY['Predictions'][0]
    
    return(round(xgb_rmse, 3), round(last_value, 3))
    
    
    
def prophet_model(df, stock):
    
    train = df[365:]
    test = df[:365]
    test_no_y = test.drop('y', axis=1)
    
    prof_model = Prophet()
    prof_model.add_regressor('Volume', standardize=True)
    prof_model.add_regressor('MACD_Signal', standardize=True)
    prof_model.add_regressor('PPO', standardize=True)
    prof_model.add_regressor('EMA', standardize=True)
    prof_model.add_regressor('DEMA', standardize=True)
    prof_model.add_regressor('MACD', standardize=True)
    prof_model.add_regressor('MACD_Hist', standardize=True)
    prof_model.add_regressor('gas', standardize=True)
    prof_model.add_regressor('cpi', standardize=True)
    prof_model.add_regressor('unemployment', standardize=True)
    prof_model.add_regressor('fed_funds', standardize=True)
    
    prof_model.fit(train)
    
    prof_forecast = prof_model.predict(test_no_y)
    
    redex = prof_forecast.reindex(index=prof_forecast.index[::-1]).reset_index()
    
    sns.lineplot(x = pd.to_datetime(test.ds), y = test['y'].astype(float), label='Actual', alpha=0.6)
    sns.lineplot(x = pd.to_datetime(redex.ds), y = redex['yhat'].astype(float), label='Predictions', color='purple')
    plt.suptitle(f'{stock}')
    plt.title('Actual vs Forecast using Prophet')
    plt.tick_params(axis='x', rotation=60)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    
    prof_rmse = mean_squared_error(y_true=test['y'], y_pred=redex['yhat'], squared=False)
    
    prof_lv = float(test[0:1]['y']) - float(redex[0:1]['yhat'])
    
    return(round(prof_rmse, 3), round(prof_lv, 3))

# Comments or Adjustments


# Main, non-stock variables to help contribute to the overall analysis of a company's performance
# Additional dataframe adjustments included below as well

def external_information_pull():
    cpi = data_sourced('CPI', 'data')
    fed_funds = data_sourced('FEDERAL_FUNDS_RATE', 'data')
    unemployment = data_sourced('UNEMPLOYMENT', 'data')
    gas = data_sourced('NATURAL_GAS', 'data')
    treasury = data_sourced('TREASURY_YIELD', 'data')
        
    
    cpi['date'] = pd.to_datetime(cpi['date'], errors='coerce')
    cpi['date'] = cpi['date'].dt.strftime('%Y-%m')
    cpi = cpi.set_index('date')
    cpi = cpi.rename(columns={'value':'cpi'})
    cpi['Adjusted_Time'] = cpi.index
    cpi['cpi'] = cpi['cpi'].apply(pd.to_numeric, errors='coerce').dropna()
    
    
    
    fed_funds['date'] = pd.to_datetime(fed_funds['date'], errors='coerce')
    fed_funds['date'] = fed_funds['date'].dt.strftime('%Y-%m')
    fed_funds = fed_funds.set_index('date')
    fed_funds = fed_funds.rename(columns={'value':'fed_funds'})
    fed_funds['Adjusted_Time'] = fed_funds.index
    fed_funds['fed_funds'] = fed_funds['fed_funds'].apply(pd.to_numeric, errors='coerce').dropna()
    
    
    unemployment['date'] = pd.to_datetime(unemployment['date'], errors='coerce')
    unemployment['date'] = unemployment['date'].dt.strftime('%Y-%m')
    unemployment = unemployment.set_index('date')
    unemployment = unemployment.rename(columns={'value':'unemployment'})
    unemployment['Adjusted_Time'] = unemployment.index
    unemployment['unemployment'] = unemployment['unemployment'].apply(pd.to_numeric, errors='coerce').dropna()
    
    
    gas['date'] = pd.to_datetime(gas['date'], errors='coerce')
    gas['date'] = gas['date'].dt.strftime('%Y-%m')
    gas = gas.set_index('date')
    gas = gas.rename(columns={'value':'gas'})
    gas['Adjusted_Time'] = gas.index
    gas['gas'] = gas['gas'].apply(pd.to_numeric, errors='coerce').dropna()
    
    
    treasury['date'] = pd.to_datetime(treasury['date'], errors='coerce')
    treasury['date'] = treasury['date'].dt.strftime('%Y-%m')
    treasury = treasury.set_index('date')
    treasury = treasury.rename(columns={'value':'treasury'})
    treasury['Adjusted_Time'] = treasury.index
    treasury['treasury'] = treasury['treasury'].apply(pd.to_numeric, errors='coerce').dropna()
    
    
    # The next cells combine the above data into one dataframe
    
    combined_fed_data = cpi.merge(fed_funds, on='Adjusted_Time')
    combined_fed_data = combined_fed_data.merge(unemployment, on='Adjusted_Time')
    combined_fed_data = combined_fed_data.merge(gas, on='Adjusted_Time')
    combined_fed_data = combined_fed_data.merge(treasury, on='Adjusted_Time')
    combined_fed_data['Year'] = pd.to_datetime(combined_fed_data['Adjusted_Time'], errors='coerce').dt.strftime('%Y')
    combined_fed_data = combined_fed_data.drop('Year', axis=1)
    
    return(combined_fed_data)


    




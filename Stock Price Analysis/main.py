from functions import *




def main():
    
    print('Hello and welcome to the Stock Analysis Hub.')
    stock_pick = user_input_func()
    
    historical_df = stock_history_pull_request(stock_pick)
    overview_df = stock_overview_pull_request(stock_pick)
    econ_df = external_information_pull()
    
    company = overview_df['Company'][0]
    industry = overview_df['Industry'][0]
    
    print(f'{company} is in the {industry} Industry.' + '\n' +
          f'Their current Price / Earnings Ratio is {overview_df.PE_Ratio[0]}' + '\n' +
          f'with current Quarterly Earnings Growth of {overview_df.Quarterly_Earnings_Growth[0]}.' + '\n' +
          f'The current price of the stock is {overview_df.Price[0]} with an Analysts Target Price of {overview_df.Analysts_Target_Price[0]} \n')
          
    new_df = combining_dataframes(historical_df, econ_df)
    
    testing_corr2 = corr_matrix_plot(new_df)
    
    news = news_sourced('NEWS_SENTIMENT', 'feed', stock_pick)
    
    print('User Sentiment Outlook:')
    print(news['overall_sentiment_label'].value_counts())
    
    lstm_rmse, lstm_lv = model_lstm(new_df, stock_pick)

    xgb_rmse, xgb_lv = model_xgboost(new_df, stock_pick)
    
    prof_rmse, prof_lv = prophet_model(new_df, stock_pick)
    
    print('\n\n')
    print('LSTM: \n')
    print(f'RMSE - {lstm_rmse}')
    print(f'Difference in the last Actual - Projected: {lstm_lv} \n\n')
    
    print('XGBoost: \n')
    print(f'RMSE - {xgb_rmse}')
    print(f'Difference in the last Actual - Projected: {xgb_lv} \n\n')
    
    print('Prophet: \n')
    print(f'RMSE - {prof_rmse}')
    print(f'Difference in the last Actual - Projected: {prof_lv} \n\n')

    
main()




import discord
import asyncio
from keras.models import Sequential, load_model
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import LSTM,Dropout,Dense
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
import pandas_datareader as pdr
import datetime as dt
import ta
import tensorflow as tf
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta, FR
import os
import psycopg2
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder


tf.autograph.set_verbosity(0)
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def gather_data_derive_vars(ticker,b_date,e_date):
    data = pdr.DataReader(ticker,'yahoo',b_date,e_date)
    security_high = data['High']
    security_low = data['Low']
    security_close = data['Close']
    rsi_val = ta.momentum.RSIIndicator(security_close,window=2)
    keltner_lower = ta.volatility.keltner_channel_lband(security_high,
                                                        security_low,
                                                        security_close,
                                                        window=21,
                                                        window_atr=2)
    keltner_upper = ta.volatility.keltner_channel_hband(security_high,
                                                        security_low,
                                                        security_close,
                                                        window=21,
                                                        window_atr=2)
    stoch = ta.momentum.StochasticOscillator(security_high,
                                             security_low,
                                             security_close,
                                             window=8,
                                             smooth_window=3)

    data['RSI'] = rsi_val.rsi()
    data['Lower Kelt'] = keltner_lower
    data['Upper Kelt'] = keltner_upper
    data['Fast Stoch'] = stoch.stoch()
    data['Slow Stoch'] = stoch.stoch_signal()
    data.dropna(inplace=True)

    data = data[['Close','High','Low','Open',
                 'RSI', 'Lower Kelt','Upper Kelt',
                 'Fast Stoch','Slow Stoch']]

    return data

def production_direction_model(tickers,b_date,e_date):
    tf.autograph.set_verbosity(0)
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    print('---------------------------------------------------------------------')
    print('Working on Price Direction Modeling\n')
    #Database Connection -----------------------------------------------------
    # local_db_conn = psycopg2.connect('REDACTED')
    # cur = local_db_conn.cursor()
    template_bucket = "**Stock Price Prediction**\n\n"

    # Date Handling -----------------------------------------------------------
    today = str(dt.datetime.today().date())
    pred_date = None
    current_year = dt.datetime.today().year
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=str(current_year)+'-01-01', end=str(current_year)+'-12-31').to_pydatetime()

    if dt.datetime.today().weekday() < 4:
        if dt.datetime.today().date() in holidays:
            pred_date = str(dt.datetime.today().date() + dt.timedelta(days=2))
        else:
            pred_date = str(dt.datetime.today().date() + dt.timedelta(days=1))
    elif dt.datetime.today().weekday() == 4:
        pred_date = str(dt.datetime.today().date() + dt.timedelta(days=3))
        if pred_date in holidays:
            pred_date = str(dt.datetime.strptime(pred_date,'%Y-%m-%d')  +dt.timedelta(days=3))
        else:
            pass
    else:
        pass

    for ticker in tickers:
        print(f'Working on {ticker}')

        model_exists = None
        try:
            model_path = 'production_modeling/current_production_'+ticker+'_price_model.h5'
            price_model = load_model(model_path)
            model_exists = True
        except OSError:
            model_exists = False


        if model_exists:
            print('Ticker model exists, moving foward with prediction.')
            data = gather_data_derive_vars(ticker,b_date,e_date)

            # Predicting Independent Variables
            # Seperate MinMax for each variable for best possible prediction of value.
            scaler = MinMaxScaler(feature_range=(0,1))

            pred_ind_stock_vars = []
            ind_vars = data.loc[:,data.columns != 'Close']
            print(f'Training on independent variables for {ticker}')
            for i in ind_vars.columns:
                train = ind_vars[i]
                train_ss = train.values.reshape(-1,1)
                train_ss = scaler.fit_transform(train_ss)

                n_input = 1
                n_features = 1
                generator = TimeseriesGenerator(train_ss, train_ss, length=n_input, batch_size=5)

                K.clear_session()
                model = Sequential()
                model.add(LSTM(12, input_shape=(n_input,n_features)))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')

                early_stop = EarlyStopping(monitor='loss',patience=1,verbose=0)
                history = model.fit(generator,epochs=100,verbose=0,callbacks=[early_stop])

                batch = train_ss[-n_input:].reshape((1, n_input, n_features))
                pred_ind_stock_vars.append(model.predict(batch)[0][0])

            for_overall_model = np.array(pred_ind_stock_vars).reshape(1,1,8)
            model_path = 'production_modeling/current_production_'+ticker+'_price_model.h5'
            price_model = load_model(model_path)
            price_pred = price_model.predict(for_overall_model)[0][0]


            # Database Interaction -----------------------------------------------------
            # Gather last prediction value to compare to today's prediction value.
            # cur.execute("SELECT pred_value FROM keras_production WHERE ticker like %s ORDER BY row_id DESC LIMIT 1",
            #     (ticker,))
            # last_pred_value = float(cur.fetchall()[0][0])

            print(f'Last Predicted Value: {last_pred_value:.3f}\nCurrent Prediction Value: {price_pred:.3f}')

            tomorrow_pred = 'UP' if last_pred_value < price_pred else 'DOWN'
            template_bucket += f"""**{ticker}** is predicted to move **{tomorrow_pred}** in the next session.\n"""
            # Database Interaction -----------------------------------------------------
            # Gather current p_id. There must be a better way...
            # cur.execute('SELECT * FROM keras_production')
            # data_pull = cur.fetchall()
            # insert_id = len(data_pull) + 1
            # cur.execute("""INSERT INTO keras_production (row_id, run_date, pred_date, ticker, pred_value, pred_direction, true_direction)
            #            VALUES (%s, %s, %s, %s, %s, %s, %s);""",
            #         (insert_id,today,pred_date,ticker,float(price_pred),tomorrow_pred,None))
            # local_db_conn.commit()

            print(f'Finished with {ticker}.\n')

        else:

            print('Ticker model not found, running initial prediction.')

            data = gather_data_derive_vars(ticker,b_date,e_date)

            split_date = pd.Timestamp('04-01-2021')

            train_df, test_df = data[:split_date],data.loc[split_date:]
            test_index = test_df.index

            ss = MinMaxScaler(feature_range=(0,1),)
            train_sc = ss.fit_transform(train_df)
            test_sc = ss.transform(test_df)

            X_train = train_sc[:,1:]
            y_train = train_sc[:,0]

            X_test = test_sc[:,1:]
            y_test = test_sc[:,0]

            X_train = X_train[:,None]
            X_test = X_test[:,None]

            model_data_df = []
            # Keras Model -----------------------------------------------------
            K.clear_session()
            price_model = Sequential()
            price_model.add(LSTM(25, input_shape=(1,8)))
            price_model.add(Dropout(0.15))
            price_model.add(Dense(15,input_shape=(10,10),activation='relu'))
            price_model.add(Dense(5,input_shape=(10,1),activation='relu'))
            price_model.add(Dense(1))
            price_model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01))

            early_stop = EarlyStopping(monitor='loss',patience=1,verbose=0)
            price_model.fit(X_train,y_train, epochs=100, batch_size=32,verbose=0,callbacks=[early_stop])

            save_path = 'production_modeling/current_production_'+ticker+'_price_model.h5'
            price_model.save(save_path)

            last_pred_value = price_model.predict(X_test)[-2][0] # Yesterday's prediction

            # Predicting Independent Variables
            scaler = MinMaxScaler(feature_range=(0,1))

            pred_ind_stock_vars = []
            ind_vars = data.loc[:,data.columns != 'Close']
            print(f'Training on independent variables for {ticker}')
            for i in ind_vars.columns:
                train = ind_vars[i]
                train_ss = train.values.reshape(-1,1)
                train_ss = scaler.fit_transform(train_ss)

                n_input = 1
                n_features = 1
                generator = TimeseriesGenerator(train_ss, train_ss, length=n_input, batch_size=5)

                K.clear_session()
                model = Sequential()
                model.add(LSTM(12, input_shape=(n_input,n_features)))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')

                early_stop = EarlyStopping(monitor='loss',patience=1,verbose=0)
                history = model.fit(generator,epochs=100,verbose=0,callbacks=[early_stop])


                batch = train_ss[-n_input:].reshape((1, n_input, n_features))
                # Actual prediction
                ind_var_pred_value = model.predict(batch)[0][0]
                # Inverse Transform
                ind_var_pred_value = scaler.inverse_transform(ind_var_pred_value.reshape(-1,1))
                # Save "True to Size" Value for Overall Model
                pred_ind_stock_vars.append(ind_var_pred_value[0][0])

            # Scale Values Similar to Original Training
            pred_ind_stock_vars = np.array(pred_ind_stock_vars).reshape(-1,1)
            pred_ind_stock_vars = scaler.transform(pred_ind_stock_vars)
            for_overall_model = np.array(pred_ind_stock_vars).reshape(1,1,8)

            price_pred = price_model.predict(for_overall_model)[0][0]

            print(f'Last Predicted Value: {last_pred_value:.3f}\nCurrent Prediction Value: {price_pred:.3f}')

            tomorrow_pred = 'UP' if last_pred_value < price_pred else 'DOWN'
            template_bucket += f"""**{ticker}** is predicted to move **{tomorrow_pred}** in the next session.\n"""
            # Database Interaction -----------------------------------------------------
            # Gather current p_id. There must be a better way...
            # cur.execute('SELECT * FROM keras_production')
            # data_pull = cur.fetchall()
            # insert_id = len(data_pull) + 1
            # cur.execute("""INSERT INTO keras_production (row_id, run_date, pred_date, ticker, pred_value, pred_direction, true_direction)
            #            VALUES (%s, %s, %s, %s, %s, %s, %s);""",
            #         (insert_id,today,pred_date,ticker,float(price_pred),tomorrow_pred,None))
            # local_db_conn.commit()

            print(f'Finished with {ticker}.\n')
    # local_db_conn.close()
    print('\nFinalized Modeling.')
    print('---------------------------------------------------------------------')
    return template_bucket

# Would be utilized in a function through discord or scheduled.
def update_true_direction():
    local_db_conn = psycopg2.connect('REDACTED')
    cur = local_db_conn.cursor()
    date = dt.datetime.today().date()
    # One Month Data Beginning
    b_date = date - relativedelta(weekday=FR(-4))
    # Gather all ticker values in the database
    cur.execute('SELECT DISTINCT ticker FROM keras_production')
    ticker_list = cur.fetchall()
    len_list = len(ticker_list)
    for i in range(0,len_list):
        ticker_list[i] = ticker_list[i][0]

    # Update True Direction for Each Ticker
    for ticker in ticker_list:
        # Gather Stock Data
        data = pdr.DataReader(ticker,'yahoo',b_date,dt.datetime.today().date())
        # Check Closing Values and Assign True Direction
        data['Close Diff'] = data['Close'] - data['Close'].shift(1)
        data.dropna(inplace=True)
        data['True Direction'] = data['Close Diff'].apply(lambda x: 'DOWN' if x < 0 else 'UP')

        # Find most Recent Prediction
        cur.execute('SELECT * FROM keras_production WHERE ticker LIKE %s ORDER BY row_id ASC', (ticker,))
        data_pull = cur.fetchall()[-1]
        last_model_date = str(data_pull[1])

        previous_day_true_direction = data[(data.index == last_model_date)]['True Direction'].values[0]

        cur.execute("UPDATE keras_production SET true_direction = %s WHERE pred_date= %s AND ticker = %s",
                (previous_day_true_direction,date,ticker))
        local_db_conn.commit()

# client = discord.Client()
# @client.event
# async def on_ready():
#     print('We have logged in as {0.user}'.format(client))
#
# @client.event
# async def on_message(message):
#     if message.author == client.user:
#         return
#
#     if message.content.startswith('$Direction Prediction'):
#         user_tickers = ast.literal_eval(str(message.content).split("-")[1])
#         e_date = str(dt.datetime.today().date() + dt.timedelta(days=1))
#         message_text = production_direction_model(user_tickers,'2015-12-31',e_date)
#         await message.channel.send(message_text)


e_date = str(dt.datetime.today().date() + dt.timedelta(days=1))
print(production_direction_model(['PYPL'],'2015-12-31',e_date))

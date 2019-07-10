# encoding: UTF-8

import gzip
import re
import csv
import os
import os.path

import pandas as pd
import numpy as np

# Set all 30 features' names
col_names = [
             "trade.date", "contract.number",
             "affair.number", "contract.name",
             "new.price", "day.high.price",
             "day.low.price", "new.vol",
             "vol", "turn.vol",
             "start.open.interest", "open.interest",
             "delta.open.interest", "today.settlement",
             "history.low.price", "history.high.price",
             "daily.limit.high", "daily.limit.low",
             "yesterday.settlement", "yesterday.close",
             "highest.buy", "highest.buy.vol",
             "implied.buy", "lowest.sell",
             "lowest.sell.vol", "implied.sell",
             "day.mean", "time",
             "open", "close"
             ]


def get_best_quotes(variety_str, path, col_names):
    '''Get day data of a variety and return a dataframe.'''

    date = re.search(r'[0-9]{8}', path).group(0)

    # Open a new csv file to write the data
    with open('C:/HTH/DFIT/reinforcement/clean/' +
              'temp_' + date + '.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(col_names)
        with gzip.open(path, 'r') as day_file:
            for line in day_file:
                data_string = str(line, encoding="utf8")
                # Check whether it is from Best_Quotes
                best_quotes = re.search(r'b;(.*?);', data_string)
                if best_quotes:
                    best_quotes_string = best_quotes.group()
                    # Check whether it is from the variety
                    if re.search((r',' + variety_str + r'[0-9]{4}'), best_quotes_string):
                        # Get rid of 'b;' and ';'
                        best_quotes_string = best_quotes_string[2:-1]
                        best_quotes_dataline =\
                            np.array(best_quotes_string.split(','))
                        writer.writerow(best_quotes_dataline)

    # Read data from csv file, save them in dataframe.
    df_variety = pd.read_csv('C:/HTH/DFIT/reinforcement/clean/' +
                             'temp_' + date + '.csv',
                             encoding='gbk')
    # Delete temporary csv
    os.remove('C:/HTH/DFIT/reinforcement/clean/' +
              'temp_' + date + '.csv')

    return df_variety


def get_donimant_contract(df):
    '''From day data get dominant contract,
    delete data before open or after close and
    complete missing parts of close price.'''

    contract_list = list(set(df['contract.number']))

    n_contracts = len(contract_list)
    initial_df = df[0:n_contracts]
    sorted_df = initial_df.sort_values(by=['open.interest'])
    dominant_contract_number = sorted_df['contract.number'].iloc[-1]
    df_dominant_contract = df[df['contract.number'] ==
                              dominant_contract_number].reset_index(drop=True)

    # Delete data before open and after close
    index = []
    for i in range(len(df_dominant_contract)):
        time = df_dominant_contract['time'].iloc[i]
        if (time >= '15:00:01') and (time <= '21:00:00'):
            index.append(i)
        elif time >= '23:30:01':
            index.append(i)
        elif time <= '09:00:00':
            index.append(i)
        elif (time >= '10:15:01') and (time <= '10:30:00'):
            index.append(i)
        elif (time >= '11:30:01') and (time <= '13:30:00'):
            index.append(i)
    df_dominant_contract =\
        df_dominant_contract.drop(index).reset_index(drop=True)

    # Complete missing parts of close price
    df_dominant_contract['close'] = df_dominant_contract['close'].iloc[-1]

    return df_dominant_contract


def get_data_per_minute(df, trade_qty):
    '''Get one dataline per minute from original day dataframe.
    Add three features: last.minute.high.price, last.minute.low.price
    and last.minute.mean.price.'''

    df['last.minute.high.price'] = 0
    df['last.minute.low.price'] = 0

    index = [0]
    df.loc[0, 'last.minute.high.price'] = df['new.price'][0]
    df.loc[0, 'last.minute.low.price'] = df['new.price'][0]
    high = df['new.price'][1]
    low = df['new.price'][1]
    for i in range(1, len(df)):
        if df['new.price'][i] <= high and df['new.price'][i] >= low:
            pass
        elif df['new.price'][i] > high:
            high = df['new.price'][i]
        else:
            low = df['new.price'][i]
        if df['time'][i][0:5] != df['time'][i-1][0:5]:
            index.append(i)
            df.loc[i, 'last.minute.high.price'] = high
            df.loc[i, 'last.minute.low.price'] = low
            if i < len(df) - 1:
                high = df['new.price'][i + 1]
                low = df['new.price'][i + 1]

    df_minutes = df.loc[index].reset_index(drop=True)

    df_minutes['last.minute.mean.price'] = 0
    if df_minutes.loc[0, 'vol'] == 0:
        mean_price = df_minutes.loc[0, 'new.price']
    else:
        mean_price = df_minutes.loc[0, 'turn.vol'] / (df_minutes.loc[0, 'vol'] * trade_qty)
        mean_price = round(mean_price, 2)
    df_minutes.loc[0, 'last.minute.mean.price'] = mean_price
    for i in range(1, len(df_minutes)):
        delta_turn_vol = df_minutes.loc[i, 'turn.vol'] - df_minutes.loc[i - 1, 'turn.vol']
        delta_vol = df_minutes.loc[i, 'vol'] - df_minutes.loc[i - 1, 'vol']
        if delta_vol == 0:
            df_minutes.loc[i, 'last.minute.mean.price'] = df_minutes.loc[i - 1, 'last.minute.mean.price']
        else:
            mean_price = delta_turn_vol / (delta_vol * trade_qty)
            mean_price = round(mean_price, 2)
            df_minutes.loc[i, 'last.minute.mean.price'] = mean_price

    return df_minutes


def get_month_data(variety_str, year, month, trade_qty_dict):
    '''Get dominant contract data of a month.
    Input is like: 201601.
    If there isn't data of such month:
        return None;
    otherwise:
        return month data of dominant contract.'''

    trade_qty = trade_qty_dict[variety_str]

    yearmonth = str(year) + str(month).zfill(2)
    if os.path.exists('C:/HTH/DFIT/data/DCE/' + yearmonth):
        pass
    else:
        print("We don't have data of such month: %d/%d." % (year, month))
        return None

    df_month_list = []
    for i in range(1, 32):
        date_day = str(i).zfill(2)
        path = 'C:/HTH/DFIT/data/DCE/' + yearmonth + '/level2Quot_' +\
               yearmonth + date_day + '.csv.tar.gz'
        if os.path.exists(path):
            df = get_best_quotes(variety_str, path, col_names)
            df_dominant = get_donimant_contract(df)
            df_dominant_minutes = get_data_per_minute(df_dominant, trade_qty)
            df_month_list.append(df_dominant_minutes)
            print('level2Quot_' + yearmonth + date_day + ' done.')

    df_month = pd.concat(df_month_list, ignore_index=True)

    return df_month


def deal_with_duplicate_mess(df):
    '''Deal with some data mistakes'''

    df_reverse = df.sort_index(ascending=False).reset_index(drop=True)
    index = []
    identity_initial = str(df_reverse['trade.date'].iloc[0]) +\
        df_reverse['time'][0][0:5]
    identity_list = [identity_initial]
    for i in range(1, len(df_reverse)):
        identity = str(df_reverse['trade.date'].iloc[i]) +\
            df_reverse['time'][i][0:5]
        if identity not in identity_list:
            identity_list.append(identity)
        else:
            index.append(i)
    df_remain = df_reverse.drop(index)
    df = df_remain.sort_index(ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# variety_list = ['a', 'b', 'bb', 'c', 'cs',
#                 'fb', 'i', 'j', 'jd', 'jm',
#                 'l', 'm', 'p', 'pp', 'v',
#                 'y']
# =============================================================================
variety_list = ['a', 'c', 'cs',
                'i', 'j', 'jd', 'jm',
                'l', 'm', 'p', 'pp', 'v',
                'y']
trade_qty_dict = {'a': 10, 'b': 10, 'bb': 500, 'c': 10, 'cs': 10,
                  'fb': 500, 'i': 100, 'j': 100, 'jd': 10, 'jm': 60,
                  'l': 5, 'm': 10, 'p': 10, 'pp': 5, 'v': 5,
                  'y': 10}
for variety_str in variety_list:
    for year in [2015, 2016]:
        for month in range(1, 13):
            final_file_name = 'C:/HTH/DFIT/reinforcement/clean/' + variety_str + '_' +\
                str(year) + str(month).zfill(2) + '_minutes.csv'
            if os.path.exists(final_file_name):
                print("We have already dealt with data of such month: %d/%d and such variety: %s."
                      % (year, month, variety_str))
                continue
    
            df_yearmonth_minutes = get_month_data(variety_str, year, month, trade_qty_dict)
            if df_yearmonth_minutes is None:
                continue
            df_yearmonth_minutes = deal_with_duplicate_mess(df_yearmonth_minutes)
            df_yearmonth_minutes.to_csv(final_file_name)

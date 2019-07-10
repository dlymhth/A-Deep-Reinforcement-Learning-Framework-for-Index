# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:51:34 2019

@author: dlymhth
"""

import pandas as pd
import os

variety_list = ['a', 'i', 'j',
                'jm', 'm', 'p',
                'y']
month_list = ['201501', '201502', '201503', '201504', '201505',
              '201506', '201507', '201508', '201509', '201510',
              '201511', '201512', '201601']

working_space = 'C:/HTH/DFIT/reinforcement/clean/'

# concat all month
result_list = []
for variety in variety_list:
    df_list = []
    for month in month_list:
        file_path = working_space + variety + '_' + month + '_minutes.csv'
        df_month = pd.read_csv(file_path, index_col=0)
        df_list.append(df_month)
    df = pd.concat(df_list, ignore_index=True)
    result_list.append(df)

    df.to_csv(working_space + 'day&night/' + variety + '_minutes.csv')

# get some basic information
variety = 'a'
df = pd.read_csv(working_space + 'day&night/' + variety + '_minutes.csv', index_col=0)
trade_date_list = list(set(df['trade.date']))
trade_date_list.sort()

minute_list = list(set([x[0:5] for x in df['time']]))
minute_list.sort()
day_minute_list = [x for x in minute_list if x < '17:00']
night_minute_list = [x for x in minute_list if x > '17:00']
whole_minute_list = night_minute_list + day_minute_list

n_minute_day = len(day_minute_list)
n_minute_whole = len(whole_minute_list)

# complete the data
for variety in variety_list:

    df = pd.read_csv(working_space + 'day&night/' + variety + '_minutes.csv', index_col=0)
    
    df_date_list = []
    for trade_date in trade_date_list:
        
        df_date = df[df['trade.date'] == trade_date].reset_index(drop=True)
        
        if len(df_date) < 300:
            # day only
            if len(df_date) == n_minute_day:
                pass
            else:
                i = 0
                while i < n_minute_day:
                    if df_date.loc[i, 'time'][0:5] == day_minute_list[i]:
                        i += 1
                    else:
                        insert_row = df_date.loc[i, :].copy().to_frame().T
                        insert_row.index = [i - 0.5]
                        insert_row['time'] = day_minute_list[i] + ':00.000'
                        df_date = df_date.append(insert_row, ignore_index=False,
                                                 sort=False).sort_index().reset_index(drop=True)
        elif len(df_date) > 300:
            # day & night
            if len(df_date) == n_minute_whole:
                pass
            else:
                i = 0
                while i < n_minute_whole:
                    if df_date.loc[i, 'time'][0:5] == whole_minute_list[i]:
                        i += 1
                    else:
                        insert_row = df_date.loc[i, :].copy().to_frame().T
                        insert_row.index = [i - 0.5]
                        insert_row['time'] = whole_minute_list[i] + ':00.000'
                        df_date = df_date.append(insert_row,  ignore_index=False,
                                                 sort=False).sort_index().reset_index(drop=True)
        
        df_date_list.append(df_date)
    
    df_clean = pd.concat(df_date_list, ignore_index=True, sort=False)
    print(df_clean.shape)
    df_clean.to_csv(working_space + 'day&night/' + variety + '_minutes_clean.csv')
    
    os.remove(working_space + 'day&night/' + variety + '_minutes.csv')

import pandas as pd
import numpy as np


def create_UniqueID(df):
    """create UniqueID combining customerID and the respective day of week"""
    df['dayOfWeek'] = df.index.dayofweek
    df['dayOfWeek'].replace({0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}, inplace=True)
    df['UniqueID'] = df['dayOfWeek'].astype(str).str[:2] + df['customer_no'].astype(str)


def load_files():
    """load csv of every day of the week"""
    df_mon = pd.read_csv('../../../Git_Data/week08/monday.csv', index_col=0, sep=';', parse_dates=True)
    df_tue = pd.read_csv('../../../Git_Data/week08/tuesday.csv', index_col=0, sep=';', parse_dates=True)
    df_wed = pd.read_csv('../../../Git_Data/week08/wednesday.csv', index_col=0, sep=';', parse_dates=True)
    df_thu = pd.read_csv('../../../Git_Data/week08/thursday.csv', index_col=0, sep=';', parse_dates=True)
    df_fri = pd.read_csv('../../../Git_Data/week08/friday.csv', index_col=0, sep=';', parse_dates=True)

    # create UniqueID for every df
    df_list = [df_mon, df_tue, df_wed, df_thu, df_fri]

    for elem in df_list:
        elem = create_UniqueID(elem)

    # create one DataFrame combining all 5 dataframes
    df = pd.DataFrame()
    for elem in df_list:
        df = df.append(elem)

    return df











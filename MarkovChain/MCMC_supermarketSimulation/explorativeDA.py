import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import df_united


def calc_timeDiff():
    """calculate the time a customer spent in a section"""
    df = df_united.load_files()
    # create new feature columns for month, hour, year
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    # Calculate time differences
    df.reset_index(inplace=True)
    df_max = df.groupby(df['UniqueID'])['timestamp'].max()
    df_min = df.groupby(df['UniqueID'])['timestamp'].min()
    #create a new DataFrame for time min, max and duration
    df_min_max = pd.DataFrame()
    df_min_max['min'] = df_min
    df_min_max['max'] = df_max
    df_min_max['duration'] = df_max - df_min
    # merge df with df_min_max
    df_merged = pd.merge(df, df_min_max, left_on='UniqueID', right_on='UniqueID', how='left')
    return df_merged

def calc_noCustPerSection(df):
    """Calculate the total number of customers in each section"""
    df_section = df.groupby('location')['UniqueID'].count()
    return df_section

def calc_noCustPerSection_ot(df):
    """Calculate the total number of customers in each section over time ('ot')"""
    df_section_ot = df.groupby(['location', 'dayOfWeek', 'hour'])['UniqueID'].count()
    return df_section_ot
    #df_section_ot.plot.bar()

def noCustAtCheckout_ot(df):
    """Display the number of customers at checkout over time ('ot')"""
    df_checkout_ot = df.reset_index()
    df_checkout_ot = df_checkout_ot.loc[df_checkout_ot['location'] == 'checkout']
    return df_checkout_ot

def calc_timeInMarket(df):
    """Calculate the time each customer spent in the market"""
    df_merged = df[['duration','UniqueID']]
    return df_merged

def noCust_ot(df):
    """Calculate the total number of customers present in the supermarket over time ('ot')"""
    df_cust_total = df.groupby(['dayOfWeek', 'hour'])['UniqueID'].unique()  # .count()
    # df_cust_total.count()
    df_cust_total = pd.DataFrame(df_cust_total)
    # df_cust_total = iterrows():
    print(df_cust_total['UniqueID'])
    return df_cust_total

def totalRevenue(df):
    """caculate the total revenue of a customer"""
    # create new dataframe with prizes
    prize = {'location': ['fruit', 'spices', 'dairy', 'drinks'],
             'prize': [4, 3, 5, 6]}
    df_prize = pd.DataFrame(prize)
    # merge dataframes
    df_merged_prize = pd.DataFrame.merge(df, df_prize, left_on='location', right_on='location', how='left')
    # total revenue per customer
    df_merged_prize.groupby('UniqueID')['prize'].sum()
    return df_merged_prize


df1 = calc_timeDiff()
df2 = calc_noCustPerSection(df1)
df3 = calc_noCustPerSection_ot(df1)
df4 = noCustAtCheckout_ot(df3)
df5 = calc_timeInMarket(df1)
df6 = noCust_ot(df1)
df7 = totalRevenue(df1)

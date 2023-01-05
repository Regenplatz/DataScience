#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "0.0.1"
__status__ = "development"

import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta


def loadData(path):

    ## load data and skip rows with null values
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    ## assign date time column
    df["datetime"] = pd.to_datetime(df["datetime"])

    ## extract date time infos to separate columns
    df["dat"] = df["datetime"].dt.strftime("%m-%Y")

    ## drop columns of no interest
    df.drop(["season", "holiday", "workingday", "weather", "atemp", "humidity", "windspeed",
             "casual", "registered", "count"], axis=1, inplace=True)

    return df


def getTodaysDate():

    ## evaluated todays date time values
    today = pd.to_datetime("today").strftime("%d/%m/%Y")
    today_year = pd.to_datetime("today").year
    today_month = pd.to_datetime("today").month

    return today_year, today_month


def extractLast12Months(y, m, no_months, df):
    ## extract last 12 months and corresponding years (separately)
    years = []
    months = []
    for i in range(1, no_months + 1):
        date = datetime.date(y, m, 1)
        months.append((date - dateutil.relativedelta.relativedelta(months=i)).month)
        years.append((date - dateutil.relativedelta.relativedelta(months=i)).year)

    ## combine month with corresponding year
    selectedMonths = []
    for idx, elem in enumerate(years):
        selectedMonths.append((months[idx], elem))

    ## reversed order of list
    selectedMonths.reverse()

    ## convert date to string combination
    lastMonths = [f"{'{:02d}'.format(elem[0])}-{elem[1]}" for elem in selectedMonths]

    ## select only rows of selected time window
    df_sel = df.loc[df["dat"].isin(lastMonths)]

    ## 12-months mean and standard deviation
    df_mean_12m = np.mean(df_sel["temp"])
    df_std_12m = np.std(df_sel["temp"])

    return lastMonths, df_sel, df_mean_12m, df_std_12m


def evaluateMeanPerMonth(lastMonths, df_sel):

    meanPerMonth = {}
    for elem in lastMonths:
        ## select data of one month
        df_1month = df_sel[df_sel["dat"] == elem]
        meanPerMonth[elem] = np.mean(df_1month)

    ## create dataframe (12 rows, containing the means of each month)
    df = pd.DataFrame(meanPerMonth).transpose()
    # df.reset_index(inplace=True)
    # df.rename(columns={"index": "dat"}, inplace=True)

    return meanPerMonth, df


def main():

    ## load data of interest
    path = "data/bikeSharing/train.csv"
    df = loadData(path)

    ## get today's year and month
    y, m = getTodaysDate()

    ## get the last twelve months in format mm-yyyy
    y = 2012
    m = 1
    lastMonths, df_sel, df_mean_12m, df_std_12m = extractLast12Months(y, m, 12, df)


if __name__ == "__main__":
    main()

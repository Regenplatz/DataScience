#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "0.0.1"


import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta


def loadData(path, colName, date_col):
    """
    Load date from file.
    :param path: String, path to data file
    :return: dataframe, containing data of interest
    """
    ## load data
    df_original = pd.read_csv(path)
    df = df_original.copy()[[date_col, colName]]

    ## assign date time column
    df[date_col] = pd.to_datetime(df.loc[:, date_col])

    ## extract date time info (format: mm-yyyy) to separate columns
    df["date_mY"] = df[date_col].dt.strftime("%m-%Y")

    return df


def extractLastMonths(y, m, no_months, df, colName):
    """
    Extract data of the time window of interest.
    :param y: int, representing the year of the latest date of interest
    :param m: int, representing the month of the latest date of interest, e.g. 11 for November
    :param no_months: int, number of months, representing the size of th time window of interest
    :param df: dataframe, of which the data of the time window of interest shall be extracted
    :return:
        lastMonths: list, containing dates as strings in format mm-yyyy, e.g. '03-2021' for March 2021
        df_sel: dataframe, containing data of the time window of interest, each data point represents the monthly mean
        mean_lastMonths: float, representing the overall mean of the chosen time window
        std_lastMonths: float, representing the overall standard deviation of the chosen time window
    """
    years = []
    months = []
    mmyyyy = []
    date = datetime.date(y, m, 1)
    for i in range(1, no_months + 1):
        months.append((date - dateutil.relativedelta.relativedelta(months=i)).month)
        years.append((date - dateutil.relativedelta.relativedelta(months=i)).year)
        mmyyyy.append(f"{'{:02d}'.format(months[i-1])}-{years[i-1]}")
    mmyyyy.reverse()

    ## select only rows of selected time window
    df_sel = df.loc[df["date_mY"].isin(mmyyyy)]

    ## mean and standard deviation of the past months
    mean_lastMonths = np.mean(df_sel[colName], axis=0)
    std_lastMonths = np.std(df_sel[colName], axis=0)

    return mmyyyy, df_sel, mean_lastMonths, std_lastMonths


def evaluateMeanPerMonth(lastMonths, df_sel):
    """
    Evaluate each month's mean.
    :param lastMonths: lastMonths: list, containing dates as strings in format mm-yyyy, e.g. '03-2021' for March 2021
    :param df_sel: dataframe, containing data of the time window of interest, each data point represents the monthly mean
    :return: dataframe, containing data of each month's mean
    """
    meanPerMonth = {}
    for elem in lastMonths:
        ## select data of one month
        df_1month = df_sel[df_sel["date_mY"] == elem]
        meanPerMonth[elem] = np.mean(df_1month, axis=0)

    ## create dataframe (12 rows, containing the means of each month)
    df = pd.DataFrame(meanPerMonth).transpose()

    return df


def main():
    pass


if __name__ == "__main__":
    main()

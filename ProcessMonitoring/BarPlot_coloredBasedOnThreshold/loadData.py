#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta


def loadData(path, date_col):
    """
    Load date from file.
    :param path: String, path to data file
    :param date_col: String, name of date column 
    :return: dataframe, containing data of interest
    """
    ## load data
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    ## assign date time column
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    ## extract date time info (format: mm-yyyy) to separate columns
    df[date_col] = pd.to_datetime(df[date_col])
    df["date_mY"] = df[date_col].dt.strftime("%Y-%m")

    return df


def extractLastMonths(y, m, no_months, df, colName):
    """
    Extract data of the time window of interest.
    :param y: int, representing the year of the latest date of interest
    :param m: int, representing the month of the latest date of interest, e.g. 11 for November
    :param no_months: int, number of months, representing the size of th time window of interest
    :param df: dataframe, of which the data of the time window of interest shall be extracted
    :param colName: String, name of column of interest
    :return:
        mmyyyy: list, containing dates as strings in format mm-yyyy, e.g. '03-2021' for March 2021
        df_sel: dataframe, containing all data of the time window of interest
        mean_lastMonths: float, representing the overall mean of the chosen time window
        std_lastMonths: float, representing the overall standard deviation of the chosen time window
    """
    years = []
    months = []
    yyyymm = []
    date = datetime.date(y, m, 1)
    for i in range(1, no_months + 1):
        years.append((date - dateutil.relativedelta.relativedelta(months=i)).year)
        months.append((date - dateutil.relativedelta.relativedelta(months=i)).month)
        yyyymm.append(f"{years[i-1]}-{'{:02d}'.format(months[i-1])}")
    yyyymm.reverse()

    ## select only rows of selected time window
    df_sel = df.loc[df["date_mY"].isin(yyyymm)]

    ## mean and standard deviation of the chosen time window
    mean_lastMonths = np.mean(df_sel[colName], axis=0)
    std_lastMonths = np.std(df_sel[colName], axis=0)

    return yyyymm, df_sel, mean_lastMonths, std_lastMonths


def evaluateMeanPerMonth(lastMonths, df_sel, colName):
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
    df = pd.DataFrame.from_dict(meanPerMonth, orient="index", columns=[colName])

    return meanPerMonth, df


def main():
    pass


if __name__ == "__main__":
    main()

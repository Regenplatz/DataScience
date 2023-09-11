#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


## load packages
import pandas as pd


def loadDataFrame():

    pathToData = "enthalpy_data.csv"

    ## load data into dataframe and create datetime-specific columns
    df = pd.read_csv(pathToData)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month
    df["week"] = df["date_time"].dt.isocalendar().week
    df["weekday"] = df["date_time"].dt.weekday + 1
    df["hour"] = df["date_time"].dt.hour
    df.dropna(axis=0, inplace=True)
    for elem in ["year", "month", "week", "weekday", "hour"]:
        df[elem] = df[elem].astype(int)

    ## select only positive energy values
    df["energy_kWh"] = df["energy_kWh"].astype(float)
    df = df[df["energy_kWh"] > 0]

    df = df[["date_time", "year", "month", "week", "weekday", "hour",
             "temperature", "pressure", "humidity", "enthalpy",
             "energy_kWh"]]

    return df

#!/usr/bin/env python3

__author__ = "WhyKiki"
__version__ = "1.0.1"


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def subsampleData(df, user_limit, media_limit):
    """
    Create subsample of data
    :param df: dataframe, data that is to be subsampled from
    :return: dataframe, the subsampled data
    """
    df = df[df["user_id"] < user_limit]
    df = df[df["media_id"] < media_limit]

    return df


def groupByData(df):
    """
    Group data in given dataframe
    :param df: dataframe, data that needs to be grouped
    :return: dataframe, in groupby format
    """
    ## group data by IDs
    df = df.groupby(["user_id", "media_id", "genre_id", "artist_id"],
                    as_index=False)["is_listened"].sum()

    ## factorize ID columns
    df["user_id"] = df["user_id"].factorize()[0]
    df["media_id"] = df["media_id"].factorize()[0]
    df["genre_id"] = df["genre_id"].factorize()[0]
    df["artist_id"] = df["artist_id"].factorize()[0]

    ## shift ids to start at 0
    for col in df.columns:
        df[col] = df[col] - df[col].min()

    ## column if song was listened to at least once or not (binary)
    df["is_listened_binary"] = np.where(df["is_listened"] == 0, 0, 1)

    # ## column for songs listened to more than once (binary)
    # df["is_listened_greater1"] = np.where(df["is_listened"] > 1, 1, 0)
    # df[df["is_listened"] > 0]

    ## drop column "is_listened" because it contains the sum of the song being listened and replace
    ## with the newly created binary "is_listened" column
    df.drop(["is_listened"], axis=1, inplace=True)
    df.rename(columns={"is_listened_binary": "is_listened"}, inplace=True)

    return df


def readData(test_csv=False, subsample=True, groupBy=False):
    """
    read data into dataframes
    :param test_csv: boolean, if test.csv shall be read in.
                     If False, train.csv is split into train and test data.
    :param subsample: boolean, if a subsample should be created
    :return: dataframes, train and test data
    """
    df_train = pd.read_csv("../Data/train.csv")
    if subsample:
        df_train = subsampleData(df_train, 550, 550000)

    if test_csv:
        df_test = pd.read_csv("../Data/test.csv")
        if subsample:
            df_test = subsampleData(df_test, 10000, 10000000)
        X_train = df_train.drop(["is_listened"], axis=1)
        y_train = df_train["is_listened"]
        X_test = df_test
        y_test = pd.DataFrame([np.nan] * len(X_test))
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        df = pd.concat([df_test, df_train])
    else:
        if groupBy:
            df_train = groupByData(df_train)
        df = df_train.copy()
        X = df_train.drop(["is_listened"], axis=1)
        y = df_train["is_listened"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                            random_state=123)

    return df, X, y, X_train, X_test, y_train, y_test


def main():
    df, X, y, X_train, X_test, y_train, y_test = readData()
    print(df.shape)
    print(X.shape, y.shape)
    print(X_test.shape, X_train.shape, y_test.shape, y_train.shape)
    None


if __name__ == "__main__":
    main()


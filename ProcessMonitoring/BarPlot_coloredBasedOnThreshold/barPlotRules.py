#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


def assignColors(df, colName):
    """
    Create new column in dataframe for color coding according to predefined thresholds.
    :param df: dataframe, containing the data of interest
    :param colName: String, column name of the column that is used to assess the colors
    :return: dataframe, containing a column for color codes
    """

    ## define thresholds for coloring
    coloredLines = {"green": 15, "yellow": 20, "red": 25}

    ## define color of bars based on defined thresholds
    df["colorCode"] = df.apply(lambda x: "Red" if x[colName] >= coloredLines["red"] else
                                         ("Yellow" if x[colName] >= coloredLines["yellow"] else "Green"),
                                         axis=1)
    return df, coloredLines


def main():
    pass


if __name__ == "__main__":
    main()

#!/usr/bin/python3

from typing import Tuple
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

## define paths to which images of plots should be saved
path_heatmap = "heatmaps"
path_scatter = "scatterplots"

## define name of date time column and target variable
dtColName = "Datetime"
col_color = "offsetWindDirection"

## define timely aggregation value
timeAgg = ["900S", "720S", "600S", "450S", "300S", "150S",
           "120S", "90S", "60S", "30S", "20S", "10S", "5S"]

## define wind speed ranges
ws_vals = [[2.0, 2.5], [2.5, 3.0], [3.0, 3.5], [3.5, 4.0],
           [4.0, 4.5], [4.5, 5.0], [5.0, 5.5], [5.5, 6.0]]

power_vals = [[0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 2.5], [2.5, 3.0],
              [3.0, 3.5], [3.5, 4.0], [4.0, 4.5], [4.5, 5.0], [5.0, 5.5], [5.5, 6.0]]

## define column names of potentially influental features
colNames = ['RotorSpeed', 'GeneratorSpeed', 'GeneratorTemperature',
            'WindSpeed', 'PowerOutput', 'SpeiseSpannung',
            'StatusAnlage', 'MaxWindHeute', 'elWindDirection',
            'offsetWindDirection', 'PitchDeg']

## define integers for weekday as text values
weekdays = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu",
            5: "Fri", 6: "Sat", 7: "Sun"}

## define bin size for each column (arbitrarily; chosen by visually checking on min and max values)
dict_colBins = {'RotorSpeed': 20, 'GeneratorSpeed': 100,
                'GeneratorTemperature': 20, 'WindSpeed': 0.5,
                'PowerOutput': 0.25, 'BeschlenigungsSensor': 5,
                'SpeiseSpannung': 5, 'StatusAnlage': 20000,
                'MaxWindHeute': 20000, 'WindDirection': 5,
                'elWindDirection': 5, 'offsetWindDirection': 5,
                'PitchDeg': 10, 'AzWin': 50,
                }

## define start of first bin
dict_startBinAt = {'PowerOutput': -0.25,
                   'offsetWindDirection': -180.0}


def createDtCols(df: pd.DataFrame, dtColName: str) -> pd.DataFrame:
    """create new columns for datetime specifics"""
    df.reset_index(inplace=True)
    df["year"] = pd.to_datetime(df[dtColName]).dt.year.astype("int")
    df["month"] = pd.to_datetime(df[dtColName]).dt.month.astype("int")
    df["weekday"] = pd.to_datetime(df[dtColName]).dt.weekday + 1
    df["hour"] = pd.to_datetime(df[dtColName]).dt.hour
    df["dayOfMonth"] = pd.to_datetime(df[dtColName]).dt.day
    df["dayOfYear"] = pd.to_datetime(df[dtColName]).dt.dayofyear
    df["weekday_text"] = df["weekday"].map(weekdays)
    return df


def resampleData(df: pd.DataFrame, timeToAggregate: str = "300S") -> pd.DataFrame:
    """Resample data for a given time period. Use average as aggregation function."""
    return df.resample(timeToAggregate, on=dtColName, label="right").mean(numeric_only=True)


def createBins(df: pd.DataFrame, colName: str, binSize: int) -> pd.DataFrame:
    """Create new column for binned attributes"""
    if colName in ("PowerOutput", "offsetWindDirection"):
        df[f"{colName}Bin"] = pd.cut(df[colName],
                                     np.arange(dict_startBinAt[colName], df[colName].max() + binSize, binSize))
    else:
        df[f'{colName}Bin'] = pd.cut(df[colName], np.arange(0.0, df[colName].max() + binSize, binSize))
    return df


def createBinsAsFloat(df: pd.DataFrame, binsize: float, colName: str, newColName: str) -> Tuple[pd.DataFrame, dict]:
    """Create bins of a continuous variable. To avoid problems with pd.Interval,
    use center of bin range being displayed as float value"""

    ## create dictionary for bins with center of bin as key and [start_value, end_value] as value
    start_val = df[colName].min() + binsize / 2
    end_val = df[colName].max() - binsize / 2
    range_val = np.arange(start_val, end_val + binsize, 5.0)
    range_val = [np.around(x, decimals=1) for x in range_val]
    bins = [[x - binsize / 2, x + binsize / 2] for x in range_val]
    dict_bins = dict(zip(range_val, bins))

    ## create list with mapped values and create as new column in dataframe
    x = []
    for idx, row in df.iterrows():
        for k, v in dict_bins.items():
            if ((row[colName] >= v[0]) & (row[colName] <= v[1])):
                x.append(k)
                break
    df[newColName] = x
    return df, dict_bins


def createScatterPlot(df: pd.DataFrame, timeAggregation: str, windspeed_interval: list,
                      metric_windDir: str = "mean", metric_powerOutput: str = "median") -> None:
    """Create scatter plots including line for metric (e.g. mean or median) and save to path"""

    ## create scatter plot
    plt.scatter(df[col_color], df["PowerOutput"])
    plt.xlim(-200, 200)
    plt.ylim(-0.5, 12.5)
    plt.ylabel("PowerOutput [kW]")
    plt.xlabel(col_color)

    ## evaluate windDirection metric of dataset and insert vertical line for metric in scatter plot
    if metric_windDir == "median":
        int_metric_ws = df[col_color].median()
    elif metric_windDir == "mean":
        int_metric_ws = df[col_color].mean()
    else:
        int_metric_ws = None
    str_metric_ws = '{:.2f}'.format(int_metric_ws)
    plt.axvline(x=int_metric_ws, color='orange')
    plt.text(30, 1, str_metric_ws, ha='right', va='center', color='orange')

    ## evaluate powerOutput metric of dataset and insert horizontal line for PowerOutput in scatter plot
    if metric_powerOutput == "median":
        int_metric_po = df["PowerOutput"].median()
    elif metric_powerOutput == "mean":
        int_metric_po = df["PowerOutput"].mean()
    else:
        int_metric_po = None
    str_metric_po = '{:.2f}'.format(int_metric_po)
    plt.axhline(y=int_metric_po, color='magenta')
    plt.text(-10, int_metric_po+0.5, str_metric_po, ha='right', va='center', color='magenta')
    text_timeAggregation = f"TA{timeAggregation}"
    text_windspeed = f"WS{windspeed_interval}"
    text_windDir = f"WD({metric_windDir}={str_metric_ws})"
    text_powerOutput = f"PO({metric_powerOutput}={str_metric_po})"
    plt.title(f"{text_timeAggregation}_{text_windspeed}_{text_windDir}_{text_powerOutput}")

    ## save to path
    base = f"{path_scatter}/{metric_windDir}/"
    path = f"{base}/{text_timeAggregation}_{text_windspeed}_{text_windDir}_{text_powerOutput}.png"
    plt.savefig(path)
    plt.figure().clear()


def createHeatmap(df: pd.DataFrame, x_col: str, y_col: str, col_colorlegend: str,
                  timeAggregation: str, figsize: tuple = (7, 6)) -> None:
    """Create heatmaps with offsetWindDirection as colored attribute"""

    ## assign min and max values for "offsetWindDirection"
    degree = 30
    wd_max = degree
    wd_min = -degree
    # wd_max = df[col_colorlegend].max()
    # wd_min = df[col_colorlegend].min()

    ## pivot data for heatmap
    dfx = df[[col_colorlegend, x_col, y_col]]
    dfx.rename(columns={col_colorlegend: "bin"}, inplace=True)
    df_pivot = dfx.pivot_table(index=y_col, columns=x_col)

    ## create heatmap
    plt.figure(figsize=figsize)
    colormap = sns.diverging_palette(220, 20, as_cmap=True)
    s = sns.heatmap(df_pivot, fmt="", cmap=colormap, linewidths=0.20,
                    cbar_kws={'label': col_colorlegend},
                    vmin=wd_min,
                    vmax=wd_max
                    )
    s.set_title(f"{timeAggregation}")
    s.set_xlabel("WindSpeed [m/s]", fontsize=10)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    s.set_ylabel("Power Output [kW]", fontsize=10)
    s.set_yticklabels(s.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(f"{path_heatmap}/{timeAggregation}.png")
    plt.figure().clear()


def cutOffOutliers(df: pd.DataFrame, col: str, quant_percent: float) -> pd.DataFrame:
    """Cut off potential outliers"""

    ## define e.g. 99.99% quantiles
    quantiles = round((100 - quant_percent) / 100, 5)
    q_halves = quantiles / 2
    q_lower = q_halves
    q_upper = 1 - q_halves

    ## defines e.g. 99.99% percentile of respective column
    q = df[col].quantile([q_lower, q_upper])

    ## set all values to nan where value meets condition --> needed for later interpolation
    df[col][(df[col] < q[q_lower]) | (df[col] > q[q_upper])] = np.nan

    ## interpolate
    df[col] = df[col].interpolate(method="time", limit=5)

    return df


def preprocessData(df: pd.DataFrame, timeAggregation: str, selectedCols: list,
                   quant_percent: float, cutoff_powerOutput: float = None) -> pd.DataFrame:
    """Process the data to averaged data for a certain time range. Create bins for some columns
       and create plots to get further insights"""

    ## resample data: averaged data for time window
    df = resampleData(df=df, timeToAggregate=timeAggregation)

    ## fill NANs
    df["WindSpeed"].fillna(method="ffill", inplace=True)
    df["PowerOutput"].fillna(method="ffill", inplace=True)
    df["offsetWindDirection"].fillna(method="ffill", inplace=True)

    ## cut off outliers
    df = cutOffOutliers(df=df,
                        col="offsetWindDirection",
                        quant_percent=quant_percent)

    ## create bins for various columns
    for col in selectedCols[2:]:
        df = createBins(df=df, colName=col, binSize=dict_colBins[col])

    ## cut off low powerOutput values
    if cutoff_powerOutput is not None:
        df = df.loc[(df["PowerOutput"] > cutoff_powerOutput), :]

    return df


def plotHistograms(df: pd.DataFrame, figsize: tuple = (12, 8)) -> None:
    plt.figure(figsize=figsize)
    fig, axes = plt.subplots(3, 1)
    sns.histplot(x="PowerOutput", data=df,
                 binwidth=0.25, kde=True, ax=axes[0])
    sns.histplot(x="WindSpeed", data=df,
                 binwidth=0.5, kde=True, ax=axes[1])
    sns.histplot(x="offsetWindDirection", data=df,
                 binwidth=5, kde=True, ax=axes[2])
    plt.tight_layout()


def timeSeriesPlot(df: pd.DataFrame, y_col: str, rollingMean_days: int) -> None:
    plt.figure(figsize=(12, 5))

    ## plot data
    sns.lineplot(x="Datetime",
                 y=y_col,
                 data=df,
                 label="Data")

    ## evaluate and plot rolling mean
    days = rollingMean_days * 60 * 24
    df[f"{y_col}_{rollingMean_days}d"] = df[y_col].rolling(days).mean()
    sns.lineplot(x="Datetime",
                 y=f"{y_col}_{rollingMean_days}d",
                 data=df,
                 label=f"Rolling Mean ({rollingMean_days}d)")
    # plt.text('2022-01', 5, df[f"{y_col}_{rollingMean_days}d"][0], ha='right', va='center', color='orange')
    plt.title(y_col)
    plt.xticks(rotation=45)
    plt.show()


def main():

    ## if folder does not exist: create
    if not os.path.exists(path_heatmap):
        os.makedirs(path_heatmap)
    if not os.path.exists(path_scatter):
        os.makedirs(path_scatter)

    ## load data and convert datetime column to data type datetime
    df = pd.read_parquet("dump.parquet")
    df.reset_index(inplace=True)
    df[dtColName] = pd.to_datetime(df[dtColName])

    ## use only selected columns
    selectedCols = ["year", "Datetime", "PowerOutput", "WindSpeed", "offsetWindDirection"]
    df_sel = df[selectedCols[1:]]
    df_sel = createDtCols(df=df_sel, dtColName=dtColName)

    ## drop NANs and map values greater 6 to 6 and values smaller 2 as 2
    df_sel = df_sel.dropna()
    df_sel["WindSpeed"] = np.where(df_sel["WindSpeed"] > 6.0, 6.0, df_sel["WindSpeed"])
    df_sel["WindSpeed"] = np.where(df_sel["WindSpeed"] <= 2.0, 2.00000001, df_sel["WindSpeed"])

    ## create plots
    for ta in timeAgg[0: -1]:

        ## preprocess data
        df_preprocessed = preprocessData(df=df_sel,
                                         timeAggregation=ta,
                                         selectedCols=selectedCols,
                                         quant_percent=99.00,
                                         cutoff_powerOutput=0.0005
                                         )
        ## create heatmaps
        createHeatmap(df=df_preprocessed,
                      x_col="WindSpeedBin",
                      y_col="PowerOutputBin",
                      col_colorlegend=col_color,
                      timeAggregation=ta,
                      figsize=(7, 6)
                      )

        # ## for each wind speed interval create scatter plot
        # for interval in ws_vals:
        #
        #     ## select data of a specific windspeed interval
        #     df_ws = df_sel.loc[(df_sel["WindSpeed"] > interval[0]) & (df_sel["WindSpeed"] <= interval[1]), :]
        #
        #     ## preprocess data
        #     df_preprocessed = preprocessData(df=df_ws,
        #                                      timeAggregation=ta,
        #                                      selectedCols=selectedCols,
        #                                      quant_percent=99.00,
        #                                      cutoff_powerOutput=0.0005
        #                                      )
        #
        #     ## create scatter plots and heatmaps
        #     createScatterPlot(df=df_preprocessed,
        #                       timeAggregation=ta,
        #                       windspeed_interval=interval,
        #                       metric_windDir="mean",
        #                       metric_powerOutput="median"
        #                       )


if __name__ == "__main__":
    main()

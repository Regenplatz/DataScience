#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"

import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns

## months and weekday mappings (integer to String)
months = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
          7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
weekdays = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def loadData(path: str, year: int) -> pd.DataFrame:
    """Load data, create columns for date specifics and select data from the year of interest"""
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df["year"] = pd.to_datetime(df["date_time"]).dt.year.astype("int")
    df["month"] = pd.to_datetime(df["date_time"]).dt.month.astype("int")
    df["weekday"] = pd.to_datetime(df["date_time"]).dt.weekday + 1
    df["hour"] = pd.to_datetime(df["date_time"]).dt.hour
    df["dayOfMonth"] = pd.to_datetime(df["date_time"]).dt.day
    df["dayOfYear"] = pd.to_datetime(df["date_time"]).dt.dayofyear
    df["wDay"] = df["weekday"].map(weekdays)
    return df.loc[df["year"] == year, :]


def createHeatmap(df, normalize: bool = False) -> None:
    """Create heatmaps for each month of the year in 1 plot"""

    ## assign min and max temperatures of the whole year
    energy_max = df["energy_kWh"].max()
    energy_min = df["energy_kWh"].min()

    ## define plot with 12 subplots
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    title = f"Energy Consumption [kWh] in 2021"
    fig.suptitle(title, fontsize=20, y=1.01)

    ## create 12 subplots (1 per month)
    for idx, m in enumerate(range(1, 13)):

        ## select columns of interest and pivot data
        df2 = df.loc[df["month"] == m, :][["energy_kWh", "dayOfMonth", "hour", "wDay"]]
        df_pivot = df2.pivot_table(index=["dayOfMonth", "wDay"], columns="hour")

        ## define row and column indices
        row_no = int(idx / 3)
        col_no = idx % 3

        ## define maximum energy value per month
        if normalize is False:
            energy_max = df2["energy_kWh"].max()
            energy_min = df2["energy_kWh"].min()

        ## create heatmap
        axes[row_no, col_no] = sns.heatmap(df_pivot, fmt="", cmap='coolwarm', linewidths=0.20,
                                           ax=axes[row_no, col_no],
                                           xticklabels=list(range(0, 24)),
                                           vmin=energy_min,
                                           vmax=energy_max)
        axes[row_no, col_no].set_title(f"{months[m]}", fontsize=16)
        axes[row_no, col_no].set_xlabel("Hour", fontsize=14)
        axes[row_no, col_no].set_ylabel("Day of Month", fontsize=14)

    fig.tight_layout()


def plotConsumption(df: pd.DataFrame, differentiated: bool = False, allPlotsInOne: bool = True) -> None:
    ## define labels for x-axis
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
               'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']

    if differentiated is True:
        d = "_diff"
        df["energy_kWh"] = df["energy_kWh"].diff()
        df["temperature"] = df["temperature"].diff()
    else:
        d = ""

    ## both plots being displayed within a single plot
    if allPlotsInOne is True:

        ax = df.plot(x="dayOfYear", y="energy_kWh", legend=False, color="blue", linewidth=2.5)
        ax2 = ax.twinx()
        ax2.set_ylabel("Temperature [°C] in 2021", fontsize=18)
        ax2.tick_params(labelsize=18)

        df.plot(x="dayOfYear", y="temperature", ax=ax2, legend=False, color="magenta", linewidth=2.5, figsize=(24, 8))
        ax.figure.legend(prop=dict(size=18))
        ax.set_title(f"Energy [kWh] and Outdoor Temperature [°C] in 2021{d}", fontsize=24)
        ax.set_ylabel("Energy [kWh]", fontsize=18)
        ax.set_xlabel("Time", fontsize=18)
        ax.tick_params(labelsize=18)
        ax.xaxis.set_ticks(np.arange(1, 366, 30.4))
        ax.set_xticklabels(labels=xlabels)

    ## both plots being displayed in separate plots
    else:

        ## define and create subplots
        fig, ax = plt.subplots(2, figsize=(20, 20))

        sns.lineplot(data=df, x="dayOfYear", y="temperature", ax=ax[0], color="magenta")
        ax[0].set_title(f"Temperature [°C] in 2021{d}", fontsize=24)
        ax[0].set_ylabel("Temperature [°C]", fontsize=18)
        ax[0].set_xlabel("Time", fontsize=18)
        ax[0].tick_params(labelsize=18)
        ax[0].xaxis.set_ticks(np.arange(1, 366, 30.4))
        ax[0].set_xticklabels(labels=xlabels)

        sns.lineplot(data=df, x="dayOfYear", y="energy_kWh", ax=ax[1], color="blue")
        ax[1].set_title(f"Energy [kWh] in 2021{d}", fontsize=24)
        ax[1].set_ylabel("Energy [kWh]", fontsize=18)
        ax[1].set_xlabel("Time", fontsize=18)
        ax[1].tick_params(labelsize=18)
        ax[1].xaxis.set_ticks(np.arange(1, 366, 30.4))
        ax[1].set_xticklabels(labels=xlabels)


def main():
    pass


if __name__ == "__main__":
    main()

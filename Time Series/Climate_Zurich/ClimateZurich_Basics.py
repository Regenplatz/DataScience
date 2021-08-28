#!/usr/local/bin/python

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_squared_log_error, \
                            mean_absolute_error, r2_score, make_scorer


class ReadConvertData:


    def __init__(self, filename):
        self.filename = filename


    def txtToDF(self):
        """ Read data and select years of interest """
        df = pd.read_csv(self.filename, header=15, index_col=0)

        # replace spaces in column names
        df.columns = [c.replace(' ', '') for c in df.columns]

        # convert given temperature values to °Celsius
        df['TG'] = df['TG'] * 0.1

        # select years of interest and create columns for year, month and week
        df['date'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
        df['year'] = pd.DatetimeIndex(df['date']).year
        df = df[df['year'] < 2020]
        df['month'] = pd.DatetimeIndex(df['date']).month
        df['week'] = pd.DatetimeIndex(df['date']).isocalendar().week.values

        # drop NAs and columns of no interest
        df.dropna(inplace=True)
        df.drop(columns=['Q_TG', 'SOUID', 'DATE'], inplace=True)

        # set date as index
        self.df = df.set_index('date')

        # return self.df


    def writeToCSV(self):
        """ save dataframe as csv file """
        self.df.to_csv("temperatesZurich.csv")



class BasicTSA:

    def __init__(self, data):
        """
        convert non-series object to series
        :param data: series of interest, to be analyzed for time series analysis
        """
        if not isinstance(data, pd.Series):
            self.series = pd.Series(data)
        else:
            self.series = data

    def dickeyFuller(self):
        """ augmented Dickey-Fuller test (adf) """
        result = adfuller(self.series.values)
        # if p-value > 0.01 calculate adf on difference
        if result[1] > 0.01:
            self.series = self.series.diff()
            self.series = self.series.dropna()
            result = adfuller(self.series)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    def acf_pacf(self):
        """ plot autocorrelation function (acf) and partial autocorrelation function (pacf) """
        fig, ax = plt.subplots(1, 2, figsize=(18, 5))
        sm.graphics.tsa.plot_acf(self.series, ax=ax[0]);
        sm.graphics.tsa.plot_pacf(self.series, ax=ax[1]);
        plt.show()



class PlotData(ReadConvertData):

    def __init__(self, filename):
        """
        initialize PlotData object, inherit from another class
        :param filename: String, name of file to be read and converted to dataframe
        """
        super().__init__(filename)
        self.txtToDF()

    def preparePlotData(self, column, time_period):
        """
        prepare for plotting data as is
        :param column: String, name of feature to be analyzed
        :param time_period: String, time interval for grouping
        """
        df_mean = self.df.groupby([time_period]).mean()
        plt.plot(df_mean[column])
        plt.xlabel(time_period)
        plt.ylabel('temperature [°C]')
        if time_period == "year":
            m, b = np.polyfit(self.df[time_period], self.df[column], 1)
            plt.plot(self.df[time_period], m * self.df[time_period] + b)
        else:
            plt.ylim(ymax=20)

    def plotData(self):
        """ plot data for year, month and week """
        fig = plt.figure(figsize=(18, 6))
        fig.add_subplot(2, 1, 1)
        self.preparePlotData('TG', 'year')
        fig.add_subplot(2, 2, 3)
        self.preparePlotData('TG', 'month')
        fig.add_subplot(2, 2, 4)
        self.preparePlotData('TG', 'week')
        plt.tight_layout()
        plt.show()



class MovingAverage:

    def __init__(self, series, window):
        """
        initialize MovingAverage object with params
        :param series: series, containing information of interest for time series analysis
        :param window: integer, time window for rolling mean (in days)
        """
        self.series = series
        self.df = series.to_frame()
        self.window = window

    def calculateMovingAverage(self):
        """
        Calculate average of last n observations
        :return: float, mean value of the analyzed series
        """
        return round(np.average(self.series[-self.window:]), 2)

    def plotMovingAverage(self, plot_intervals=False, scale=1.96, plot_anomalies=False):
        """
        plot with moving average
        :param plot_intervals: boolean, show confidence intervals or not
        :param scale: float, scaling factor for plotting
        :param plot_anomalies: boolean, show anomalies or not
        """
        rolling_mean = self.series.rolling(window=self.window).mean()
        trend = self.series.rolling(window=self.window * 10).mean()

        plt.figure(figsize=(15, 5))
        plt.title("Moving average\n window size = {}".format(self.window))

        # Plot confidence intervals for smoothed values
        if plot_intervals:
            mae = mean_absolute_error(self.series[self.window:], rolling_mean[self.window:])
            deviation = np.std(self.series[self.window:] - rolling_mean[self.window:])
            lower_bond = rolling_mean - (mae + scale * deviation)
            upper_bond = rolling_mean + (mae + scale * deviation)
            plt.plot(upper_bond, "m--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond, "m--")

            # Having the intervals, find abnormal values
            if plot_anomalies:
                # first, creates series frame with index of df, filled with NAs !!
                anomalies = pd.DataFrame(index=self.df.index, columns=[self.df.columns])
                # write anomalies greater/lower than corresponding bonds to dataframe "anomalies"
                anomalies[self.series < lower_bond] = self.df[self.series < lower_bond]
                anomalies[self.series > upper_bond] = self.df[self.series > upper_bond]
                plt.plot(anomalies, "ro", markersize=10)

        plt.plot(self.series[self.window:], label="Actual values")
        plt.plot(rolling_mean, "r", label="Rolling mean trend (1 yr)")
        plt.plot(trend, "b", label="Rolling mean trend (10 yrs)")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()



class LagData:

    def __init__(self, df, feature_name):
        """
        initialize LagData object with params
        :param df: dataframe, containing data to be analyzed
        :param feature_name: String, name of feature containing information of interest for time series analysis
        """
        self.df = df
        self.feature_name = feature_name

    def generateLagData(self):
        """ generate lag data """
        for i in range(1, 13):
            self.df[f'{self.feature_name} -{i}month'] = self.df[self.feature_name].shift(i)
        self.df.dropna(inplace=True)

    def plotHeatmap(self):
        """ plot heatmap """
        sns.heatmap(self.df.drop(['year', 'month', 'week'], axis=1).corr())
        plt.show()


class Decompose_Transform:

    def __init__(self, df, column):
        """
        initialize Decompose_Transform object
        :param df: dataframe, containing data to be analyzed
        :param column: String, name of feature of interest for time series analysis
        """
        self.df = df
        self.column = column


    def decompose_ts(self):
        """ decompose time series to seasonal, trend and residuals """
        sdr = seasonal_decompose(self.df[self.column], freq=12, model='additive')
        sdr.plot()
        plt.show()
        None  # to avoid getting 2 similar plots (bug)

        # create new dataFrame with temp, seasonal, trend, residuals
        self.results = pd.DataFrame(data={self.column: self.df[self.column], \
                                     'seasonal': sdr.seasonal, 'trend': sdr.trend, 'residuals': sdr.resid})


    def transform_ts(self):
        """ transform data to difference, percentage change and logarithmic values """

        # diff: calculate difference to the past year as absolut value
        self.df['diff'] = self.df[self.column].diff()

        # pct change: calculate difference to the past year as relative value
        self.df['pct_change'] = self.df[self.column].pct_change()

        # log: calculate log values to get rid of heteroscedasticity and to remove exponential growth
        mask = (self.df[self.column] > 0)
        self.df['log'] = 0
        df2 = self.df[mask]
        self.df.loc[mask, 'log'] = np.log(df2[self.column])

        self.df.head()


def main():

    # plot data
    plot1 = PlotData('TG_STAID000244.txt')
    plot1.plotData()

    # calculate and plot moving average
    movingAvg = MovingAverage(plot1.df["TG"], 365)
    moving_average = movingAvg.calculateMovingAverage()
    print("Moving Average: ", moving_average)
    movingAvg.plotMovingAverage(plot_intervals=True, plot_anomalies=True)

    # generate lag data
    laggedData = LagData(plot1.df, "TG")
    laggedData.generateLagData()
    laggedData.plotHeatmap()

    # Basic Time Series Analysis: check for stationarity and plot acf and pacf
    basicTSA = BasicTSA(plot1.df["TG"])
    basicTSA.dickeyFuller()
    basicTSA.acf_pacf()

    # Decompose and Transform
    decomp_transform = Decompose_Transform(plot1.df, "TG")
    decomp_transform.decompose_ts()
    decomp_transform.transform_ts()


if __name__ == "__main__":
    main()

#!/usr/local/bin/python

import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import itertools
import warnings
warnings.filterwarnings("ignore")
import ClimateZurich_Basics as czb

plot1 = czb.PlotData('TG_STAID000244.txt')
plt.style.use('fivethirtyeight')


class Sarima:

    def __init__(self, data):
        """ initialize Sarima object
            :param data: series, to be analyzed
        """
        self.data = data
        self.sarima_params()

    def sarima_params(self):
        """ generate combinations of
               - p (autoregressive (AR)),
               - d (integrated) and
               - q triplets (moving average (MA))
        """

        # Define the p, d and q parameters to take any value between 0 and 2
        p = d = q = range(0, 2)

        # Generate all different combinations of p, q and q triplets
        self.pdq = list(itertools.product(p, d, q))

        # Generate all different combinations of seasonal p, q and q triplets
        self.seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        print('Examples of parameter combinations for Seasonal ARIMA...')
        print('SARIMAX: {} x {}'.format(self.pdq[1], self.seasonal_pdq[1]))
        print('SARIMAX: {} x {}'.format(self.pdq[1], self.seasonal_pdq[2]))
        print('SARIMAX: {} x {}'.format(self.pdq[2], self.seasonal_pdq[3]))
        print('SARIMAX: {} x {}'.format(self.pdq[2], self.seasonal_pdq[4]))


    def calculate_AIC(self):
        """ calculate Akaike Information Criterion (AIC) """
        for param in self.pdq:
            for param_seasonal in self.seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.data,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()
                    print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                except:
                    continue

    def fitModel(self):
        """ plot residuals, qqPlot, histogram with density plot, correlogram """
        mod = sm.tsa.statespace.SARIMAX(self.data,
                                        order=(1, 1, 1),
                                        seasonal_order=(1, 1, 1, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.results = mod.fit()
        print(self.results.summary().tables[1])

    def visualizeSarima(self):
        """ plot sarima """
        self.results.plot_diagnostics(figsize=(15, 12))
        plt.show()


class SarimaPrediction(Sarima):

    def __init__(self, data, dynamic):
        """ initialize SarimaPrediction object
            :param data: series, to be analyzed
            :param dynamic: boolean, model's hyperparameter setting
        """
        self.data = data
        super().__init__(self.data)
        self.fitModel()      # generates self.results
        self.pred = self.results.get_prediction(start=pd.to_datetime('2005-01-01'), dynamic=dynamic)
        self.pred_ci = self.pred.conf_int()

    def plotForecast(self):
        """ plot forecast """
        plt.figure(figsize=(15, 6))
        ax = self.data['2000':].plot(label='observed')
        self.pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
        ax.fill_between(self.pred_ci.index,
                        self.pred_ci.iloc[:, 0],
                        self.pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('CO2 Levels')
        plt.legend()
        plt.show()

    def calculateMSE(self):
        """ calculate mean squared error (MSE) """
        y_forecasted = self.pred.predicted_mean
        y_truth = self.data['1998-01-01':]
        mse = ((y_forecasted - y_truth) ** 2).mean()
        print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))



def main():
    series = plot1.df["TG"]
    # sar = Sarima(series)
    # sar.calculate_AIC()
    # sar.visualizeSarima()

    staticSar = SarimaPrediction(series, dynamic=False)
    staticSar.plotForecast()
    staticSar.calculateMSE()



if __name__ == "__main__":
    main()
#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import numpy as np


class TrendingRules:

    def __init__(self, df, colName):
        """
        Initialize class variables.
        :param df: dataframe, containing the data of interest
        :param colName: String, column name containing data of interest to be processed / analyzed
        """
        self.df = df
        self.colName = colName
        self.data = self.df[self.colName]
        self.mean = np.mean(df[colName])
        self.sigma = np.std(df[colName])
        ## initialize color code as green and overwrite outlier's color code later
        self.df["colorCode"] = "Green"


    def colorYellow(self, lastElements, idx):
        """
        Evaluate the minimum of a given list of numbers and assign corresponding color as yellow.
        :param lastElements: list, containing numbers (floats) of which the minimum is to be evaluated.
        :param idx: integer, index of value to be processed
        :return: NA
        """
        dict_lastElements = {}
        for ind, el in lastElements:
            dict_lastElements[idx - (len(lastElements)-1) + ind] = el
        self.df["colorCode"].iloc[[k for k, v in dict_lastElements.items() if v == min(lastElements)]] = "Yellow"


    def rule01(self):
        """
        One point is more than 3 standard deviations from the mean.
        :return: dataframe containing outlier color coding.
        """
        self.df["colorCode"] = self.df.apply(lambda x: ("Red" if x[self.colName] < (self.mean - 3 * self.sigma) or
                                             x[self.colName] > (self.mean + 3 * self.sigma) else "Green"), axis=1)

        return self.df


    def rule02(self):
        """
        Nine (or more) points in a row are on the same side of the mean.
        :return: dataframe containing outlier color coding.
        """
        for idx, elem in enumerate(self.data):
            # last9 = list(self.data[:idx+1])[-9:]
            last9 = list(self.data.iloc[idx-8:idx+1])
            if len(last9) == 9:
                if min(last9) > self.mean or max(last9) < self.mean:
                    self.df["colorCode"].iloc[idx-8:idx+1] = "Red"

        return self.df


    def rule03(self):
        """
        Six (or more) points in a row are continually increasing (or decreasing).
        :return: dataframe containing outlier color coding.
        """
        for idx, elem in enumerate(self.data):
            last6 = list(self.data.iloc[idx-5:idx+1])
            if len(last6) == 6:
                ## continually increasing or decreasing
                if last6[5] > last6[4] > last6[3] > last6[2] > last6[1] > last6[0] \
                        or last6[5] < last6[4] < last6[3] < last6[2] < last6[1] < last6[0]:
                    self.df["colorCode"].iloc[idx-5:idx+1] = "Red"

        return self.df


    def rule04(self):
        """
        Fourteen (or more) points in a row alternate in direction, increasing then decreasing.
        :return: dataframe containing outlier color coding.
        """
        for idx, elem in enumerate(self.data):
            last14 = list(self.data.iloc[idx-13:idx+1])
            if len(last14) == 14:
                if (last14[13] > last14[12] < last14[11] > last14[10] < last14[9] > last14[8] < last14[7] > last14[6] < last14[5] > last14[4] < last14[3] > last14[2] < last14[1] > last14[0]) \
                        or (last14[13] < last14[12] > last14[11] < last14[10] > last14[9] < last14[8] > last14[7] < last14[6] > last14[5] < last14[4] > last14[3] < last14[2] > last14[1] < last14[0]):
                    self.df["colorCode"].iloc[idx-14:idx+1] = "Red"

        return self.df


    def rule05(self):
        """
        Check if 2 out of 3 elements in a row are more than 2 std away from the mean in the same direction.
        :return: dataframe containing outlier color coding.
        """
        for idx, elem in enumerate(self.data):
            last3 = list(self.data.iloc[idx-2:idx+1])
            if len(last3) == 3:
                min2 = last3.copy()
                min2.remove(min(last3))

                ## if elements are greater than the mean
                if min(last3) > self.mean:
                    ## color points in red if condition fulfilled
                    if min(min2) > self.mean + 2*self.sigma:
                        self.df["colorCode"].iloc[idx-2:idx+1] = "Red"
                        ## in case only 2 out of 3 points are greater 2*sigma, color the remaining one in yellow
                        if min(last3) < 2*self.sigma:
                            self.colorYellow(last3, idx)

                ## if elements are smaller than the mean
                elif min(last3) < self.mean:
                    ## color points in red if condition fulfilled
                    if min(min2) < self.mean - 2 * self.sigma:
                        self.df["colorCode"].iloc[idx-2:idx+1] = "Red"
                        ## in case only 2 out of 3 points are greater 2*sigma, color the remaining one in yellow
                        if min(last3) > 2 * self.sigma:
                            self.colorYellow(last3, idx)

        return self.df


    def rule06(self):
        """
        check if 4 out of 5 elements in a row are more than 1 std away from the mean in the same direction.
        :return: dataframe containing outlier color coding.
        """
        for idx, elem in enumerate(self.data):
            last5 = list(self.data.iloc[idx-4:idx+1])
            if len(last5) == 5:
                min4 = last5.copy()
                min4.remove(min(last5))

                ## if elements are greater than the mean
                if min(last5) > self.mean:
                    if min(min4) > self.mean + self.sigma:
                        self.df["colorCode"].iloc[idx-4:idx+1] = "Red"
                        ## in case only 2 out of 3 points are greater 2*sigma, color the remaining one in yellow
                        if min(last5) < 2 * self.sigma:
                            self.colorYellow(last5, idx)

                ## if elements are smaller than the mean
                if min(last5) < self.mean:
                    if min(min4) < self.mean - self.sigma:
                        self.df["colorCode"].iloc[idx-4:idx+1] = "Red"
                        ## in case only 2 out of 3 points are greater 2*sigma, color the remaining one in yellow
                        if min(last5) < 2 * self.sigma:
                            self.colorYellow(last5, idx)

        return self.df


    def rule07(self):
        """
        15 points in a row are within 1 std of the mean on either side of the mean.
        :return: dataframe containing outlier color coding.
        """
        for idx, elem in enumerate(self.data):
            last15 = list(self.data.iloc[idx-14:idx+1])
            if len(last15) == 15:
                if min(last15) > (self.mean - self.sigma) and max(last15) < (self.mean + self.sigma):
                    self.df["colorCode"].iloc[idx-14:idx+1] = "Red"

        return self.df


    def rule08(self):
        """
        8 points in a row exist but none within 1 standard deviation of the mean,
        and the points are in both directions from the mean.
        :return: dataframe containing outlier color coding.
        """
        foundWithin = False
        for idx, elem in enumerate(self.data):
            last8 = list(self.data.iloc[idx-7:idx+1])
            if len(last8) == 8:
                for el in last8:
                    if el > (self.mean - self.sigma) and el < (self.mean + self.sigma):
                        foundWithin = True
                        break
                if not foundWithin:
                    self.df["colorCode"].iloc[idx-7:idx+1] = "Red"
            foundWithin = False

        return self.df


    def getRule(self, ruleName):
        """
        Apply trending rule of interest (chosen via drop down menu).
        :param ruleName: String, name of the chosen trending rule (drop down menu)
        :return: dataframe containing the outlier color code of interest
        """
        if ruleName == "Rule 01":
            df = self.rule01()
        elif ruleName == "Rule 02":
            df = self.rule02()
        elif ruleName == "Rule 03":
            df = self.rule03()
        elif ruleName == "Rule 04":
            df = self.rule04()
        elif ruleName == "Rule 05":
            df = self.rule05()
        elif ruleName == "Rule 06":
            df = self.rule06()
        elif ruleName == "Rule 07":
            df = self.rule07()
        elif ruleName == "Rule 08":
            df = self.rule08()
        else:
            df = self.rule01()

        return df


def main():
    pass


if __name__ == "__main__":
    main()

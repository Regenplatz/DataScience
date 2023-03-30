#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"

import numpy as np


class TrendingRules:

    def __init__(self, df_meanPerMonth, df_sel, colName):
        """
        Initialize class variables.
        :param df: dataframe, containing the data of interest
        :param colName: String, column name containing data of interest to be processed / analyzed
        """
        self.df = df_meanPerMonth
        self.colName = colName
        self.data = self.df[self.colName]
        self.mean = np.mean(df_sel[colName], axis=0)
        self.sigma = np.std(df_sel[colName], axis=0)
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
            dict_lastElements[idx - (len(lastElements) - 1) + ind] = el
        self.df["colorCode"].iloc[[k for k, v in dict_lastElements.items() if v == min(lastElements)]] = "Yellow"


    def rule05_rule06(self, noPoints_window, noPointsOutOf_window, std_value):
        """
        Check if a certain number of points (e.g. 2) from a given window (e.g. 3) are more than x standard deviations
        (e.g. 2 std) away from the mean and the points are in the same direction (all above or all below mean).
        :param noPoints_window: int, number of points for window of interest, e.g. 3 in "2 out of 3 points in a row"
        :param noPointsOutOf_window: int, number of points within this window that fulfill a certain condition,
                e.g. 2 in "2 out of 3 points in a row"
        :param std_value: int, number of standard deviations, e.g. 2 for 2 std
        :return: dataframe containing outlier/trend color coding
        """
        for idx, elem in enumerate(self.data):
            last_noPoints = list(self.data.iloc[idx - (noPoints_window - 1):idx + 1])
            if len(last_noPoints) == noPoints_window:
                helperX = sorted(last_noPoints)
                minX = helperX[len(helperX) - noPointsOutOf_window:]

                ## if elements are greater than the mean
                if min(last_noPoints) > self.mean:
                    ## color points in red if condition fulfilled
                    if min(minX) > self.mean + (std_value * self.sigma):
                        self.df["colorCode"].iloc[idx - (noPoints_window - 1):idx + 1] = "Red"
                        ## in case only m out of n points are greater x*sigma, color the remaining ones in yellow
                        if min(last_noPoints) < (std_value * self.sigma):
                            self.colorYellow(last_noPoints, idx)

                ## if elements are smaller than the mean
                elif min(last_noPoints) < self.mean:
                    ## color points in red if condition fulfilled
                    if min(minX) > self.mean + (std_value * self.sigma):
                        self.df["colorCode"].iloc[idx - (noPoints_window - 1):idx + 1] = "Red"
                        ## in case only m out of n points are greater x*sigma, color the remaining ones in yellow
                        if min(last_noPoints) > (std_value * self.sigma):
                            self.colorYellow(last_noPoints, idx)

                else:
                    pass

        return self.df


    def rule01(self, std_value=3):
        """
        One point is more than x standard deviations from the mean.
        :param std_value: int, number of standard deviations, e.g. 3 for 3 std
        :return: dataframe containing outlier color coding.
        """
        self.df["colorCode"] = self.df.apply(
            lambda x: ("Red" if (x[self.colName] < (self.mean - (std_value * self.sigma))) or
                                (x[self.colName] > (self.mean + (std_value * self.sigma))) else "Green"),
            axis=1)

        return self.df


    def rule02(self, no_points=9):
        """
        Number of points (or more) in a row are on the same side of the mean.
        :param no_points: int, minimum number of points to fulfill the condition
        :return: dataframe containing outlier color coding.
        """
        for idx, elem in enumerate(self.data):
            last_noPoints = list(self.data.iloc[idx - (no_points - 1):idx + 1])
            if len(last_noPoints) == no_points:
                if min(last_noPoints) > self.mean or max(last_noPoints) < self.mean:
                    self.df["colorCode"].iloc[idx - (no_points - 1):idx + 1] = "Red"

        return self.df


    def rule03(self, no_points=6):
        """
        Number of points (or more) in a row are continually increasing (or decreasing).
        :param no_points: int, minimum number of points to fulfill the condition
        :return: dataframe containing outlier color coding.
        """
        for idx, elem in enumerate(self.data):

            last_direction = "start"
            last_elem = -100
            cnt = 0
            last_noPoints = list(self.data.iloc[idx - (no_points - 1):idx + 1])
            if len(last_noPoints) == no_points:

                for elem2 in last_noPoints:
                    if last_direction == "start":
                        direction = None
                    else:
                        if elem2 > last_elem:
                            direction = "up"
                        elif elem2 < last_elem:
                            direction = "down"
                        else:
                            direction = "equal"

                        if ((last_direction == direction) or (last_direction is None)) and (direction != "equal"):
                            cnt += 1
                        else:
                            break

                    last_direction = direction
                    last_elem = elem2

            if cnt == (no_points-1):
                self.df["colorCode"].iloc[idx-(no_points-1):idx+1] = "Red"

        return self.df


    def rule04(self, no_points=14):
        """
        Fourteen (or more) points in a row alternate in direction, increasing then decreasing.
        :return: dataframe containing outlier color coding.
        """
        for idx, elem in enumerate(self.data):

            last_direction = "start"
            last_elem = -100
            cnt = 0
            last_noPoints = list(self.data.iloc[idx-(no_points-1):idx+1])
            if len(last_noPoints) == no_points:

                for elem2 in last_noPoints:
                    if last_direction == "start":
                        direction = None
                    else:
                        if elem2 > last_elem:
                            direction = "up"
                        elif elem2 < last_elem:
                            direction = "down"
                        else:
                            direction = "equal"

                        if (last_direction != direction) and (direction != "equal"):
                            cnt += 1
                        else:
                            break

                    last_direction = direction
                    last_elem = elem2

                if cnt == (no_points-1):
                    self.df["colorCode"].iloc[idx-(no_points-1):idx+1] = "Red"

        return self.df


    def rule05(self, noPoints_window=3, noPointsOutOf_window=2, std_value=2):
        """
        Check if 2 out of 3 elements in a row are more than 2 std away from the mean in the same direction.
        :param noPoints_window: int, number of points for window of interest, e.g. 3 in "2 out of 3 points in a row"
        :param noPointsOutOf_window: int, number of points within this window that fulfill a certain condition,
                e.g. 2 in "2 out of 3 points in a row"
        :param std_value: int, number of standard deviations, e.g. 2 for 2 std
        :return: dataframe containing outlier color coding.
        """
        self.df = self.rule05_rule06(noPoints_window, noPointsOutOf_window, std_value)

        return self.df


    def rule06(self, noPoints_window=5, noPointsOutOf_window=4, std_value=1):
        """
        check if 4 out of 5 elements in a row are more than 1 std away from the mean in the same direction.
        :param noPoints_window: int, number of points for window of interest, e.g. 3 in "2 out of 3 points in a row"
        :param noPointsOutOf_window: int, number of points within this window that fulfill a certain condition,
                e.g. 2 in "2 out of 3 points in a row"
        :param std_value: int, number of standard deviations, e.g. 2 for 2 std
        :return: dataframe containing outlier color coding.
        """
        self.df = self.rule05_rule06(noPoints_window, noPointsOutOf_window, std_value)

        return self.df


    def rule07(self, no_points=15, std_value=1):
        """
        Number of points in a row (e.g. 15) are within 1 std of the mean on either side of the mean.
        :param no_points: int, minimum number of points to fulfill the condition
        :param std_value: int, number of standard deviations, e.g. 2 for 2 std
        :return: dataframe containing outlier color coding.
        """
        for idx, elem in enumerate(self.data):
            last_noPoints = list(self.data.iloc[idx - (no_points - 1):idx + 1])
            if len(last_noPoints) == no_points:
                if min(last_noPoints) > (self.mean - (std_value * self.sigma)) and \
                        max(last_noPoints) < (self.mean + (std_value * self.sigma)):
                    self.df["colorCode"].iloc[idx - (no_points - 1):idx + 1] = "Red"

        return self.df


    def rule08(self, no_points=8, std_value=1):
        """
        Number of points in a row (e.g. 8) exist but none within x standard deviations of the mean,
        and the points are in both directions from the mean.
        :param no_points: int, minimum number of points to fulfill the condition
        :param std_value: int, number of standard deviations, e.g. 2 for 2 std
        :return: dataframe containing outlier color coding.
        """
        foundWithin = False
        for idx, elem in enumerate(self.data):
            last_noPoints = list(self.data.iloc[idx - (no_points - 1):idx + 1])
            if len(last_noPoints) == no_points:
                for el in last_noPoints:
                    if (self.mean - (std_value * self.sigma)) < el < (self.mean + (std_value * self.sigma)):
                        foundWithin = True
                        break
                if not foundWithin:
                    self.df["colorCode"].iloc[idx - (no_points - 1):idx + 1] = "Red"
            foundWithin = False

        return self.df


    def getRule(self, ruleName, no_points=None, noPoints_window=None, noPointsOutOf_window=None, std_value=None):
        """
        Apply trending rule of interest (chosen via drop down menu).
        :param ruleName: String, name of the chosen trending rule (drop down menu)
        :param no_points: int, minimum number of points to fulfill the condition
        :param noPoints_window: int, number of points for window of interest, e.g. 3 in "2 out of 3 points in a row"
        :param noPointsOutOf_window: int, number of points within this window that fulfill a certain condition,
                e.g. 2 in "2 out of 3 points in a row"
        :param std_value: int, number of standard deviations, e.g. 2 for 2 std
        :return: dataframe containing the outlier color code of interest
        """
        if ruleName == "Rule 01":
            if std_value is None:
                std_value = 3
            df = self.rule01(std_value)

        elif ruleName == "Rule 02":
            if no_points is None:
                no_points = 9
            df = self.rule02(no_points)

        elif ruleName == "Rule 03":
            if no_points is None:
                no_points = 6
            df = self.rule03(no_points)

        elif ruleName == "Rule 04":
            df = self.rule04()

        elif ruleName == "Rule 05":
            if noPoints_window is None:
                noPoints_window = 3
            if noPointsOutOf_window is None:
                noPointsOutOf_window = 2
            if std_value is None:
                std_value = 2
            df = self.rule05(noPoints_window, noPointsOutOf_window, std_value)

        elif ruleName == "Rule 06":
            if noPoints_window is None:
                noPoints_window = 5
            if noPointsOutOf_window is None:
                noPointsOutOf_window = 4
            if std_value is None:
                std_value = 1
            df = self.rule06(noPoints_window, noPointsOutOf_window, std_value)

        elif ruleName == "Rule 07":
            if no_points is None:
                no_points = 15
            if std_value is None:
                std_value = 1
            df = self.rule07(no_points, std_value)

        elif ruleName == "Rule 08":
            if no_points is None:
                no_points = 8
            if std_value is None:
                std_value = 1
            df = self.rule08(no_points, std_value)

        else:
            df = self.rule01(std_value=3)

        return df


def main():
    pass


if __name__ == "__main__":
    main()

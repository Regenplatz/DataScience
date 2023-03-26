#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import numpy as np
import pandas as pd
import math


class CalculateSampleSize():

    def __init__(self, population_size, ci, me, p):
        """
        Initialize class variables.
        :param population_size: int, size of population (absolute number)
        :param ci: int, confidence interval in %
        :param me: int, margin of error in %
        :param p: int, probability that the sample size is big enough in terms of confidence interval,
            default usually 50% (for p of 0.5)
        """
        self.N = population_size
        self.ci = ci / 100
        self.me = me / 100
        self.p = p / 100
        self.z = None
        self.path_zScoreTable = "zScore.xlsx"
        if ci < 0.5:
            self.sheet_zScore = "zscore_right"
        else:
            self.sheet_zScore = "zscore_left"


    def evaluate_zScore(self):
        """
        Evaluate z-score from table. Therefore, check on the value that is closest to the calculated
        area_left value. Take care of the line feet so that you also compare the area_left value to
        the last value of the row and to the first value of the next row.
        :return: NA
        """
        ## percentile (needed to read z value from z-score table)
        area_left = (1 + self.ci) / 2

        ## read z-score table to assess the z_score later (by summing up column header and row index)
        df = pd.read_excel(self.path_zScoreTable, sheet_name=self.sheet_zScore)

        ## unpivot z-score table
        zScoreTable = pd.melt(df, id_vars="Z")
        zScoreTable.sort_values(by=["Z", "variable"], inplace=True)
        zScoreTable["Z"] = np.arange(0, 3.5, 0.01)
        zScoreTable.set_index(np.arange(0, 350), inplace=True)
        zScoreTable.drop(["variable"], axis=1, inplace=True)

        def evaluateClosestValue(lst, a):
            """
            Find the closest value to the target.
            :param lst: list, containing the closest values to the target
            :param a: float, target value for which z-score needs to be evaluated
            """
            ## in case target is exactly between two available values
            x1 = lst[min(range(len(lst)), key=lambda i: abs(lst[i] - a))]
            x2 = lst[lst.index(x1) + 1]
            if np.round(abs(x1 - a), 6) == np.round(abs(x2 - a), 6):
                return x2
            else:
                return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - a))]


        def lookUpZScore(zScoreTable, areaVal):
            """
            Evaluate value with minimum distance to the desired value.
            """
            z_score = None

            ## in case value is not available in zScoreTable, check for the closest one(s)
            lst = zScoreTable["value"]
            closest1_val = lst[min(range(len(lst)), key=lambda i: abs(lst[i] - areaVal))]
            closest1_idx = zScoreTable[zScoreTable["value"] == closest1_val].index[0]
            closest2_val = zScoreTable.iloc[closest1_idx + 1]["value"]
            closest_val = evaluateClosestValue([closest1_val, closest2_val], areaVal)

            ## evaluate z-score
            for idx, row in zScoreTable.iterrows():

                ## in case an value is available in several cells,
                ## then take the one with the higher index
                if row["value"] > closest_val:
                    # print("zScore:", np.round(row["Z"], 2), "; value:", row["value"])
                    break

                z_score = np.round(row["Z"], 2)

            return z_score

        self.z = lookUpZScore(zScoreTable, area_left)


    def formulaStandard(self):
        """
        Calculate sample size based on stardard formula.
        :return: int, sample size
        """
        sample_size = (((self.z ** 2 * self.p * (1 - self.p)) / (self.me ** 2)) /
                       (1 + ((self.z ** 2 * self.p * (1 - self.p)) / (self.me ** 2 * self.N))))
        return math.ceil(sample_size)


    def formulaUnknownHugePopulation(self):
        """
        Calculate sample size for unknown or huge population.
        :return: int, sample size
        """
        sample_size = (self.z ** 2 * self.p * (1 - self.p)) / self.me ** 2
        return math.ceil(sample_size)


    def formulaSlovin(self):
        """
        Calculate sample size for population with unknown behavior.
        :return: int, sample size
        """
        sample_size = self.N / (1 + (self.N * self.me ** 2))
        return math.ceil(sample_size)


def main():

    N = 1000
    ci = 99
    me = 5
    p = 50

    sample_size_class = CalculateSampleSize(N, ci, me, p)
    sample_size_class.evaluate_zScore()
    print("Required Sample Size:", sample_size_class.formulaStandard())



if __name__ == "__main__":
    main()

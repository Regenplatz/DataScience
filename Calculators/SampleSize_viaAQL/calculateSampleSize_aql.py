#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CalculateSampleSizeAQL:

    lot_quantity: int
    inspec_level: str
    aql: str
    filename: str = "AQL.xlsx"
    sheet_letterCode = "letterCode"
    sheet_samplingPlan = "samplingPlan"
    indexCol_letterCode = "Lot Size"
    indexCol_samplingPlan = "Sample Size Code Letter"
    df_lc = None
    df_sp = None
    dict_letterCode = None
    dict_samplingPlan = None


    def loadTables(self, filename: str, sheet_name: str, header: int, index_col: str) -> pd.DataFrame:
        return pd.read_excel(io=filename, sheet_name=sheet_name, header=header, index_col=index_col)


    def dataToDict(self, df) -> dict:
        return df.T.to_dict(orient="dict")


    def loadDataToDict(self):
        """
        Load tables for AQL to dataframe and convert to dictionaries.
        """
        ## for data on letter code
        self.df_lc = self.loadTables(filename=self.filename,
                                     sheet_name=self.sheet_letterCode,
                                     header=1,
                                     index_col=self.indexCol_letterCode)
        self.dict_letterCode = self.dataToDict(self.df_lc)
        for k, v in self.dict_letterCode.items():
            self.dict_letterCode[k]["numRange"] = [int(x) for x in k.split("-")]

        ## for data on sampling plan
        self.df_sp = self.loadTables(filename=self.filename,
                                     sheet_name=self.sheet_samplingPlan,
                                     header=1,
                                     index_col=self.indexCol_samplingPlan)
        self.dict_samplingPlan = self.dataToDict(self.df_sp)


    def evaluateSampleSize(self):

        sampleSize = None
        maxAcceptLevel = None
        minRejectLevel = None
        for k, v in self.dict_letterCode.items():

            if not self.lot_quantity > v['numRange'][1] or len(v['numRange']) == 1:

                ## print input variables -----------------------------------------------------------
                self.aql = "Ac " + str(self.aql)
                str_interval_lotSize = k
                lst_interval_lotSize = v['numRange']
                #print("----- INITIAL VALUES -------------------------------------------")
                #print(f"lot quantity {self.lot_quantity}           ||   in interval {lst_interval_lotSize}")

                inspecLevel_directRead = v[self.inspec_level]
                #print(f"initial inspec level ({self.inspec_level}) ||   {inspecLevel_directRead}")

                elem_directReadAql = self.dict_samplingPlan[v[self.inspec_level]][self.aql]
                #print(f"initial aql ({self.aql[3:]})         ||   {elem_directReadAql}")
                #print("\n----- FINAL VALUES ---------------------------------------------")

                ## directly readable from table; Ac / Re values available -------------------------
                if elem_directReadAql not in ["x", "z"]:
                    sampleSize = self.dict_samplingPlan[v[self.inspec_level]]['Sample Size']
                    #print(f"new sample size        ||   {sampleSize}")

                    maxAcceptLevel = elem_directReadAql
                    minRejectLevel = elem_directReadAql + 1


                ## not directly readable from table; Ac / Re values not available -----------------
                else:
                    size_interim = self.dict_samplingPlan[v[self.inspec_level]]['Sample Size']
                    indx = self.df_sp[self.df_sp["Sample Size"] == size_interim].index.to_list()[0]
                    i = ""
                    inspecLevel_new = None

                    if elem_directReadAql == "x":
                        for i, elem in self.df_sp.loc[indx:, self.aql].items():
                            if elem != "x":
                                sampleSize = self.df_sp.loc[i, 'Sample Size']
                                if self.lot_quantity < sampleSize:
                                    sampleSize = self.lot_quantity
                                #print(f"new inspec level          ||   {i}")
                                #print(f"new sample size           ||   {sampleSize}")
                                break

                    elif elem_directReadAql == "z":
                        for i, elem in self.df_sp[::-1].loc[indx:, self.aql].items():
                            if elem != "z":
                                sampleSize = self.df_sp.loc[i, 'Sample Size']
                                if self.lot_quantity < sampleSize:
                                    sampleSize = self.lot_quantity
                                #print(f"new inspec level          ||   {i}")
                                #print(f"new sample size           ||   {sampleSize}")
                                break

                    else:
                        pass

                        ## print number of Acceptance and Rejection samples --------------------------
                    col_idx = self.df_sp.columns.get_loc(self.aql)
                    if self.aql[0:2] == "Ac":
                        col_ac = col_idx
                        col_re = col_idx + 1
                    else:
                        col_ac = col_idx - 1
                        col_re = col_idx
                    maxAcceptLevel = self.df_sp.loc[i][col_ac]
                    minRejectLevel = self.df_sp.loc[i][col_re]
                    # print(f"Max. Acceptance Level     ||   {maxAcceptLevel}")
                    # print(f"Min. Rejection Level      ||   {minRejectLevel}")

                break

        return sampleSize, maxAcceptLevel, minRejectLevel


def main():

    ## Example
    lot_quantity = 20
    inspec_level = "G2"
    aql = str(0.4)

    classObject = CalculateSampleSizeAQL(lot_quantity=lot_quantity, inspec_level=inspec_level, aql=aql)
    classObject.loadDataToDict()
    sampleSize, maxAcceptLevel, minRejectLevel = classObject.evaluateSampleSize()

    print(f"\nThe REQUIRED SAMPLE SIZE is {sampleSize} "
          f"\nfor LOT SIZE {lot_quantity} and INSPECTION LEVEL {aql}"
          f"\nwith MAX. ACCEPTANCE LEVEL {maxAcceptLevel} and MIN. REJECTION LEVEL {minRejectLevel}! ")

if __name__ == "__main__":
    main()

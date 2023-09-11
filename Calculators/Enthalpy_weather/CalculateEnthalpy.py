#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import pandas as pd
import numpy as np
import math
import openpyxl


pathSaturationTable = "saturation_table.xlsx"


## DEFINE GLOBAL VARIABLES (constants for enthalpy calculation) --------------------------
## c(pL): specific heat capacity of air [kJ/(kgK)]
C_air = 1.006
## c(pd): specific heat capacity of steam [kJ/(kgK)]
C_steam = 1.86
## c(e): specific heat capacity of ice [kJ/(kgK)]
C_ice = 2.05
## c(w): specific heat capacity of water [kJ/(kgK)]
C_water = 4.19
## l(s): heat of fusion (German: Schmelzwärme) [kJ/kg]
L_s = 333
## R0: heat of vaporization (German: Verdampfungswärme) [kJ/kg]
R0 = 2502
## ---------------------------------------------------------------------------------------


def roundData(the_data: float, decimal_no: int) -> float:
    """
    Round data to the number of decimals provided.
    :param the_data: float, data to be rounded
    :param decimal_no: int, number of decimals in the rounded value
    :return: float, rounded data
    """
    if not np.isnan(the_data):
        x1 = the_data * pow(10, decimal_no) + 0.5
        x2 = math.floor(x1) / pow(10, decimal_no)
        return x2
    else:
        None


def ps_xs_lookup(temp: int, colName: str, saturation_table: pd.DataFrame) -> float:
    """
    Look up ps (saturation_vapor_pressure_ps [Pa]) and xs (watercontent_xs [g/kg]) at a given temperature.
    :param temp: int, temperature in [°C]
    :param colName: String, column name in saturation_table ("xs" or "ps")
    :param saturation_table: dataframe, saturation table
    :return: "xs" or "ps" value at a given temperature (int)
    """
    return saturation_table[colName][saturation_table["temp"] == temp].values[0]


def calculateEnthalpy(phi: float, temp: int, x: float, xs: float) -> float:
    """
    Calculate enthalpy based on conditions.
    :param phi: float, relative humidity (as value between 0 and 1)
    :param temp: int, temperature in °C
    :param x: float, actual water content in outdoor air
    :param xs: float, saturation water content [g/kg] at a given temperature t
    :return: float, calculated specific enthalpy [kJ/(kg(air))]
    """
    ## physical states: (e)-ice, (w)-water, (d)-steam

    ## unsaturated and saturated air ---------------------------------------
    ## x(w)=0, x(e)=0
    if (phi <= 1) and (x <= xs):
        enthalpy = C_air * temp + x * (R0 + (C_steam * temp))

    ## oversaturated air ---------------------------------------------------
    elif (phi == 1) and (x > xs):

        ## oversaturated air with water
        ## x(d)=x(s), x(w)!=0, x(e)=0, temp in [°C]
        if temp > 0:
            enthalpy = C_air * temp + xs * (R0 + (C_steam * temp)) + (x - xs) * C_water * temp

        ## oversaturated air with ice
        ## x(d)=x(s), x(w)=0, x(e)!=0, temp in [°C]
        elif temp < 0:
            enthalpy = C_air * temp + xs * (R0 + (C_steam * temp)) + (x - xs) * (-L_s + C_ice * temp)

    ## humid air at triple point: ------------------------------------------
    ## all physical states (ice (e), water (w), steam (d)) are present (), temp in [°C]
    elif temp == 0:
        enthalpy = xs * R0 + (x - xs) * (-L_s)

    else:
        # print("No temperature data available! No calculation possible")
        enthalpy = None

    return enthalpy


def calculateMetrics(df: pd.DataFrame, airTemp: str, airPressure: str,
                     airHumidity: str) -> pd.DataFrame:
    """
    Calculate or evaluate (look-up) all metrics that are needed for enthalpy calculation and calculate enthalpy.
    :param df: dataframe, containing weather information such as outdoor-temperature and relative humidity
    :param airTemp: String, name of column for , air temperature [°C]
    :param airPressure: String, name of column for air pressure [Pa]
    :param airHumidity: String, name of column for , relative humidity [g/kg]
    :return: dataframe, containing original as well as all calculated metrics
    """
    ## load saturation table
    saturation = pd.read_excel(pathSaturationTable, engine="openpyxl")
    saturation.rename(columns={"temperature_t [°C]": "temp",
                               "saturation_vapor_pressure_ps [Pa]": "ps",
                               "watercontent_xs [g/kg]": "xs"}, inplace=True)

    ## round temperature to integer value as preparation for look-up
    # df["temp"] = df[airTemp].apply(lambda x: roundTemp(x) if pd.notnull(x) else None)
    df["temp"] = df[airTemp].apply(lambda x: roundData(x, 0) if pd.notnull(x) else None)

    ## look up values for "ps" and "xs" [g/kg] in saturation table
    df["ps"] = df["temp"].apply(lambda x: ps_xs_lookup(x, "ps", saturation) if pd.notnull(x) else None)
    df["xs"] = df["temp"].apply(lambda x: ps_xs_lookup(x, "xs", saturation) if pd.notnull(x) else None)

    ## for air pressure: convert hPa in Pa
    df["p"] = df[airPressure].apply(lambda x: x * 100)

    ## for relative humidity: convert percent into values between 0 and 1
    df["phi"] = df[airHumidity].apply(lambda x: x / 100)

    ## calculate actual water content x [g/kg]
    df["x"] = 0.622 * df["phi"] * df["ps"] / (df["p"] - (df["phi"] * df["ps"]))

    ## calculate specific enthalpy [kJ/(kg(air))]
    df["enthalpy"] = df.apply(lambda x: calculateEnthalpy(x["phi"], x["temp"], x["x"], x["xs"]), axis=1)

    ## round to 1 decimal
    df["enthalpy"] = df["enthalpy"].apply(lambda x: roundData(x, 1))

    return df


def main():

    ## define path to data table and column names
    path_toWeatherData = "weather_data.csv"
    colName_temperature = "temperature"
    colName_pressure = "pressure"
    colName_humidity = "humidity"

    ## load data and calculate enthalpy
    df = pd.read_csv(path_toWeatherData, sep=",")
    df_enthalpy = calculateMetrics(df=df,
                                   airTemp=colName_temperature,
                                   airPressure=colName_pressure,
                                   airHumidity=colName_humidity)
    print(df_enthalpy)

    ## save data including enthalpy to file
    df_enthalpy.to_csv("enthalpy_data.csv", index=False)


if __name__ == "__main__":
    main()

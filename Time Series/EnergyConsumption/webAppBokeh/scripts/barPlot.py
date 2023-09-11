#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


## load packages
import pandas as pd
import numpy as np
from datetime import datetime
from sigfig import round as sigfig_round
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Select, Panel, PreText
import loadData as ld


## load data
df_all = ld.loadDataFrame()
#df_all["enthalpy"] = 15.6


def barplot_tab():

    ## ***** DEFINITIONS (GLOBAL VARIABLES) ************************************************************
    ## *************************************************************************************************

    ## TO BE DISPLAYED IN DROPDOWNS (KEYS), PARAMETER TO BE ADDRESSED (VALUES) -------------------

    ## define temperature/enthalpy from Roche's/Meteoswiss' weather data
    weather_measurand = {"Temperature [°C]": "temperature",
                         "Enthalpy [kJ/kg]": "enthalpy",
                         "Relative Humidity [%]": "humidity",
                         "Air Pressure [hPa]": "pressure",
                         }

    ## define years
    years = {"2021": 2021,
             "2020": 2020,
             "2019": 2019,
             "2018": 2018,
             "2017": 2017
             }

    hours = {"0 o'clock": 0, "2 o'clock": 2, "4 o'clock": 4, "6 o'clock": 6, "8 o'clock": 8, "10 o'clock": 10,
             "12 o'clock": 12, "14 o'clock": 14, "16 o'clock": 16, "18 o'clock": 18, "20 o'clock": 20, "22 o'clock": 22
             }


    ## DEFINE DROPDOWNS ---------------------------------------------------------------------
    ## title: title, value: default value, options: selection menu

    ## YEAR (y)
    sel_y = Select(title="YEAR",
                   value="2021",
                   options=["2021", "2020", "2019", "2018", "2017"])

    ## HOUR (h)
    sel_h = Select(title="TIME",
                   value="12 o'clock",
                   options=["0 o'clock", "2 o'clock", "4 o'clock", "6 o'clock", "8 o'clock", "10 o'clock",
                            "12 o'clock", "14 o'clock", "16 o'clock", "18 o'clock", "20 o'clock", "22 o'clock"])

    ## WEATHER, originally temperature / enthalpy (te)
    sel_weather = Select(title="WEATHER CONDITION, colored",
                         value="Temperature [°C]",
                         options=["Temperature [°C]",
                                  "Enthalpy [kJ/kg]",
                                  "Relative Humidity [%]",
                                  "Air Pressure [hPa]"])


    ## DEFINE DESCRIPTIVE STATISTICS ----------------------------------------------------------

    ## initialize text field
    stats = PreText(text='Initialize Descriptive Statistics', width=500)



    ## ***** LOAD DATA *********************************************************************************
    ## *************************************************************************************************

    def loadData(df_all, weatherFeature: str, year: str, hour: str) -> pd.DataFrame:
        # global df
        df = df_all[["date_time", "energy_kWh", weatherFeature, "year", "hour"]]

        if not df.empty:
            ## sort datetime column
            df = df.sort_values(by=weatherFeature)
            df = df[~df[weatherFeature].isna()]

            ## show only data that was selected for year and hour
            df = df.loc[((df["year"] == year) & (df["hour"] == hour)), :]

            ## update descriptive statistics and replace column names accordingly
            descStat = df[["energy_kWh", weatherFeature]].describe().applymap(lambda x: sigfig_round(x, 2))
            new_colnames = []
            for elem in list(descStat.columns):
                if "energy_kWh" in elem:
                    newName = ["Energy [kWh]"]
                elif elem in ["year", "month", "weekday", "hour"]:
                    newName = ["Date Time"]
                else:
                    newName = elem
                new_colnames.append(newName[0])
            descStat.columns = new_colnames
            ## convert to string (for displaying in bokeh app)
            stats.text = str(descStat)

        return df



    ## ***** UPDATE DISPLAYED DATA TO CHOICE ON DROPDOWN MENU ******************************************
    ## *************************************************************************************************

    def update(attr, old, new):
        """
        Update plotted data according to choice via dropdown menu
        :param attr: String, attribute that was changed on dropdown menu
        :param old: array, old values (before changes via dropdown menu)
        :param new: array, new values (after changes via dropdown menu)
        :return: NA
        """
        dd_plot.children[1] = createFigure()


    ## UPDATE DROPDOWNS ACCORDING TO USER'S CHOICES -----------------------------------------

    ## update on change: YEAR (y)
    sel_y.on_change("value", update)

    ## update on change: HOUR (h)
    sel_h.on_change("value", update)

    ## update on change: WEATHER
    sel_weather.on_change("value", update)


    ## ***** PLOT DATA *********************************************************************************
    ## *************************************************************************************************

    def createFigure():
        """
        Load data of interest (use selection from the drop down menus for this).
        Create scatter plot.
        :return: plot, to be displayed on bokeh web server (port 5006)
        """
        ## check time before loading
        bl = datetime.now()
        before_loading = bl.strftime("%H:%M:%S")
        print("BEFORE LOADING:", before_loading)

        ## load data
        df = loadData(df_all,
                      weather_measurand[sel_weather.value],
                      years[sel_y.value],
                      hours[sel_h.value])

        ## if no data available, initialize an empty dataframe
        if df.empty:
            return PreText(text=f"No Data Available!", width=900)
        print("DATAFRAME ----------------------------------\n", df)

        ## check time after loading
        al = datetime.now()
        after_loading = al.strftime("%H:%M:%S")
        print("DATA LOADED:", after_loading)
        print("TIME FOR LOADING:", al - bl, "-----------------------------\n")

        ## define bins for weather data
        binSize_weather = 10
        wm = weather_measurand[sel_weather.value]
        df[f"{wm}_bin"] = [([binSize_weather * int(x / binSize_weather) - binSize_weather,
                             binSize_weather * int(x / binSize_weather)])
                             if x < 0
                             else ([binSize_weather * int(x / binSize_weather),
                                    binSize_weather * int(x / binSize_weather) + binSize_weather]) for x in df[wm]]
        df[f"{wm}_bin"] = df[f"{wm}_bin"].astype(str)
        colors = ["#2554C7", "#5CB3FF", "#E9AB17", "#FF8C00", "#C04000"]
        lst_allWeatherBins = ['[-10, 0]', '[0, 10]', '[10, 20]', '[20, 30]', '[30, 40]']

        ## define bins for energy data
        binSize_energy = 30
        df["energy_bin"] = [([binSize_energy * int(x / binSize_energy),
                              binSize_energy * int(x / binSize_energy) + binSize_energy]) for x in df["energy_kWh"]]
        df["energy_bin"] = df["energy_bin"].astype(str)
        lst_allEnergyBins = ['[0, 30]', '[30, 60]', '[60, 90]', '[120, 150]',
                             '[150, 180]', '[180, 210]', '[210, 240]', '[240, 270]']

        df = df[(~df[f"{wm}_bin"].isna()) & (~df["energy_bin"].isna())]
        print("NEW DATAFRAME ----------------------------------------------\n", df)

        energy_bins = df["energy_bin"].unique().tolist()
        weather_bins = df[f"{wm}_bin"].unique().tolist()

        ## load count per bin to dictionary
        dict_en_wm = df.groupby(by=["energy_bin", "temperature_bin"]).count()["energy_kWh"].to_dict()

        ## create all possible combinations of energy_bin and weather_bin
        lst_allCombinations = []
        for en in lst_allEnergyBins:
            for w in lst_allWeatherBins:
                lst_allCombinations.append((en, w))

        for elem in lst_allCombinations:
            if elem not in dict_en_wm.keys():
                dict_en_wm[elem] = 0
        print("LST_ALL COMBINATIONS --------------------------------------------\n", lst_allCombinations)
        print("DICT EN WM EXTENDED ---------------------------------------------\n", dict_en_wm)

        ## loop through all possible power-weatherCondition combinations and fill in corresponding value.
        ## If value not found, fill up  with zero (--> length of list must always be constant)
        lst_energy = []
        for idx, wb in enumerate(lst_allWeatherBins):
            lst_energy2 = []
            print("dict ------------------------------------")
            for idxp, eb in enumerate(lst_allEnergyBins):
                ## define keys for dictionary (power bin value / weather condition bin value)
                dKey = (eb, wb)
                val = dict_en_wm.get(dKey, 0)
                print(idx, idxp, "key:", dKey, "val:", val)
                lst_energy2.append(val)
            lst_energy.append(lst_energy2)

        data = {"energy_bins": lst_allEnergyBins,
                lst_allWeatherBins[0]: lst_energy[0],
                lst_allWeatherBins[1]: lst_energy[1],
                lst_allWeatherBins[2]: lst_energy[2],
                lst_allWeatherBins[3]: lst_energy[3],
                lst_allWeatherBins[4]: lst_energy[4]
                }
        data2 = pd.DataFrame(data)
        print("DATA --------------------------------------------------------\n", data)

        p = figure(x_range=data2.energy_bins,
                   height=500, width=800, title=f"Power Counts by {sel_weather.value}",
                   toolbar_location=None, tools="hover", tooltips="bin $name, count: @$name")

        p.vbar_stack(lst_allWeatherBins,
                     x="energy_bins",
                     width=0.9, color=colors, source=data2,
                     legend_label=lst_allWeatherBins)

        return p


    ## DISPLAY IN BROWSER (DROPDOWNS & PLOT TOGETHER) ---------------------------------------

    ## define alignment of dropdowns (beneath each other)
    w = 300
    controls = column(sel_y, sel_h, sel_weather, width=w, margin=(0, 0, 100, 0))

    ## define alignment of dropdown section and plot
    dd_plot = row(controls, createFigure())

    ## define alignment of descriptive statistics table
    col_stats = column(stats)

    ## define alignment of dd_plot (dropdowns and plot) and descriptive statistics
    layout = column(dd_plot, col_stats)

    ## display as tab in browser
    tab = Panel(child=layout, title="Energy Consumption per Weather Condition")

    return tab

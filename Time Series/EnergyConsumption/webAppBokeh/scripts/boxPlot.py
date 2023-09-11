#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


## load packages
import pandas as pd
from sigfig import round as sigfig_round
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Select, Panel, PreText
import loadData as ld


## load data
df_all = ld.loadDataFrame()


def boxplot_tab():

    ## ***** DEFINITIONS (GLOBAL VARIABLES) ************************************************************
    ## *************************************************************************************************

    ## define time subject (and axis labels) -------------------------
    time_subj = {"Year": "year",
                 "Month": "month",
                 "Weekday": "weekday",
                 "Hour": "hour"}

    ## define parameters for renaming x-axis ticks
    timeCategoricals = {"year": {2017: "2017", 2018: "2018", 2019: "2019", 2020: "2020", 2021: "2021"},
                        "month": {1: " 1 Jan", 2: " 2 Feb", 3: " 3 Mar", 4: " 4 Apr", 5: " 5 May", 6: " 6 Jun",
                                  7: " 7 Jul", 8: " 8 Aug", 9: " 9 Sep", 10: "10 Oct", 11: "11 Nov", 12: "12 Dec"},
                        "weekday": {1: "1 Mon", 2: "2 Tue", 3: "3 Wed", 4: "4 Thu",
                                    5: "5 Fri", 6: "6 Sat", 7: "7 Sun"},
                        "hour": {0: " 0", 1: " 1", 2: " 2", 3: " 3", 4: " 4", 5: " 5", 6: " 6", 7: " 7",
                                 8: " 8", 9: " 9", 10: "10", 11: "11", 12: "12", 13: "13", 14: "14", 15: "15",
                                 16: "16", 17: "17", 18: "18", 19: "19", 20: "20", 21: "21", 22: "22", 23: "23"}
                        }


    ## DEFINE DROPDOWNS ---------------------------------------------------------------------
    ## title: title, value: default value, options: selection menu

    ## TIME CATEGORICALS (time)
    sel_time = Select(title="TIME GRANULARITY",
                      value="Month",
                      options=["Year", "Month", "Weekday", "Hour"])


    ## DEFINE DESCRIPTIVE STATISTICS ----------------------------------------------------------

    ## initialize text field
    stats = PreText(text='Initialize Descriptive Statistics', width=500)



    ## ***** LOAD DATA *********************************************************************************
    ## *************************************************************************************************

    def loadData(df_all, timeGranularity: str) -> pd.DataFrame:
        #global df
        df = df_all[["date_time", "energy_kWh", timeGranularity]]

        if not df.empty:
            ## sort datetime column
            df.sort_values(by=timeGranularity, inplace=True)

            ## take only observations where timeGranularity not NAN
            df = df[~df[timeGranularity].isna()]
            df[timeGranularity] = df[timeGranularity].astype(str)
            df[timeGranularity] = df[timeGranularity].astype("category")

            ## update descriptive statistics and replace column names accordingly
            descStat = df.describe().applymap(lambda x: sigfig_round(x, 4))
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
        print("in update")
        """
        Update plotted data according to choice via dropdown menu
        :param attr: String, attribute that was changed on dropdown menu
        :param old: array, old values (before changes via dropdown menu)
        :param new: array, new values (after changes via dropdown menu)
        :return: NA
        """
        layout.children[1] = createFigure()


    ## UPDATE DROPDOWNS ACCORDING TO USER'S CHOICES ------------------------------------------

    ## update on change: YEAR (y)
    sel_time.on_change("value", update)


    ## ***** PLOT DATA *********************************************************************************
    ## *************************************************************************************************

    def createFigure():

        ## LOAD DATA  ----------------------------------------------------

        df = loadData(df_all, timeGranularity=time_subj[sel_time.value])

        ## if no data available, initialize a notification
        if df.empty:
            return PreText(text=f"No Data Available!", width=900)
        print(df)

        ## take only the rows where x and y-axis parameters are not NaN
        df = df[(~df[time_subj[sel_time.value]].isna()) & (~df["energy_kWh"].isna())]

        ## define variables for x and y parameters
        x_col = time_subj[sel_time.value]
        y_col = "energy_kWh"

        ## define axis labels
        x_title = f"{sel_time.value}"
        y_title = f"Energy [kWh]"


        ## DEFINE BOXPLOT -------------------------------------------------

        ## define categoricals to be plotted
        cats = df[x_col].unique().tolist()

        ## define dataset whose data is to be plotted
        df = pd.DataFrame(dict(score=df[y_col], group=df[x_col]))

        ## find the quartiles and IQR for each category
        groups = df.groupby('group')
        q1 = groups.quantile(q=0.25)
        q2 = groups.quantile(q=0.5)
        q3 = groups.quantile(q=0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr

        ## find the outliers for each category
        def outliers(group):
            cat = group.name
            return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']

        out = groups.apply(outliers).dropna()

        ## prepare outlier data for plotting, we need coordinates for every outlier.
        if not out.empty:
            outx = list(out.index.get_level_values(0))
            outy = list(out.values)

        TOOLTIPS = [
            ("box lower:", "@top"),
            ("box upper:", "@bottom"),
            ("whisker lower:", "@y1"),
            ("whisker upper:", "@y0"),
        ]

        p = figure(tools="", background_fill_color="#efefef", x_range=cats, toolbar_location=None,
                   title=f"{y_title}", height=600, width=900
                   , tooltips=TOOLTIPS)

        ## if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
        qmin = groups.quantile(q=0.00)
        qmax = groups.quantile(q=1.00)
        upper.score = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, 'score']), upper.score)]
        lower.score = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, 'score']), lower.score)]

        ## stems
        p.segment(cats, upper.score, cats, q3.score, line_color="black")
        p.segment(cats, lower.score, cats, q1.score, line_color="black")

        ## boxes
        p.vbar(cats, 0.7, q2.score, q3.score, fill_color="#E08E79", line_color="black")
        p.vbar(cats, 0.7, q1.score, q2.score, fill_color="#3B8686", line_color="black")

        ## whiskers (almost-0 height rects simpler than segments)
        p.rect(cats, lower.score, 0.2, 0.01, line_color="black")
        p.rect(cats, upper.score, 0.2, 0.01, line_color="black")

        ## outliers
        if not out.empty:
            p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = "white"
        p.grid.grid_line_width = 2
        p.xaxis.major_label_text_font_size = "16px"

        return p


    ## DISPLAY IN BROWSER (DROPDOWNS & PLOT TOGETHER) ---------------------------------------

    ## define alignment of dropdowns and sliders (beneath each other)
    w = 300
    controls = column(sel_time, width=w, margin=(0, 0, 100, 0))
    controls2 = column(controls, stats, width=w)

    ## define alignment of dd_plot (dropdowns and plot) and descriptive statistics
    layout = row(controls2, createFigure())

    ## display as tab in browser
    tab = Panel(child=layout, title="BoxPlot_EnergyPerTimeGranularity")

    return tab

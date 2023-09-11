#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import pandas as pd
import numpy as np
from datetime import date, timedelta
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from loadData import loadData, evaluateMeanPerMonth
import warnings
warnings.filterwarnings("ignore")
import TrendingRules as tr


## load the last 12 months' data, the 12 months' mean and std ------------------------
colName = "temp"
date_col = "datetime"
df_raw = loadData("data/bikeSharing/train.csv", date_col)


#########################################################################
##### DEFINITIONS / PREPARATIONS ########################################
#########################################################################

## dates for selection menu in dashboard --------------------------------------------

## assign today's date and split to access year, month and date separately later on
today_date = pd.to_datetime("today").strftime("%d/%m/%Y")
today_date_list = today_date.split("/")

## define initial date as starting point (e.g. today - 2 years)
initial_date = date.today() - timedelta(days=2*365)
initial_date_list = initial_date.strftime("%d/%m/%Y").split("/")

minAllowedDateInDatePicker = date(2010, 1, 1)
maxAllowedDateInDatePicker = date(int(today_date_list[2]), int(today_date_list[1]), int(today_date_list[0]))
# startDateInDatePicker = date(int(initial_date_list[2]), int(initial_date_list[1]), int(initial_date_list[0]))
# endDateInDatePicker = date(int(today_date_list[2]), int(today_date_list[1]), int(today_date_list[0]))
startDateInDatePicker = date(2010, 1, 1)
endDateInDatePicker = date(2012, 12, 31)


## define the parameters for calculation. For e.g. rule 3, the default number of points is set to 6 for displaying
## "Six (or more) points in a row are continually increasing (or decreasing)". You can adjust the number to
## tailor it towards your own needs.
paramsPerRule = {"Rule 01": {"std_value": 3},
                 "Rule 02": {"no_points": 9},
                 "Rule 03": {"no_points": 6},
                 "Rule 04": {"no_points": 14},
                 "Rule 05": {"noPoints_window": 3, "noPointsOutOf_window": 2, "std_value": 2},
                 "Rule 06": {"noPoints_window": 5, "noPointsOutOf_window": 4, "std_value": 1},
                 "Rule 07": {"no_points": 15, "std_value": 1},
                 "Rule 08": {"no_points": 8, "std_value": 1}
                 }

## map trending rule number to trending rule details: Key, Value will be displayed in drop down menu
rules = {"Rule 01": f"""1 point is more than {paramsPerRule['Rule 01']['std_value']} standard deviations 
                    from the mean""",
         "Rule 02": f"""{paramsPerRule['Rule 02']['no_points']} (or more) points in a row are on the 
                    same side of the mean""",
         "Rule 03": f"""{paramsPerRule['Rule 03']['no_points']} (or more) points in a row are continually 
                    increasing (or decreasing)""",
         "Rule 04": f"""{paramsPerRule['Rule 04']['no_points']} (or more) points in a row alternate in direction, 
                    increasing then decreasing""",
         "Rule 05": f"""{paramsPerRule['Rule 05']['noPointsOutOf_window']} (or 
                    {paramsPerRule['Rule 05']['noPoints_window']}) out of 
                    {paramsPerRule['Rule 05']['noPoints_window']} points in a row are more 
                    than {paramsPerRule['Rule 05']['std_value']} standard deviations from the mean in the same direction""",
         "Rule 06": f"""{paramsPerRule['Rule 06']['noPointsOutOf_window']} (or 
                    {paramsPerRule['Rule 06']['noPoints_window']}) out of {paramsPerRule['Rule 06']['noPoints_window']} 
                    points in a row are more than {paramsPerRule['Rule 06']['std_value']} standard deviation from the 
                    mean in the same direct""",
         "Rule 07": f"""{paramsPerRule['Rule 07']['no_points']} points in a row are all within 
                    {paramsPerRule['Rule 07']['std_value']} standard deviation of the mean on either side of the mean""",
         "Rule 08": f"""{paramsPerRule['Rule 08']['no_points']} points in a row exist, but none within 
                    {paramsPerRule['Rule 08']['std_value']} standard deviation of the mean, and the point are in 
                    both directions from the mean"""
         }

## define color code for legend
color_legend = {"Green": "No obvious trend recognizable!",
                "Yellow": "Within time window of recognized trend!",
                "Red": "Trend recognized!"
                }


#########################################################################
##### LAYOUT ############################################################
#########################################################################

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Nelson Trending Rules"),
    html.Div([
        dcc.DatePickerRange(
            id="chosen_date",
            display_format="DD.MM.YYYY",
            min_date_allowed=minAllowedDateInDatePicker,
            max_date_allowed=maxAllowedDateInDatePicker,
            start_date=startDateInDatePicker,
            end_date=endDateInDatePicker
        )
    ], id="section_datePicker"),
    html.Div([
        dcc.Dropdown(id="dd_meanOrAllValues",
                     options=[f"Monthly Mean Values",
                              f"All Values"],
                     value=f"Monthly Mean Values"),
        dcc.Dropdown(id="dd_rules",
                     options=[f"Rule 01: {rules['Rule 01']}",
                              f"Rule 02: {rules['Rule 02']}",
                              f"Rule 03: {rules['Rule 03']}",
                              f"Rule 04: {rules['Rule 04']}",
                              f"Rule 05: {rules['Rule 05']}",
                              f"Rule 06: {rules['Rule 06']}",
                              f"Rule 07: {rules['Rule 07']}",
                              f"Rule 08: {rules['Rule 08']}"],
                     value=f"Rule 01: {rules['Rule 01']}"),
        dcc.Graph(animate=False,
                  id="graph"),
        dcc.Textarea(id="text_colorCode",
                     value=f"Green: {color_legend['Green']}, Yellow: {color_legend['Yellow']}, Red: {color_legend['Red']}",
                     style={"width": "100%", "height": 50})
    ], id="section_graph"),
])


@app.callback(
    Output(component_id="graph", component_property="figure"),
    Input(component_id="dd_meanOrAllValues", component_property="value"),
    Input(component_id="dd_rules", component_property="value"),
    Input(component_id="chosen_date", component_property="start_date"),
    Input(component_id="chosen_date", component_property="end_date")
)
def updateDashboard(meanOrAllValues, ruleName, start_date, end_date):

    ##### DEFINE PLOT PARAMETERS -----------------------------------------

    print("START DATE, END DATE", start_date, end_date)

    ## evaluate mean per month for each of the last months or use all values
    if meanOrAllValues == "Monthly Mean Values":
        ## select time window according to selection in calendar
        df_dateSelection = df_raw[(df_raw["date_mY"] >= start_date) & (df_raw["date_mY"] <= end_date)]
        dict_lastMonths_meanPerMonth, df_meanOrAllValues = evaluateMeanPerMonth(df_raw["date_mY"].unique(),
                                                                                df_dateSelection, colName)
    else:
        ## select time window according to selection in calendar
        df_dateSelection = df_raw[(df_raw[date_col] >= start_date) & (df_raw[date_col] <= end_date)]
        df_meanOrAllValues = df_dateSelection[[colName, date_col]].set_index(date_col)

    ## create class object for dataframe according to trending rules
    df_object = tr.TrendingRules(df_meanOrAllValues, df_dateSelection, colName)

    ## retrieve data according to chosen trending rule
    df = df_object.getRule(ruleName[:7], **paramsPerRule[ruleName[:7]])

    ## define mean and standard deviations
    if df.empty:
        return None
    else:
        overallMean = np.mean(df[colName])
        overallStd = np.std(df[colName])
        sigmas = {"+1 std": (overallMean + 1 * overallStd, "dash"),
                  "-1 std": (overallMean - 1 * overallStd, "dash"),
                  "+2 std": (overallMean + 2 * overallStd, "dashdot"),
                  "-2 std": (overallMean - 2 * overallStd, "dashdot"),
                  "+3 std": (overallMean + 3 * overallStd, "dot"),
                  "-3 std": (overallMean - 3 * overallStd, "dot"),
                  "mean": (overallMean, None)
                  }


        ##### CREATE PLOT --------------------------------------------------

        ## define line plot
        yAxisParam = list(df[colName])
        xAxisParam = list(df.index)
        color_code = list(df["colorCode"])

        ## create line plot
        fig_line = px.line(x=xAxisParam, y=yAxisParam)
        fig_line.update_traces(line=dict(color="blue", width=2))

        ## create scatter plots with different scatter colors (according to exceeding thresholds)
        fig_scatter = px.scatter(x=xAxisParam, y=yAxisParam)
        fig_scatter.update_traces(marker=dict(color=color_code, size=10))

        ## overlay line and scatter plot to create a single plot
        fig_combined = go.Figure(data=fig_line.data+fig_scatter.data)


        ##### ADD LINES & ANNOTATIONS --------------------------------------

        ## add line for the mean
        fig_combined.add_shape(type='line',
                               x0=list(df.index)[0],
                               y0=overallMean,
                               x1=list(df.index)[-1],
                               y1=overallMean,
                               line=dict(color='Gray', ),
                               xref='x',
                               yref='y'
                               )

        ## add lines for x*sigma
        for k, v in sigmas.items():
            fig_combined.add_shape(type='line',
                                   x0=list(df.index)[0],
                                   y0=v[0],
                                   x1=list(df.index)[-1],
                                   y1=v[0],
                                   line=dict(color='Gray', dash=v[1]),
                                   xref='x',
                                   yref='y'
                                   )

            ## add annotations to the lines for mean and standard deviations
            fig_combined.add_trace(go.Scatter(
                x=[list(df.index)[-1]],
                y=[v[0]],
                text=[k + ": " + '{0:.2f}'.format(v[0])],
                mode="text",
            ))

        ## hide legend
        fig_combined.update_traces(showlegend=False)

        ## rotate x-axis tick labels and assign axis label
        fig_combined.update_xaxes(
            tickangle=-45,
            title_text="Time",
            title_font={"size": 20},
            title_standoff=25
        )

        ## update y-axis and assign axis label
        fig_combined.update_yaxes(
            title_text="Measurand",
            title_font={"size": 20},
            title_standoff=25
        )

        return fig_combined


if __name__ == '__main__':
    app.run_server(debug=True)

#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from loadData import loadData, extractLast12Months, evaluateMeanPerMonth
from barPlotRules import assignColors


## load data ----------------------------------------------------------

## extract the last 12 months' data, the 12 months' mean and std
df = loadData("data/bikeSharing/train.csv")
y = 2012
m = 1
lastMonths, df_sel, df_mean_12m, df_std_12m = extractLast12Months(y, m, 12, df)

## evaluate mean per month
meanPerMonth, df_12months = evaluateMeanPerMonth(lastMonths, df_sel)

## assign outlier color codes based on predefined thresholds
df, coloredLines = assignColors(df_12months, "temp")


##### BARPLOT ###########################################################
#########################################################################

barplot = Dash(__name__)

## define plot
yAxisParam = df["temp"]
xAxisParam = list(df.index)
fig = px.bar(x=xAxisParam, y=yAxisParam,
             labels=dict(x="Time", y="count"),
             color=xAxisParam,
             color_discrete_sequence=list(df["colorCode"])
             )

## add red line
fig.add_shape(type='line',
                x0=list(df.index)[0],
                y0=coloredLines["red"],
                x1=list(df.index)[-1],
                y1=coloredLines["red"],
                line=dict(color='Red',),
                xref='x',
                yref='y'
)

## add yellow line
fig.add_shape(type='line',
                x0=list(df.index)[0],
                y0=coloredLines["yellow"],
                x1=list(df.index)[-1],
                y1=coloredLines["yellow"],
                line=dict(color='Yellow',),
                xref='x',
                yref='y'
)

# ## add green line
# fig.add_shape(type='line',
#                 x0=list(df.index)[0],
#                 y0=coloredLines["green"],
#                 x1=list(df.index)[-1],
#                 y1=coloredLines["green"],
#                 line=dict(color='Green',),
#                 xref='x',
#                 yref='y'
# )

## create plot
barplot.layout = html.Div([
    html.H4("BAR PLOT"),
    dcc.Graph(id="graph", figure=fig)
    ])


if __name__ == '__main__':
    barplot.run_server()

#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from loadData import loadData, extractLast12Months, evaluateMeanPerMonth
import TrendingRules as tr


##### PREPARATION ------------------------------------------------------------------------

## load the last 12 months' data, the 12 months' mean and std
df_raw = loadData("data/bikeSharing/train.csv")
y = 2012
m = 1
lastMonths, df_sel, df_mean_12m, df_std_12m = extractLast12Months(y, m, 12, df_raw)

## evaluate mean per month for each of the last 12 months
dict_meanPerMonth, df_12months = evaluateMeanPerMonth(lastMonths, df_sel)

## assign values and dash types for the different standard deviations and the mean
sigmas = {"+1 std": (df_mean_12m + 1 * df_std_12m, "dash"),
          "-1 std": (df_mean_12m - 1 * df_std_12m, "dash"),
          "+2 std": (df_mean_12m + 2 * df_std_12m, "dashdot"),
          "-2 std": (df_mean_12m - 2 * df_std_12m, "dashdot"),
          "+3 std": (df_mean_12m + 3 * df_std_12m, "dot"),
          "-3 std": (df_mean_12m - 3 * df_std_12m, "dot"),
          "mean": (df_mean_12m, None)
          }


##### LINE PLOT #########################################################
#########################################################################

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Live control of annotations'),
    html.P("Select text position:"),
    dcc.Dropdown(["Rule 01", "Rule 02", "Rule 03", "Rule 04", "Rule 05", "Rule 06",
                  "Rule 07", "Rule 08"], "Rule 01", id="TrendingRule_DD"),
    dcc.Graph(id="graph"),
    dcc.Markdown(id="myMarkdown")
])


@app.callback(
    Output("graph", "figure"),
    Output("myMarkdown", "children"),
    Input("TrendingRule_DD", "value")
)
def modify_legend(ruleName):

    ##### DEFINE PLOT -------------------------------------------------

    ## create class object for dataframe according to trending rules
    df_object = tr.TrendingRules(df_12months, "temp")

    ## retrieve data according to chosen trending rule
    df = df_object.getRule(ruleName)

    ## define line plot
    yAxisParam = df["temp"]
    xAxisParam = list(df.index)
    fig = px.line(x=xAxisParam, y=yAxisParam,
                  title="LinePlot",
                  labels=dict(x="Time", y="Monthly Mean"),
                  markers=True,
                  color=xAxisParam,
                  color_discrete_sequence=list(df["colorCode"])
                  )


    ##### ADD LINES ----------------------------------------------------

    ## add line for the mean
    fig.add_shape(type='line',
                  x0=list(df.index)[0],
                  y0=df_mean_12m,
                  x1=list(df.index)[-1],
                  y1=df_mean_12m,
                  line=dict(color='Gray', ),
                  xref='x',
                  yref='y'
                  )

    ## add lines for x*sigma
    for k, v in sigmas.items():
        fig.add_shape(type='line',
                      x0=list(df.index)[0],
                      y0=v[0],
                      x1=list(df.index)[-1],
                      y1=v[0],
                      line=dict(color='Gray', dash=v[1]),
                      xref='x',
                      yref='y'
                      )

        ## add text to the lines for mean and standard deviations
        fig.add_trace(go.Scatter(
            x=[""],
            y=[v[0]],
            text=[k],
            mode="text",
        ))


    ##### ADD TEXT --------------------------------------------------------

    text1 = f"""+3 std: {'{:.2f}'.format(df_mean_12m + 3 * df_std_12m)}
             \n+2 std: {'{:.2f}'.format(df_mean_12m + 2 * df_std_12m)}
             \n+1 std: {'{:.2f}'.format(df_mean_12m + 1 * df_std_12m)}
             \nMean of the last 12 months: {'{:.2f}'.format(df_mean_12m)}
             \n-1 std: {'{:.2f}'.format(df_mean_12m - 1 * df_std_12m)}
             \n-2 std: {'{:.2f}'.format(df_mean_12m - 2 * df_std_12m)}
             \n-3 std: {'{:.2f}'.format(df_mean_12m - 3 * df_std_12m)}"""

    return fig, text1


if __name__ == '__main__':
    app.run_server()

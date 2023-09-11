#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import calculateSampleSize_aql as cs
from dash import Dash, html, dcc, Input, Output, State
import openpyxl


######################################################################################################
##### DEFINE SAMPLE SIZE CALCULATION #################################################################
######################################################################################################

def calculateWithInput(lot_quantity: int, inspec_level: str, aql: str) -> int:
    """
    Instantiate class object and evaluate sample size.
    """
    classObject = cs.CalculateSampleSizeAQL(lot_quantity=lot_quantity, inspec_level=inspec_level, aql=aql)
    classObject.loadDataToDict()
    sampleSize, maxAcceptLevel, minRejectLevel = classObject.evaluateSampleSize()
    return sampleSize, maxAcceptLevel, minRejectLevel


######################################################################################################
##### BUILD APP VIA DASH #############################################################################
######################################################################################################

app = Dash(__name__)

## define app layout
app.layout = html.Div([
    html.H2("Calculate Sample Size of Population"),
    html.Div([
        html.H3("Parameters"),
        html.H4("Lot Size [Absolute Number]"),
        dcc.Input(id="input_lotSize", type="number", placeholder="Enter Lot Size"),
        html.Br(),
        html.H4("Inspection Level"),
        dcc.Dropdown(options=["G1", "G2", "G3", "S1", "S2", "S3", "S4"],
                     value="G2",
                     id="dd_inspecLevel"),
        html.Br(),
        html.H4("AQL [%]"),
        dcc.Dropdown(options=["0.065", "0.10", "0.15", "0.25", "0.4", "0.65", "1.0", "1.5", "2.5", "4.0", "6.5"],
                     value="0.065",
                     id="dd_aql"),
        html.Br(),
    ], id="section_parameters"),
    html.Div([
        html.H3("Start Calculation"),
        html.Button("Calculate", id="btn_calculate", n_clicks=0),
        html.Br(),
    ], id="section_pressButton"),
    html.Div([
        html.H3("Result"),
        dcc.Markdown(id="result")
    ], id="section_result"),
])

## initialize number of button clicks
n_clicks_btnCalculate = 0

## define callbacks
@app.callback(
    Output("result", "children"),
    Input("input_lotSize", "value"),
    Input("dd_inspecLevel", "value"),
    Input("dd_aql", "value"),
    Input("btn_calculate", "n_clicks"),
)
def cb_render(inp_lotSize, ddChoice_inspecLevel, ddChoice_aql, btnClicks_calculate):
    """
    Render input field, dropdown menus and calculation button. Calculate sample size and output.
    """
    ## provide console output
    print("Lot Size:", inp_lotSize,
          "    Inspection Level:", ddChoice_inspecLevel,
          "    AQL [%]:", ddChoice_aql,
          "    Button n_clicks:", btnClicks_calculate)

    global n_clicks_btnCalculate

    ## if no button was clicked, return (--> otherwise error messages by data input)
    text_result = """Please choose *Lot Size*, *Inspection Level* and *AQL*. 
                  \nStart your search by pressing the *Calculate* button!"""
    if (btnClicks_calculate == n_clicks_btnCalculate):
        return text_result

    ## Calculate required sample size from input
    elif inp_lotSize != None:
        if btnClicks_calculate > 0:
            n_clicks_btnCalculate += 1
            sampleSize, maxAcceptLevel, minRejectLevel = calculateWithInput(inp_lotSize, ddChoice_inspecLevel, ddChoice_aql)
            text_result = f"""The REQUIRED SAMPLE SIZE is **{sampleSize}** 
                              \nfor LOT SIZE **{inp_lotSize}**, INSPECTION LEVEL 
                               **{ddChoice_inspecLevel}** and AQL **{ddChoice_aql}** 
                               \nwith MAX. ACCEPTANCE LEVEL
                               **{maxAcceptLevel}** and MIN. REJECTION LEVEL **{minRejectLevel}**.
                           """

        else:
            text_result = "The sample size could not be evaluated due to missing input!"

    return text_result


if __name__ == "__main__":
    app.run_server(debug=True)

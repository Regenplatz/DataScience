

__author__ = "WhyKiki"
__version__ = "1.0.0"


import calculateSampleSize as cs
from dash import Dash, html, dcc, Input, Output
import openpyxl


######################################################################################################
##### DEFINE SAMPLE SIZE CALCULATION #################################################################
######################################################################################################

def calculateWithNumbers(population_size, ci, me, p, formula):
    """
    :param population_size: int, size of population (absolute number)
    :param ci: int, confidence interval in %
    :param me: int, margin of error in %
    :param p: int, probability that the sample size is big enough in terms of confidence interval,
        default usually 50% (for p of 0.5)
    :param formula: String, implies which formula should be applied for calculation
    """
    ## instantiate class object
    size_obj = cs.CalculateSampleSize(population_size, ci, me, p)

    ## calculate sample size (according to chosen formula)
    if formula == "standard":
        size_obj.evaluate_zScore()
        sample_size = size_obj.formulaStandard()
    elif formula == "hugePopulation":
        size_obj.evaluate_zScore()
        sample_size = size_obj.formulaUnknownHugePopulation()
    else:
        sample_size = size_obj.formulaSlovin()

    return sample_size


######################################################################################################
##### BUILD APP VIA DASH #############################################################################
######################################################################################################

app = Dash(__name__)

## define app layout
app.layout = html.Div([
    html.H2("Calculate Sample Size of Population"),
    html.Div([
        html.H3("Parameters"),
        html.H4("Population [Absolute Number]"),
        dcc.Input(id="input_population", type="number", placeholder="Population"),
        html.Br(),
        html.H4("Confidence Interval [%]"),
        dcc.Input(id="input_ci", type="number", placeholder="Confidence Interval [%]"),
        html.Br(),
        html.H4("Margin of Error [%]"),
        dcc.Input(id="input_me", type="number", placeholder="Maring of Error [%]"),
        html.Br(),
        html.H4("Probability [%]"),
        dcc.Input(id="input_p", type="number", placeholder="Probability [%]"),
        html.Br(),
    ], id="section_parameters"),
    html.Div([
        html.H3("Start Calculation"),
        html.Button("Standard Formula", id="btn_stdFormula", n_clicks=0),
        html.Button("For Huge Population", id="btn_hugeFormula", n_clicks=0),
        html.Button("Slovin's Formula", id="btn_slovinFormula", n_clicks=0),
        html.Br(),
    ], id="section_pressButton"),
    html.Div([
        html.H3("Result"),
        dcc.Markdown(id="result")
    ], id="section_result"),
])

## initialize number of clicks per button
n_clicks_stdFormula = 0
n_clicks_hugeFormula = 0
n_clicks_slovinFormula = 0

## define callbacks
@app.callback(
    Output("result", "children"),
    Input("input_population", "value"),
    Input("input_ci", "value"),
    Input("input_me", "value"),
    Input("input_p", "value"),
    Input("btn_stdFormula", "n_clicks"),
    Input("btn_hugeFormula", "n_clicks"),
    Input("btn_slovinFormula", "n_clicks"),
)
def cb_render(i_population, i_ci, i_me, i_p, btn_stdF, btn_hugeF, btn_slovinF):
    """
    Render input fields, calculation button per formula. Calculate sample size and output.
    :param i_population:
    :param i_ci:
    :param i_me:
    :param i_p:
    :param btn_stdF:
    :param btn_hugeF:
    :param btn_slovinF:
    :return:
    """
    ## provide console output
    print(i_population, i_ci, i_me, i_p, btn_stdF, btn_hugeF, btn_slovinF)

    ## activate global variables
    global n_clicks_stdFormula
    global n_clicks_hugeFormula
    global n_clicks_slovinFormula

    ## if no button was clicked, return (--> otherwise error messages by data input)
    if ((btn_stdF == n_clicks_stdFormula) and
            (btn_hugeF == n_clicks_hugeFormula) and
            (btn_slovinF == n_clicks_slovinFormula)):
        return ""

    ## Calculate required sample size from input
    elif (i_population != None and i_ci != None and i_me != None and i_p != None):
        formula = ""
        if btn_stdF > n_clicks_stdFormula:
            n_clicks_stdFormula += 1
            formula = "standard"
        elif btn_hugeF > n_clicks_hugeFormula:
            n_clicks_hugeFormula += 1
            formula = "hugePopulation"
        elif btn_slovinF > n_clicks_slovinFormula:
            n_clicks_slovinFormula += 1
            formula = "slovin"

        sample_size = calculateWithNumbers(i_population, i_ci, i_me, i_p, formula)
        text_result = f"""The required sample size for a population of {i_population}, a confidence interval of 
                           {i_ci}, error margin of {i_me} and a probability of {i_p} is {sample_size}.
                           This was calculated with {formula} formula.    
                       """

    else:
        text_result = "The sample size could not be evaluated due to missing input!"

    return text_result


if __name__ == "__main__":
    app.run_server(debug=True)

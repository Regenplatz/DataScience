## Calculate Sample Size of a Population

This project was designed to build a sample size calculator as Web App. For this, [plotly Dash](https://plotly.com/dash/) was used. The app provides input fields for  `population size`, the desired `confidence interval`, `margin of error` and  `probability`.

It furthermore provides three buttons for the different formulas:
1. Standard Formula
2. Formula for Unknown or Hugh Populations
3. Slovin's Formula (when no further knowledge about the population is available)

[calculateSampleSize_Dash.py](calculateSampleSize_Dash.py) contains the code for web app layout and functionality. For `z-score` lookup and sample size calculation, it calls a class in [calculateSampleSize.py](calculateSampleSize.py).

[CalculationSamplesSize.ipynb](CalculationSampleSize.ipynb) serves as demonstration of the building blocks for z-score lookup and sample size calculation.

### Data
Data was obtained from [European Climate Assessment & Dataset](https://www.ecad.eu/dailydata/predefinedseries.php)
in July 2021. Only data for weather station 244 ('STAID000244'), which is located in ZUERICH/FLUNTERN, SWITZERLAND
was further processed. The dataset consisted of the daily mean temperature ("TG").

#       
### Data Processing
Data was processed to analyze basics, SARIMA as well as Machine Learning approaches. Therefore, different files were created.

##### Basics
[ClimateZurich_Basics.py](ClimateZurich_Basics.py) was built for basic time series analysis. It includes test/plots for stationarity (Augmented Dickey-Fuller Test), (partial) autocorrelation function (ACF, PACF), moving average and time series decomposition.
The is also the basis for the other two files because it includes the reading of data into a dataframe. Moreover, it also demonstrates the usage of object-oriented programming (classes, inheritance) in python.


##### Seasonal ARIMA
[ClimateZurich_SARIMA.py](ClimateZurich_SARIMA.py) deals with seasonal autoregressive integrated moving average (ARIMA). It requires the import of *ClimateZurich_Basics.py* and also includes object-oriented programming including inheritance.


##### Machine Learning Approach
[ClimateZurich_MachineLearning.py](ClimateZurich_MachineLearning.py) provides the Machine Learning approach. It was designed to demonstrate how search for and prediction with the evaluated best algorithm could be automated. This file also requires the import of *ClimateZurich_Basics.py* to get access to the preprocessed dataframe. Unlike the other two file, it does not include object-oriented programming.

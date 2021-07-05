### About the Project
The project's goal was to illuminate influences of the time of Elon Musk's tweets as well as Google trends on the asset returns bitcoin, dogecoin and ethereum in USD. It focused on the analysis of correlation and causality between a) cryptocurrency asset returns and Musk’s tweets and b) cryptocurrency asset returns and Google trends. Note: The project did not focus on sentiment analysis.

<br>

### Data Collection and Preprocessing
For this project, data from different data sources was obtained. Therefore, various techniques were applied. All data was obtained from June 1st 2016 onwards.

##### Asset Returns
Data concerning cryptocurrency prizes was obtained from [yahoo! Finance](https://finance.yahoo.com) via API. Observations where no data was available were replaced with the data from the day before, in order to have a continuous data set with daily values.

##### Elon Musk's Tweets
Data on Elon Musk's tweets was obtained from [kaggle](https://www.kaggle.com/ayhmrba/elon-musk-tweets-2010-2021) as csv files (accessed: 01 May 2021). The data was provided as one file per year, so that tweet data from the selected years 2016 (starting from 1st June) to 2021 (ending at the last available tweet from 22nd March) was bundled in a single data frame. Data was then reduced to the observations that contained (independent from upper and lower case) the corresponding keywords (bitcoin, dogecoin, ethereum) or their abbreviations or wrong spelling (btc, doge, etherium). Note: Data on Musk's tweets were limited to March 2021.

##### Google Trends
Data on google trends was obtained from [Google Trends](https://trends.google.de/trends) using R's gtrendsR package. Its gtrends() function allowed the trend collection according to maximum 5 keywords. Those were set to "Bitcoin", "bitcoin", "btc", "BTC", "BITCOIN" for Bitcoin, "Dogecoin", "dogecoin", "doge", "DOGE", "DOGECOIN" for Dogecoin and "Ethereum", "ethereum", "ETHEREUM", "Etherium",  "ETHERIUM" for Ethereum. Note: Data on google trends was collected until May 2021.

<br>

#### Analysis
Statistical analysis focused on vector autoregression (VAR) and Granger causality to identify correlations between Google trends or Musk's tweets and asset returns.

##### Asset Returns and Musk's Tweets
Cryptocurrency prices were assigned to two different groups: The first one contained the prices after a tweet within a predefined lag in days, whereas the other one contained the remaining observations (time periods without tweets). Those two groups were analyzed in the following to check for significant differences. The datasets of the asset returns as well as the tweets data have been tested for and transformed to stationarity where needed, in order to apply the VAR model and Granger causality test to identify significant effects between the variables

##### Asset Returns and Google Trends
The data sets of the asset returns as well as the Google trends data have been tested for and transformed to stationarity where needed, in order to apply the VAR model and Granger causality test to identify significant effects between the variables.

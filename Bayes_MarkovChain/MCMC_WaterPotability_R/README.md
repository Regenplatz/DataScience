### Bayesian Data Analysis on Water Potability
A Bayesian data analysis project in R using JAGS.

#### Data Collection and Preprocessing
Data was obtained from [kaggle](https://www.kaggle.com/adityakadiwal/water-potability) in June 2021. The 3276 observations contained information on *potability* (binary) as well as *ph*, *hardness*, *solids*, *chloramines*, *sulfate*, *conductivity*, *organic carbon*, *trihalomethanes* and *turbidity*.

Preprocessing and calculations can be found in the script [waterQuality.R](waterQuality.R).

Three columns contained NULL values: *ph* (491), *trihalomethanes* (162) and *sulfate* (781). Instead of dropping those columns or observations, normally distributed data was simulated based on the respective non-missing values.


#### The Model
Sampling density for the dependent variable potability (binary) was set to Bernouille distribution (*dbern*). The independent variables’ prior distributions were defined as normal (*dmnorm*). Two starting points with reverse signs were defined to start the chains from different directions. The posterior was calculated with a burn in period of 1000 followed by 10000 iterations.

#### Results
The following results were observed:

| Independent variable | beta (std.error) | 95% confidence interval  | sample size | effective sample size |
| ------------- |:-------------:| -----:|-----:|-----:|
| pH              |  0.000 (0.024) | [-0.049, 1.405] | 3276 | 1000 |
| hardness        | -0.001 (0.001) | [-0.003, 0.001] | 3276 | 2000 |
| chloramines     |  0.030 (0.022) | [-0.013, 0.073] | 3276 | 1000 |
| sulfate         | -0.001 (0.001) | [-0.003, 0.001] | 3276 | 1600 |
| conductivity    |  0.000 (0.000) | [-0.001, 0.001] | 3276 | 2000 |
| organic carbon  | -0.018 (0.011) | [-0.040, 0.003] | 3276 | 1000 |
| trihalomethanes |  0.001 (0.002) | [-0.004, 0.005] | 3276 | 2000 |
| turbidity       |  0.002 (0.047) | [-0.089, 0.091] | 3276 | 2000 |

Calculations revealed deviance being 190370844.104 (±4.282, confidence interval: [190370837.883, 1.903709e+08]), deviance information criteria (DIC) being 190370853.3 and pD being 9.2. Gelman Rubin statistics were represented as Rhat, meaning the ratio of between-chain-variances to within-between-chain-variances. All Rhat values were below 1.1, indicating strong convergence. As further convergence diagnostics, traceplots were produced which accordingly showed strong evidence of convergence.

#### Conclusion
All calculated beta values were found to be approximately zero. This result implies that none of the analyzed parameters has any effect on the water potability. Further analysis should take other potential influential factors into account like e.g. bacteria count or water pollution.

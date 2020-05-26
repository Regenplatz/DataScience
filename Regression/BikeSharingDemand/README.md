### About the Project

"Help a metropolitan bicycle sharing system to find out how many bicycles need to be ready at a given time"

Data was taken from https://www.kaggle.com/c/bike-sharing-demand/data      
For further information see also file [info_on_data](https://github.com/Regenplatz/DataScience/blob/master/Regression/BikeSharingDemand/info_on_data)

This regression project was tried with different approaches regarding preprocessing of the self-introduced features `hour` and `month` (feature 'season' was eliminated to try to get more specific predictions per month). This finally addressed the issue *one-hot-encoding* vs *multicolinearity*.

#  
### About the Notebooks

- **bicycleSharing1**:               

	No one-hot-encoding used for 'hour' and 'month' features. This approach is not right at all as categorical variables like 'hour 11' is not worth more than 'hour 3'. However, it was tried for later comparison with the other approaches (which I like to call 'try to learn (and to see the differences)').

- **bicycleSharing2**:

	One-hot-encoding used for 'hour' and 'month' features.
	Before further processing, drop the last new feature to avoid perfect multicolinearity (as one of those new created features can be indirectly represented by the other new created features (were all entries are zero)).

- **bicycleSharing3**:    

	First, introduce another column for 4-hour intervals ('hour4h') and finally one-hot-encode this feature.
	Before further processing, drop the last new feature to avoid perfect multicolinearity (as one of those new created features can be indirectly represented by the other new created features (were all entries are zero)).

Apart from this differing preprocessing, further processing was performed in the same way. Different regression models were tried such as *RidgeCV*, *ElasticNet*, *SVR* and *RandomForestRegressor*.

You get different scores, different RMSLE and different likelihood to over-fit!

At first sight, *bicycleSharing1* seems to work best, although it is not the way you should handle it!      
Don't fall into the `Dummy Variable Trap` when you deal with categorical features!

Further analysis to be done ...

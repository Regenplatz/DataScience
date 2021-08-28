### About the Project

"Help a metropolitan bicycle sharing system to find out how many bicycles need to be ready at a given time"

Data was taken from https://www.kaggle.com/c/bike-sharing-demand/data      
For further information see also file [info_on_data](info_on_data)


#  
### About the Notebooks

- [**bikeSharing1**](bikeSharing1.ipynb):               

	`Initial data exploration` was performed to get familiar with the kaggle training data set and their underlying distributions. Plots showed the absence of linear relations between feature *count* and the rest. *Count* also showed a right `skewed distribution` which might imply outliers. Features *casual* and *registered* explain 100% of feature *count* while *holiday* and *workingday* were not found to be mutually exclusive. Furthermore, `variance inflation factor` revealed `multicollinearity` for *temp* and *atemp* (which could also be seen in a heatmap). As `OLS assumptions` were not fulfilled, the machine learning part focused on non-linear regression models (as shown in [bikeSharing2](bikeSharing2.ipynb)). In the interest of completeness, also linear regression models using different regularization techniques were tried (see [bikeSharing3](bikeSharing3.ipynb)). As expected, the results were far from good.

- [**bikeSharing2**](bikeSharing2.ipynb):

  The preprocessed DataFrame from [bikeSharing1](bikeSharing1.ipynb) was further processed using the following non-linear regression models: *RandomForestRegressor*, *Gradient Boosting Regressor*, *AdaBoost Regressor*, *Support Vector Regressor (SVR)*. Except for SVR, they showed prediction scores >90%. *GradientBoostingRegressor* and *RandomForestRegressor* revealed RMSLE values <0.4, while SVR showed the worst result with a value greater than 0.7.

- [**bikeSharing3**](bikeSharing3.ipynb):    

	This notebook shows linear regression models with different regularization techniques: *LinearRegression*, *RidgeCV*, *LassoCV*, *ElasticNet*. Although not expected to perform well, it is still shown here for the sake of completeness. All models performed bad with training and test scores <40%. RMSLE were found >1%.

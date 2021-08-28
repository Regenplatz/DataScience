#!/usr/local/bin/python

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# models
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# dataframe to be analyzed
import ClimateZurich_Basics as czb
data = czb.PlotData('TG_STAID000244.txt')
laggedData = czb.LagData(data.df, "TG")
laggedData.generateLagData()


def assignData(df, column, year):
    """
    assign train and test data
    :param df: dataframe,  contains the time series data to be analyzed (among other things)
    :param column: series, data to be analyzed
    :param year: integer, cuts data into train (< year) and test data (>= year)
    :return: dataframes (x_train, x_test), series (y_train, y_test)
    NOTE::  x refers to lag data, y refers to real data
    """
    features = [f'{column} -2month',
                f'{column} -3month',
                f'{column} -4month',
                f'{column} -5month']
    # assign train data
    x_train = df[features][df['year'] < year]
    y_train = pd.DataFrame(df.loc[df['year'] < year, column])
    # assign test data
    x_test = df[features][df['year'] >= year]
    y_test = pd.DataFrame(df.loc[df['year'] >= year, column])
    return x_train, y_train, x_test, y_test


def rmse(x, y):
    """
    calculate root mean squared error (RMSE) as quality metrics for time series analysis
    :param x: dataframe, independent variable
    :param y: series, dependent variable
    :return: float, RMSE score
    """
    y = np.array(y)
    x = np.array(x)
    distance = y - x
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    rmseScore = np.sqrt(mean_square_distance)
    return rmseScore


def evaluateBestParams(model, model_params, x_train, y_train):
    """
    evaluate best hyperparameters for a given model
    :param model: selected model
    :param model_params: model's hyperparameters to be screened for the best suitable ones
    :param x_train: dataframe, train data (independent variable)
    :param y_train: series, train data (dependent variable)
    :return: model (best estimator) and dictionary (best parameters)
    """
    rmse_score = make_scorer(rmse, greater_is_better=False)
    tscv = TimeSeriesSplit(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=model_params, scoring=rmse_score)
    gsearch.fit(x_train, np.ravel(y_train))
    print("Best estimator: ", gsearch.best_estimator_)
    print("Best score: ", round(gsearch.best_score_, 4))
    print("Best Parameters: ", gsearch.best_params_)
    return gsearch.best_estimator_, gsearch.best_params_


def fit_predict(model, x_train, y_train, x_test, y_test):
    """
    fit data and predict
    :param model: selected model
    :param x_train: dataframe, train data (independent variable)
    :param y_train: series, train data (dependent variable)
    :param x_test: dataframe, test data (independent variable)
    :param y_test: series, test data (dependent variable)
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mScore = mean_squared_error(y_test, y_pred)
    # mScore = mean_squared_error(np.array(x_test).flatten(), np.array(y_pred).flatten())
    return y_pred, mScore


def compareDifferentModels(model_dict, y):
    """
    compare different algorithms visually
    :param model_dict: dictionary, info about model (key) and
           best model, best parameters, y_pred and MSE score (values as single dictionaries)
    :param y: series, test data (dependent variable), also referred to as y_test
    """
    # initialize new dataFrame with y_test data (as control)
    df_new = pd.DataFrame(y)
    # convert y_pred and y_test to usable format for further analysis
    for key, value in model_dict.items():
        # define y_pred (per model) and y_test (once)
        y_pred = np.array(value["y_pred"]).flatten()
        y_test = np.array(y).flatten()
        print(y_pred)
        print(y_test)
        # create new features in dataframe for y_pred (per model)
        df_new[key] = y_pred

    # scatter plots
    y_test = df_new["TG"]
    plt.figure(figsize=(15, 15))
    for i, elem in enumerate(df_new.columns[1:], 1):
        plt.subplot(4, 2, i)
        plt.plot(y_test, df_new[elem], 'o')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
        plt.xlabel("y_test")
        plt.ylabel("y_pred")
        plt.title(f"{df_new[elem].name}: y_pred vs y_test")
        # plot linear regression line: m=slope, b=intercept
        m, b = np.polyfit(y_test, df_new[elem], 1)
        plt.plot(y_test, b + (m * y_test))
    plt.show()

    # boxplots for general comparison of algorithms
    plt.boxplot(df_new, labels=df_new.columns)
    plt.title('Algorithm Comparison')
    plt.show()

    # cat plot for score comparison of algorithms:
    # first, create new dataframe with model names and corresponding scores
    score_list = []
    model_list = []
    for key, value in model_dict.items():
        score = value["score"]
        model_list.append(key)
        score_list.append(score)
    score_dict = {"Regression Model": model_list, "score": score_list}
    df_scores = pd.DataFrame(score_dict)
    print(df_scores)
    fig = sns.catplot(y="Regression Model", x="score",
                      data=df_scores, kind="bar", height=5, aspect=3)
    fig.set(xlim=(0.0, 12.0))
    plt.show()


def main():

    # assign data
    x_train, y_train, x_test, y_test = assignData(data.df, "TG", 2016)

    # assign model hyperparameters to be screened for
    linR_param_search = {"n_jobs": [-1]}
    rfR_param_search = {"n_estimators": [20, 50, 100],
                        "max_features": ["auto", "sqrt", "log2"],
                        "max_depth": [i for i in range(5, 15)],
                        "n_jobs": [-1],
                        "random_state": [12345]}
    knR_param_search = {"n_neighbors": range(3,7),
                        "weights": ["uniform", "distance"],
                        "n_jobs": [-1]}
    mlpR_param_search = {"solver": ["lbfgs", "adam"],
                         "activation": ["relu", "logistic", "tanh"],
                         "hidden_layer_sizes": [(50,), (150,), (250,), (350,), (450,)],
                         "random_state": [12345]}
    svR_param_search = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                        "degree": range(1,4),
                        "gamma": ["scale", "auto"]}

    # create dictionaries per model with model names and
    # corresponding hyperparameters to be screened in order to find the best suitable ones
    paramSearch = {}
    paramSearch["LinR"] = {"model": LinearRegression(), "param": linR_param_search}
    paramSearch["RFR"] = {"model": RandomForestRegressor(), "param": rfR_param_search}
    paramSearch['KNR'] = {"model": KNeighborsRegressor(), "param": knR_param_search}
    paramSearch['MLPR'] = {"model": MLPRegressor(), "param": mlpR_param_search}
    paramSearch['SVR'] = {"model": SVR(), "param": svR_param_search}

    # for each model evaluate best estimator, best params, evaluated y_pred and score via prediction,
    # write results to dictionary
    model_param = {}
    for key, value in paramSearch.items():
        bestOptimizedModel, bestParams = evaluateBestParams(value["model"], value["param"], x_train, y_train)
        y_pred, modelScore = fit_predict(bestOptimizedModel, x_train, y_train, x_test, y_test)
        model_param[key] = {"model": bestOptimizedModel,
                            "bestParams": bestParams,
                            "y_pred": y_pred,
                            "score": modelScore}

    # compare the different models visually with regard to MSE (mean squared error) score
    compareDifferentModels(model_param, y_test)


if __name__ == "__main__":
    main()

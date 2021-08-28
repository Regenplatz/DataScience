#!/usr/local/bin/python

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
import warnings
warnings.filterwarnings("ignore")

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# read data and drop unnecessary features (to avoid high levels of multicollinearity)
df = pd.read_csv("df_preprocessed.csv", index_col=None)
df.drop(["signal"], axis=1, inplace=True)
df.drop(["channel_pattern"], axis=1, inplace=True)


def assignData(df, feature):
    """
    assign train and test data
    :param df: dataframe, containing the data to be analyzed
    :param feature: feature, data to be analyzed (dependent variable, y)
    :return: train and test data (both dependent (series) and independent (dataframes) variables)
    """
    # assign X and y
    X = df.drop(feature, axis=1)
    y = df[feature]
    # scale X data
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
    return X_train, y_train, X_test, y_test


def rmse(x, y):
    """
    calculate root mean squared error (RMSE) as quality metrics for grid search on time series analysis
    :param x: dataframe, independent variable
    :param y: series, dependent variable
    :return: float, RMSE score
    """
    x = np.array(x)
    y = np.array(y)
    distance = y - x
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    rmseScore = np.sqrt(mean_square_distance)
    return rmseScore


def crossValModel(model, x_train, y_train):
    """
    try models with best determined hyperparameters and use cross-validation within evaluation
    :param model: given model for supervised machine learning
    :param x_train: dataframe, independent train data
    :param y_train: series, dependent train data
    :return: cross-validation split algorithm
    """
    results = []
    cv = StratifiedKFold(n_splits=10)
    cv_results = cross_val_score(model, x_train, np.ravel(y_train), cv=cv, scoring="f1")
    results.append(cv_results)
    print(f"{model}: mean={cv_results.mean()}, std={cv_results.std()}")
    return cv


def evaluateBestParams(model, model_params, x_train, y_train):
    """
    evaluate best hyperparameters for a given model
    :param model: selected model
    :param model_params: model's hyperparameters to be screened for the best suitable ones
    :param x_train: dataframe, train data (independent variable)
    :param y_train: series, train data (dependent variable)
    :return: evaluated best estimator (model) and best params (dictionary)
    """
    rmse_score = make_scorer(rmse, greater_is_better=False)
    # cv = crossValModel(model, x_train, y_train)
    cv = StratifiedKFold(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=cv, param_grid=model_params, scoring=rmse_score)
    gsearch.fit(x_train, np.ravel(y_train))
    print("Best estimator: ", gsearch.best_estimator_)
    print("Best score: ", round(gsearch.best_score_, 4))
    print("Best Parameters: ", gsearch.best_params_)
    return gsearch.best_estimator_, gsearch.best_params_


def fit_predict(model, x_train, y_train, x_test, y_test):
    """
    fit data and predict dependent variable
    :param model: selected model
    :param X_train: dataframe, train data (independent variable)
    :param y_train: dataframe, train data (dependent variable)
    :param X_test: series, test data (independent variable)
    :param y_test: series, test data (dependent variable)
    :return: series, predicted dependent variable and calculated accuracy
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy


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

    # boxplots for general comparison of algorithms
    plt.boxplot(df_new, labels=df_new.columns)
    plt.title('Algorithm Comparison')
    plt.show()

    # cat plot for score comparison of algorithms:
    # first, create new dataframe with model names and corresponding scores
    score_list = []
    model_list = []
    for key, value in model_dict.items():
        score = value["accuracy"]
        model_list.append(key)
        score_list.append(score)
    score_dict = {"Regression Model": model_list, "accuracy": score_list}
    df_scores = pd.DataFrame(score_dict)
    print(df_scores)
    fig = sns.catplot(y="Regression Model", x="accuracy",
                      data=df_scores, kind="bar", height=5, aspect=3)
    fig.set(xlim=(0.0, 1.05))
    plt.show()


def main():

    # assign data
    x_train, y_train, x_test, y_test = assignData(df, "open_channels")

    # assign model's hyperparameters to be screened for
    logR_param_search = {"n_jobs": [-1],
                         "random_state": [12345]}
    # rfC_param_search = {"n_estimators": [20, 50, 100],
    #                     "max_features": ['auto', 'sqrt', 'log2'],
    #                     "max_depth": [i for i in range(5, 15)],
    #                     "criterion": ["gini", "entropy"],
    #                     "n_jobs": [-1],
    #                     "random_state": [12345]}
    # knC_param_search = {"n_neighbors": range(3, 7),
    #                     "weights": ["uniform", "distance"],
    #                     "n_jobs": [-1]}
    # svC_param_search = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
    #                     "degree": range(1, 4),
    #                     "gamma": ["scale", "auto"],
    #                     "random_state": [12345]}
    # mlpC_param_search = {"solver": ["lbfgs", "adam"],
    #                      "activation": ["relu", "logistic", "tanh"],
    #                      "hidden_layer_sizes": [(50,), (150,), (250,), (350,), (450,)] ,
    #                      "random_state": [12345]}

    # create dictionary with model names and corresponding screening hyperparameters
    paramSearch = {}
    paramSearch["LogR"] = {"model": LogisticRegression(),
                           "param": logR_param_search}
    # paramSearch["RFC"] = {"model": RandomForestClassifier(),
    #                       "param": rfC_param_search}
    # paramSearch['KNC'] = {"model": KNeighborsClassifier(),
    #                       "param": knC_param_search}
    # paramSearch['SVC'] = {"model": SVC(),
    #                       "param": svC_param_search}
    # paramSearch['MLPC'] = {"model": MLPClassifier(),
    #                        "param": mlpC_param_search}

    # for each model evaluate best estimator, best params, evaluated y_pred and R2 score via prediction,
    # write results to dictionary
    model_param = {}
    for key, value in paramSearch.items():
        bestOptimizedModel, bestParams = evaluateBestParams(value["model"], value["param"], x_train, y_train)
        y_pred, accuracy = fit_predict(bestOptimizedModel, x_train, y_train, x_test, y_test)
        model_param[key] = {"model": bestOptimizedModel,
                            "bestParams": bestParams,
                            "y_pred": y_pred,
                            "accuracy": accuracy}

    # compare the different models visually with regard to accuracy score
    compareDifferentModels(model_param, y_test)


if __name__ == "__main__":
    main()
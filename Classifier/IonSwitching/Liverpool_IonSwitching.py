#!/usr/local/bin/python

__author__ = "WhyKiki"
__version__ = "1.0.2"


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import sfg
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


def readData(filename):
    """
    Read data into dataframe, drop unimportant columns and one-hot-encode feature "batch"
    :param filename: String, name of the file to be read in
    :return: dataframe, dataframe to be further processed and analyzed
    """
    ## read data and drop unnecessary features (to avoid high levels of multicollinearity)
    df_preprocessed = pd.read_csv(filename, index_col=None)
    df_preprocessed.drop(["time", "signal_width", "channel_pattern", "signal_timeDelta"],
                         axis=1, inplace=True)

    ## one-hot-encode feature "batch" and drop one of the new created features to avoid perfect multicollinearity
    df_ohe = pd.get_dummies(df_preprocessed["batch"])
    df_ohe.rename(columns={1: 'batch1', 2: 'batch2', 3: 'batch3', 4: 'batch4', 5: 'batch5',
                           7: 'batch7', 8: 'batch8', 9: 'batch9', 10: 'batch10'}, inplace=True)
    df = pd.concat([df_preprocessed, df_ohe], axis=1)
    df.drop([6, "batch"], inplace=True, axis=1)

    ## create subset to reduce number of observations (to exemplarily show that the code runs)
    df_subset = df.sample(10000, random_state=123)

    return df_subset


def assignData(df, feature):
    """
    assign train and test data
    :param df: dataframe, containing the data to be analyzed
    :param feature: feature, data to be analyzed (dependent variable, y)
    :return: train and test data (both dependent (series) and independent (dataframes) variables)
    """
    ## assign X and y
    X = df.drop(feature, axis=1)
    y = df[feature]
    ## scale X data
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
    return X_train, y_train, X_test, y_test


def calculateScores(y_test, y_pred):
    """
    calculate recall, precision, f1_score as quality metrics for grid search on time series analysis
    :param x: dataframe, independent variable
    :param y: series, dependent variable
    :return: float, score
    """
    ## convert to series / extract column of interest
    y_test = y_test.reset_index()["open_channels"]
    y_pred = pd.Series(y_pred.reshape(-1))

    ## concat both series to dataframe
    df_y = pd.concat([y_test, y_pred], axis=1)
    df_y.rename(columns={"open_channels": "y_test", 0: "y_pred"}, inplace=True)

    ## initialize score metric lists
    recall_list = []
    precision_list = []
    f1_score_list = []

    ## calculate scores for the actual and the next line
    for i in df_y.index:
        if i < df_y.index.max():
            row1 = [df_y.loc[i, "y_test"], df_y.loc[i + 1, "y_test"]]
            row2 = [df_y.loc[i, "y_pred"], df_y.loc[i + 1, "y_pred"]]
            recall_list.append(recall_score(row1, row2, average="macro"))
            precision_list.append(precision_score(row1, row2, average="macro"))
            f1_score_list.append(f1_score(row1, row2, average="macro"))

    ## calculate overall scores
    ts_recall = sum(recall_list) / len(recall_list)
    ts_precision = sum(precision_list) / len(precision_list)
    ts_f1_score = sum(f1_score_list) / len(f1_score_list)

    print("recall: ", round(ts_recall, 4))
    print("precision: ", round(ts_precision, 4))
    print("f1_score: ", round(ts_f1_score, 4), "\n")

    return ts_recall, ts_precision, ts_f1_score


def evaluateBestParams(model, model_params, x_train, y_train):
    """
    evaluate best hyperparameters for a given model
    :param model: selected model
    :param model_params: model's hyperparameters to be screened for the best suitable ones
    :param x_train: dataframe, train data (independent variable)
    :param y_train: series, train data (dependent variable)
    :return: evaluated best estimator (model) and best params (dictionary)
    """
    cv = StratifiedKFold(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=cv, param_grid=model_params, scoring="accuracy")
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
    ts_recall, ts_precision, ts_f1_score = calculateScores(y_test, y_pred)

    print("usual recall score: ", recall_score(y_test, y_pred, average="macro"))
    print("usual precision score: ", precision_score(y_test, y_pred, average="macro"))
    print("usual f1 score: ", f1_score(y_test, y_pred, average="macro"), "\n")

    return y_pred, ts_recall, ts_precision, ts_f1_score


def compareDifferentModels(model_dict, y):
    """
    compare different algorithms visually
    :param model_dict: dictionary, info about model (key) and
           best model, best parameters, y_pred and MSE score (values as single dictionaries)
    :param y: series, test data (dependent variable), also referred to as y_test
    """
    ## initialize new dataFrame with y_test data (as control)
    df_new = pd.DataFrame(y)

    ## convert y_pred and y_test to usable format for further analysis
    for key, value in model_dict.items():
        # define y_pred (per model) and y_test
        y_pred = np.array(value["y_pred"]).flatten()
        y_test = np.array(y).flatten()
        # create new features in dataframe for y_pred (per model)
        df_new[key] = y_pred

    ## boxplots for general comparison of algorithms
    plt.boxplot(df_new, labels=df_new.columns)
    plt.title("Algorithm Comparison")
    plt.show()

    ## cat plot for score comparison of algorithms:
    ## first, create new dataframe with model names and corresponding scores
    fig = plt.figure(figsize=(12, 4), tight_layout=True)
    gs = gridspec.GridSpec(3, 1)
    i = 0

    for elem in ["recall", "precision", "f1_score"]:
        score_list = []
        model_list = []
        for key, value in model_dict.items():
            score = value[elem]
            model_list.append(key)
            score_list.append(score)
        score_dict = {"Regression Model": model_list, elem: score_list}
        df_scores = pd.DataFrame(score_dict)

        ## plot classification metrics for model comparison
        i += 1
        sfg.SeabornFig2Grid(sns.catplot(data=df_scores, y="Regression Model", x=elem,
                                        kind="bar", height=5, aspect=3
                                        ), fig, gs[i - 1])
        plt.xlim(0.4, 1.0)
    plt.show()


def assignModel():
    """
    Assign model and respective hyperparameters to be screened for evaluating the best ones
    :return: dictionary, containing model and respective hyperparameters to be screened for
    """
    ## assign model's hyperparameters to be screened for
    logR_param_search = {"n_jobs": [-1],
                         "random_state": [12345],
                         "solver": ["sag"]}
    rfC_param_search = {"n_estimators": [20, 50, 100],
                        "max_features": ['auto', 'sqrt', 'log2'],
                        "max_depth": [i for i in range(5, 15)],
                        "criterion": ["gini", "entropy"],
                        "n_jobs": [-1],
                        "random_state": [12345]}
    knC_param_search = {"n_neighbors": range(3, 7),
                        "weights": ["uniform", "distance"],
                        "n_jobs": [-1]}
    svC_param_search = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                        "degree": range(1, 4),
                        "gamma": ["scale", "auto"],
                        "random_state": [12345]}
    mlpC_param_search = {"solver": ["lbfgs", "adam"],
                         "activation": ["relu", "logistic", "tanh"],
                         "hidden_layer_sizes": [(50,), (150,), (250,), (350,), (450,)] ,
                         "random_state": [12345]}

    ## create dictionary with model names and corresponding screening hyperparameters
    paramSearch = {}
    paramSearch["LogR"] = {"model": LogisticRegression(),
                           "param": logR_param_search}
    paramSearch["RFC"] = {"model": RandomForestClassifier(),
                          "param": rfC_param_search}
    paramSearch["KNC"] = {"model": KNeighborsClassifier(),
                          "param": knC_param_search}
    paramSearch["SVC"] = {"model": SVC(),
                          "param": svC_param_search}
    paramSearch["MLPC"] = {"model": MLPClassifier(),
                           "param": mlpC_param_search}

    return paramSearch



def getCodeDocumentation(functionName):
    """
    Access code documentation
    :param functionName: class function, function of which the documentation shall be derived from
    """
    funcName = str(functionName).split(" ")[1]
    print(funcName, functionName.__doc__)



def main():
    """
    Run supervised machine learning and print code documentation
    """
    ## read data
    df = readData("../Data/df_preprocessed.csv")

    ## assign data
    x_train, y_train, x_test, y_test = assignData(df, "open_channels")

    ## get the model parameters
    paramSearch = assignModel()

    ## get current time
    start_time = datetime.now()

    ## for each model evaluate best estimator, best params, evaluated y_pred and R2 score via prediction,
    ## write results to dictionary
    model_param = {}
    for key, value in paramSearch.items():
        bestOptimizedModel, bestParams = evaluateBestParams(value["model"], value["param"], x_train, y_train)
        y_pred, ts_recall, ts_precision, ts_f1_score = fit_predict(bestOptimizedModel, x_train, y_train, x_test, y_test)
        model_param[key] = {"model": bestOptimizedModel,
                            "bestParams": bestParams,
                            "y_pred": y_pred,
                            "recall": ts_recall,
                            "precision": ts_precision,
                            "f1_score": ts_f1_score}

    ## compare the different models visually with regard to their scores
    compareDifferentModels(model_param, y_test)

    ## get current time and show duration of calculation
    end_time = datetime.now()
    diff_time = relativedelta(end_time, start_time)
    print("Calculation duration: %d hours, %d minutes, %d seconds\n\n" % (diff_time.hours,
                                                                            diff_time.minutes, diff_time.seconds))

    ## access code documentation
    print("-----------------------------------------------------------")
    print("-------------------- Code Documentation -------------------\n")
    getCodeDocumentation(readData)
    getCodeDocumentation(assignData)
    getCodeDocumentation(calculateScores)
    # getCodeDocumentation(crossValModel)
    getCodeDocumentation(evaluateBestParams)
    getCodeDocumentation(compareDifferentModels)
    getCodeDocumentation(assignModel)
    getCodeDocumentation(getCodeDocumentation)
    getCodeDocumentation(main)


if __name__ == "__main__":
    main()
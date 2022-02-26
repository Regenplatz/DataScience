#!/usr/bin/env python3

__author__ = "WhyKiki"
__version__ = "1.0.1"


import pandas as pd
from sklearn.utils import shuffle
from numpy import mean, std
from datetime import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from skopt.searchcv import BayesSearchCV
import warnings
warnings.filterwarnings("ignore")


def dataPreprocessing(df):
    """
    Create new features, drop features of no interest.
    :param df: dataframe, dataframe to be processed
    :return: dataframe, processed dataframe
    """
    ## get new features out of timestamp (listened)
    df["listen_dateTime"] = [dt.fromtimestamp(x) for x in df["ts_listen"]]
    df["listen_month"] = df["listen_dateTime"].dt.month
    df["listen_week"] = df["listen_dateTime"].dt.isocalendar().week.astype("int64")
    df["listen_weekday"] = df["listen_dateTime"].dt.weekday
    df["listen_hour"] = df["listen_dateTime"].dt.hour

    ## drop columns of no interest (high VIF for "album_id")
    df = df.drop(["album_id", "listen_dateTime"], axis=1)

    return df


def scaleDataColumns(df):
    """
    Scale columnwise, only certain columns
    :param df: dataframe, no scaled data
    :return: dataframe, with scaled data
    """
    enc = ColumnTransformer(
        remainder="passthrough",
        transformers=[("std", StandardScaler(), ["ts_listen", "media_id", "release_date",
                                                 "media_duration", "user_id", "artist_id"])]
    )
    df_scaled = enc.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=enc.get_feature_names_out())
    return df


def prepareData(test_csv=False, subsample=True):
    """
    Read data, scale and assign to test and train data
    :param test_csv: boolean, if test.csv should be read in or not
    :param subsample: boolean, if the prediction should run on a subsample or not
    :return: dataframes: X_train, X_test, Series: y_train, y_test
    """
    ## read data from train.csv and preprocess
    df_train = pd.read_csv("../Data/train.csv")
    df_train = dataPreprocessing(df_train)
    if test_csv:
        ## read data from test.csv and preprocess
        df_test = pd.read_csv("../Data/test.csv")
        df_test = dataPreprocessing(df_test)
    else:
        ## assign feature "sample_id" and reorder columns in dataframe
        df_train["sample_id"] = df_train.index
        if subsample:
            ## create subsample
            df = df_train[df_train["user_id"] < 200]
            df = df_train[df_train["media_id"] < 400000]
        # split train.csv data into train and test data
        df = shuffle(df)
        cutoff = int(0.8 * len(df))
        df_train = df.iloc[:cutoff]
        df_test = df.iloc[cutoff:]

    ## scale data, assign data to test and train data
    # scaler = RobustScaler()
    X_train = df_train.drop(["is_listened"], axis=1)
    X_train = scaleDataColumns(X_train)
    X_test = df_test.drop(["is_listened"], axis=1)
    X_test = scaleDataColumns(X_test)
    y_train = df_train["is_listened"]
    if not test_csv:
        y_test = df_test["is_listened"]
    else:
        y_test = 0
    return X_train, X_test, y_train, y_test, test_csv


def evaluateBestParams(model, X_train, y_train):
    """
    Evaluate the best hyperparameters for the model to predict. Use Bayesian approach.
    :param params: dictionary, hyperparameters of which the best are being evaluated
    :param X_train: dataframe, train data (independent variable)
    :param y_train: Series, train data (dependent variable)
    :return: model: best estimator, dictionary: best parameters
    """
    ## define search space for Bayes search
    search_space = BayesSearchCV(
        estimator=model["model"],
        search_spaces=model["params"],
        n_jobs=1,
        cv=5,
        n_iter=7,
        scoring="roc_auc",
        verbose=4,
        random_state=42
    )

    ## run Bayes search and evaluate best hyperparameters
    search_space.fit(X_train, y_train)
    print(search_space.best_score_)
    print(search_space.best_params_, "\n")

    return search_space.best_estimator_, search_space.best_params_


def crossValidate(model, X_train, y_train):
    """
    Cross validate the model
    :param model: optimized model
    :param X_train: dataframe, train data (independent variable)
    :param y_train: Series, train data (dependent variable)
    :return:
    """
    ## cross validation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_train, y_train, scoring='roc_auc',
                               cv=cv, n_jobs=-1, error_score='raise')
    print(f"ROC AUC [mean (sd)]: {mean(n_scores).round(4)} ({std(n_scores).round(4)})\n")


def fitPredict(model, X_train, y_train, X_test, y_test, qualMetrics=True):
    """
    Fit the model, make prediction and evaluate quality of prediction
    :param model:
    :param X_train: dataframe, train data (independent variable)
    :param y_train: Series, train data (dependent variable)
    :param X_test: dataframe, test data (independent variable)
    :param y_test: Series, test data (dependent variable)
    :param qualMetrics: boolean, if quality metrics should be evaluated or not
    :return:
    """
    ## fit model
    model.fit(X_train, y_train)

    ## make prediction
    y_pred = model.predict(X_test)

    if qualMetrics:
        ## evaluate quality of prediction
        print("ROC AUC score:", metrics.roc_auc_score(y_test, y_pred))

    return y_pred


def dfPrediction(X_test, y_pred):
    """
    Create dataframe for sample ID and prediction (as required for kaggle submission)
    :param X_test: dataframe, test data (independent variable)
    :param y_pred: Series, predicted data (dependent variable)
    :return: dataframe, results
    """
    df_pred = pd.DataFrame([X_test["remainder__sample_id"], y_pred]).transpose()
    df_pred.rename({"remainder__sample_id": "sample_id", "Unnamed 0": "y_pred"}, axis=1, inplace=True)
    return df_pred


def main():
    """
    Read data, find best hyperparameters, cross validate optimized model, fit and predict
    """
    ## assign train and test data
    X_train, X_test, y_train, y_test, test_csv = prepareData()

    ## assign base model and set of hyperparameters to be screened for the best suitable ones
    model_dict = {"model_name": "RandomForestClassifier",
                  "model": RandomForestClassifier(),
                  "params": {"n_estimators": [100],  # , 200, 300, 400, 500],
                            "max_depth": (1, 2),  # [i for i in range(5, 11)],
                            "criterion": ["entropy"], # , "gini"]
                            "n_jobs": [-1],
                            "random_state": [12345]
                             }
                  }
    # model_dict = {"model_name": "LogisticRegression",
    #               "model": LogisticRegression(),
    #               "params": {"n_jobs": [-1],
    #                          "random_state": [12345],
    #                          "solver": ["saga"],
    #                          "penalty": ["l1", "l2", "elasticnet", "none"]
    #                          }
    #               }
    # model_dict = {"model_name": "SVC",
    #               "model": SVC(),
    #               "params": {"kernel": ["linear", "poly", "rbf", "sigmoid"],
    #                         "degree": range(1, 4),
    #                         "gamma": ["scale", "auto"],
    #                         "random_state": [12345]
    #                          }
    #               }
    # model_dict = {"model_name": "KNeighborsClassifier",
    #               "model": KNeighborsClassifier(),
    #               "params": {"n_neighbors": range(3, 7),
    #                         "weights": ["uniform", "distance"],
    #                         "n_jobs": [-1]
    #                          }
    #               }
    # model_dict = {"model_name": "MLPClassifier",
    #               "model": MLPClassifier(),
    #               "params": {"solver": ["lbfgs", "adam"],
    #                         "activation": ["relu", "logistic", "tanh"],
    #                         "hidden_layer_sizes": [(50,), (150,), (250,), (350,), (450,)] ,
    #                         "random_state": [12345]
    #                          }
    #               }

    ## run GridSearch to evaluate the best hyperparameters for prediction
    optimizedModel, bestParams = evaluateBestParams(model_dict, X_train, y_train)

    ## assign model
    model = optimizedModel

    ## cross validate
    crossValidate(model, X_train, y_train)

    ## fit model and make prediction
    y_pred = fitPredict(optimizedModel, X_train, y_train, X_test, y_test)

    ## create new dataframe with sample_id and prediction only
    df_pred = dfPrediction(X_test, y_pred)
    # print("shape", df_pred.shape)
    print(df_pred.head())


if __name__ == "__main__":
    main()

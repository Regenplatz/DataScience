#!/usr/bin/env python3

__author__ = "WhyKiki"
__version__ = "1.0.1"

## hyperparameter tuning according to https://www.tensorflow.org/tutorials/keras/keras_tuner and
## https://keras.io/api/keras_tuner/hyperparameters/


import keras.callbacks
import pandas as pd
import numpy as np
from keras_tuner import Objective
from numpy import mean, std
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, RandomizedSearchCV
from skopt.searchcv import BayesSearchCV
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.optimizers import SGD
from keras.metrics import AUC, Precision, Recall, BinaryAccuracy
from keras.losses import BinaryCrossentropy
from tensorflow.python.keras import Sequential
import keras_tuner as kt
# from tensorboard.plugins.hparams import api as hp
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from deezerData import readData
import warnings
warnings.filterwarnings("ignore")


## define ID of user to whom you want to recommend songs
userID_to_rec = 30

## initialize variables for matrix dimensionality
N = 100  ## number of users, gets overwritten later
M = 100  ## number of users, gets overwritten later
K = 50   ## latent dimensionality

## define file location specific attributes for the neural net
## (necessary for the kerasTuner, saves computed data to it)
directory = "KerasTuner"
folder_name = "kerasTuner"   ## folder needs to be set up in advance (within this stated directory)


def modelBuilder(hp):
    """
    Assign input and output tensors, build neural net and compile model
    :param hp: hyperparameters, argument needed to call this function from evaluateBestParams function
               (see also https://keras.io/api/keras_tuner/hyperparameters/)
    :return: model, compiled neural net model
    """
    ## keras model
    u = Input(shape=(1,))
    s = Input(shape=(1,))
    u_embedding = Embedding(N, K)(u)   ## (N, 1, K)
    s_embedding = Embedding(M, K)(s)   ## (N, 1, K)
    u_embedding = Flatten()(u_embedding)  ## (N, K)
    s_embedding = Flatten()(s_embedding)  ## (N, K)
    x = Concatenate()([u_embedding, s_embedding])  ## (N, 2K)

    ## Tune the number of units in the first Dense layer
    ## Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    x = Dense(units=hp_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1, activation="sigmoid")(x)

    ## define model and compile. Use BinaryCrossEntropy for binary classification approach
    model = Model(inputs=[u, s], outputs=x)
    model.compile(
      loss=BinaryCrossentropy(from_logits=True),
      optimizer=SGD(lr=0.08, momentum=0.9),
      metrics=[AUC(thresholds=[0.0, 0.5, 1.0]),
               BinaryAccuracy(threshold=0.5),
               Precision(),
               Recall()],
    )

    ## print model summary
    # model.summary()

    return model


def evaluateBestParams(X_train, y_train):
    """
    Evaluate the best hyperparameters found for the training dataset
    :param X_train: numpy array, train data (independent variable)
    :param y_train: numpy array, train data (dependent variable)
    :return: optimized model, best found hyperparameters for the optimized model, tuned model
    """
    tuner = kt.BayesianOptimization(
        hypermodel=modelBuilder,
        objective="val_loss",
        max_trials=5,
        num_initial_points=4,
        alpha=0.0001,
        beta=2.6,
        seed=568,
        # hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        directory=directory,
        project_name=folder_name
    )

    ## stop training early after reaching a certain value for validation
    stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    ## Run hyperparameter search
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
    print("get_best_params:", tuner.get_best_hyperparameters(num_trials=1)[0])
    print("get_best_model:", tuner.get_best_models())

    return tuner.get_best_models(), tuner.get_best_hyperparameters(num_trials=1)[0], tuner


def fitPredict(X_train, y_train, X_test, y_test, tuner, bestParams):
    """
    Fit data with the compiled model, predict dependent variable
    :param X_train: numpy array, train data (independent variable)
    :param y_train: numpy array, train data (dependent variable)
    :param X_test: numpy array, test data (independent variable)
    :param y_test: numpy array, test data (dependent variable)
    :param tuner: tuned model, optimized model
    :param bestParams: hyperparameters (hp), best found hp for the optimized model
    :return: dataframe to compare y_test and y_pred, fitted model history, fitted tuner
    """
    ## build model with optimal hyperparameters and train it on data for 50 epochs
    model = tuner.hypermodel.build(bestParams)
    modelHist = model.fit(X_train, y_train, epochs=50, validation_split=0.2)
    val_acc_per_epoch = modelHist.history["val_auc_1"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch))
    print(f"Best epoch: {best_epoch}")

    ## Re-instantiate hypermodel and train it with optimal number of epochs from above
    hypermodel = tuner.hypermodel.build(bestParams)
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)

    ## evaluate result
    eval_result = hypermodel.evaluate(X_test, y_test)
    print("[test loss, test accuracy]:", eval_result)

    ## predict, create dataframe to compare y_pred and y_test
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred < 0.5, 0, 1)
    print(type(y_pred))
    df_compare = pd.DataFrame([y_test, y_pred.flatten()]).transpose()
    df_compare.rename(columns={0: "y_test", 1: "y_pred"}, inplace=True)

    ## create dataframe for recommendations
    df_rec = pd.DataFrame([X_test[0], X_test[1], y_pred.flatten()]).transpose()
    df_rec.rename(columns={0: "user_id", 1: "media_id", 2: "y_pred"}, inplace=True)
    df_rec = df_rec.loc[df_rec["y_pred"] == 1]
    df_rec = df_rec.groupby(["media_id"], as_index=False)["user_id"].count()
    df_rec.rename(columns={"user_id": "user_cnt"}, inplace=True)
    df_rec.sort_values(by=["user_cnt"], ascending=False, inplace=True)
    df_rec.set_index(["media_id"], inplace=True)

    return df_compare, df_rec, modelHist, best_epoch


def printPlotMetrics(modHist, epochs):
    """
    Plot various quality metrics related to the binary classification approach
    :param modHist: History object, model history
    :param epochs: Integer, number of epoch for the best evaluated quality metric
    :return: NA
    """
    print(modHist.history.keys())

    metric_dict = {'loss': 'val_loss',
                   'auc_1': 'val_auc_1',
                   'binary_accuracy': 'val_binary_accuracy',
                   'precision_1': 'val_precision_1',
                   'recall_1': 'val_recall_1'}

    ## plot metrics (separate plots)
    plt.figure()
    for key, value in metric_dict.items():
        print(f"epoch {epochs}:: {key}:", modHist.history[key][-1], f", {value}:", modHist.history[value][-1])
        plt.plot(modHist.history[key], label=f"train {key}")
        plt.plot(modHist.history[value], label=f"test {key}")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.title(f"model {key}")
        plt.show()

    ## one plot including all metrics
    pd.DataFrame(modHist.history).plot(figsize=(8, 5))
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title("model metric")
    plt.show()


def main():
    """
    Run song recommendation
    :return: NA
    """
    ## read data and convert train and test data into numpy arrays
    df, X, y, df_X_train, df_X_test, df_y_train, df_y_test = readData(groupBy=True)
    X_train = [df_X_train["user_id"].values, df_X_train["media_id"].values]
    y_train = df_y_train.values
    X_test = [df_X_test["user_id"].values, df_X_test["media_id"].values]
    y_test = df_y_test.values

    ## assign number of users and songs
    global N
    N = df["user_id"].nunique()   ## number of users
    global M
    M = df["media_id"].nunique()  ## number of songs

    ## assign assign base model and set of hyperparameters to be screened for the best suitable ones
    optimizedModel, bestParams, tuner = evaluateBestParams(X_train, y_train)

    # model = tuner.hypermodel.build(bestParams)
    df_compare, df_rec, fittedModHist, best_epoch = fitPredict(X_train, y_train, X_test, y_test, tuner, bestParams)
    print("DONE")

    ## fit and predict
    # df_compare, df_rec, fittedModelHist = fitPredict(model, X_train, y_train, X_test, y_test, epochs)
    print("\n------------------ y_test vs y_pred ------------------\n", df_compare.head(10))

    # recommend songs to a user (10 of the highest frequency (number of users who listened to the song)
    already_listened = list((df_X_train.loc[df_X_train["user_id"] == userID_to_rec]["media_id"]).unique())
    print("\n------------------ RECOMMENDATION ------------------\n")
    print(f"User {userID_to_rec} already listened to songs with media_id being: {sorted(already_listened)}")
    print(f"\nRECOMMENDATIONS for user {userID_to_rec}:\n", df_rec.head(10))

    ## plot results for supervised approach
    printPlotMetrics(fittedModHist, best_epoch)


if __name__ == "__main__":
    main()

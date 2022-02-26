#!/usr/bin/env python3

__author__ = "WhyKiki"
__version__ = "1.0.1"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.optimizers import SGD
from keras.metrics import AUC, Precision, Recall, BinaryAccuracy
from keras.losses import BinaryCrossentropy
from deezerData import readData
import warnings
warnings.filterwarnings("ignore")


## define ID of user to whom you want to recommend songs
userID_to_rec = 30


def assignModel(N, M, K):
    """
    Assign input and output tensors, build neural net and compile model
    :param N: integer, number of users
    :param M: integer, number of songs
    :param K: integer, latent dimensionality
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

    ## the neural network (use sigmoid activation function in output layer for binary classification)
    x = Dense(400)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100)(x)
    x = BatchNormalization()(x)
    # x = Activation('sigmoid')(x)
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

    return model


def fitPredict(model, X_train, y_train, X_test, y_test, epochs):
    """
    Fit data with the compiled model, predict dependent variable
    :param model: model, compiled neural net model
    :param X_train: numpy array, train data (independent variable)
    :param y_train: numpy array, train data (dependent variable)
    :param X_test: numpy array, test data (independent variable)
    :param y_test: numpy array, test data (dependent variable)
    :param epochs: integer, number of epochs
    :return: dataframe to compare y_test and y_pred, fitted model history
    """
    ## fit model
    fittedModHist = model.fit(
      x=X_train,
      y=y_train,
      epochs=epochs,
      batch_size=128,
      validation_data=(X_test, y_test)
    )

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

    return df_compare, df_rec, fittedModHist


def printPlotMetrics(modHist, epochs):
    """
    Plot various quality metrics related to the binary classification approach
    :param modHist: History object, model history
    :return: NA
    """
    # print(modHist.history.keys())

    metric_dict = {'loss': 'val_loss',
                   'auc': 'val_auc',
                   'binary_accuracy': 'val_binary_accuracy',
                   'precision': 'val_precision',
                   'recall': 'val_recall'}

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
    N = df["user_id"].nunique()   ## number of users
    M = df["media_id"].nunique()  ## number of songs

    ## initialize hyperparameters:
    ## K=latent dimensionality, epochs=number of epochs, reg=regularization penalty
    K = 10
    epochs = 50
    # reg = 0.0001

    ## assign model
    model = assignModel(N, M, K)

    ## print model summary
    model.summary()

    ## fit and predict
    df_compare, df_rec, fittedModelHist = fitPredict(model, X_train, y_train, X_test, y_test, epochs)
    print("\n------------------ y_test vs y_pred ------------------\n", df_compare.head(10))

    ## recommend songs to a user (10 of the highest frequency (number of users who listened to the song)
    already_listened = list((df_X_train.loc[df_X_train["user_id"] == userID_to_rec]["media_id"]).unique())
    print("\n------------------ RECOMMENDATION ------------------\n")
    print(f"User {userID_to_rec} already listened to songs with media_id being: {sorted(already_listened)}")
    print(f"\nRECOMMENDATIONS for user {userID_to_rec}:\n", df_rec.head(10))

    ## plot results for supervised approach
    printPlotMetrics(fittedModelHist, epochs)


if __name__ == "__main__":
    main()

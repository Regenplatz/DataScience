#!/usr/bin/env python3

## according to https://beckernick.github.io/matrix-factorization-recommender/

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


##### define global variables
###########################################################################################

## user ID to whom you want to recommend songs
USER_ID = 79

## number of songs to be recommended
NO_RECOMMENDATIONS = 10

## read data
df = pd.read_csv("../Data/train.csv")
df["sample_id"] = df.index

## create subsample
df = df[df["user_id"] < 200]
df = df[df["media_id"] < 400000]

## create feature sample_id
df["sample_id"] = df.index
# df = shuffle(df)
# print(df["user_id"].values)

## dataframe containing dependent variable ("is_listened") and join parameter ("media_id")
df_isListened = df[["user_id", "media_id", "is_listened"]]
df_isListened = df_isListened.groupby(["user_id", "media_id"], as_index=False)["is_listened"].sum()
df_isListened = df_isListened.sort_values(by = ["user_id", "media_id"], axis=0)

## dataframe containing independent variables and join parameter ("media_id")
## if only groupby on media_id
# df_songs = df.groupby(["media_id"], as_index=False)["is_listened"].count()
## if groupby on media_id and genre_id
df["genre_id_txt"] = df["genre_id"].astype(str)
df_songs = df.groupby(["media_id", "genre_id_txt"], as_index=False)["is_listened"].count()
df_songs = df_songs.groupby(["media_id"], as_index=False).agg({"genre_id_txt": " | ".join, "is_listened": "sum"})



def preprocessData(df_isListened):
    """
    Pivot data of dependent variable, pivot and convert to matrix, demean matrix
    :param df_isListened: dataframe, containing the dependent variable ("is_listened")
    :return:
    """
    ## pivot and convert to matrix
    df_isListened_pivot = df_isListened.pivot(index="user_id", columns="media_id", values="is_listened").fillna(0)
    m_isListened = df_isListened_pivot.values

    ## demean data
    isListened_mean = np.mean(m_isListened, axis=1)
    m_isListened_demeaned = m_isListened - isListened_mean.reshape(-1, 1)

    return df_isListened_pivot, m_isListened, isListened_mean, m_isListened_demeaned


def svd_calculation(m_isListened_demeaned):
    """
    Singular value decomposition: (M x N) --> (M X K) (K x K) (K X N)
    :param m_isListened_demeaned: matrix to be decomposed into singular components
    :return:
    """
    U, sigma, Vt = svds(m_isListened_demeaned, k=50)
    sigma = np.diag(sigma)

    return U, sigma, Vt


def makePrediction(U, sigma, Vt, isListened_mean, df_isListened_pivot):
    """
    Predict songs to user
    :params U, sigma, Vt: matrices, derived from singular value decomposition
    :param isListened_mean: double, mean of dependent variable
    :param df_isListened_pivot: dataframe, pivot of dependent variable's dataframe
    :return: dataframe, prediction
    """
    ## making predictions from decomposed matrices
    all_user_pred_isListened = np.dot(np.dot(U, sigma), Vt) + isListened_mean.reshape(-1, 1)
    df_pred = pd.DataFrame(all_user_pred_isListened, columns=df_isListened_pivot.columns)

    return df_pred


def recommend_songs(df_pred, userID, df_songs, df_isListened, no_recommendations=5):
    """
    Recommend songs
    :param df_pred: dataframe, with predictions for songs
    :param userID: integer, ID from a user
    :param df_songs: dataframe, containing info on songs
    :param df_isListened: dataframe, containing info if song was listened to or not
    :param no_recommendations: integer, number of recommendations to be made
    :return:
    """
    ## get and sort user's predictions (contains also songs that user already listened to)
    user_pred_sorted = df_pred.iloc[userID].sort_values(ascending=False)

    ## merge data of all songs the user listened to with further data of the songs matrix (e.g. containing info on genre)
    user_allSongs = df_isListened[df_isListened.user_id == (userID)]
    user_allSongs_enriched = (user_allSongs.merge(df_songs
                     , how="left"
                     , left_on="media_id"
                     , right_on="media_id"
                    ).sort_values(["is_listened_x"], ascending=False))

    ## (1) get songs which user did not yet listen to, (2) merge with those predicted for user (inner join drops
    ## out predictions for the user for songs user already listened to), (3) sort values according to predictions
    df_songsNotListenedByUser = df_songs[~df_songs["media_id"].isin(user_allSongs_enriched["media_id"])]
    df_recommendations = df_songsNotListenedByUser.merge(pd.DataFrame(user_pred_sorted).reset_index()
                    , how="inner"
                    , left_on="media_id"
                    , right_on="media_id"
                    ).rename(columns={userID: "Predictions"})
    df_recommendations = df_recommendations.sort_values("Predictions", ascending=False).iloc[:no_recommendations, :-1]

    ## reduce dataframe of songs already having been listened to to features of interest
    user_allSongs_enriched.drop(["is_listened_x", "is_listened_y"], axis=1, inplace=True)

    return userID, user_allSongs_enriched, df_recommendations, no_recommendations


def main():
    """
    Perform matrix facorization for song recommendations
    :return: NA
    """
    ## Restructure Array: Pivot data of dependent variable, pivot and convert to matrix, demean matrix
    df_isListened_pivot, m_isListened, isListened_mean, m_isListened_demeaned = preprocessData(df_isListened)

    ## Perform "singular value decomposition"
    U, sigma, Vt = svd_calculation(m_isListened_demeaned)

    ## predict songs to a user
    df_pred = makePrediction(U, sigma, Vt, isListened_mean, df_isListened_pivot)

    ## make recommendations to user
    userID, user_allSongs_enriched, df_recommendations, no_recommendations = recommend_songs(df_pred, USER_ID, df_songs,
                                                                                             df_isListened,
                                                                                             NO_RECOMMENDATIONS)

    ## console output of songs already listened to plus further song recommendations
    print(f"\nUser {userID} has already listened to the following {user_allSongs_enriched.shape[0]} songs:")
    print(user_allSongs_enriched, "\n")
    print(f"Recommending the highest rated {no_recommendations} songs not yet having been listened to: \n",
          df_recommendations.head(no_recommendations))


if __name__ == "__main__":
    main()

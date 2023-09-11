#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import pandas as pd
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Tuple
import defineQuery as dq
import pythonConnectSnowflake as con


def loadData(path_toEnvFile: str, query: str) -> pd.DataFrame:
    """Get password for database connection. Query data, load to dataframe and close connection"""
    pw = con.lookUpPW(path_toEnvFile)
    return con.queryData(query, pw)


def preprocessData(df: pd.DataFrame, colName_X: str, colName_idx: str,
                   languageEncoding: str, colName_y: str = None) -> Tuple[pd.DataFrame, dict]:
    """Remove special characters from text, lower text, tokenize and lemmatize words"""

    ## drop duplicates and rows where column X is empty
    if colName_y is None:
        df = df[[colName_idx, colName_X]].copy().drop_duplicates()
    else:
        df = df[[colName_idx, colName_X, colName_y]].copy().drop_duplicates()
    df.dropna(inplace=True)

    ## check for class imbalance
    # tbd

    ## clean text: remove special characters and white spaces
    df[f"{colName_X}_cleaned"] = df[colName_X].replace(r"[^A-Za-zäÄöÖüÜ0-9]", " ", regex=True)\
        .replace(r"\s+[a-zA-Z]\s+", " ", regex=True)

    ## remove stopwords and lemmatize
    languageModel = spacy.load(languageEncoding)
    docs_lemmatized = []
    for idx, row in df.iterrows():
        str_lemmatized = ""
        doc = languageModel(row[f"{colName_X}_cleaned"])
        for token in doc:
            if not token.is_stop:
                str_lemmatized += token.lemma_ + " "
            else:
                str_lemmatized += ""
        docs_lemmatized.append(str_lemmatized)
    df[f"{colName_X}_lemmatized"] = docs_lemmatized
    df[f"{colName_X}_lemmatized"] = df[f"{colName_X}_lemmatized"].str.lower()

    ## create dictionary for unconverted, cleaned and lemmatized text
    dict_perID = {}
    if colName_y is not None:
        for idx, row in df.iterrows():
            dict_perID[row[colName_idx]] = {colName_X: row[colName_X],
                                            f"{colName_X}_cleaned": row[f"{colName_X}_cleaned"],
                                            f"{colName_X}_lemmatized": row[f"{colName_X}_lemmatized"],
                                            colName_y: row[colName_y]
                                            }
    else:
        for idx, row in df.iterrows():
            dict_perID[row[colName_idx]] = {colName_X: row[colName_X],
                                            f"{colName_X}_cleaned": row[f"{colName_X}_cleaned"],
                                            f"{colName_X}_lemmatized": row[f"{colName_X}_lemmatized"]
                                            }

    return df, dict_perID


def vectorizeText(vectorizer: object, data: list) -> Tuple[pd.DataFrame, np.array, list, np.array, dict, object]:
    """Fit model and convert text to array"""
    vectorMatrix = vectorizer.fit_transform(data)
    array_X_features = vectorMatrix.toarray()
    tokens = vectorizer.get_feature_names_out()
    ## vocabulary: A mapping of terms to feature indices (this is called "vocabulary" in sklearn vectorizer)
    vocabulary = vectorizer.vocabulary_
    doc_names = [f"doc_{i+1}" for i, _ in enumerate(array_X_features)]
    df = pd.DataFrame(data=array_X_features, index=doc_names, columns=tokens)
    return df, array_X_features, tokens, vectorMatrix, vocabulary, vectorizer


def findMostCorrelatedTerms(dict_perID: dict, colName_y: str, features: np.array,
                            ngram_range: tuple, fitted_vectorizer: object, n: int) -> dict:
    """Find the most correlated terms in text"""
    dict_grams = {(1, 1): 1,
                  (2, 2): 2}
    dict_gramsPerID = {}
    for id, val in dict_perID.items():
        features_chi2 = chi2(features, labels=val[colName_y])
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(fitted_vectorizer.get_feature_names_out())[indices]
        grams = [val for val in feature_names if len(val.split(" ")) == dict_grams[ngram_range]]
        topN_grams = grams[-n:]
        dict_gramsPerID[id] = {"category": val[colName_y],
                               "grams": grams,
                               f"top{n}_grams": topN_grams
                               }
    return dict_gramsPerID


def returnVectorizedArrays(vectorizer: object, X_train: pd.Series, X_test: pd.Series) -> Tuple[np.array, np.array]:
    """Vectorize text and return as array"""
    vectorizer.fit(X_train)
    array_X_train = vectorizer.transform(X_train).toarray()
    array_X_test = vectorizer.transform(X_test).toarray()
    return array_X_train, array_X_test


def classifyText(X_train: pd.Series, X_test: pd.Series, y_train: pd.Series, y_test: pd.Series,
                 vectorizer: object, clf: object, dict_res: dict, idx_testData: list) \
        -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Vectorize text, fit classifier, predict and save prediction as well as model performance to dictionary"""

    ## vectorize text
    array_X_train, array_X_test = returnVectorizedArrays(vectorizer=vectorizer,
                                                         X_train=X_train,
                                                         X_test=X_test)

    ## fit classifier
    clf.fit(array_X_train, np.ravel(y_train))
    training_score = clf.score(array_X_train, np.ravel(y_train))

    ## predict probabilities
    pred_proba = clf.predict_proba(array_X_test).round(5)
    df_predProba = pd.DataFrame(columns=clf.classes_, data=pred_proba, index=idx_testData)

    ## predict category (y)
    y_pred = clf.predict(array_X_test)
    df_yPred = pd.DataFrame(data=y_pred, index=idx_testData)

    ## load quality metrics / model performances to dictionary
    dict_res[clf] = {"score_training": training_score,
                     "accuracy": accuracy_score(y_test, y_pred)
                     }

    return dict_res, df_predProba, df_yPred


def dimReductionPCA(X_train: pd.Series, X_test: pd.Series, idx_testData: list,
                    vectorizer: object, n_components: int) -> np.array:
    """Dimensionality Reduction via Principle Component Analysis (PCA)"""
    ## vectorize text
    vectorizer.fit(X_train)
    array_X_train = vectorizer.transform(X_train).toarray()
    array_X_test = vectorizer.transform(X_test).toarray()

    ## create dataframe to also include infos on IDs and tokens
    df_testData = createDF_testData(array_X_test=array_X_test,
                                    idx_testData=idx_testData,
                                    fitted_vectorizer=vectorizer
                                    )

    pca = PCA(n_components=n_components)
    pca.fit(df_testData.iloc[:, :-1])

    return pca.explained_variance_ratio_


def plotPCAExplainableVariance(figsize: tuple, explained_variance_ratio: np.array) -> None:
    """Create plot to visualize Principle Components in relation to Explainable Variance"""
    plt.figure(figsize=figsize)
    plt.plot(range(1, 10), explained_variance_ratio, c="red", label="Per Component Explained Variance")
    plt.bar(range(1, 10), height=np.cumsum(explained_variance_ratio), label="Cumulative Explained Variance")
    plt.axhline(y=0.9, c="g", label="Cut Off")
    plt.title("Explained Variance in PCA")
    plt.xticks(range(1, 10))
    plt.legend(loc=0)
    plt.show()


def clusterText(X_train: pd.Series, X_test: pd.Series, vectorizer: object,
                clustAlg: object) -> Tuple[list, np.array, np.array]:
    """Vectorize text, fit classifier, predict and save prediction as well as model performance to dictionary"""
    ## vectorize text
    array_X_train, array_X_test = returnVectorizedArrays(vectorizer=vectorizer,
                                                         X_train=X_train,
                                                         X_test=X_test)

    ## cluster text
    clustAlg.fit(array_X_train)
    clustAlg.predict(array_X_test)
    y_pred = clustAlg.labels_

    return y_pred, array_X_train, array_X_test


def createDF_testData(array_X_test: np.array, idx_testData: list, fitted_vectorizer: object) -> pd.DataFrame:
    """Create dataframe with IDs as rows and tokens as columns"""
    return pd.DataFrame(data=array_X_test,
                        index=idx_testData,
                        columns=fitted_vectorizer.get_feature_names_out()
                        )


def plotElbowPlot(nClusters: list, hyperparams: dict, array_X_train: np.array, cv: int) -> None:
    elbow_results = []
    for i in range(nClusters[0], nClusters[1]):
        kmeans = KMeans(n_clusters=i, **hyperparams)
        results = cross_validate(estimator=kmeans, X=array_X_train, cv=cv)
        elbow_results.append(results["test_score"].mean()*-1)

    plt.plot(elbow_results)
    plt.title("ELBOW PLOT")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within Clusters Sum of Squares")
    plt.show()


def plotClusterMap(array_X_test: np.array, idx_testData: list,
                   fitted_vectorizer: object, metric: str) -> None:
    """Create clustermap with IDs as rows and tokens as columns"""
    df_testData = createDF_testData(array_X_test=array_X_test,
                                    idx_testData=idx_testData,
                                    fitted_vectorizer=fitted_vectorizer
                                    )
    sns.clustermap(data=df_testData,
                   metric=metric)
    plt.show()


def main():

    ##### PREPARATION #########################################################################################
    ###########################################################################################################

    ## load data
    str_path_toEnvFile = ".env"
    str_query = dq.returnQuery()
    df = loadData(str_path_toEnvFile, str_query)

    ## define names of columns of interest and language encoding
    str_colName_X = "<colName_X>"
    str_colName_y = "<colName_y>"
    str_colName_idx = "<colName_idx>"
    languageEncoding = "de_core_news_md"

    ## preprocess data: remove special characters and additional white spaces. Tokenize and lemmatize words.
    df, dict_perID = preprocessData(df=df,
                                    colName_X=str_colName_X,
                                    colName_y=str_colName_y,
                                    colName_idx=str_colName_idx,
                                    languageEncoding=languageEncoding)

    ## define vectorizer
    hyperparams_vectorizer = {"analzyer": "word",
                              "ngram_range": (2, 2)}
    vectorizer = TfidfVectorizer(**hyperparams_vectorizer)

    ## vectorize text
    df_vectorized, array_X, lst_tokens, array_matrix, dict_vocabulary, fitted_vectorizer = vectorizeText(
        vectorizer=vectorizer,
        data=df[f"{str_colName_X}_lemmatized"]
    )


    ##### MOST CORRELATED TERMS IN TEXT #######################################################################
    ###########################################################################################################

    ## find most correlated terms
    n = 10
    ngram_range = (2, 2)
    dict_gramsPerID = findMostCorrelatedTerms(dict_perID=dict_perID,
                                              colName_y=str_colName_y,
                                              features=array_X,
                                              ngram_range=ngram_range,
                                              fitted_vectorizer=fitted_vectorizer,
                                              n=n)
    print(f"TOP {n} GRAMS {ngram_range}:", dict_gramsPerID[df[str_colName_idx][0]][f"top{n}_grams"])


    ##### CLASSIFICATION ######################################################################################
    ###########################################################################################################

    ## define classifier
    hyperparams_classifier = {"n_estimators": 100,
                              "max_depth": 5,
                              "random_state": 1569}
    clf = RandomForestClassifier(**hyperparams_classifier)

    ## assign training and test data and keep track of IDs (as index)
    df.set_index(str_colName_idx, inplace=True)
    X_vals = df[f"{str_colName_X}_lemmatized"]
    y_vals = df[str_colName_y]
    X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.25, random_state=892)
    idx_trainData = X_train.index
    idx_testData = X_test.index

    ## classify
    dict_clfResults = {}
    dict_clfResults, df_predProba, df_yPred = classifyText(X_train=X_train,
                                                           X_test=X_test,
                                                           y_train=y_train,
                                                           y_test=y_test,
                                                           vectorizer=vectorizer,
                                                           clf=clf,
                                                           dict_res=dict_clfResults,
                                                           idx_testData=idx_testData)
    print("ACCURACY:", dict_clfResults[clf]["accuracy"])
    print("PREDICTION:", df_yPred)


    ##### DIMENSIONALITY REDUCTION ############################################################################
    ###########################################################################################################

    ## dimensionality reduction via Principle Component Analysis (PCA)
    explained_variance_ratio = dimReductionPCA(X_train=X_train,
                                               X_test=X_test,
                                               idx_testData=idx_testData,
                                               vectorizer=fitted_vectorizer,
                                               n_components=9)

    ## plot Explainable Variance per Principle Component
    plotPCAExplainableVariance(figsize=(15, 8),
                               explained_variance_ratio=explained_variance_ratio)


    ##### CLUSTERING ##########################################################################################
    ###########################################################################################################

    ## define clustering algorithm
    hyperparams_clustering = {"init": "k-means++",
                              "n_init": 10}
    clustAlg = KMeans(n_clusters=5, **hyperparams_clustering)

    ## cluster text
    y_pred, array_X_train, array_X_test = clusterText(X_train=X_train,
                                                      X_test=X_test,
                                                      vectorizer=vectorizer,
                                                      clustAlg=clustAlg)

    ## create Elbow Plot to visually assess the best suitable number of clusters
    cv = 5
    nClusters = [2, 10]
    plotElbowPlot(nClusters=nClusters,
                  hyperparams=hyperparams_clustering,
                  array_X_train=array_X_train,
                  cv=cv)

    ## plot Cluster Map
    plotClusterMap(array_X_test=array_X_test,
                   idx_testData=idx_testData,
                   fitted_vectorizer=fitted_vectorizer,
                   metric="euclidean")


if __name__ == "__main__":
    main()

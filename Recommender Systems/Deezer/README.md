## Music Recommender

Build a system to recommend songs to users. Songs were not rated explicitely but implicitely by listening to a given song more or less than 30 seconds. Therefore, there is potentially several ratings on a single song by the same user.
Deezer data was taken from [kaggle](https://www.kaggle.com/c/dsg17-online-phase/data). For this project, only data from the *train.csv* was imported and used for recommendations.


##### Explanatory Data Analysis

Initial [exploratory data analysis](EDA.ipynb) showed the presence of several IDs, e.g. for media, album, genre, user. Visualization was therefore limited. This data analysis is examplary and provides information on the size of the dataset, NaNs, data types, descriptive statistics, age of users, check for high multicollinearity.


##### Classification Models

In a next step, [various classifiers](variousClassifiers.py) were tried (*Random Forest*, *Logistic Regression*, *SVC*, *K-Nearest Neighbor*, *Multi Layer Perceptron*) with different hyperparameter settings. The algorithm evaluates the best hyperparameters per model via grid search. From those optimized models the best performing one is then used to predict on test data. However, this common classification approach does not display a usual recommendation system and is therefore provided supplementally.


##### Recommender System: Matrix Factorization and SVD

One possibility to build a recommender system requires [*matrix factorization*](MatrixFactorization.py) and *singular value decomposition* (SVD). This approach also requires pivotting, conversion into matrix and demeaning.


##### Recommender System: Deep Learning Approach

A recommender system was then built using a [keras deep learning](keras_deepLearning.py) approach. A single perceptron as well as sigmoid activation function are used on output layer for binary classification. As loss function *Binary Crossentropy* was set. Several quality metrics such as *AUC*, *BinaryAccuracy*, *Precision* and *Recall* are being evaluated.

The code was expanded to include demonstration of hyperparameter using the [keras tuner](keras_deepLearning_KerasTuner.py).

In both scripts, data was imported via [deezerData.py](deezerData.py). This script allows for importing a subset of the data, as well as grouped format (depending on the settings of the optional parameters).

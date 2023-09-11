#!/usr/bin/env python


__author__ = "WhyKiki"
__version__ = "1.0.0"


from matplotlib import pyplot as plt
import seaborn as sns
import keras
import keras_tuner
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras import regularizers


class ConvNet:

    def __init__(self, X_train, train_filenames, X_test, test_filenames):
        """
        Class constructor. Instantiate variables
        :param X_train:
        :param train_filenames:
        :param X_test:
        :param test_filenames:
        """
        ## training and test data. X (images as array), y (list of filenames (Strings))
        self.X_train = X_train
        self.y_train = train_filenames
        self.X_test = X_test
        self.y_test = test_filenames

        ## neural net specifics ------------------------------------------
        ## number of neurons for input layer
        self.nNeurons = 500
        ## number of epochs
        self.nEpochs = 50
        ## batch size
        self.nBatch = 64
        ## initialize model and history (to be overwritten later)
        self.model = None
        self.modHist = None
        self.history = None
        self.tuner = None

        ## define directory for checkpoints ------------------------------
        self.directory = "."
        self.folder_name = "cnn_kerasTuner"


    def model_builder(self, hp):
        """
        Assign input and output tensors, build neural net and compile model.
        :param hp: hyperparameters, argument needed to call this function from evaluateBestParams function
                   (see also https://keras.io/api/keras_tuner/hyperparameters/)
        :return: model, compiled neural net model
        """
        ##### DEFINE MODEL ###############################################

        self.model = Sequential()

        ## input layer --------------------------------------------
        self.model.add(Conv2D(filters=self.nNeurons,
                              kernel_size=1,
                              strides=1,
                              activation="softmax",
                              ## input shape (height, width, #color channels (3 for RGB))
                              input_shape=(self.X_train.shape[1:]),
                              #input_shape=(self.X_train.shape[1], self.X_train.shape[2], 3),
                              name="block1_conv1"))
        self.model.add(MaxPooling2D(pool_size=2, name="block1_maxPooling1"))
        self.model.add(Flatten(name="block1_flat1"))
        self.model.add(BatchNormalization(momentum=0.9,
                                          epsilon=1e-5,
                                          axis=1))
        self.model.add(Dropout(0.5, name="block1_drop1"))
        self.model.add(Dense(units=hp.Int("units", min_value=32, max_value=512, step=32),
                             activation='softmax',
                             name='block1_dense1',
                             kernel_regularizer=regularizers.L2(0.001),
                             activity_regularizer=regularizers.L2(0.001)))

        ## output layer ----------------------------------------------
        self.model.add(Dense(14, activation='softmax', name='block3_dense1',
                             kernel_regularizer=regularizers.L2(0.001),
                             activity_regularizer=regularizers.L2(1e-5)))

        ## tune learning rate and set up optimization of the weights
        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        adam = keras.optimizers.Adam(learning_rate=lr,
                                     beta_1=0.95,
                                     epsilon=1e-7)

        ## compile model (use "categorical accuracy" because of one-hot-encoded figures)
        self.model.compile(optimizer=adam,
                           loss="categorical_crossentropy",
                           metrics=["categorical_accuracy"])


        ##### HISTORY: FIT & PREDICT ########################################

        ## fit model
        self.history = self.model.fit(self.X_train,
                                      self.y_train,
                                      epochs=self.nEpochs,
                                      batch_size=self.nBatch,
                                      validation_data=(self.X_test, self.y_test),
                                      verbose=2,
                                      shuffle=True)

        ## plot history
        #     plt.figure(figsize=(14, 8))
        plt.plot(self.history.history["loss"], label="train")
        plt.plot(self.history.history["val_loss"], label="test")
        plt.legend()
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel("Loss", fontsize=18)
        plt.show()

        ## model summary
        # model.summary

        return self.model


    def evaluateBestParams(self):
        """
        Evaluate the best hyperparameters found for the training dataset
        :param X_train: numpy array, train data (independent variable)
        :param y_train: numpy array, train data (dependent variable)
        :return: optimized model, best found hyperparameters for the optimized model, tuned model
        """
        self.tuner = keras_tuner.BayesianOptimization(
            hypermodel=self.model_builder,
            objective="val_loss",
            max_trials=5,
            num_initial_points=4,
            alpha=0.0001,
            beta=2.6,
            seed=568,
            tune_new_entries=True,
            allow_new_entries=True,
            directory=self.directory,
            project_name=self.folder_name
        )

        ## stop training early after reaching a certain value for validation
        stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

        ## Run hyperparameter search
        self.tuner.search(self.X_train, self.y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
        print("get_best_params:", self.tuner.get_best_hyperparameters(num_trials=1)[0])
        print("get_best_model:", self.tuner.get_best_models())

        return self.tuner.get_best_models(), self.tuner.get_best_hyperparameters(num_trials=1)[0], self.tuner


    def fitPredict(self, bestParams):
        """
        Fit data with the compiled model, predict dependent variable
        :param X_train: numpy array, train data (independent variable)
        :param X_test: numpy array, test data (independent variable)
        :param y_train: numpy array, train data (dependent variable)
        :param y_test: numpy array, test data (dependent variable)
        :param tuner: tuned model, optimized model
        :param bestParams: hyperparameters (hp), best found hp for the optimized model
        :return: quality metrics on MAE, CV, R2, RMSE (dictionary), model history, number of best epoch (integer),
                 dependent variable of test and training data (both pandas Series objects)
        """
        ## build model with optimal hyperparameters and train it on data for 50 epochs
        self.model = self.tuner.hypermodel.build(bestParams)
        self.modelHist = self.model.fit(self.X_train,
                                        self.y_train,
                                        epochs=self.nEpochs,
                                        validation_split=0.5,
                                        batch_size=128,
                                        verbose=2)
        print(self.modelHist.history.keys())

        ## evaluate validation loss
        val_loss_per_epoch = self.modelHist.history["val_loss"]
        self.best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch))
        print(f"Best epoch (loss): {self.best_epoch}")

        ## evaluate validation accuracy
        val_acc_per_epoch = self.modelHist.history["val_categorical_accuracy"]
        self.best_epoch = val_loss_per_epoch.index(max(val_acc_per_epoch))
        print(f"Best epoch (accuracy): {self.best_epoch}")

        ## Re-instantiate hypermodel and train it with optimal number of epochs from above
        self.hypermodel = self.tuner.hypermodel.build(bestParams)

        ## fit model
        self.hypermodel.fit(self.X_train, self.y_train, epochs=self.best_epoch, validation_split=0.2)

        ## evaluate result
        eval_result = self.hypermodel.evaluate(self.X_test, self.y_test)
        print("[test loss, test accuracy]:", eval_result)

        ## predict response variable
        self.y_pred = self.model.predict(self.X_test)

        return self.modelHist, self.best_epoch, self.y_test, self.y_pred


    def printPlotMetrics(self):
        """
        Plot various quality metrics related to the binary classification approach
        :param modHist: History object, model history
        :param epochs: Integer, number of epoch for the best evaluated quality metric
        :return: NA
        """
        ## define size of plot
        #     plt.figure(figsize=(14,8))

        ## Plot Loss ------------------------------------------------------------------

        plt.plot(self.modHist.history["loss"], label="train")
        plt.plot(self.modHist.history["val_loss"], label="test")
        plt.legend()
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel("Loss", fontsize=18)
        plt.savefig('CNN_Loss.png')
        plt.show()


        ## Plot Accuracy --------------------------------------------------------------

        plt.plot(self.modHist.history["categorical_accuracy"], label="train")
        plt.plot(self.modHist.history["val_categorical_accuracy"], label="test")
        plt.legend()
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel("Accuracy", fontsize=18)
        plt.savefig('CNN_Accuracy.png')
        plt.show()


        # ## Separate plots per metric ------------------------------------------------
        #
        # metric_dict = {'loss': 'val_loss',
        #                'mae': 'val_categorical_accuracy'}
        #
        # for key, value in metric_dict.items():
        #     print(f"epoch {epochs}:: {key}:", modHist.history[key][-1], f", {value}:", modHist.history[value][-1])
        #     plt.plot(modHist.history[key], label=f"train {key}")
        #     plt.plot(modHist.history[value], label=f"test {key}")
        #     plt.legend()
        #     plt.xlabel("epoch")
        #     plt.ylabel(key)
        #     plt.title(f"model {key}")
        #     plt.show()
        #
        #
        # ## 1 Plot including all metrics -------------------------------------------
        #
        # pd.DataFrame(modHist.history).plot(figsize=(8, 5))
        # plt.xlabel("epoch")
        # plt.ylabel("metric")
        # plt.title("model metric")
        # plt.show()


    def plotPerformance(self):
        """
        Plot predicted vs true values on energy consumption.
        :param y_test: array, true energy consumption
        :param y_pred: array, predicted energy consumption
        :return: NA
        """
        plt.figure(figsize=(14, 8))
        ax = sns.regplot(x=self.y_test, y=self.y_pred, ci=95)
        ax.set_title("Energy: True vs Predicted Values", fontsize=20)
        ax.set_xlabel("Predicted Energy [kWh]", fontsize=14)
        ax.set_ylabel("True Energy [kWh]", fontsize=14)
        plt.savefig("CNN_Performance.png")
        plt.show()


def main():
    pass


if __name__ == "__main__":
    main()

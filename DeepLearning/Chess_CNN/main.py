#!/usr/bin/env python


__author__ = "WhyKiki"
__version__ = "1.0.0"


import numpy as np
import prepareData as prepData
import fieldOccupation as fo
import FENfromFilename as fenFile
import CNN as cnn
import keras_tuner


def preprocessBoardImages(purpose, subsample_size, new_ImgSize):
    """
    Preprocess data: Load filenames, shuffle & subsample. Rescale images, convert
    to numpy array and demean.
    :param purpose: String, "train" for training data or "test" for test data.
    :param subsample_size: integer, size of the subsample,
    :param new_ImgSize: integer, length and width of the compressed images
    :return: X: numpy array, demeaned. filenames_subsample: y, list containing Strings, each representing a FEN
    """
    ## load filenames from files
    filenames = prepData.loadFilenames(f"data/{purpose}/*.jpeg")

    ## shuffle filenames
    filenames = prepData.shuffleDataSet(filenames)

    ## create a subsample
    filenames_subsample = prepData.subSample(filenames, subsample_size)

    ## rescale images
    data = prepData.rescaleImages(filenames_subsample, new_ImgSize)

    ## convert data from PIL image type to numpy arrays
    X = prepData.convertImagesToArrays(data)

    ## demean images
    X = prepData.demeanImages(X)

    ## create the dependent variable y from filenames (FEN)
    y = []
    for filename in filenames_subsample:
        fen = prepData.splitFilename(filename)
        y.append(fen)

    return X, y


def fromFENtoArray(fen):
    """
    Generate a numpy array from FEN that is representing a board's field occupations.
    :param fen: String, FEN
    :return: numpy array (list of strings) representing the whole board, including info on field occupations
    """
    ## split the FEN in the 8 rows' field occupations
    dic_fen = fenFile.extractFENpartsFromFilename(fen)

    ## translate FEN to String-array
    fen_parts = dic_fen[fen]
    y_board = fenFile.fenToArray(fen_parts)

    return y_board, fen_parts


def createDatasetFieldImages(board, fen, path):
    """
    Chop image of board into 64 (8*8) images and save to path.
    :param board: String-based array (8*8), each sub-array (1*8) representing a row with its corresponding field occupations
    :param fen: String, representing a FEN (dependent variable y)
    :param path: String, path where created images shall be saved to
    :return: NA
    """
    ## assign all locations of white fields and black fields
    white_fields, dark_fields = fo.defineColorOfField()

    ## create folders to save images of chessmen, white & black fields (for both train and test data)
    fo.createFolderForChessmen("train")
    fo.createFolderForChessmen("test")

    ## assign location of an exemplary free white and an exemplary free dark field
    dic_chessmen = fo.mapPositionsToFigures(board, white_fields)

    ## cut each image of a board into 64 images of its fields and save to path
    fo.cutBoardImageIntoFields(fen, dic_chessmen, path)


def assignDataForNN(purpose, fieldSize):
    """
    From the new created dataset consisting of the separate field images, create X and y.
    :param purpose: String, "train" for training data, "test" for test data
    :return: X: array of images (numpy arrays), y: list of Strings (each representing the figure the image displays)
    """
    ## map character-based figure labels to list of binaries
    label_dict = fenFile.oneHotEncodeFigure()

    ## load filenames from path
    filenames = prepData.loadFilenames(f"data/{purpose}_chops/*.jpg")

    ## extract y from each filename, e.g. 'b' for black bishop from the ['b'] in
    ## ['b']__1b1b1b2-3r4-1rK4b-R7-R2R1k2-2Bp4-2P5-2r5_03.jpg"
    y = []
    for filename in filenames:
        figAbbrev = filename.split("'")[1]
        y.append(label_dict[figAbbrev])

    ## rescale images (X)
    data = prepData.rescaleImages(filenames, fieldSize)

    ## convert data from PIL image type to numpy arrays
    X = prepData.convertImagesToArrays(data)

    return X, np.array(y)


def figureRecognitionViaCNN(X_train, y_train, X_test, y_test):
    """
    Generate FEN from image using CNN.
    :param X_train: numpy array (from images), training data, independent variable
    :param y_train: String, FEN, training data, dependent variable
    :param X_test: numpy array (from images), test data, independent variable
    :param y_test: String, FEN, test data, dependent variable
    :return: NA
    """
    ## create object from class
    cnn_obj = cnn.ConvNet(X_train, y_train, X_test, y_test)

    ## evaluate best model and hyperparameters
    best_model, best_hyperParams, tuner = cnn_obj.evaluateBestParams()

    ## tune hyperparameters
    hp = keras_tuner.tuner
    hypermodel = cnn_obj.model_builder(hp)
    model = hypermodel.build(hp)

    ## fit and predict
    modelHist, best_epoch, y_test, y_pred = hypermodel.fitPredict(hp, best_hyperParams)

    ## visualize loss
    cnn_obj.printPlotMetrics(modelHist, best_epoch)

    ## visualize y_pred vs y_test
    cnn_obj.plotPerformance(y_test, y_pred)


def convertListToArray(lst):
    """
    Convert list to numpy array (required for input into neural net)
    :param lst: list, to be converted to numpy array
    :return: array
    """
    return np.asarray(lst)


def main():

    ## load data, shuffle, subsample, resize, convert to numpy array and demean ---------------------
    new_BoardImgSize = 400
    subsample_size_trainData = 6
    subsample_size_testData = 2
    X_train_boardImages, y_train_fens = preprocessBoardImages("train", subsample_size_trainData, new_BoardImgSize)
    X_test_boardImages, y_test_fens = preprocessBoardImages("test", subsample_size_testData, new_BoardImgSize)

    ## define field image size
    field_ImgSize = int(new_BoardImgSize/8)


    ## convert y (FEN) to an array of Strings -------------------------------------------------------
    path_part = "data/train_chops"
    for idx, elem in enumerate(y_train_fens):
        y_train_boardStringArrays, y_train_fenParts = fromFENtoArray(elem)

        ## cut board images into field images (8x8 --> 64 fields per board) and save to path
        createDatasetFieldImages(y_train_boardStringArrays, elem, path_part)


    ## convert X (board images) to an array of field images -----------------------------------------
    path_part = "data/test_chops"
    for idx, elem in enumerate(y_test_fens):
        y_test_boardStringArrays, y_test_fenParts = fromFENtoArray(elem)

        ## cut board images into field images (8x8 --> 64 fields per board) and save to path
        createDatasetFieldImages(y_test_boardStringArrays, elem, path_part)


    ###################################################################################################################
    ##### CNN: Figure Recognition #####################################################################################
    ###################################################################################################################

    ## Assign data for figure recognition
    X_train_figure, y_train_figure = assignDataForNN("train", fieldSize=50)
    X_test_figure, y_test_figure = assignDataForNN("test", fieldSize=50)

    ## Perform figure recognition per field image and assess quality metrics
    figureRecognitionViaCNN(X_train_figure, y_train_figure, X_test_figure, y_test_figure)


if __name__ == '__main__':
    main()

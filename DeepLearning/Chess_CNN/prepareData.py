#!/usr/bin/env python


__author__ = "WhyKiki"
__version__ = "1.0.0"


import numpy as np
from random import shuffle
from tqdm import tqdm
from PIL import Image
import glob


def loadFilenames(path):
    """
    Load data from path.
    :param path: String, location of images to be processed.
    :return: Strings, filenames of the images in path.
    """
    filenames = glob.glob(path)
    return filenames


def splitFilename(filename):
    """
    Cut off extension from filename --> FEN.
    :param filename: String, filename including extension
    :return: String, filename without extension
    """
    return filename.split("\\")[1].split(".")[0]


def shuffleDataSet(data):
    """
    Shuffle Data.
    :param data: String, filenames of images to be processed.
    :return: Strings, shuffled filenames
    """
    shuffle(data)
    return data


def subSample(data, size):
    """
    Create a subsample to set up the code initially.
    :param data: Strings, filenames.
    :param size: integer, size of the subsample,
    :return: filenames, subsample.
    """
    return data[:size]


def rescaleImages(filenames, new_size):
    """
    Rescale images.
    :param filenames: String, filenames of interest.
    :param new_size: integer, length and width of the compressed images
    :return: images, compressed.
    """
    images = []
    for filename in tqdm(filenames):
        img = Image.open(filename)
        ## rescale image to new_size * new_size pixels
        resized_img = img.resize((new_size, new_size), Image.ANTIALIAS)
        ## create set to images
        images.append(resized_img)
    return images


def convertImagesToArrays(images):
    """
    Convert images to numpy arrays.
    :param images: images.
    :return: numpy array.
    """
    X = []
    for i in range(len(images)):
        X.append(np.array(images[i]))

    ## reshape dimension from e.g. (240, 240, 3) to (5000, 240, 240, 3)
    X = np.concatenate([img[np.newaxis] for img in X])

    return X


def demeanImages(X):
    """
    Demean images, zero-center images.
    :param X: numpy array.
    :return: numpy array, demeaned.
    """
    ## convert pixels to type float
    X = X.astype('float32')

    ## calculate mean
    mean = X.mean(axis=(0, 1, 2))

    ## demean images: zero center (zc)
    X_zc = X - mean
    return X_zc


def main():
    pass


if __name__ == '__main__':
    main()

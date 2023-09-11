#!/usr/bin/env python


__author__ = "WhyKiki"
__version__ = "1.0.0"


import os
from PIL import Image
import FENfromFilename as fenFile
import re


def createFolderForChessmen(purpose):
    """
    Create a new folder in path for the chopped images. (Each chop containing the image of a single field.)
    :param purpose: String, "train" for training data or "test" for test data.
    :return: NA
    """
    path = f"data/{purpose}_chops"
    if not os.path.exists(path):
        os.makedirs(path)


def defineColorOfField():
    """
    Create list of lists for both, white and black fields separately. These contain the positions [row, column] for
    each of these fields. For the white fields it would be [[0,0], [0,2], [0,4], ...]
    Each inner list contains of two integers representing the location of all available fields.
    :return:
    """
    white_fields = []
    dark_fields = []
    for row in range(8):
        col = range(0, 7, 2)
        for c in col:
            ## define all white fields on a plain chess board
            white_fields.append([row, c + row % 2])
            ## define all dark fields on a plain chess board
            dark_fields.append([row, c + (row+1) % 2])

    return white_fields, dark_fields


def mapPositionsToFigures(board, white_fields):
    """
    Extract single representatives for both, white and black free fields.
    :param board: array (string-based), containing information about field occupation
    :param white_fields: list, containing lists of all white fields on chess board encoded in a list of
           two integers (row, col) per board, like [[0,0], [0,2], [0,4],...]
    :return: lists, one for each of free white and dark fields. Each list contains 2 integers for the info on location.
    """
    ## initialize dictionary for chessmen (or white/dark fields) positions
    dic_figures = {"whiteField": [], "darkField": [],
                   "p": [], "r": [], "n": [], "b": [], "q": [], "k": [],
                   "P": [], "R": [], "N": [], "B": [], "Q": [], "K": []}

    ## for each chessman / empty field: write position to corresponding key in dictionary
    rows = 0
    cols = 0
    for row in board:
        for elem in row:
            ## empty fields
            if elem == "":
                ## for free white fields
                if [rows, cols] in white_fields:
                    dic_figures["whiteField"].append([rows, cols])
                ## for free dark fields
                else:
                    dic_figures["darkField"].append([rows, cols])
            ## chessman
            else:
                dic_figures[elem].append([rows, cols])
            cols += 1
        cols = 0
        rows += 1

    return dic_figures


def openImage(filename):
    """
    Open image of board
    :return: image, filename (String)
    """
    return Image.open(filename), filename


def cutBoardImageIntoFields(fen, dic_figures, path_part):
    """
    Cut a board into 64 images (8*8) and save one representative of each chessman type, black or white field to path.
    :param fen: Strings, FEN.
    :param dic_figures: dictionary, containing information on location of each chessman type, black or white field
    :param path_part: String, part of the path where created images shall be saved to
    :return: NA
    """
    purpose = path_part.split("/")[1].split("_")[0]
    filename = f"data/{purpose}/{fen}.jpeg"
    img, name = openImage(filename)
    width, height = img.size

    ## define size of image chunks
    chopSize = int(width / 8)

    ## save chops of original image
    cnt = 0
    filenames_with_pos = []
    for x0 in range(0, width, chopSize):
        for y0 in range(0, height, chopSize):
            box = (x0, y0,
                   x0 + chopSize if x0 + chopSize < width else width - 1,
                   y0 + chopSize if y0 + chopSize < height else height - 1)

            ## define position (e.g. [3,5]) on chess board and adapt path to include this position
            row = int(cnt % 8)
            col = int(cnt / 8)

            ## check if position is one of the fig_dict_unique's values, then save image. The image is saved as e.g.
            ## "['B']__1bn1R3-KB4p1-2b5-7k-1B6-8-8-r7_11.jpeg" where ['B'] represents the white bishop and _11 in the
            ## end represents the position [1, 1] of which the bishop was located.
            for elem in dic_figures.values():
                if [row, col] in elem:
                    chessman = [key for key in dic_figures if ([row, col] in dic_figures[key])]
                    img_path = f'{path_part}/{chessman}__{fen}.jpg'.replace(".jpg", f"_{row}{col}.jpg")
                    filenames_with_pos.append(img_path)
                    img.crop(box).save(img_path)

            cnt += 1


def main():
    pass


if __name__ == '__main__':
    main()

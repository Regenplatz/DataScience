#!/usr/bin/env python


__author__ = "WhyKiki"
__version__ = "1.0.0"


import numpy as np
from sklearn.preprocessing import LabelBinarizer


def extractFENpartsFromFilename(filename):
    """
    Cut off filename extension (--> fen).
    Separate each row's composition from filename. Split by "-" into separate parts (--> fen_parts).
    :param filename: String, FEN, containing "-"
    :return: dic_fen: dictionary, with FEN as key and fen parts (each rows' occupation) as value (list of String).
    """
    fen_parts = filename.split("-")
    dic_fen = {}
    dic_fen[filename] = fen_parts

    return dic_fen


def fenToArray(fen_occupation):
    """
    Convert chessmen positions from FEN to board.
    :fen_occupation: String, FEN notation per chess board.
    :return: array representing the chess board, containing chessmen positions.
    """
    ## initialize chess board
    board = np.zeros((8, 8), dtype=np.str)

    ## replace empty Strings with chessmen if appropriate
    for idx, row in enumerate(fen_occupation, start=0):
        pos = 0
        for elem in row:
            if elem in "12345678":
                pos += int(elem)
            if elem.lower() in "prnbqk" and idx < 8:
                board[idx][pos] = elem
                pos += 1

    return board


def oneHotEncodeFigure():
    """
    One hot encode String label per figure as binary.
    :param figureAbbreviation:
    :return:
    """
    ## define reference list for each possible chessman or free field --> later possible to translate back
    reference_list = ["darkField", "whiteField",
                      "p", "r", "n", "b", "q", "k",
                      "P", "R", "N", "B", "Q", "K"]

    ## create a list for the binary mapping lists
    encoder = LabelBinarizer()
    transformed_label_list = encoder.fit_transform(reference_list)

    ## map original String-based figure name to corresponding binary list
    label_dict = dict(zip(reference_list, transformed_label_list))

    return label_dict


# def oneHotEncodeFieldOccupations(filename):
#     """
#     Convert array containing characters to array with numbers 0 and 1 --> computations possible
#     :param filename: list of Strings, each String represents a FEN
#     :return: array, containing one-hot-encoded chessboard's field occupations.
#              Size of array: (no_boards, 8, len(reference_list)*8)
#     """
#     ## define reference list for each possible chessman or free field --> later possible to translate back
#     reference_list = ["",
#                       "p", "r", "n", "b", "q", "k",
#                       "P", "R", "N", "B", "Q", "K"]
#
#     ## assess number of all possible chessman and free field
#     no_chessmen = len(reference_list)
#
#     ## translate FEN to array
#     board_array = fromFENtoArray(filename)
#
#     y_ohe = []
#     fields = []
#     ## for each row per board
#     for row in board_array:
#         ## for each field per row
#         for field in row:
#             ## initialize list of length (number of chessmen and empty field) with zeros
#             occupation = no_chessmen * [0]
#             ## assign a 1 at the position where the corresponding field occupation is located in reference_list
#             for idx, chessman in enumerate(reference_list):
#                 if field == chessman:
#                     occupation[idx] = 1
#             fields.append(occupation)
#     boardOccupation = np.array(fields).reshape(8, no_chessmen * 8)
#     y_ohe.append(boardOccupation)
#
#     return y_ohe, reference_list


def fromArrayToFen(board):
    """
    Convert chessmen positions from board to FEN.
    :param board: array of arrays , each subarray (8*1) representing a row with its corresponding field occupations
    :return: String, FEN (position of chessmen on board).
    """
    ## initialize FEN
    fen = ""

    ## create FEN from array
    for row in board:
        emptyFields = 0

        for elem in row:
            if elem == "":
                emptyFields += 1
            else:
                if emptyFields > 0:
                    fen += str(emptyFields)
                fen += elem
                emptyFields = 0

        if emptyFields > 0:
            fen += str(emptyFields)
        fen += "-"
    fen = fen[:-1]
    return fen


def main():
    pass


if __name__ == '__main__':
    main()

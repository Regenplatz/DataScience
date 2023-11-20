#!/usr/bin/python3

import numpy as np


##### GLOBAL VARIABLES #################################################################################
##### Dictionaries relate to 6 dot BRAILLE (3x2) characters and need to be adjusted
##### if 8 dot BRAILLE (4x2, Unicode) is wanted

brailleArray_length = 8

dict_numbers = {"#": [3, 4, 5, 6],
                "1": [1], "2": [1, 2], "3": [1, 4], "4": [1, 4, 5], "5": [1, 5],
                "6": [1, 2, 4], "7": [1, 2, 4, 5], "8": [1, 2, 5], "9": [2, 4], "0": [2, 4, 5]
                }

dict_punctuationMark = {".": [2, 5, 6], ",": [2], ";": [2, 3], ":": [2, 5], "?": [2, 6], "!": [2, 3, 5],
                        "(": [2, 3, 5, 6], ")": [2, 3, 5, 6], '„': [2, 3, 6], '“': [3, 5, 6], '"': [3, 5, 6]
                        }

dict_specialCharacters = {"*": [3, 5], "_": [3, 6], "'": [3], "capital": [6], "/": [3, 4], " ": []
                         }

dict_mathsSymbols = {"=": [2, 3, 5, 6], "+": [2, 3, 5], "-": [3, 6], "*": [3, 5], "/": [2, 5, 6],
                     ":": [2, 5], "<": [5, 6], ">": [4, 5]
                     }

dict_latinLetters = {"a": [1], "b": [1, 2], "c": [1, 4], "d": [1, 4, 5], "e": [1, 5],
                     "f": [1, 2, 4], "g": [1, 2, 4, 5], "h": [1, 2, 5], "i": [2, 4], "j": [2, 4, 5],
                     "k": [1, 3], "l": [1, 2, 3], "m": [1, 3, 4], "n": [1, 3, 4, 5], "o": [1, 3, 5],
                     "p": [1, 2, 3, 4], "q": [1, 2, 3, 4, 5], "r": [1, 2, 3, 5], "s": [2, 3, 4],
                     "t": [2, 3, 4, 5], "u": [1, 3, 6], "v": [1, 2, 3, 6], "w": [2, 4, 5, 6],
                     "x": [1, 3, 4, 6], "y": [1, 3, 4, 5, 6], "z": [1, 3, 5, 6]
                     }

dict_germanAddition = {"ä": [3, 4, 5], "ö": [2, 4, 6], "ü": [1, 2, 5, 6], "au": [1, 6],
                       "äu": [3, 4], "ei": [1, 4, 6], "ie": [3, 4, 6], "eu": [1, 2, 6],
                       "ch": [1, 4, 5, 6], "sch": [1, 5, 6], "st": [2, 3, 4, 5, 6], "β": [2, 3, 4, 6]
                       }

dict_esperantoAddition = {"cx": [1, 4, 6], "gx": [1, 2, 4, 5, 6], "hx": [1, 2, 5, 6],
                          "jx": [2, 4, 5, 6], "sx": [2, 3, 4, 6], "ux": [3, 4, 6]
                          }

dict_irishAddition = {"á": [1, 2, 3, 4, 6], "é": [1, 2, 3, 4, 5, 6], "í": [1, 2, 3, 5, 6],
                      "ó": [2, 3, 4, 6], "ú": [2, 3, 4, 5, 6],
                      "bh": [1, 2, 6], "ch": [1, 4, 6], "dh": [1, 4, 5, 6], "fh": [1, 5, 6],
                      "gh": [1, 2, 4, 6], "mh": [1, 2, 4, 5, 6], "ph": [1, 2, 5, 6],
                      "sh": [2, 4, 6], "th": [2, 4, 5, 6]
                      }

dict_germanShorts = {"der": [1, 2, 3, 5], "die": [3, 4, 6], "das": [1, 4, 5],
                     "immer": [1, 3, 4, 6], "mit": [2, 3, 4, 5],
                     "über": [1, 2, 5, 6], "vor": [2, 6], "uns": [1, 3, 5, 6]
                     }

dict_combined = None


def adaptDictToArrayLength8(dictionary: dict) -> dict:
    """In case UTF8 should be used, shift is needed after position 3, because structure would be:
    column1: [1, 2, 3, 7], column2: [4, 5, 6, 8]"""
    for key, value in dictionary.items():
        for idx, elem in enumerate(value):
            if elem > 3:
                value[idx] = elem + 1
            else:
                value[idx] = elem
    return dictionary


def adaptDictsToArrayLength():
    """In case UTF8 format shall be used, adapt character encoding accordingly
    (from 6 to 8 dots for a Braille array)"""
    if brailleArray_length == 8:
        global dict_numbers
        dict_numbers = adaptDictToArrayLength8(dictionary=dict_numbers)
        global dict_punctuationMark
        dict_punctuationMark = adaptDictToArrayLength8(dictionary=dict_punctuationMark)
        global dict_specialCharacters
        dict_specialCharacters = adaptDictToArrayLength8(dictionary=dict_specialCharacters)
        global dict_mathsSymbols
        dict_mathsSymbols = adaptDictToArrayLength8(dictionary=dict_mathsSymbols)
        global dict_latinLetters
        dict_latinLetters = adaptDictToArrayLength8(dictionary=dict_latinLetters)
        global dict_germanAddition
        dict_germanAddition = adaptDictToArrayLength8(dictionary=dict_germanAddition)
        global dict_esperantoAddition
        dict_esperantoAddition = adaptDictToArrayLength8(dictionary=dict_esperantoAddition)
        global dict_irishAddition
        dict_irishAddition = adaptDictToArrayLength8(dictionary=dict_irishAddition)
        global dict_germanShorts
        dict_germanShorts = adaptDictToArrayLength8(dictionary=dict_germanShorts)


def idxListToListOfBinaries(idxList: list) -> list:
    """Translate letters to a list of binaries to encode for full and emtpy dots."""
    binaryEncoded = brailleArray_length * [0]
    for i in idxList:
        binaryEncoded[i-1] = 1
    return binaryEncoded


def listOfBinariesToIdxList(lst: list) -> list:
    """Translate letters to a list of binaries to encode for full and emtpy dots."""
    idxList = []
    for idx, elem in enumerate(lst):
        if elem == 1:
            idxList.append(idx+1)
    return idxList


def listOfBinariesToArray(lst: list) -> np.array:
    """Convert list of binary entries (1x8) to array (4x2) or (3x2) to converge to Braille representation."""
    return np.array(lst).reshape(2, int(brailleArray_length/2)).T


def letterToBrailleArray(letter: str, dictionary: dict) -> np.array:
    """Translate letter to Braille array."""
    letterAsBinaryList = idxListToListOfBinaries(dictionary[letter])
    return listOfBinariesToArray(letterAsBinaryList)


def brailleArrayToListOfBinaries(brailleArray: np.array) -> list:
    """Convert Braille array (4x2) or (3x2) to list of binaries (1x8)."""
    return brailleArray.T.reshape(-1, ).tolist()


def brailleArrayToLetter(brailleArray: np.array, dictionary: dict) -> str:
    """Translate a Braille character to the corresponding letter."""
    braille_binaryList = brailleArrayToListOfBinaries(brailleArray)
    braille_idxList = [idx+1 for idx, elem in enumerate(braille_binaryList) if elem == 1]
    return [k for k, v in dictionary.items() if v == braille_idxList][0]


def numberToBrailleSequence(number: int) -> np.array:
    """Translate a number to Braille array"""
    str_number = str(number)
    idxList_firstElement = idxListToListOfBinaries(idxList=dict_numbers["#"])
    lst_numberArrays = [np.array(idxList_firstElement).reshape(2, int(brailleArray_length / 2)).T]
    for num in str_number:
        idxList = dict_numbers[num]
        binaryList = idxListToListOfBinaries(idxList=idxList)
        binaryArray = listOfBinariesToArray(lst=binaryList)
        lst_numberArrays.append(binaryArray)
    return lst_numberArrays


def brailleSequenceToNumber(brailleSequence: np.array) -> int:
    """Convert Braille sequence to number"""
    lst_number = []
    for binaryArray in brailleSequence:
        binaryList = brailleArrayToListOfBinaries(brailleArray=binaryArray)
        idxList = listOfBinariesToIdxList(lst=binaryList)
        num = [k for k, v in dict_numbers.items() if v == idxList][0]
        lst_number.append(num)
    number = "".join(lst_number)
    print(f"NUMBER:\n{number}\n")
    return int(number[1:])


def brailleSequenceToText(brailleSequence: list, dictionary: dict) -> str:
    text = ""
    for elem in brailleSequence:
        char = brailleArrayToLetter(brailleArray=elem,
                                              dictionary=dictionary)
        text += char
    return text


def main():

    ## adapt to Braille array length
    adaptDictsToArrayLength()
    print(f"Braille Array Lenght:", brailleArray_length, "\n-> indices for numbers:", dict_numbers, "\n")

    ## combine dictionaries for everything but numbers of 1 language in a single dictionary
    global dict_combined
    dict_combined = {**dict_latinLetters,
                     **dict_germanAddition,
                     **dict_punctuationMark,
                     **dict_specialCharacters,
                     **dict_mathsSymbols}


    ##### SINGLE CHARACTER ############################################################

    ## translate letter to Braille array
    letter = "m"
    brailleArray = letterToBrailleArray(letter=letter,
                                        dictionary=dict_latinLetters)
    print(f"LETTER {letter}:\n-> braille\n {brailleArray}\n")

    ## translate Braille array to letter
    if brailleArray_length == 6:
        brailleList = [0, 1, 1, 1, 1, 0]
    else:
        brailleList = [0, 1, 1, 0, 1, 1, 0, 0]
    brailleArray = np.array(brailleList).reshape(2, int(brailleArray_length/2)).T
    letter = brailleArrayToLetter(brailleArray=brailleArray,
                                  dictionary=dict_latinLetters)
    print(f"BRAILLE ARRAY (representing a STRING)\n{brailleArray}:\n-> letter {letter}\n")


    ##### NUMBER ######################################################################

    ## translate number to Braille array
    number = 123
    brailleNum = numberToBrailleSequence(number=number)
    print(f"NUMBER: {number}\n-> BRAILLE Sequence:\n{brailleNum}\n")

    ## translate Braille array to number (therefore first create Braille array for #629)
    if brailleArray_length == 6:
        numSign = [0, 0, 1, 1, 1, 1]
        num6 = [1, 1, 0, 1, 0, 0]
        num2 = [1, 1, 0, 0, 0, 0]
        num9 = [0, 1, 0, 1, 0, 0]
    else:
        numSign = [0, 0, 1, 0, 1, 1, 1, 0]
        num6 = [1, 1, 0, 0, 1, 0, 0, 0]
        num2 = [1, 1, 0, 0, 0, 0, 0, 0]
        num9 = [0, 1, 0, 0, 1, 0, 0, 0]
    brailleArray_numSign = np.array(numSign).reshape(2, int(brailleArray_length / 2)).T
    brailleArray6 = np.array(num6).reshape(2, int(brailleArray_length / 2)).T
    brailleArray2 = np.array(num2).reshape(2, int(brailleArray_length / 2)).T
    brailleArray9 = np.array(num9).reshape(2, int(brailleArray_length / 2)).T
    brailleNum = [brailleArray_numSign, brailleArray6, brailleArray2, brailleArray9]
    number = brailleSequenceToNumber(brailleSequence=brailleNum)
    print(f"BRAILLE ARRAY (representing a NUMBER):\n{brailleNum}\n-> Number: {number}\n")


    ##### CHARACTER SEQUENCE ##########################################################

    ## translate characters to Braille array
    text = "Some weird text example for 5 seconds to Mars in 12 words: Schön"
    lst_text_lower = text.lower()
    brailleSequences = []
    for char in lst_text_lower:
        if char in dict_combined:
            braille_elem = letterToBrailleArray(letter=char,
                                                dictionary=dict_combined)
            brailleSequences.append(braille_elem)
        elif char in dict_numbers:
            braille_elem = numberToBrailleSequence(number=int(char))
            brailleSequences.append(braille_elem)
    print(f"SENTENCE: {text}\n-> Braille Sequences:\n{brailleSequences}\n")

    ## translate Braille sequence to characters (therefore first create Braille array for "some")
    if brailleArray_length == 6:
        charS = [0, 1, 1, 1, 0, 0]
        charO = [1, 0, 1, 0, 1, 0]
        charM = [1, 0, 1, 1, 0, 0]
        charE = [1, 0, 0, 0, 1, 0]
    else:
        charS = [0, 1, 1, 0, 1, 0, 0, 0]
        charO = [1, 0, 1, 0, 0, 1, 0, 0]
        charM = [1, 0, 1, 0, 1, 0, 0, 0]
        charE = [1, 0, 0, 0, 0, 1, 0, 0]
    brailleArray_s = np.array(charS).reshape(2, int(brailleArray_length / 2)).T
    brailleArray_o = np.array(charO).reshape(2, int(brailleArray_length / 2)).T
    brailleArray_m = np.array(charM).reshape(2, int(brailleArray_length / 2)).T
    brailleArray_e = np.array(charE).reshape(2, int(brailleArray_length / 2)).T
    brailleText = [brailleArray_s, brailleArray_o, brailleArray_m, brailleArray_e]
    print(f"BRAILLE ARRAY (representing a TEXT):\n{brailleText}\n")
    text = brailleSequenceToText(brailleSequence=brailleText,
                                 dictionary=dict_combined)
    print(f"BRAILLE Sequence:\n{brailleText}\n-> Text: {text}\n")


if __name__ == "__main__":
    main()

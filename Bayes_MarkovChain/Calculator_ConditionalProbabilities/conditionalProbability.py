#!/usr/bin/env


###### Assign field indices
x = 0
notX = 1
y = 2
notY = 3
x_y = 4
x_notY = 5
notX_y = 6
notX_notY = 7
y_x = 8
y_notX = 9
notY_x = 10
notY_notX = 11


def totalProbability(items, ind1, ind2, ind3, ind4, ind5):
    """Calculate P(A) = P(A|B)*P(B) + P(A|notB)*P(notB)"""
    if str(items[ind1].value) == "" and \
            str(items[ind2].value) != "" and \
            str(items[ind3].value) != "" and \
            str(items[ind4].value) != "" and \
            str(items[ind5].value) != "":
        result = float(items[ind2].value) * float(items[ind3].value) \
                 + float(items[ind4].value) * float(items[ind5].value)
        items[ind1].value = str(round(result, 3))


def singleValsProba(items, ind1, ind2):
    """Calculate P(A) = 1 - P(notA).
    If both, P(A) and P(notA) are not given, call function singleVals."""
    if str(items[ind1].value) != "" and str(items[ind2].value) == "":
        items[ind2].value = str(round(1 - float(items[ind1].value), 3))
    elif str(items[ind1].value) == "" and str(items[ind2].value) != "":
        items[ind1].value = str(round(1 - float(items[ind2].value), 3))
    else:
        if ind1 == 0:
            totalProbability(items, ind1, 4, 2, 5, 3)
            totalProbability(items, ind2, 6, 2, 7, 3)
        elif ind1 == 2:
            totalProbability(items, ind1, 8, 0, 9, 1)
            totalProbability(items, ind2, 10, 0, 11, 1)


def b_of_a_first(items, ind1, ind2, ind3, ind4):
    """Calculate P(B|A) = (P(A|B) * P(B)) / P(A) """
    if str(items[ind1].value) == "" and \
            str(items[ind2].value) != "" and \
            str(items[ind3].value) != "" and \
            str(items[ind4].value) != "":
        result = float(items[ind2].value) * float(items[ind3].value) / float(items[ind4].value)
        items[ind1].value = str(round(result, 3))


def b_of_a_second(items, ind1, ind2, ind3, ind4, ind5):
    """Calculate P(B|A) = ( P(B) - P(B|notA) * P(notA) ) / P(A) """
    if str(items[ind1].value) == "" and \
            str(items[ind2].value) != "" and \
            str(items[ind3].value) != "" and \
            str(items[ind4].value) != "" and \
            str(items[ind5].value) != "":
        result = (float(items[ind2].value) - float(items[ind3].value) \
                  * float(items[ind4].value)) / float(items[ind5].value)
        items[ind1].value = str(round(result, 3))


def condProbs(items):
    """Calculate P(A|B) and P(B|A)."""

    ### P(X|Y)
    if str(items[y_x].value) != "":
        b_of_a_first(items, x_y, y_x, x, y)
    elif str(items[x_notY].value) != "":
        b_of_a_second(items, x_y, x, x_notY, notY, y)

    ### P(X|notY)
    if str(items[notY_x].value) != "":
        b_of_a_first(items, x_notY, notY_x, x, notY)
    elif str(items[x_y].value) != "":
        b_of_a_second(items, x_notY, x, x_y, y, notY)

    ### P(notX|Y)
    if str(items[y_notX].value) != "":
        b_of_a_first(items, notX_y, y_notX, notX, y)
    elif str(items[notX_notY].value) != "":
        b_of_a_second(items, notX_y, notX, notX_notY, notY, y)

    ### P(notX|notY)
    if str(items[notY_notX].value) != "":
        b_of_a_first(items, notX_notY, notY_notX, notX, notY)
    elif str(items[notX_y].value) != "":
        b_of_a_second(items, notX_notY, notX, notX_y, y, notY)

    ### P(Y|X)
    if str(items[x_y].value) != "":
        b_of_a_first(items, y_x, x_y, y, x)
    elif str(items[y_notX].value) != "":
        b_of_a_second(items, y_x, y, y_notX, notX, x)

    ### P(Y|notX)
    if str(items[notX_y].value) != "":
        b_of_a_first(items, y_notX, notX_y, y, notX)
    elif str(items[y_x].value) != "":
        b_of_a_second(items, y_notX, y, y_x, x, notX)

    ### P(notY|X)
    if str(items[x_notY].value) != "":
        b_of_a_first(items, notY_x, x_notY, notY, x)
    elif str(items[notY_notX].value) != "":
        b_of_a_second(items, notY_x, notY, notY_notX, notX, x)

    ### P(notY|notX)
    if str(items[notX_notY].value) != "":
        b_of_a_first(items, notY_notX, notX_notY, notY, notX)
    elif str(items[notY_x].value) != "":
        b_of_a_second(items, notY_notX, notY, notY_x, x, notX)


def fillFields(items):
    singleValsProba(items, x, notX)
    singleValsProba(items, y, notY)
    condProbs(items)


def calc_all(items):
    for i in range(7):
        fillFields(items)
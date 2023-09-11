#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import pandas as pd
import numpy as np
from typing import Tuple


##### VECTORIZERS #########################################################################################
###########################################################################################################

def tfidf(word, sentence, docs):
    freq = sentence.count(word)

    ## term frequency
    tf = freq / len(sentence)

    ## inverse document frequency
    idf = np.log10(len(docs) / sum([1 for doc in docs if word in doc]))

    return round((tf * idf), 4)


def bm25(docs, lst, word, sentence, k=1.2, b=0.75):
    avgDist = sum(len(sentence) for sentence in lst / len(docs))
    N = len(docs)

    ## term frequency
    freq = sentence.count(word)
    tf = (freq * (k + 1)) / (freq + k * (1 - b + b * len(sentence) / avgDist))

    ## inverse document frequency
    N_q = sum([1 for doc in docs if word in doc])

    ## docs that contain the word
    idf = np.log(((N - N_q + 0.5) / (N_q + 0.5)) + 1)

    return round((tf * idf), 4)



##### SIMILARITIES / DISTANCES ############################################################################
###########################################################################################################

def jaccardSimDist(text1: list, text2: list) -> Tuple[float, float]:
    """Calculate Jaccard Similarity and Distance"""

    ## convert lists to sets (otherwise intersection() and union() won't work)
    set1 = set(text1)
    set2 = set(text2)

    ## find intersection and union of words list between the different documents
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    ## calculate JACCARD SIMILARITY SCORE: (length of intersection set) / (length of union set) &
    ## JACCARD DISTANCE: 1 - JACCARD SIMILARITY
    jaccSim = float(len(intersection)) / len(union)
    jaccDist = 1 - jaccSim

    return jaccSim, jaccDist



def main():
    pass


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import transMatrix_df
import supermarket as sm



def create_tpMatrix():
    """load DataFrame and convert to Transition Probability Matrix 'tpm' """
    df = transMatrix_df.create_tpm()
    tpm = np.array(df)
    return tpm
#     print(tpm)
# create_tpMatrix()
# exit()

dict_initState = {'dairy': [0, 1, 0, 0, 0, 0],
                 'drinks': [0, 0, 1, 0, 0, 0],
                  'fruit': [0, 0, 0, 1, 0, 0],
                 'spices': [0, 0, 0, 0, 1, 0],
               'checkout': [0, 0, 0, 0, 0, 0]}


def get_randomState():
    states = ['dairy', 'drinks', 'fruit', 'spices', 'checkout']
    entry = random.randint(0,4)
    return states[entry],dict_initState[states[entry]]

# initState, initState_vector = get_randomState('fruit')
# print(initState, initState_vector)


def calc_allStates():
    """constructing Markov-Chain according to following equation:
       next state = transition possibility * initial state"""

    # Transition Probability Matrix 'tpm' as numpy array
    tpm = create_tpMatrix()

    # get the first location after entering the supermarket by random choice
    currentState, currentVector = get_randomState()
    print('First location: ', currentState)

    # assign different lists for later use
    nextState_results = []          # for results of next_state matrix
    list_of_states = ['entrance']
    list_columnNames = ['checkout', 'dairy', 'drinks', 'fruit', 'spices', 'entrance']

    # initially, isv is set to current vector
    # isv = np.array(currentVector)
    next_vector = np.array(currentVector)

    while currentState != 'checkout':
        list_of_states.append(currentState)
        nextState_results.append(np.array(next_vector).flatten())

        isv = next_vector

        # calculate next location
        next_vector = isv.dot(tpm)  # result is a vector

        # get (feature) index of highest value
        maxNextState = np.argmax(next_vector)
        currentState = list_columnNames[maxNextState]

        print('Next location (vector): ', next_vector)
        print('Next Location: ', currentState)

    # append checkout data
    list_of_states.append(currentState)
    nextState_results.append(np.array(next_vector).flatten())

    # list_of_states.append(currentState)
    print('All states of a customer: ', list_of_states)
    print('All state results (vector): ', nextState_results, '\n')

    return list_of_states

# calc_allStates()
# exit()



def customerInSupermarket():
    """various customers move their locations in the supermarket"""

    cust_resultsDict = {}

    # 3 customers in the supermarket
    for customer in range(7):
        print('Customer No: ', customer)
        res = calc_allStates()
        cust_resultsDict.update({customer: res})

    print(f'Result dictionary for all customers: {cust_resultsDict}')

    show_paths = sm.migrate_paths(cust_resultsDict)
    sm.show_supermarket(show_paths)



customerInSupermarket()
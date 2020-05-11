import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#import seaborn as sns
import random
import transMatrix_df


# We have a customer!
# We have a transition matrix and initial distribution of first aisles.

# ### Example.
# I have 100 customers:


# #### Step 1
# Between 7-9am a customer comes every 10 minutes.
# Between 9-11 there is a customer coming every minute.

# #### Step 2
# 40 are in fruits first (aisle 0)
# 30 are in spices first (aisle 1)
# 20 are in dairy first (aisle 2)
# 10 are in drinks first (aisle 3)

# Write a function that gives random integers (numpy has a function to generate random integers) between 0 and 3.
# So if 0 then send the customer to fruit first. At the end, you can put if conditions so that if you already have
# 44 customers in fruits you generate a new random number so that they don't go to fruits any more).'

# #### Step 3
# Transition matrix to follow one customer. And then many...

# #### Step 4
# Visualise this day for your bosses to show how at each hour customers moved....



def create_tpMatrix():
    """load DataFrame and convert to Transition Probability Matrix 'tpm' """
    df = transMatrix_df.create_tpm()
    tpm = np.array(df)
    return tpm
#     print(tpm)
#
# create_tpMatrix()
# exit()


def get_randomState(currentState):
    states = ['dairy', 'drinks', 'fruit', 'spices', 'checkout']
    entry = random.randint(0,4)
    # if current state is next state (entry) --> get another next state
    while currentState == states[entry]:
        entry = random.randint(0, 4)
    dict_initState = {'dairy': [0, 1, 0, 0, 0, 0],
                     'drinks': [0, 0, 1, 0, 0, 0],
                      'fruit': [0, 0, 0, 1, 0, 0],
                     'spices': [0, 0, 0, 0, 1, 0],
                   'checkout': [0, 0, 0, 0, 0, 0]}

    return states[entry],dict_initState[states[entry]]

# initState, initState_vector = get_randomState('fruit')
#print(initState, initState_vector)


def calc_allStates():
    """constructing Markov-Chain according to following equation:
       next state = transition possibility * initial state"""

    # Transition Probability Matrix 'tpm' as numpy array
    tpm = create_tpMatrix()

    # set default initial state for every customer: 'entrance'
    currentVector = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]       # vector for 'entrance'
    currentState = 'entrance'

    nextState_results = []
    nextStateVector_results = []
    while currentState != 'checkout':
        #print('point 1: ', currentState)
        isv = currentVector
        # assumption: tpm does not change, state is assumed to be stable
        next_state = tpm * isv
        print('point 5: ', next_state)

        # for the whole nextState matrices
        nextState_results.append(np.array(next_state).flatten())

        # print('point 2: ', currentState, '\n', next_state)
        currentState, currentVector = get_randomState(currentState)
        # print('point 3: ', currentState)
    # print('point 4: ', nextState_results)

    return nextState_results

# nextStateResults = calc_allStates()
# print(nextStateResults)



def customerInSupermarket():
    """various customers move their locations in the supermarket"""

    cust_resultsDict = {}

    # 10 customers in the supermarket
    for customer in range(3):
        res = calc_allStates()
        cust_resultsDict.update({customer: res})
        print('Customer No: ', customer)
    print('customerSup')
    print(cust_resultsDict)

customerInSupermarket()
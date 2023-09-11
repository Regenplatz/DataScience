### Project Desciption

This project aimed to simulate customer movements in a supermarket.


##### Data

Data was obtained from Spiced Academy as five separate csv files, each displaying customer movements on one day of the week.

Each of the files (Mon-Fri) consisted of the following features:
- timestamp
- customerID
- location

As locations customers could move to within the supermarket, the following were given:
- fruit
- dairy
- drinks
- spices
- checkout

#  
##### General approach

For `Markov Chain` calculations, at first the transition probability matrix need to be calculated based on the actual data. Then, matrix multiplication with the `transition probability matrix` and the `initial state vector` (the customer's *actual location*) is performed to finally obtain the respective `end state vector` (which means the *next location* the customer moves to).

##### First data processing
For calculating the transition probability matrix, the five csv files were read to pandas as DataFrame and initially edited (e.g. new customer IDs ('UniqueID') were generated as customerNo 1 on Monday was not the same as customerNo1 on Tuesday). Afterwards, those 5 DataFrames were merged to one single DataFrame for further processing (see [df_united.py](df_united.py)).


##### The Transition Probability Matrix

In this matrix, rows display the initial state while columns symbolize the next location. To display an n x n matrix where every state can be found as initial state (row) as well as next state (column), the matrix was appropriately expanded (including 'entrance' and 'checkout').
In this project, the transition probability matrix (see [transMatrix_df.py](transMatrix_df.py)) was assumed to be stationary.


#  
### Project

For this project, different approaches were tried and might be extended later:

- [explorativeDA.py](explorativeDA.py): <br>
For initial data analysis various functions were built. This file does not contain a Markov Chain model.

- [simulationSupermarket_rc.py](simulationSupermarket_rc.py): (*rc* for *total random choice*)     
Get every next state (next location) by random choice, which means that the last calculated next state was not set as the initial state for the next step

- [simulationSupermarket_mc.py](simulationSupermarket_mc.py): (*mc* for *markov chain calculations*)    
Only get the first state (behind the entrance) by random choice and calculate all further customer movements by setting the last calculated next state as initial state for the next calculation. As a consequence, the movement will be the same for customers with the same first location.


#   
### Visualizing Simulation
For both simulations [simulationSupermarket_rc.py](simulationSupermarket_rc.py) and [simulationSupermarket_mc.py](simulationSupermarket_mc.py) the file
[supermarket.png](supermarket.png) was used as visual basis. Customers were presented by using png files of *frau* (woman) and *mann* (man).

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns
import df_united


def create_tpm():

    df = df_united.load_files()

    # create new columns
    df['location_next'] = df.groupby(df['UniqueID'])['location'].transform('shift',periods=-1) #.loc[df['UniqueID']=='Tu2']
    df['location_before'] = df.groupby(df['UniqueID'])['location'].transform('shift',periods=+1)
    #df.loc[df['UniqueID']=='Tu2']

    # replace NaN in feature 'location_before' with 'entrance'
    df = df.fillna(value={'location_before':'entrance'})

    # create crosstab for 'location_before' and 'location'
    df_cross_entrance = pd.crosstab(df.location_before, df.location, normalize='index') #, margins=True)

    # create new column for entrance with values 0 (as nobody moves from other locations to entrance)
    entr_values = [0.0,0.0,0.0,0.0,0.0]
    df_cross_entrance['entrance'] = entr_values

    # need to append a row for checkout (all values are set to zero as you cannot move from checkout to another location)
    checkout_dict = {'checkout':0.0,'dairy':0.0,'drinks':0.0,'fruit':0.0,'spices':0.0, 'entrance':0.0}
    #df_cross_entrance
    df_cross_entrance.loc['checkout']=checkout_dict

    # sort matrix to get a symmetrical matrix
    # therefore, a new feature 'feature_sort' is created as helper. Afterwards this column is deleted
    df_cross_entrance['feature_sort'] = [1, 2, 5, 3, 4, 0]
    df_cross_entrance.sort_values(by=['feature_sort'], inplace=True)
    df_cross_entrance.drop(columns=['feature_sort'], inplace=True)

    # # save crosstab as excel file
    # df_cross_entrance.to_excel('transMatrix_df.xlsx')
    #
    # # grouped bar plot
    # stacked = df_cross_entrance.stack().reset_index().rename(columns={0:'value'})
    # sns.barplot(x=stacked.location_before, y=stacked.value,hue=stacked.location)

    return df_cross_entrance

    # show final crosstab
#     print(df_cross_entrance)
#
#
# create_tpm()



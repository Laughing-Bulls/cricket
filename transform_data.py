# This program takes the raw data as a dataframe and returns the pre-processed inputs.
import numpy as np
import pandas as pd
#import sklearn
#from sklearn.compose import ColumnTransformer
#from sklearn import preprocessing
#from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split


def label_transform(df):
    # transform categories (e.g., venues, teams) into categorical integers
    print("label transform")
    print(df.head())
    coded = pd.get_dummies(data=df, columns=['team1', 'team2', 'outcome','winner','toss winner', 'toss decision',])



    #print(df.loc[1, 'batsman'])
   # df = df.drop(columns = columnsRemoved)

 
   # print(df.head())
    # ENCODE toss winner, toss decision, outcome,
    #ct = ColumnTransformer([('encode', OneHotEncoder(categories='auto'))], remainder='passthrough') # column transformation
    #transformed = np.array(ct.fit_transform(df), dtype=np.str)
    #transformed_df = pd.DataFrame(transformed) # convert back to dataframe
    #le = preprocessing.LabelEncoder()
    #encoded = df.apply(le.fit_transform)  # each unique category assigned an integer
    #enc = preprocessing.OneHotEncoder()
    #enc.fit(encoded)
    #transformed = enc.transform(encoded).toarray()
    # convert each category into its own column with value 0 or 1
    print('Transform Data: The shape of the transformed array is:', coded.shape)
    return coded


def transform(df):
    # Transforms the raw data.
    print('Transform Data: The raw data headers are:')
    print(df.columns)
    columnsRemoved = ['batsman','non striker', 'bowler', 'extras', 'extra kind', 'wicket kind', 'player out', 'fielders', 
    'by', 'player of match','match type', 'venue', 'city', 'gender', 'umpire1', 'umpire2']
    df.drop(columnsRemoved, axis =1, inplace = True)
    consistent_teams = ['Kings XI Punjab', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad', 'Chennai Super Kings',
    'Kolkata Knight Riders', 'Delhi Daredevils', 'Rajasthan Royals', 'Mumbai Indians']

    df = df[(df['team1'].isin(consistent_teams)) & (df['team2'].isin(consistent_teams)) & (df['winner'].isin(consistent_teams)) 
    & (df['toss winner'].isin(consistent_teams))] 


    print(df.head())
    df = df[df['delivery'] >= 5.0]  # Remove pitches during first 5 overs
    df = df.select_dtypes(include=[object])  # isolate categorical data
    transformed = label_transform(df)  # function returns transformed array
    #print(transformed.head())
    #print('Transform Data: The transformed data size is:')
    #print(transformed.shape)
    return df
    #return transformed



def split_data(df):
    y = df["total"]
    X = df.drop(labels="total", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
    print("Data successfully split.")
    print("X_train:", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)
    return X_train, X_test, y_train, y_test

# This program takes the raw data as a dataframe and returns the pre-processed inputs.
import numpy as np
import pandas as pd
#import sklearn
#from sklearn.compose import ColumnTransformer
#from sklearn import preprocessing
#from sklearn.preprocessing import Normalizer, OneHotEncoder


def read_data():
    path = './data/'
    name = 'ipldata.csv'
    df = pd.read_csv(path + name, header=0)
    return df


def write_data(df):
    path = './data/'
    name = 'cricket-processed-data.csv'
    df.to_csv(path + name)
    return True


def prepare_input():
    # Calls functions to read, process and transform raw data, and save it as csv
    raw_data = read_data()
    print("transform_data: Raw data: ")
    print(raw_data.head())  # Print raw data
    print(raw_data.shape)
    cricket_input = transform(raw_data)
    print("transform_data: The processed cricket inputs matrix: ")
    print(cricket_input.columns)
    print(cricket_input.head())  # Print transformed data
    print(raw_data.shape)
    write_data(cricket_input)
    return True


def transform(df):
    # Transforms the raw data.
    print('transform_data: The raw data headers are:')
    print(df.columns)
    columnsRemoved = ['batsman','non striker', 'bowler', 'extras', 'extra kind', 'wicket kind', 'player out', 'fielders', 
    'by', 'player of match','match type', 'venue', 'city', 'gender', 'umpire1', 'umpire2']
    df.drop(columnsRemoved, axis =1, inplace = True)
    df.drop(columns=df.columns[0], axis = 1, inplace = True)
    print('Columns Dropped:::')
    print(df.columns)
    consistent_teams = ['Kings XI Punjab', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad', 'Chennai Super Kings',
    'Kolkata Knight Riders', 'Delhi Daredevils', 'Rajasthan Royals', 'Mumbai Indians']
    df = df[(df['team1'].isin(consistent_teams)) & (df['team2'].isin(consistent_teams))] 
    print(df.head())
    #df.drop(index=df.index[1:300000], inplace=True)  # Make data set smaller for testing
    #df.drop(labels='date', axis=1, inplace=True)  # Remove date column
    df = df[df['delivery'] >= 5.0]  # Remove pitches during first 5 overs
    transformed = label_code(df)  # function returns transformed array
    return transformed


def label_code(df):
    # transform categories (e.g., venues, teams) into categorical integers
    # convert each category into its own column with value 0 or 1
    coded = pd.get_dummies(data=df, columns=['team1', 'team2', 'outcome','winner','toss winner', 'toss decision',])
    #ct = ColumnTransformer([('encode', OneHotEncoder(categories='auto'))], remainder='passthrough') # column transformation
    #transformed = np.array(ct.fit_transform(df), dtype=np.str)
    #transformed_df = pd.DataFrame(transformed) # convert back to dataframe
    #le = preprocessing.LabelEncoder()
    #encoded = df.apply(le.fit_transform)  # each unique category assigned an integer
    #enc = preprocessing.OneHotEncoder()
    #enc.fit(encoded)
    #transformed = enc.transform(encoded).toarray()
    print('transform_data: The shape of the transformed array is:', coded.shape)
    return coded

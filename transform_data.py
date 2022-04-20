# This program takes the raw data as a dataframe and returns the pre-processed inputs.
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split

def label_transform(df):
    # transform categories (e.g., venues, teams) into categorical integers
    coded = pd.get_dummies(data=df, columns=['venue', 'bat_team', 'bowl_team'])
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
    df.drop(index=df.index[1:340000], inplace=True)  # Make data set smaller for testing
    df.drop(labels='date', axis=1, inplace=True)  # Remove date column
    df = df[df['overs'] >= 5.0]  # Remove pitches during first 5 overs
    #df = df.select_dtypes(include=[object])  # isolate categorical data
    transformed = label_transform(df) # function returns transformed array
    print(transformed.head())
    #print('Transform Data: The transformed data size is:')
    #print(transformed.shape)
    return transformed

def split_data(df):
    y = df["total"]
    X = df.drop(labels="total", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
    print("Data successfully split.")
    print("X_train:", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)
    return X_train, X_test, y_train, y_test
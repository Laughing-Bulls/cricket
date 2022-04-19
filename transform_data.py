# This program takes the raw data as a dataframe and returns the pre-processed inputs.
import numpy as np
import pandas as pd


def transform(df):
    # Transforms the raw data.
    print('Transform Data: The raw data headers are:')
    print(df.columns)
    df.drop(labels='date', axis=1, inplace=True)  # Transforms the data
    print('Transform Data: The transformed data is:')
    print(df.head())  # Print transformed data.
    return df

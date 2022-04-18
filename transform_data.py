# This program takes the raw data as a dataframe and returns the pre-processed inputs.
import numpy as np
import pandas as pd


def transform(df):
    # Transforms the raw data.
    processed = df / 2  # Transforms the data
    print('Transform Data: The transformed data is:')
    print(processed)  # Print transformed data.
    return processed

# This is the MAIN Python script. It provides the user interface to run subroutines.
import numpy as np
import pandas as pd

from transform_data import transform
from transform_data import split_data
from ml_models import construct_model
from final_model import predict_score

def read_data(name):
    path = './data/'
    df = pd.read_csv(path + name, header=0)
    return df

def prepare_input():
    # Returns the answer.
    raw_data = read_data("cricket-raw-data.csv")
    print("Main: Raw data: ")
    print(raw_data.head())
    print(raw_data.shape)
    cricket_input = transform(raw_data)
    print("Main: The cricket inputs matrix: ")
    print(cricket_input.head())  # Print transformed data.
    return cricket_input

    """ 

    pred = predict_score(input)
    print('Main: The calculated prediction is:', pred)  # Print final answer."""


# Runs the script.
if __name__ == '__main__':
    input = prepare_input()
    construct_model(input)


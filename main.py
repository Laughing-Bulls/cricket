# This is the MAIN Python script. It provides the user interface to run subroutines.
import numpy as np
import pandas as pd

from transform_data import transform
from final_model import predict_score

def read_data(name):
    path = './data/'
    df = pd.read_csv(path + name, header=0)
    return df

def retrieve_result():
    # Returns the answer.
    raw_data = read_data("cricket-raw-data.csv")
    print("Main: Raw data: ")
    print(raw_data.head())
    cricket_input = transform(raw_data)
    print("Main: The cricket inputs matrix: ")
    print(cricket_input.head())

    """ 


    pred = 0
    input = np.zeros((1, 4))
    input[0, 0] = cricket_input.iat[0, 0]
    input[0, 1] = cricket_input.iat[0, 1]
    input[0, 2] = cricket_input.iat[0, 2]
    input[0, 3] = cricket_input.iat[0, 3]
    print("Main: The inputs are: ", input)
    pred = predict_score(input)
    print('Main: The calculated prediction is:', pred)  # Print final answer."""


# Runs the script.
if __name__ == '__main__':
    retrieve_result()


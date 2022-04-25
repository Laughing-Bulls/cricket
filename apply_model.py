# This contains the final trained model. It supplies the predicted answer given provided inputs.
import numpy as np
import pandas as pd
import pickle

def read_model():
    path = './model/'
    filename = 'final_model.sav'
    saved_model = pickle.load(open(path + filename, 'rb'))
    return saved_model

def predict_score():
    # Calculates the predicted score, based on saved ML model
    model = read_model()
    print("apply_model: The Model parameters are from the currently saved model.")
    print("apply_model: The Model weightings are: ")
    """ 
    model_inputs = match_data()
    print(model_parameters)

    pred = input + var
    print('apply_model: The model predicts:', pred)  # Print final answer.

        pred = predict_score(input)
        print('apply_model: The calculated prediction is:', pred)  # Print final answer.
        """
    result = 100
    # print(type(input_data))
    # print(input_data.head())
    # print(input_data.loc[[300004]])
    # print(input_data.iloc[300004]) - dont work
    print("The predicted score is: ", result)
    return True


def match_data():
    print("apply_model: This match data is from the training set.")
    print("apply_model: The inputs are: ")
    X = []
    X = [1 for i in range(108)]
    print(X)
    return X


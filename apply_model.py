# This contains the final trained model. It supplies the predicted answer given provided inputs.
import numpy as np
import pandas as pd
import pickle
from ml_models import read_input_file

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
    print(model.coef_)

    model_inputs = match_data()
    # print(type(input_data))

    prediction = model.predict(model_inputs)
    print("apply_model: The predicted score is: ", prediction)
    return True


def match_data():
    # obtains a sample set of inputs from the processed training data
    print("apply_model: This match data is from the training set.")
    all = read_input_file()
    X = all.iloc[[201]]
    y = X["total"]
    X = X.drop(labels="total", axis=1)  # remove output column
    print("apply_model: The inputs are: ")
    print(X)
    print("apply_model: The actual score was: ", y.values[0])
    return X


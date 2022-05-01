# This contains the final trained model. It supplies the predicted answer given provided inputs.
import numpy as np
import pandas as pd
import pickle
from ml_models import read_input_file
from sklearn.metrics import explained_variance_score

def read_model():
    path = './model/'
    filename = 'final_model.sav'
    saved_model = pickle.load(open(path + filename, 'rb'))
    return saved_model

def predict_score():
    # Calculates the predicted score, based on saved ML model
    model = read_model()
    print("apply_model: The Model parameters are from the currently saved model.")
    try:
        print("apply_model: The Model weightings are: ")
        print(model.coef_)
    except:
        print("apply_model: The saved model does not have linear weightings.")
    match = input("Do you want to predict a single match (m) or the entire 2007 season (2007)? ")
    if match == 'm':
        model_inputs = match_data()
        prediction = model.predict(model_inputs)
        print("apply_model: The predicted score is: ", prediction)
    else:
        model_inputs, actual_y = season_data()
        y_prediction = model.predict(model_inputs)
        print("Explained Variance Score: ", explained_variance_score(actual_y, y_prediction))

    return True


def match_data():
    # obtains a sample set of inputs from the processed training data
    print("apply_model: This match data is from the training set.")
    all = read_input_file()
    X = all.loc[[125541]]
    y = X["total"]
    X = X.drop(labels="total", axis=1)  # remove output column
    print("apply_model: The inputs are: ")
    print(X)
    print("apply_model: The actual score was: ", y.values[0])
    return X


def season_data():
    # obtains the 2007 samples from the processed training data
    print("apply_model: This match data is from the training set.")
    all = read_input_file()
    X = all.iloc[300549:313653]
    y = X["total"]
    X = X.drop(labels="total", axis=1)  # remove output column
    print("apply_model: The inputs are: ")
    print(X)
    print("apply_model: The actual scores were: ", y.values[0])
    return X, y
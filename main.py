# This is the MAIN Python script. It provides the user interface to run subroutines.
import numpy as np
import pandas as pd

from transform_data import transform
#from transform_data import split_data
from ml_models import construct_model
#from final_model import predict_score


def read_data():
    path = './data/'
    name = "cricket-raw-data.csv"
    df = pd.read_csv(path + name, header=0)
    return df


def prepare_input():
    # Returns the answer.
    raw_data = read_data()
    print("Main: Raw data: ")
    print(raw_data.head())
    print(raw_data.shape)
    cricket_input = transform(raw_data)
    print("Main: The cricket inputs matrix: ")
    print(cricket_input.columns)
    print(cricket_input.head())  # Print transformed data.
    return cricket_input


def user_input():
    choice = input("Select a model number (1,2,3,4,5,6,7,8): ")  # user can choose ML model
    model_name = "quit"
    if choice == '1':
        model_name = "Gaussian Naive Bayes"
    if choice == '2':
        model_name = "Support Vector Classifier"
    if choice == '3':
        model_name = "Random Forest"
    if choice == '4':
        model_name = "Linear Regression"
    if choice == '5':
        model_name = "Lasso Regression"
    if choice == '6':
        model_name = "Stochastic Gradient Descent"
    if choice == '7':
        model_name = "Baysian Ridge Regression"
    if choice == '8':
        model_name = "Ridge Regression"
    if choice == '9':
        model_name = "Linear SVR"
    print("You chose: ", choice, model_name)
    return model_name


# Runs the script.
if __name__ == '__main__':
    input_data = prepare_input()  # load and prepare input data
    choice = user_input()
    while choice != 'quit':
        model = construct_model(input_data, choice)  # run and evaluate selected model
        choice = user_input()
    print("That's all, Folks!")

    """ 
    pred = predict_score(input)
    print('Main: The calculated prediction is:', pred)  # Print final answer.
    """

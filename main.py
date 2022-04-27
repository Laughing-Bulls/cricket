# This is the MAIN Python script. It provides the user interface to run subroutines.
import numpy as np
import pandas as pd
from transform_data import prepare_input
from ml_models import construct_model
from apply_model import predict_score


def model_choice():
    choice = input("Select a model number (1,2,3,4,5,6,7,8,9,10,11): ")  # user can choose ML model
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
    if choice == '10':
        model_name = "KNN Regression"
    if choice == '11':
        model_name = "Decision Tree"
    print("You chose: ", choice, model_name)
    return model_name


# Runs the script.
if __name__ == '__main__':

    input_choice = input("Do you want to transform a new dataset? (y/n)")  # process raw data or not?
    if input_choice == "y":
        prepare_input()  # load and prepare raw data

    training_choice = input("Do you want to train a new model? (y/n)")  # process raw data or not?
    if training_choice == "y":
        choice = model_choice()
        while choice != 'quit':
            model = construct_model(choice)  # run and evaluate selected model
            choice = model_choice()

    output_choice = input("Do you want to apply the model to data? (y/n)")  # apply the saved model?
    if output_choice == "y":
        predict_score()

    print("That's all, Folks!")


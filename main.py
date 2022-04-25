# This is the MAIN Python script. It provides the user interface to run subroutines.
import numpy as np
import pandas as pd
import pickle
from transform_data import prepare_input
from ml_models import construct_model


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


def runtest():
    filename = "final_model.sav"
    saved_model = pickle.load(open(filename, 'rb'))
    X = []
    X = [1 for i in range(108)]
    #result = X
    result = saved_model.predict(X)
    print("The predicted score is: ", result)
    return True


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
        #print(type(input_data))
        #print(input_data.head())
        #print(input_data.loc[[300004]])
        #print(input_data.iloc[300004]) - dont work
        runtest()

    print("That's all, Folks!")

    """ 
    pred = predict_score(input)
    print('Main: The calculated prediction is:', pred)  # Print final answer.
    """

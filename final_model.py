# This contains the final trained model. It supplies the predicted answer given provided inputs.
import numpy as np
import pandas as pd

def read_model(name):
    path = './model/'
    df = pd.read_csv(path + name)
    return df

def predict_score(input):
    # Calculates the predicted answer.
    model_parameters = read_model("parameters.csv")
    var = np.zeros((1, 4))
    print("Predict Score: The Model parameters: ")
    print(model_parameters)
    var[0, 0] = model_parameters.iat[0, 0]
    var[0, 1] = model_parameters.iat[0, 1]
    var[0, 2] = model_parameters.iat[0, 2]
    var[0, 3] = model_parameters.iat[0, 3]
    print("Predict Score: The Model weightings: ", var)
    pred = input + var
    print('Final Model: The model predicts:', pred)  # Print final answer.
    return pred

# This contains the final trained model. It supplies the predicted answer given provided inputs.
import numpy as np
import pandas as pd


def predict_score(input1, input2, input3):
    # Calculates the predicted answer.
    pred = input1 + input2 + input3
    print('Final Model: The model predicts:', pred)  # Print final answer.
    return pred

# This program runs the selected machine learning model on the pre-processed inputs.
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def split_data(df):
    # split data into ttraining and test data (80% / 20%)
    y = df["total"]
    X = df.drop(labels="total", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
    print("ML_Models: Data successfully split.")
    print("X_train:", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)
    return X_train, X_test, y_train, y_test

def construct_model(input):
    # split data and run through ML model
    X_train, X_test, y_train, y_test = split_data(input)
    print("ML_Models: y_test")
    print(y_test)
    ridge = Ridge(alpha=1)
    ridge.fit(X_train, y_train)
    print("ML_Models: Ridge regression model built.")
    print(ridge.get_params())
    ridge.predict(X_test)
    print("R-squared accuracy: ", ridge.score(X_test, y_test))
    return


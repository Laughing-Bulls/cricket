# This program runs the selected machine learning model on the pre-processed inputs.
import numpy as np
import pandas as pd
#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score


def split_data(df):
    # split data into training and test data (80% / 20%)
    y = df["total"]
    X = df.drop(labels="total", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
    print("ML_Models: Data successfully split.")
    print("X_train:", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)
    return X_train, X_test, y_train, y_test


def construct_model(input_data, model_name):
    # call split data function and run through ML model
    X_train, X_test, y_train, y_test = split_data(input_data)

    nonlinear = False
    if model_name == "Gaussian Naive Bayes":
        model = GaussianNB()
        nonlinear = True
    if model_name == "Support Vector Classifier":
        model = SVC(C=100, kernel='rbf')
        # alternative => SVC(C=100, kernel='poly', degree=3)
        nonlinear = True
    if model_name == "Random Forest":
        model = RandomForestClassifier(max_depth=10, random_state=0)
        nonlinear = True
    if model_name == "Lasso Regression":
        model = Lasso(alpha=1.0)
    if model_name == "Stochastic Gradient Descent":
        model = SGDRegressor(max_iter=1000, tol=1e-3)
    if model_name == "Baysian Ridge Regression":
        model = BayesianRidge()
    if model_name == "Ridge Regression":
        model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)
    print("ML_Models: ", model_name, " model built.")
    print(model.get_params())
    y_pred = model.predict(X_test)
    if nonlinear:
        analyze_model(y_test, y_pred)
    else:
        print(model_name, ": R-squared accuracy: ", model.score(X_test, y_test))
    return


def analyze_model(y_test, y_pred):
    print("Prediction Accuracy: %1.3f" %accuracy_score(y_test, y_pred))
    return

# This program runs the selected machine learning model on the pre-processed inputs.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def read_input_file():
    path = './data/'
    name = 'cricket-processed-data.csv'
    df = pd.read_csv(path + name, index_col=0)
    return df


def construct_model(model_name):
    # get input file, call split data function, and train model with selected ML algorithm
    input_data = read_input_file()
    X_train, X_test, y_train, y_test = split_data(input_data)

    classification = False
    if model_name == "Gaussian Naive Bayes":
        model = GaussianNB()
        classification = True
    if model_name == "Support Vector Classifier":
        model = SVC(C=100, kernel='rbf')
        # alternative => SVC(C=100, kernel='poly', degree=3)
        classification = True
    if model_name == "Random Forest":
        model = RandomForestClassifier(max_depth=10, random_state=0)
        classification = True
    if model_name == "Linear Regression":
        model = LinearRegression()
    if model_name == "Lasso Regression":
        model = Lasso(alpha=1.0)
    if model_name == "Stochastic Gradient Descent":  #KEEP
        model = SGDRegressor(max_iter=1000, tol=1e-3)
    if model_name == "Baysian Ridge Regression":
        model = BayesianRidge()
    if model_name == "Ridge Regression":
        model = Ridge(alpha=1.0)
    if model_name == "Linear SVR":
        model = LinearSVR()
    if model_name == "KNN Regression":
        model = KNeighborsRegressor()
    if model_name == "Decision Tree":
        model = tree.DecisionTreeRegressor()

    model.fit(X_train, y_train)
    print("ML_Models: ", model_name, " model built.")
    print(model.get_params())
    y_pred = model.predict(X_test)
    analyze_model(classification, y_test, y_pred)
    save_model(model_name, model)
    return True


def split_data(df):
    # split data into training and test data (80% / 20%)
    y = df["total"]
    X = df.drop(labels="total", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    print("ML_Models: Data successfully split.")
    print("X_train:", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)
    return X_train, X_test, y_train, y_test


def analyze_model(classification, y_test, y_pred):
    # tests classification models and regression models
    if classification:
        #print("Precision: %.3f" % precision_score(y_test, y_pred))
        #print("Recall: %.3f" % recall_score(y_test, y_pred))
        #print("F1 Score: %.3f" % f1_score(y_test, y_pred))
        print("Prediction Accuracy: %.3f" % accuracy_score(y_test, y_pred))
    else:
        #print("Model score: ", model.score(X_test, y_test))
        print("R-squared: ", r2_score(y_test, y_pred))
        print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
        print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
        print("Explained Variance Score: ", explained_variance_score(y_test, y_pred))
        runplot(y_test, y_pred)
    return True


def runplot(y_test, y_pred):
    # plots graph of actual vs. predicted
    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_test, edgecolors=(0, 0, 1))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()
    return True


def save_model(choice, model):
    path = './model/'
    filename = "final_model.sav"
    parameter_filename = "parameters.csv"
    yes = input("Save this model? (y/n)")
    if yes == "y":
        pickle.dump(model, open(path + filename, 'wb'))
        print(choice, " : Model has been saved as final_model.sav")
        weights = model.coef_
        np.savetxt(path + parameter_filename, weights, delimiter=',')
        print("Model weights/coefficients have been saved as parameters.csv")
    return True

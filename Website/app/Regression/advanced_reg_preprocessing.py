"""
Author: Vishal Kundar

advanced_reg_preprocessing.py is used for advanced techniques to improve 
model performace. 

The techniques used are:
    1. Feature selection 
    2. Hyperparameter tuning    
"""
# Packages
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
import copy  # To deep copy model with parameters
import math
import numpy as np

# models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
import xgboost as xgb


def hyperparameter_tuning(data, result):
    """
    This will be carried out only for specific models:
        1. XGBoost
        2. Adaboost
        3. Decision Tree
        4. Random forest
        5. KNN
        6. SVM
    """
    X_train, y_train = data["Xtrain"], data["Ytrain"]
    X_test, y_test = data["Xtest"], data["Ytest"]
    X, Y = data["X"], data["Y"]

    # XGboost
    print("XGBoost tuning..")
    tuned_parameters = [
        {
            "max_depth": [5, 15, 25, 30],
            "learning_rate": [0.001, 0.01, 0.1],
            "n_estimators": [100, 200, 300],
        }
    ]
    search = GridSearchCV(
        result["xgbreg"],
        tuned_parameters,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=3,
    )
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)

    r2 = round((r2_score(y_test, y_pred)), 3)
    adjr2 = round((1 - (1 - r2) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)), 3)

    if result["performance"]["xgbreg"]["adj_r2"] < adjr2:
        result["xgbreg"] = search

    # AdaBoost
    print("AdaBoost tuning..")
    tuned_parameters = [
        {
            "learning_rate": [0.1, 1, 2, 3, 4, 5],
            "n_estimators": [100, 200, 300, 400, 500],
        }
    ]
    search = GridSearchCV(
        result["adbreg"],
        tuned_parameters,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=3,
    )
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)

    r2 = round((r2_score(y_test, y_pred)), 3)
    adjr2 = round((1 - (1 - r2) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)), 3)

    if result["performance"]["adbreg"]["adj_r2"] < adjr2:
        result["adbreg"] = search

    # Decision Tree
    print("Decision tree tuning..")
    tuned_parameters = [{"max_depth": [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200]}]
    search = GridSearchCV(
        result["decreg"],
        tuned_parameters,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=3,
    )
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)

    r2 = round((r2_score(y_test, y_pred)), 3)
    adjr2 = round((1 - (1 - r2) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)), 3)

    if result["performance"]["decreg"]["adj_r2"] < adjr2:
        result["decreg"] = search

    # Random Forests
    print("Random forest tuning..")
    tuned_parameters = [
        {
            "max_depth": [5, 10, 15, 20, 50, 70],
            "n_estimators": [10, 25, 50, 100, 150, 200, 250],
        }
    ]
    search = GridSearchCV(
        result["forestreg"],
        tuned_parameters,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=3,
    )
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)

    r2 = round((r2_score(y_test, y_pred)), 3)
    adjr2 = round((1 - (1 - r2) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)), 3)

    if result["performance"]["forestreg"]["adj_r2"] < adjr2:
        result["forestreg"] = search

    # KNN
    print("KNN tuning..")
    tuned_parameters = [{"n_neighbors": [1, 2, 3, 4, 5, 10, 15, 20], "p": [1, 2]}]
    search = GridSearchCV(
        result["knnreg"],
        tuned_parameters,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=3,
    )
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)

    r2 = round((r2_score(y_test, y_pred)), 3)
    adjr2 = round((1 - (1 - r2) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)), 3)

    if result["performance"]["knnreg"]["adj_r2"] < adjr2:
        result["knnreg"] = search

    # SVM
    print("SVM tuning..")
    tuned_parameters = [{"C": [1, 2, 3, 5, 6], "gamma": [0.0001, 0.001, 0.01, 0.1]}]
    search = GridSearchCV(
        result["svmreg"],
        tuned_parameters,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=3,
    )
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)

    r2 = round((r2_score(y_test, y_pred)), 3)
    adjr2 = round((1 - (1 - r2) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)), 3)

    if result["performance"]["svmreg"]["adj_r2"] < adjr2:
        result["svmreg"] = search

    return result


def feature_selection(data, result):
    """
    We carry out feature selection using mutual information statistics
    Mutual information statistics: Mutual information from the field of information 
    theory is the application of information gain to feature selection. 
    
    Mutual information is calculated between two variables and measures the 
    reduction in uncertainty for one variable given a known value of the other 
    variable.
    
    IMPORTANT: Will be using adj r2 for evaluation
    
    Returns
    --------
    updated models
    """
    # We will discard 20% of redundant columns
    fs = SelectKBest(score_func=mutual_info_regression)
    X = data["X"]
    Y = data["Y"]
    X_train, y_train = data["Xtrain"], data["Ytrain"]
    X_test, y_test = data["Xtest"], data["Ytest"]
    num_discarded_features = math.ceil(X.shape[1] * 0.2)

    if X.shape[1] < 4:
        # If less than 4 features exist then do not remove any features
        return result

    # defining grid
    grid = dict()
    grid["sel__k"] = [
        i for i in range(X.shape[1] - num_discarded_features, X.shape[1] + 1)
    ]

    """
    we will evaluate models using the negative mean absolute error. It is 
    negative because the scikit-learn requires the score to be maximized, so 
    the MAE is made negative, meaning scores scale from -infinity to 0 (best).
    """
    for key in result.keys():

        if key != "performance":
            # Setting pipeline
            model = copy.deepcopy(result[key])
            pipeline = Pipeline(steps=[("sel", fs), ("model", model)])

            # defining grid search
            search = GridSearchCV(
                pipeline, grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=3
            )

            # Fitting model and predicting result
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)

            # Checking adj r2
            r2 = round((r2_score(y_test, y_pred)), 3)
            adjr2 = round((1 - (1 - r2) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)), 3)

            # if model is better we replace it else leave the old model
            if result["performance"][key]["adj_r2"] < adjr2:
                result[key] = search

    return result


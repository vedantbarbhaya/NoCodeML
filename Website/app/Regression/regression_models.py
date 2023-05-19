"""
Author: Vishal Kundar

regression_models.py is used to run various regression models on the data.
The results of the models is used to determine further data processing steps.

Imports file advanced_reg_preprocessing.py that is used to carry out futher 
methods to improve model performance.

The models being used are:
    1. Linear Regression
    2. Decision tree regressor
    3. Random forest regressor
    4. KNN regressor
    5. Adaboost regressor
    6. XGBoost regressor
    7. CatBoost regressor
    8. SVM regressor
    
Models with hyperparameters will be tuned using Gridsearch

The metrics used to evaluate the models are:
    1. R-squared
    2. Adjusted R-squared
    3. Mean Square Error
    4. Root Mean Square Error
"""
# Packages
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import advanced_reg_preprocessing as arp

# models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
import xgboost as xgb
import catboost as cb


def model_performance(result, data):
    # Function to check metrics like R2, adjusted R2, MSE and RMSE
    X_test, y_test = data["Xtest"], data["Ytest"]
    X, y = data["X"], data["Y"]

    performance = {}
    for key in result.keys():
        if key != "performance":
            y_pred = result[key].predict(X_test)

            # Metrics
            r2 = round((r2_score(y_test, y_pred)), 3)
            adj_r2 = round((1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)), 3)
            mse = mean_squared_error(y_test, y_pred)
            rmse = round(np.sqrt(mse))

            # Storing results
            temp = {"r2": r2, "adj_r2": adj_r2, "mse": mse, "rmse": rmse}
            performance[key] = temp

    return performance


def run_models(data, var):
    # var tells which model to run. 'first-run' runs all models
    # only the improved models are added and rest are discarded.
    x_train, y_train = data["Xtrain"], data["Ytrain"]

    if var == "first-run":
        # Linear regression
        linreg = LinearRegression()
        linreg.fit(x_train, y_train)

        # Decision tree regressor
        decreg = DecisionTreeRegressor()
        decreg.fit(x_train, y_train)

        # Random forest regressor
        forestreg = RandomForestRegressor()
        forestreg.fit(x_train, y_train)

        # KNN regressor
        knnreg = KNeighborsRegressor()
        knnreg.fit(x_train, y_train)

        # Adaboost regressor
        adbreg = AdaBoostRegressor()
        adbreg.fit(x_train, y_train)

        # XGBoost regressor
        xgbreg = xgb.XGBRegressor()
        xgbreg.fit(x_train, y_train)

        # Catboost regressor
        cbreg = cb.CatBoostRegressor(logging_level="Silent")
        cbreg.fit(x_train, y_train)

        # SVM regressor
        svmreg = SVR()
        svmreg.fit(x_train, y_train)

        # Adding to dictionary as base models
        result = {
            "linreg": linreg,
            "decreg": decreg,
            "forestreg": forestreg,
            "knnreg": knnreg,
            "adbreg": adbreg,
            "xgbreg": xgbreg,
            "cbreg": cbreg,
            "svmreg": svmreg,
        }

        # Function to check performance
        performance = model_performance(result, data)
        result["performance"] = performance

        return result

    return


def bestModel(result):
    """Function to find best model. Custom algorithm used:
        
    Check adjusted r2 and select top 3 models and among them pick the model 
    with least rmse
    """
    # Dict holding top 3 models with adj r2 and rmse and model. Intialization:
    top_models = {1: [-999, 999, {}], 2: [-999, 999, {}], 3: [-999, 999, {}]}
    # Finding top 3 models with highest adj r2
    for key in result.keys():
        if key != "performance":
            temp = result["performance"][key]["r2"]

            if top_models[1][0] < temp:
                top_models[1][0] = temp
                top_models[1][1] = result["performance"][key]["rmse"]
                top_models[1][2] = {}
                top_models[1][2][key] = result[key]
            elif top_models[2][0] < temp:
                top_models[2][0] = temp
                top_models[2][1] = result["performance"][key]["rmse"]
                top_models[2][2] = {}
                top_models[2][2][key] = result[key]
            elif top_models[3][0] < temp:
                top_models[3][0] = temp
                top_models[3][1] = result["performance"][key]["rmse"]
                top_models[3][2] = {}
                top_models[3][2][key] = result[key]

    # Selecting model with lowest rmse
    temp = 999999999999
    bestkey = ""
    for key in top_models.keys():
        if top_models[key][1] < temp:
            temp = top_models[key][1]
            bestkey = key

    if bestkey == "":
        print("All models fail!")
        return 0, 0, 0
    print("\n-------------------------")
    print("\nBest model found! Training complete!")
    adjr_best = top_models[bestkey][0]
    rmse_best = top_models[bestkey][1]
    model = top_models[bestkey][2]
    return model, adjr_best, rmse_best


def regression_run_models(data):
    # data is dictionary result of the data preprocessing stage
    # Calling function to run models
    result = run_models(data, "first-run")

    # Carrying out advanced data preprocessing
    print("\nCarrying out hyperparameter tuning.. ")
    result = arp.hyperparameter_tuning(data, result)
    perf = model_performance(result, data)
    result["performance"] = perf

    print("\nCarrying out feature selection.. ")
    #result = arp.feature_selection(data, result)
    #perf = model_performance(result, data)
    #result["performance"] = perf

    """
    Generally, when you have a model with the highest Adjusted R² and high 
    RMSE, you would be better off with the one that has moderate Adjusted R² 
    and low RMSE as the latter is the absolute measure of fit.
    """
    print("\nFinding best model..", end="\n")
    bestregmodel, adjr, rmse = bestModel(result)
    result = [bestregmodel, adjr, rmse]
    return result

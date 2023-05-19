"""
Author: Vishal Kundar

regression_results.py is used to display the results to be put on the user's
personalized dashboard

The results being displayed are:
    1. Data properties
    2. Feature analysis
    3. Predict function 
"""
# Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas_profiling import ProfileReport
from reg_data_validation import data_check as dc
from reg_data_preprocessing import dataPrep as dpr


def predict_function(dpr_output, model, test_df):
    df_flag = False
    # Encoding
    encoder = dpr_output["encoder"]
    dpr_obj = dpr(test_df, dpr_output["dependentFeature"], df_flag, encoder)
    output = dpr_obj.preprocess()

    # Scaling
    scaler = dpr_output["scaler"]
    X = output["X"]
    X_index = output["Xindex"]
    X[:, X_index] = scaler.transform(X[:, X_index])

    # Predicting
    for key, value in model.items():
        y_pred = model[key].predict(X)

    # Displaying results
    result = pd.DataFrame(y_pred, columns=[dpr_output["dependentFeature"]])
    test_df = pd.concat([test_df, result], axis=1)
    test_df.to_csv(r'/Users/vishalkundar/Downloads/Website/predicted_data/results.csv', index = False, header = True)
    return test_df


def display(dpr_results, model_results):
    # Data properties and feature analysis
    # Storing data in html file to be rendered later
    f = open("/Users/vishalkundar/Downloads/Website/app/templates/model-info.html", "w")
    html_template = """
    <div>

    """
    dep_feature = dpr_results["dependentFeature"]
    if len(dpr_results["numericalFeatures"]) > 0:
        df = dpr_results["datasetOriginal"]

        corr_df = df.corr()
        corr_df.drop(dep_feature, axis=0, inplace=True)

        best = corr_df[dep_feature].idxmax()
        worst = corr_df[dep_feature].idxmin()
        best = best + " with score = " + str(round(corr_df[dep_feature][best], 2))
        worst = worst + " with score = " + str(round(corr_df[dep_feature][worst], 2))

        html_template = (
            html_template
            + "<p> Best feature realtion with "
            + str(dep_feature)
            + " is: "
            + str(best)
            + "<br/ >"
            + "<br/ >"
            + "</p>"
        )
        html_template = (
            html_template
            + " <p> Worst feature realtion with "
            + str(dep_feature)
            + " is: "
            + str(worst)
            + "<br/ >"
            + "<br/ >"
            + "</p>"
        )

        corr_df.sort_values(by=dep_feature, axis=0, ascending=False, inplace=True)
        html_template = (
            html_template
            + "<h2>Features sorted based on correlation (best to worst) </h2>"
            + "<br/ >"
            + "<br/ >"
        )
        html_template = (
            html_template
            + corr_df[dep_feature].to_frame().to_html(classes="table table-hover table-dark")
            + "<br/ >"
            + "<br/ >"
        )

    # Model details
    html_template = html_template + "<h2>Model performance </h2>" + "<br/ >"
    bestmodel = ""
    for key in model_results[0]:
        if key == "linreg":
            bestmodel = "Linear Regression"
        elif key == "decreg":
            bestmodel = "Decision Tree Regression"
        elif key == "forestreg":
            bestmodel = "Random Forest Regression"
        elif key == "knnreg":
            bestmodel = "K Nearest Neighbors Regression"
        elif key == "adbreg":
            bestmodel = "Adaboost Regression"
        elif key == "xgbreg":
            bestmodel = "XGBoost Regression"
        elif key == "cbreg":
            bestmodel = "Catboost Regression"
        elif key == "svmreg":
            bestmodel = "Support Vector Machine Regression"
            
    html_template = (
        html_template + "<p> Best Model: " + bestmodel + "<br/ >" + "<br/ >" + "</p>"
    )
    html_template = (
        html_template
        + " <p> Adjusted R-squared: "
        + str(model_results[1])
        + "<br/ >"
        + "<br/ >"
        + "</p>"
    )
    html_template = (
        html_template
        + " <p> Root mean square error: "
        + str(model_results[2])
        + "<br/ >"
        + "<br/ >"
        + "</p>"
    )
    html_template = (
        html_template
        + """
    
    </div>"""
    )

    # writing the code into the file
    f.write(html_template)
    f.close()

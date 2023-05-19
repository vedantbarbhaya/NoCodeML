import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas_profiling import ProfileReport
from cla_data_validation import data_check as dc
from cla_data_preprocessing import datapreprocess

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



# these models do not require mutliproccessing as it is inbuilt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings

def predict_function(dpc_output, modeln,modeld, test_df):
    
    #print(dpc_output)
    #print(type(dpc_output))
    encoder = dpc_output["encoder"]
    scaler = dpc_output["scaler"]
    num_feat = dpc_output["num_feat"]

    print("***********************************")
    print(modeln)
    print(modeld)
    
    
    dpc_obj = datapreprocess()
    df = dpc_obj.preprocess(test_df, encoder = encoder, scaler = scaler,num_feat = num_feat)


    '''
    key = modeln
    if key == "LogisticRegression": 
        bestmodel = LogisticRegression()
    elif key == "SVC":
        bestmodel = SVC()
    elif key == "LinearSVC":
        bestmodel = LinearSVC()
    elif key == "NuSVC":
        bestmodel = NuSVC()
    elif key == "PassiveAggressiveClassifier":
        bestmodel = PassiveAggressiveClassifier()
    elif key == "KNeighborsClassifier":
        bestmodel = KNeighborsClassifier()
    elif key == "DecisionTreeClassifier":
        bestmodel = DecisionTreeClassifier()
    elif key == "MLPClassifier":
        bestmodel = KNeighborsClassifier()
    elif key == "QuadraticDiscriminantAnalysis":
        bestmodel = QuadraticDiscriminantAnalysis()
    elif key == "RandomForestClassifier":
        bestmodel = RandomForestClassifier()
    elif key == "AdaBoostClassifier":
        bestmodel = AdaBoostClassifier()
    elif key == "GradientBoostingClassifier":
        bestmodel = GradientBoostingClassifier()
    elif key == "XGBClassifier":
        bestmodel = XGBClassifier()
    elif key == "CatBoostClassifier":
        bestmodel = CatBoostClassifier()
    '''

    bestmodel = modeld[0]
    
    # Setting up the model
    X = df
    model = bestmodel
    print(modeld[2])
    model.set_params(**modeld[2])
   
    # Predicting
    y_pred = model.predict(X)

    # Displaying results
    result = pd.DataFrame(y_pred, columns=[dpc_output["target"]])
    test_df = pd.concat([test_df, result], axis=1)
    return test_df


def display(dpc_results, model_results):
    # Data properties and feature analysis
    # Storing data in html file to be rendered later
    f = open("/Users/vishalkundar/Downloads/Website/app/templates/model-info.html", "w")
    
    html_template = """
    <div>

    """
    
    dep_feature = dpc_results["target"]
    if len(dpc_results["numericalFeatures"]) > 0:
        df = dpc_results["ogdata"]

        corr_df = df.corr()
        corr_df.drop(dep_feature, axis=0, inplace=True)

        best = corr_df[dep_feature].idxmax()
        worst = corr_df[dep_feature].idxmin()
        best = best + " with score = " + str(round(corr_df[dep_feature][best], 2))
        worst = worst + " with score = " + str(round(corr_df[dep_feature][worst], 2))

        html_template = (
            html_template
            + "Best feature realtion with "
            + str(dep_feature)
            + " is: "
            + str(best)
            + "<br/ >"
            + "<br/ >"
        )
        html_template = (
            html_template
            + "Worst feature realtion with "
            + str(dep_feature)
            + " is: "
            + str(worst)
            + "<br/ >"
            + "<br/ >"
        )

        corr_df.sort_values(by=dep_feature, axis=0, ascending=False, inplace=True)
        html_template = (
            html_template
            + "<h2>Features sorted based on correlation (best to worst): </h2>"
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
    html_template = html_template + "<h2>Model performance: </h2>" + "<br/ >"
    bestmodel = ""
    i = 0
    for key,val in model_results.items():
        if i >=1:
            break
        else:
            if key == "logisitic_regression":
                bestmodel = "Logisitic Regression"
                modeln = key
            elif key == "SVC":
                bestmodel = "Support Vector Classifier"
                modeln = key
            elif key == "LinearSVC":
                bestmodel = "Linear Support Vector Classifier"
                modeln = key
            elif key == "NuSVC":
                bestmodel = "NU Support Vector Classifier"
                modeln = key
            elif key == "PassiveAggressiveClassifier":
                bestmodel = "Passive Aggressive Classifier"
                modeln = key
            elif key == "KNeighborsClassifier":
                bestmodel = "K-Neighbors Classifier"
                modeln = key
            elif key == "DecisionTreeClassifier":
                bestmodel = "Decision Tree Classifier"
                modeln = key
            elif key == "MLPClassifier":
                bestmodel = "Neural Network Classifier"
                modeln = key
            elif key == "QuadraticDiscriminantAnalysis":
                bestmodel = "Quadratic Discriminant Analysis"
                modeln = key
            elif key == "RandomForestClassifier":
                bestmodel = "Random Forest Classifier"
                modeln = key
            elif key == "AdaBoostClassifier":
                bestmodel = "AdaBoost Classifier"
                modeln = key
            elif key == "GradientBoostingClassifier":
                bestmodel = "GradientBoosting Classifier"
                modeln = key
            elif key == "XGBClassifier":
                bestmodel = "XG Boost Classifier"
                modeln = key
            elif key == "CatBoostClassifier":
                bestmodel = "CatBoost Classifier"
                modeln = key
            
            i = i + 1
            

    html_template = html_template + "Best Model: " + bestmodel + "<br/ >" + "<br/ >"
    html_template = (
        html_template
        + "Best Estimator for the data: "
        + str(model_results[modeln][0])
        + "<br/ >"
        + "<br/ >"
    )
    html_template = (
        html_template
        + "Accuracy: "
        + str(model_results[modeln][1])
        + "<br/ >"
        + "<br/ >"
    )
    html_template = (
        html_template
        + "Best parameters for the model: "
        + str(model_results[modeln][2])
        + "<br/ >"
        + "<br/ >"
    )
    html_template = (
        html_template
        + "F1score of the model: "
        + str(model_results[modeln][3])
        + "<br/ >"
        + "<br/ >"
    )
    html_template = (
        html_template
        + """
    
    </div>"""
    )

    # writing the code into the file
    f.write(html_template)
    f.close()

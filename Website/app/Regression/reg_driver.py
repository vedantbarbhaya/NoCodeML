"""
Author: Vishal Kundar
Driver function
"""
import sys
from reg_data_validation import data_check as dc
from reg_data_preprocessing import dataPrep as dpr
import regression_models as rm
import regression_results as regres
import os


def runNoCodeML(df, depCol):
    ############################
    # Data preprocessing
    ############################

    # Data preprocessing for regression
    df_copy = df.copy()
    dpr_obj = dpr(df, depCol)
    output = dpr_obj.preprocess()
    output["datasetOriginal"] = df_copy
    print("\nData preprocessing complete!")
    # print(output['dataset'].head())



    ############################
    # Model
    ############################
    print("\nMoving on to model building!") 
    # Model for regression
    model_results = rm.regression_run_models(output)


    ############################
    # Results
    ############################
    print("---------------------------------------------------------------------------")
    print("\t\t\t\t\t\t\t\t RESULTS")
    print("---------------------------------------------------------------------------")

    regres.display(output, model_results)

    return output, model_results

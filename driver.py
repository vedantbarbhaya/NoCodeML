"""
Author: Vishal Kundar
Driver function
"""
import sys
from data_validation import data_check as dc
from data_preprocessing_regression import dataPrep as dpr
import regression_models as rm
import regression_results as regres
import os

def runNoCodeML(dataPath):
    ############################
    #Data validation
    ############################
    print("\nChecking if file and data is supported!")
    dc_obj = dc(dataPath)

    fileCheck = dc_obj.identify_file()
    if(fileCheck == "None"):
        print("\nFile type is supported")
    else:
        print("Error: ", fileCheck)
        print("\nSupported file types: - [csv, tsv, xlsx, json]")  
        sys.exit()     

    df = dc_obj.file_to_dataframe()
    dataValidation = dc_obj.validation_check(df)
    if(dataValidation == "None"):
        print("\nData format is supported")
    else:
        sys.exit(dataValidation)

    #The dependent variable needs to be selected by user from the GUI. 
    #For now we are taking the last column as dependent variable
    depCol = df.columns[-1]
    problemType = dc_obj.identify_problem(df, depCol)
    if(problemType != "Regression" and problemType != "Classification"):
        sys.exit(problemType)

    print("\nProblem type identified as: ", problemType)
    print("\nMoving on to data preprocessing!")
    
    ############################
    #Data preprocessing
    ############################
    if(problemType == "Regression"):
        #Data preprocessing for regression
        df_copy = df.copy()
        dpr_obj = dpr(df, depCol)
        output = dpr_obj.preprocess()
        output['datasetOriginal'] = df_copy
        print("\nData preprocessing complete!")
        #print(output['dataset'].head())

    else:
        #Data preprocesing for classification
        pass

    ############################
    #Model
    ############################
    print("\nMoving on to model building!")
    if(problemType == "Regression"):
        #Model for regression
        model_results = rm.regression_run_models(output)
    else:
        #Model for classification
        pass
    
    ############################
    #Results
    ############################
    print("\nPress enter to view results: ")
    input()
    os.system("clear")
    print("---------------------------------------------------------------------------")
    print("\t\t\t\t\t\t\t\t RESULTS")
    print("---------------------------------------------------------------------------")
    print("\n\n")
    if(problemType == "Regression"):
        #Result for regression
        regres.display(output, model_results)
    else:
        #Result for classification
        pass
    

if __name__ == "__main__":
    runNoCodeML("/Users/vishalkundar/Desktop/ML/Datasets/CarPrice_Assignment.csv")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
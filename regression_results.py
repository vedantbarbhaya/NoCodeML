"""
Author: Vishal Kundar

regression_results.py is used to display the results to be put on the user's
personalized dashboard

The results being displayed are:
    1. Data properties
    2. Feature analysis
    3. Predict function 
    
    /Users/vishalkundar/Desktop/ML/Datasets/admission_predict_test.csv
"""
#Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas_profiling import ProfileReport
from data_validation import data_check as dc
from data_preprocessing_regression import dataPrep as dpr

def predict_function(choice, dpr_output, model):
    df_flag = False
    if choice == 1:
      #Let us get value for each feature and then convert it 
      #into a single row dataframe
      feature_list = list(dpr_output['datasetOriginal'].columns)
      feature_list.remove(dpr_output['dependentFeature'])
      test_df = {}
    
      for x in feature_list:
        temp = input("Enter value for " + x + " : ")
        if temp.strip() == "":
            print("Error field left blank")
            return True
        
        if temp.isdigit():
            test_df[x] = int(temp)
        else:
            try:
                test_df[x] = float(temp)
            except:
                test_df[x] = str(temp)
      test_df = pd.DataFrame(test_df, index=[1])  
    
    else:
      data_path = input("Data path: ")
      dc_obj = dc(data_path)
      
      #Validation check
      fileCheck = dc_obj.identify_file()
      if(fileCheck == "None"):
        print("\nFile type is supported")
      else:
        print("Error: ", fileCheck)
        print("\nSupported file types: - [csv, tsv, xlsx, json]")  
        return True

      test_df = dc_obj.file_to_dataframe()
      dataValidation = dc_obj.validation_check(test_df)
      if(dataValidation == "None"):
        print("\nData format is supported")
      else:
        print(dataValidation)
        return True
    
    #Data preprocessing
    #Encoding
    encoder = dpr_output['encoder']
    dpr_obj = dpr(test_df, dpr_output['dependentFeature'], df_flag, encoder)
    output = dpr_obj.preprocess()
    
    if output['missingDataFlag']:
        print("Missing data found in dataset")
        print("Missing data contribution: ", output['missingDataContribution'])
    
    #Scaling
    scaler = dpr_output['scaler']
    X = output['X']
    X_index = output['Xindex']
    X[:, X_index] = scaler.transform(X[:, X_index])
    
    #Predicting
    for key in model:
        y_pred = model[key].predict(X)
    
    #Displaying results
    result = pd.DataFrame(y_pred, columns=[dpr_output['dependentFeature']])
    print("Result : \n\n")
    print(result)
    return False
    

def display(dpr_results, model_results):
    #Data properties and feature analysis
    print("Data properties: ")
    print("--------------------------")
    df = dpr_results['datasetOriginal']
    
    
    if df.shape[0] > 2000:
        profile = ProfileReport(df, minimal=True)
    else:
        profile = ProfileReport(df)
      
    profile.to_file("user_report.html")
    
    
    print("\n")
    print("Open user_report.html")
    
    #Feature correlation
    print("\n\n")
    print("Feature correlations: ")
    print("--------------------------")
    dep_feature = dpr_results['dependentFeature']
    if len(dpr_results['numericalFeatures']) > 0: 
        df = dpr_results['datasetOriginal']
        plt.figure(figsize=(15, 15))
        sns.heatmap(df.corr(),annot=True,cmap='inferno',mask=np.triu(df.corr(),k=1))
        plt.show()
        
        corr_df = df.corr()
        corr_df.drop(dep_feature, axis = 0, inplace = True)
        
        best = corr_df[dep_feature].idxmax()
        worst = corr_df[dep_feature].idxmin()
        best = best + " with score = " + str(round(corr_df[dep_feature][best],2))
        worst = worst + " with score = " + str(round(corr_df[dep_feature][worst],2))
        
        print("\n\nBest feature relation with " + dep_feature + " is: " + best)
        print("\nWorst feature relation with " + dep_feature + " is: " + worst)
        
        corr_df.sort_values(by=dep_feature, axis=0, ascending=False, inplace=True)
        print("\n\nFeatures sorted based on correlation (best to worst): ")
        print("\n")
        print(corr_df[dep_feature])       
        
    #Model details
    print("\n")
    print("Model performance: ")
    print("--------------------------")
    print("Best model: ", str(model_results[0]))
    print("\nAdjusted R-squared: ", model_results[1])
    print("\nRoot mean square error: ", model_results[2])
    
    #Plotting results
    bestregmodel = model_results[0]
    for key in bestregmodel:
        y_pred = bestregmodel[key].predict(dpr_results['Xtest'])
    y_test = dpr_results["Ytest"]

    minimum = int(min(y_test) -1)
    maximum = int(max(y_test) + 1)
    y_col = list(np.linspace(minimum, maximum, len(y_test)))   
    
    #Plotting y_pred vs y_test
    plt.plot(y_pred, y_col, '-r', label="Y pred")
    plt.plot(y_test, y_col, '-b', label="Y test")
    plt.title("y_pred vs y_test")
    plt.legend()
    plt.show()
    
    #Prediction function
    condition = True
    while(condition):
      print("\n\n")
      print("--------------------------")
      print("Predict values using model: ")
      print("\n")
      print("1. Predict single data point")
      print("2. Predict in batch using file")
      print("3. Exit")
      choice = int(input("Select your choice: "))
      if choice == 1 or choice == 2:
        condition = predict_function(choice, dpr_results, model_results[0])
      elif choice == 3:
        break
      else:
        print("Invalid choice!")
    
    
    
    
        
    

















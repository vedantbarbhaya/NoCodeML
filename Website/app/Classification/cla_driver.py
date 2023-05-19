import pandas as pd
import numpy as np
from cla_data_validation import data_check
from cla_data_preprocessing import datapreprocess
from cla_models import classif_models
from cla_HPO import hyperparameter_tuning
import ctypes, sys
from cla_predict import display

#filename = input("please enter the full path including the file name of the file")
#target = input("enter the name of the column you want to predict")



def runNoCodeML(df,target):
    
        ############################
        # Data preprocessing
        ############################
        df_flag = True
        pp = datapreprocess()
        classificationPrepOutput = pp.preprocess(df,target,df_flag)
        
        training_set = classificationPrepOutput["trainingset"]
        training_set_bal = classificationPrepOutput["trainingsetbal"]
        test_set = classificationPrepOutput["testset"]
        classif_type = classificationPrepOutput["classiftype"]
        encoder = classificationPrepOutput["encoder"]
        scaler = classificationPrepOutput["scaler"]
        
  
        ############################
        # Model
        ############################
        if isinstance(training_set_bal,int):
            modelcall = classif_models()
            top_models = modelcall.main_caller(0,classif_type,training_set,test_set)
            tune = hyperparameter_tuning()
            tuned_top_models = tune.optimal_param(0,top_models,classif_type,training_set,test_set)
        

    
        else:
            modelcall = classif_models()
            top_models = modelcall.main_caller(1,classif_type,training_set,training_set_bal,test_set)
            tune = hyperparameter_tuning()
            tuned_top_models = tune.optimal_param(1,top_models,classif_type,training_set,training_set_bal,test_set)
            
        ############################
        # Results
        ############################            
        print("TUNED TOP_MODELS DRIVER")
        print(tuned_top_models)
        display(classificationPrepOutput, tuned_top_models)
        return classificationPrepOutput, tuned_top_models

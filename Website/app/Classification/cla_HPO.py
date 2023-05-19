import time
import numpy as np
#from tune_sklearn import TuneSearchCV
#rom tune_sklearn import TuneGridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from collections import defaultdict

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class hyperparameter_tuning():
    
        # ****************class variables:********************
            
        # logistic regression
        param_grid_lr = {
        'penalty': ['l2'],
        'C': [0.8, 1, 1.2],
        'tol': [0.00005, 0.0001, 0.00015, 0.0002],
        'fit_intercept' : [True, False],
        'warm_start' : [True, False],
        'solver': [ 'lbfgs', 'liblinear'],
        'class_weight':['balanced', None]
        }
        
        # SVC
        param_grid_svc = {
        'C': [0.8, 1, 1.2],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel' :['linear', 'poly', 'rbf', 'sigmoid'],
        'tol': [0.00005, 0.0001, 0.00015, 0.0002],
        'class_weight':['balanced', None]
            
        }
        
        # Linear SVC
        param_grid_lsvc = {
        'C': [0.8, 1, 1.2],
        'loss' :['hinge', 'squared_hinge'],
        'tol': [0.00005, 0.0001, 0.00015, 0.0002],
        'class_weight':['balanced', None]
        }
        
        # NuSVC
        param_grid_nusvc = {
        'kernel' : [ 'poly', 'rbf'],
         'nu' : [0.01,0.1,1,10],
        'gamma' : ['scale','auto'],
        'tol': [0.00005, 0.0001, 0.00015, 0.0002],
        'class_weight':['balanced', None]
        }

        
        # SGD
        #self.param_grid_sgd ={ 
        #    'loss': ['hinge', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive'],
        #    'C': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
        #    'penalty': ['l2', 'l1', 'elasticnet'],
        #    'epsilon':[0.01, 0.1],
        #    'class_weight':['balanced', None]
        #    #'n_jobs': [-1]
        #    "multiclass":['ovr','ova']
        #}
        
        # PAC   
        param_grid_pac = {
        'loss' : ['squared_hinge'],
        'C': [0.8, 1, 1.2],
        'tol': [0.00005, 0.0001, 0.00015],
        'warm_start' : [True, False],
        'class_weight':['balanced', None],
         'average': [1,10,100,1000]
        }
        
        # KNN
        param_grid_knn = {     
            'n_neighbors' : [2,3,4,5,6,7,8,9,10],
            'weights' : ["uniform",'distance'],
            'metric' : ['euclidian','manhattan','minkowski']
            }
        
        # DTC
        max_depth = [i for i in range(5,50,5)]
        param_grid_dtc = {
            'criterion':['gini','entropy'],
            'max_depth': max_depth,
            'class_weight':['balanced', None]
            }
        
        
        #MLP
        alpha =  list(10.0 ** -np.arange(1, 10))
        hidden_layer_size = list(np.arange(2, 10))   
        
        param_grid_mlp = {
            #'solver': ['lbfgs','adam'],
            'solver': ['adam'], 
            #'activation': ['logistic','relu','softmax','tanh'],
            'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000], 
            'alpha': alpha,
            'hidden_layer_sizes':hidden_layer_size,
            'learning_rate': ['constant', 'invscaling','adaptive'],
            'random_state':[0,1,2,3,4,5,6,7,8,9],
            'warm_start' : [True, False],
            }
        
        
        # QDA  
        param_grid_qda = {
            'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
         
        # RFC
        n_estimators = [100,200,300,400,500]
        max_depth =  [i for i in range(5,10)]
        
        param_grid_rfc = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "bootstrap": [True,False]
        }
        
        
        # xgboost
        param_grid_xgb = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate':[0.001, 0.01, 0.1,1],
            #"min_child_weight": [1, 5, 10],
            "gamma": [0.5, 1, 1.5, 2, 5],
            #"subsample": [0.6, 0.8, 1.0],
            #"colsample_bytree": [0.6, 0.8, 1.0],
            'early_stopping': [True],
            'use_label_encoder': [False]
            
        }
        
        # gradient_boosting
        param_grid_gb = {    
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate':[0.01,0.1,1],
            }
    
        #catboost
        param_grid_cb = {
            "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "depth": [3, 4, 5],
            "verbose": [0]
        }
        
        # adaboost
        param_grid_ada = {
            'learning_rate': [0.1,1,2,3,4,5], 
             'n_estimators': [100,200,300, 400, 500]
            }

        perf_dict = defaultdict(list)
        score_dict = defaultdict(float)
        
        hpomodels = {
                  "logisitic_regression": [LogisticRegression(), param_grid_lr],
                  "SVC": [SVC(), param_grid_svc],
                  "LinearSVC": [LinearSVC(), param_grid_lsvc],
                  "NuSVC": [NuSVC(), param_grid_nusvc],
                  "PassiveAggressiveClassifier": [PassiveAggressiveClassifier(),param_grid_pac],
                  "KNeighborsClassifier": [KNeighborsClassifier(), param_grid_knn],
                  "DecisionTreeClassifier": [DecisionTreeClassifier(), param_grid_dtc],
                  "MLPClassifier": [MLPClassifier(), param_grid_mlp],
                  "QuadraticDiscriminantAnalysis": [QuadraticDiscriminantAnalysis(),param_grid_qda],
                  "RandomForestClassifier": [RandomForestClassifier(), param_grid_rfc],
                  "AdaBoostClassifier": [AdaBoostClassifier(), param_grid_ada],
                  "GradientBoostingClassifier": [GradientBoostingClassifier(),param_grid_gb],
                  "XGBClassifier": [XGBClassifier(), param_grid_xgb],
                  "CatBoostClassifier": [CatBoostClassifier(), param_grid_cb]
                  } 
        
        def best_models(self,dict,flag = 0):
            
              top_models = {}  
              score_dict = {}
  
              for modelname,modelres in dict.items():
                  score = modelres['best_score'] + modelres['f1score']
                  score_dict[modelname] = score
            
              final_list = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
              
              final_score_dict = {}

              for model,score in final_list:
                 final_score_dict.setdefault(model, []).append(score)
              
             
                     
              i = 0 
              for key,val in final_score_dict.items():
                  top_models[key] = (dict[key]['best_estimator'],
                                    dict[key]['best_score'],dict[key]['best_params'],
                                    dict[key]['f1score']
                                    )
                  i=i+1
                  print("i",i)
                  if i == 5:
                    break
                     
                    
   
              return top_models




        def optimal_param(self,flag,base_models,classif_type,*args):
                
              if flag == 0:
                  training_set = args[0]
                  test_set = args[1]
                  self.__X_train = training_set.iloc[:,:-1]
                  self.__Y_train = training_set.iloc[:,-1]
               
              
              if flag == 1:   
                  training_set = args[0]
                  training_set_bal = args[1]
                  test_set = args[2]
                  self.__X_train = training_set.iloc[:,:-1]
                  self.__Y_train = training_set.iloc[:,-1]
                  self.__X_train_bal = training_set_bal.iloc[:,:-1]
                  self.__Y_train_bal = training_set_bal.iloc[:,-1]
              
              self.__X_test = test_set.iloc[:,:-1]
              self.__Y_test = test_set.iloc[:,-1]
              
            # start time for multiprocessing of tuning models
            
              if classif_type == 0:
                  self.param_grid_xgb["objective"] = ["binary:logistic","binary:hinge"]
                  self.param_grid_mlp["activation"] = ["logistic"]
                  
              if classif_type == 1:
                  self.param_grid_xgb["objective"] = ["multi:softprob"]
                  self.param_grid_mlp["activation"] = ["softmax"    ]
                  

              start = time.perf_counter()
              print("Started tuning of the models")
              
              
              count = 0
              for mod_n,mod_d in self.hpomodels.items():
                    
                    for base_model,stats in base_models.items():
                        
                        if mod_n == base_model:
                            count+=1
                            self.tune_model(mod_n,mod_d)
                            
                    if count == 5:
                        break
        
               
              finish = time.perf_counter()
              print(f'Finished tunning of models in {round(finish-start, 2)} second(s)')
              
              print("--------------------- PERFORMANCE AFTER TUNING ---------------------")
              
              
              res = self.best_models(self.perf_dict)
              print("TOP MODELS AFTER TUNING hpo file")
              print(res)

              return res
          
            
             
         
        def tune_model(self,modeln,modeld):  
              
              start = time.perf_counter()    
              modelname = modeln
              print(f'Starting tuning of model {modelname}')
              modelcall = modeld[0]
              params = modeld[1]
              randm = RandomizedSearchCV(estimator=modelcall, param_distributions = params, 
                                         cv = 5, n_iter = 5, n_jobs=-1, refit = True, 
                                         scoring = 'roc_auc', error_score = 0, verbose = 3,
                                         return_train_score = True)
                        

              randm.fit(self.__X_train, self.__Y_train)
              Y_pred = randm.predict(self.__X_test)
              
              f1score  = f1_score(self.__Y_test,Y_pred)
              
              
              finish = time.perf_counter()
                       
              print(" Results from Random Search " )
              print("The best estimator across ALL searched params: ", randm.best_estimator_)
              print("The best score across ALL searched params: ", randm.best_score_)
              print("The best parameters across ALL searched params:", randm.best_params_)
              
              
              self.perf_dict[modelname] = {
                                      "best_estimator":randm.best_estimator_,
                                      "best_score":randm.best_score_,
                                      "best_params": randm.best_params_,
                                      "f1score":f1score}
              
              self.score_dict[modelname] = {"best_score":randm.best_score_}
              
             
              
              print(f'Finished tuning of model {modelname} in time {finish-start}')
              
              
              
              
              



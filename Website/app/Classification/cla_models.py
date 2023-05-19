"""
Author: Vedant Barbhaya

models.py is used to run various regression models on the data.
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



# from sklearn.linear_model import TweedieRegressor # GLM
#from sklearn.linear_model.ridge import RidgeClassifierCV
#from sklearn.linear_model.ridge import RidgeClassifier
#from sklearn.ensemble.forest import ExtraTreesClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.linear_model import SGDClassifier

from multiprocessing import Manager
import concurrent.futures
import time
from collections import defaultdict
from collections import OrderedDict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


# these models will be passsed through multiprocessing code
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


# these algorithms are meta estimators and use other estimators
#from sklearn.multioutput import ClassifierChain
#from sklearn.multioutput import MultiOutputClassifier
#from sklearn.multiclass import OutputCodeClassifier
#from sklearn.multiclass import OneVsOneClassifier
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.ensemble.bagging import BaggingClassifier
#from sklearn.ensemble.voting_classifier import VotingClassifier


#kernel = 1.0 * RBF(1.0)
#gpc = GaussianProcessClassifier(kernel=kernel,random_state=0)

class classif_models:

        def main_caller(self,flag,classif_type,*args):
            
            self.flag = flag
            
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
            self.__classif_type = classif_type
            self.performance_dict = Manager().dict()

            bestmodels = self.driver()
            return bestmodels

        def driver(self):

              # creating model_space
              # 1. models requiring multiprocessing module
              
              if self.__classif_type == 0:
                  self.mutlipr_models = {
                  "LogisticRegression(": [LogisticRegression(), {}],
                  "SVC": [SVC(), {"random_state":0}],
                  "LinearSVC": [LinearSVC(), {"random_state":0}  ],
                  "NuSVC": [NuSVC(),{}  ],
                  "PassiveAggressiveClassifier": [PassiveAggressiveClassifier(),{"random_state":0,}  ],
                  "KNeighborsClassifier": [KNeighborsClassifier(), {}  ],
                  "DecisionTreeClassifier": [DecisionTreeClassifier(), {"random_state":0}  ],
                  "MLPClassifier": [MLPClassifier(), {"random_state":1}  ],
                  "QuadraticDiscriminantAnalysis": [QuadraticDiscriminantAnalysis(), {}  ]
                  }
              
              else:
                  self.mutlipr_models = {
                  "LogisticRegression(": [LogisticRegression(), {'multi_class':'multinomial', 'solver':'lbfgs' }],
                  "SVC": [SVC(), {"random_state":0}],
                  "LinearSVC": [LinearSVC(), {"random_state":0, "multi_class":"ovr"}  ],
                  "NuSVC": [NuSVC(),{}  ],
                  "PassiveAggressiveClassifier": [PassiveAggressiveClassifier(),{"random_state":0,}  ],
                  "KNeighborsClassifier": [KNeighborsClassifier(), {}  ],
                  "DecisionTreeClassifier": [DecisionTreeClassifier(), {"random_state":0}  ],
                  "MLPClassifier": [MLPClassifier(), {"random_state":1}  ],
                  "QuadraticDiscriminantAnalysis": [QuadraticDiscriminantAnalysis(), {}  ]
                  }
                  

              # 2. models with inbuilt multiprocessing capability
              self.singlepr_models = {
              "RandomForestClassifier": [RandomForestClassifier(),{"random_state":0, 'n_jobs':-1} ],
              "AdaBoostClassifier": [AdaBoostClassifier(), {} ],
              "GradientBoostingClassifier": [GradientBoostingClassifier(), {"random_state":0,} ],
              "XGBClassifier": [XGBClassifier(), {"nthread":-1} ],
              "CatBoostClassifier": [CatBoostClassifier(), {"verbose":0} ]

              }


              self.multiprocess(self.mutlipr_models)
              self.singleprocess(self.singlepr_models)
              bestmodels =  self.best_models(self.performance_dict)
              return bestmodels


        def multiprocess(self,models):

              print("********************MUTLIPROCESSING OF MODELS STARTED***********************", end = "\n\n")
              # start time for multiprocessing
              start = time.perf_counter()
              print("Starting multiprocessing of the models")
              

              with concurrent.futures.ProcessPoolExecutor() as executor:
                     
                     results = executor.map(self.run_model,models.items())
                     #print("RESULTS")
                     #print(next(results))
                     #for result in results:
                     #       print(result)

              finish = time.perf_counter()
              print(f'Finished multiprocessed running of models in {round(finish-start, 2)} second(s)')
              print("********************MUTLIPROCESSING OF MODELS ENDED***********************", end = "\n\n")


        def singleprocess(self,models):

              print("********************SINGLE PROCESSING OF MODELS STARTED***********************", end = "\n\n")
              start = time.perf_counter()

              for modelname,model in models.items():
                  self.run_model((modelname,model))

              finish = time.perf_counter()
              print(f'Finished single processing of models in {round(finish-start, 2)} second(s)')
              print("********************SINGLE PROCESSING OF MODELS ENDED***********************", end = "\n\n")


        def run_model(self,model):
            
              try:
                  modelname = model[0]
                  model_data = model[1]
                  
                  print("RUNNING MODEL - " + modelname)
                  modeln = model_data[0]
                  clf = modeln
    
                  if model_data[1] != {}:
                      params = model_data[1]
                      clf.set_params(**params)
    
                  start = time.perf_counter()
                  clf_unb = clf.fit(self.__X_train,self.__Y_train)
                  self.performance(modelname,clf_unb)
                  finish_un = time.perf_counter()
                  print(f'Completed processing of {modelname} in {round(finish_un-start, 2)} second(s)')
                
              except Exception as e:
                  print(e)
                
              warnings.filterwarnings("ignore", category=UserWarning) 

              if self.flag == 1:
                  print("==============================================================================")
                  try:
                      modelname_bal = modelname + '_bal'
                      print("RUNNING MODEL - " +  modelname_bal)
                      start = time.perf_counter()
                      clf_bal =  clf.fit(self.__X_train_bal,self.__Y_train_bal)
                      self.performance(modelname_bal,clf_bal)
                      finish_bal = time.perf_counter()
                      print(f'Completed processing of {modelname_bal} in {round(finish_bal-start, 2)} second(s)')
                  
                  except Exception as e:
                      print("ERROR")
                      print(e)


        def performance(self,model_name,model):
              #print(f"############ CALCULATING PERFORMANCE FOR MODEL {model_name} ##############")
              results = defaultdict(float)

              Y_pred =  model.predict(self.__X_test)
              #print("******************* Y_ PRED *************************")
              #print(Y_pred)
              #print("******************* Y_ PRED and Y_test shape *************************")
              #print(Y_pred.shape)
              #print(self.__Y_test.shape)
              
              
              results['acc'] = accuracy_score(self.__Y_test,Y_pred)
              results['precision'] = precision_score(self.__Y_test,Y_pred)
              results['recall'] = recall_score(self.__Y_test,Y_pred)
              results['f1score'] = f1_score(self.__Y_test,Y_pred)
              #results['roc-auc-score'] = roc_auc_score(self.__Y_test, clf.predict_proba(self.__X_train), multi_class='ovr')
              print(model_name)
              print("acc", end = " ")
              print(results['acc'])
              print("f1-score", end = " ")
              print(results['f1score'])
              self.performance_dict[model_name] = results


              
        def best_models(self,dict,flag = 0):
            
              top_models = {}  
              score_dict = {}
  
              for modelname,modelres in dict.items():
                  score = modelres['acc'] + modelres['f1score']
                  score_dict[modelname] = score
            
              final_list = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
              
              final_dict = {}

              for model,score in final_list:
                 final_dict.setdefault(model, []).append(score)
              
              i = 0   
              if flag == 0:
                  for key,val in final_dict.items():
                     top_models[key] = (dict[key]['acc'],dict[key]['f1score'])
                     i=i+1
                     print("i",i)
                     if i == 5:
                         break
                     
              
                    
   
              return top_models
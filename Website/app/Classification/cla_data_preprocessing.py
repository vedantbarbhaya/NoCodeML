import pandas as pd
import numpy as np
import re
import datetime as datetime
from scipy import stats
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import LeaveOneOutEncoder
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

class datapreprocess:

        def preprocess(self,data,target = None,df_flag=False,encoder=None,scaler=None,num_feat = None):
            
                # things required when predicting new data
                self.__df_flag = df_flag
                self.encoder = encoder
                self.scaler = scaler
                self.__data = data
                self.target = target  
                self.X_train = ""
                self.X_test =  ""
                self.y_train = ""
                self.y_test  = ""
                self.training_set = ""
                self.training_set_bal = ""
                self.test_set = ""
                self.num_feat = num_feat
                
                
                __df = self.__data.copy()
                
                

                """
                Driver function which will call all other functions when training the model

                @params:
                data - dataframe containing all the data
                target - column name of the dependant variable
                """
                
                print("====================================")
                print("Initial dataframe")
                print("====================================")
                print(self.__data.head(), end = "\n\n")
                
                print("====================================")
                print("dataframe characterstics")
                print("====================================")
                print(self.__data.info(), end = "\n\n")
                print(self.__data.describe(), end = "\n\n")
                
                
                print(self.__df_flag)
                
                # checking for binary or multiclass classification
                if self.__df_flag:
                    print("SELF OF DF FLAG TRUE")
                    self.__class = list(self.__data[target].unique())
                    self.__no_class = len(self.__class) #no of classes in case of mutliclass classification

                    # if self.__cltype = 0 -> binary classification and if self.__cltype = 1 -> multiclass classification
                    self.__cltype = None

                    if self.__no_class == 2:
                        self.__cltype = 0
                        print("====================================")
                        print("Classification type: binary classification",end = "\n\n")
                    else:
                        self.__cltype = 1
                        print("====================================")
                        print("Classification type: mutliclass classification")

        
            
                         
                    classificationPrepOutput = self.data_qual_assessment(__df,target)    
                    return classificationPrepOutput
                    #return self.__cltype, training_set,training_set_bal,test_set
                
                else:
                         print("IM HEEERRRRREEEEEE!!!!")
                         __df = self.data_qual_assessment(__df)
                         return __df

        def balance(self,X,y):

                #do after splitting only on train if possible
                # Oversample with SMOTE and random undersample for imbalanced dataset
                # define pipeline
                over = SMOTE(sampling_strategy=0.5)
                under = RandomUnderSampler(sampling_strategy=0.5)
                steps = [('o', over), ('u', under)]
                pipeline = Pipeline(steps=steps)

                # transform the dataset
                X, y = pipeline.fit_resample(X, y)
                return X,y



        def split(self,__df):
                X = __df.iloc[:,0:-1]
                Y = __df.iloc[:,-1]
                X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.2, random_state=0)
                
                X_train.reset_index(inplace = True)
                X_train.drop("index",axis =1, inplace = True)
                X_test.reset_index(inplace = True)
                X_test.drop("index",axis =1, inplace = True)
                y_train = y_train.reset_index()
                y_train.drop("index",axis =1, inplace = True)
                y_test = y_test.reset_index()
                y_test.drop("index",axis =1, inplace = True)

                return X_train, X_test, y_train, y_test

        '''
        def encode(self,__df,target,cat_feat,card,type2="OneHotEncoder"):

                print("====================================")
                print("dataframe before encoding:")
                print(__df.head(20))
                    
                for col,type1 in card.items():
                    
                    if type1 == True:
                        X = __df[col]
                        y = __df.iloc[:,-1]
                        encoder = TargetEncoder()
                        __df[col] = encoder.fit_transform(X,y)
                        
                    else:
                        
                        if type2 == " LeaveOneOutEncoder":
                            enc = LeaveOneOutEncoder(cols=cat_feat)
                            X = __df.iloc[:,0:-1]
                            y = __df.iloc[:,-1]
                            __df = enc.fit_transform(X, y)
        
        
                        if type2 == "OneHotEncoder":
                            dummies = pd.get_dummies(__df[col], drop_first=True)
                            __df = pd.concat([__df, dummies], axis=1)
                            __df.drop(col, axis=1, inplace=True)
                            
                            target_se = __df[target]
                            __df.drop([target], axis=1,inplace = True)   
                            __df[target] = target_se
                        
                return __df
      '''
      
        def encode(self,df,cat_feat):
            if not cat_feat:
                return [], None #No categrocial features to encode
        
            
            if self.__df_flag:

                encoder_list = [0] * len(cat_feat) #List of encoders for each categorical feature
                for i in range(len(cat_feat)):
    
                    
                    encoder_list[i] = LabelBinarizer()
                    encoder_list[i].fit(df[cat_feat[i]])  
                    transformed = encoder_list[i].transform(df[cat_feat[i]])
                    columns = [encoder_list[i].classes_[j] for j in range(len(encoder_list[i].classes_))]
                    
                    if len(encoder_list[i].classes_) == 2:
                        del columns[-1]
                        
                    ohe_df = pd.DataFrame(transformed, columns=columns,index = df.index)
                    df = pd.concat([df, ohe_df], axis=1).drop(cat_feat[i], axis=1)
                    
                    target_se =  df[self.target]
                    df.drop([self.target], axis=1,inplace = True)
                    df[self.target] = target_se
                
                        
                encoder = encoder_list
                return df, encoder

            # if encoding new data after training the model
            else:
            
                encoder = self.encoder
                print(self.encoder)
                for i in range(len(cat_feat)):
                    transformed = encoder[i].transform(df[cat_feat[i]])
                    columns = [encoder[i].classes_[j] for j in range(len(encoder[i].classes_))]
                    if len(encoder[i].classes_) == 2:
                        del columns[-1]
                        
                    ohe_df = pd.DataFrame(transformed, columns=columns, index=df.index)
                    self.__dataset = pd.concat([df, ohe_df], axis=1).drop(cat_feat[i], axis=1)

            #convCatFeatureList = list(__df.select_dtypes(include=['object']).columns)
                return df,encoder
        
        def balance_check(self,training_set,target,X_train,y_train,__colnames):
                        tar_val = self.__class
                        #no_targ = self.__no_class
                        #imb_dict = defaultdict(int)
                        #per_dict = defaultdict(float)
                        imb_li = []
                        per_li = []
                        imbalance = False
                        threshold_per = 30
        
                        for classno in range(len(tar_val)):
                            #imb_dict[class] = __df[__df[target]==class][target].count()
                            #per_dict[class] = imb_dict[class] / __df.shape[0] * 100
                            imb_li.append(training_set[training_set[target]==tar_val[classno]][target].count())
                            per_li.append(imb_li[classno]/training_set.shape[0] * 100)
        
                        per_li.sort(reverse=True)
                        ideal_per = ((training_set.shape[0] // len(tar_val) ) / training_set.shape[0]) * 100
                        range_class_per = per_li[0] - per_li[-1]
        
                        print(per_li)
                        print(range_class_per)
                        
                        
                        for x in X_train.columns:
                            print(x)
                            print(X_train[x].isna().sum())
        
                        if range_class_per > threshold_per:
                            print('Data is imbalanced data set')
                            X_train_bal, y_train_bal = self.balance(X_train,y_train)
                            X_train_bal = pd.DataFrame(X_train_bal)
                            X_train_bal.columns = __colnames
                            y_train_bal = pd.DataFrame(y_train_bal)
                            y_train_bal.columns = [target]
                            training_set_bal =  pd.concat([X_train_bal, y_train_bal], axis=1)
                            
                            
        
                            print("-----------X_train_bal-----------")
                            print(X_train_bal.head())
                            print("Shape of X_train_bal is ", end = "")
                            print(X_train_bal.shape)
                            print("-----------y_train_bal-----------")
                            print(y_train_bal.head())
                            print("Shape of y_train_bal is ", end = "")
                            print(y_train_bal.shape)
        
                            return training_set,training_set_bal
                        
                        else:
                          print('Data is balanced data set')
                          return training_set,-1

            
                

        def data_qual_assessment(self,__df,target = None):

   
                # making the dependant variable the last column in the dataframe
                if self.__df_flag:
                    target_se = __df[target]
                    __df.drop([target], axis=1,inplace = True)
                    __df[target] = target_se

                    
                    print("====================================")
                    print("making target variable the last variable",end = "\n\n")
                print(__df.head(10))

                # ****************************** STEP 1: BASIC KNOWLEDGE ABOUT THE DATA ******************************
                __rows = __df.shape[0]
                __cols = __df.shape[1]
                __colnames = list(__df.columns)
                __datatypes = list(__df.dtypes)
                __dindex = list(__df.index) # list of index values

                __indextype = None
                if isinstance(__dindex[0], int):
                    __indextype = "int"
                elif isinstance(_dindex[0], np.object_):
                    __indextype = "object"
                elif isinstance(__dindex[0],str):
                    __indextype = "str"
                elif isinstance(__dindex[0],datetime):
                    __indextype = "datetime "
                else:
                    __indextype = None
                
                print("====================================")
                print("indextype ==" + __indextype)
          

                # ****************************** STEP 3: REMOVING UNNCESSARY INDEXING COLUMNS  ******************************
                print("====================================")
                print("removing unwanted columns",end = "\n\n")
          
                index_names = ['id','ID','Id',"Sr. No","SERNO","SN","S. No.","S No","Serial No.", "index","Index","INDEX"]
               
                
                a_set = set(index_names)
                b_set = set(__colnames)
                rem_cols = []
                
                if len(a_set.intersection(b_set)) > 0:
                        rem_cols.extend(list(a_set.intersection(b_set)))
                        
                            
                __colnames = [i for i in  __colnames if i not in rem_cols]
                
                
                if self.__df_flag:
                    __colnames = [i for i in  __colnames if i != target]
                        
                # using regular expressions for more refined check
                patterns1 =   ['.*' + x + "$" for x in index_names] 
                patterns2 =   ['^' + x + ".*$" for x in index_names] 
                
                test_strings = __colnames
                
                for x in test_strings:
                    for pat in patterns1:
                        
                        if re.match(pat,x):
                            print(f"removing column {x}")
                            rem_cols.append(x)
                    
                    for pat in patterns2:
                        if re.match(pat,x):
                            print(f"removing column {x}")
                            rem_cols.append(x)
                            

                print("Cols to be removed are:")
                print(rem_cols)
                
                __df.drop(rem_cols, axis = 1, inplace = True)

                # ****************************** STEP 3: IDENTIFYING NUMERICAL AND CATEGORICAL VARIABLES  ******************************
                #num feat is only calculated for training data
                if self.__df_flag:
                        self.num_feat = list(__df.select_dtypes(exclude=['object']).columns)
                        print(f"num_feat: {self.num_feat}", end = "\n\n")
                
                    # removing target variables from num_feat
                        if target in self.num_feat:
                            self.num_feat.remove(target)
                        
                    # removing already encoded categorical variables from num_feats
                        temp = self.num_feat.copy()
                        for x in temp:
                            tmp = len(__df[x].unique())
                            print(f"unique values of {x} are {tmp}")
                            if len(__df[x].unique()) <= 10:
                                print(f"removing {x} from num_feat")
                                self.num_feat.remove(x)
                                        
                        
                self.cat_feat = list(__df.select_dtypes(include=['object']).columns)
                print("====================================")
                print("numerical features are")
                print(self.num_feat)
                print('categorical features are')
                print(self.cat_feat,end = "\n\n")
                
                # ****************************** STEP 4: DEALING WITH NULL VALUES  ******************************

                # DEALING WITH NULL VALUES FOR NUMERICAL FEATURES

                for col in self.num_feat:
                    if __df[col].isnull().sum() != 0:
                        print(col)
                        temp = __df[col].values
                        temp = np.reshape(temp, (-1, 1))
                        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                        imputer.fit(temp)
                        __df[col] = imputer.transform(temp)

                # DEALING WITH NULL VALUES FOR CATEGORICAL VARIABLES WITH MODE values
                for col in self.cat_feat:
                    if __df[col].isnull().sum() != 0:
                        modeval = __df[col].mode()[0]
                        __df[col].fillna(modeval, inplace=True)
                        print(__df[col].unique())
                                   
                
                print("====================================")
                print("imputing null values done",end = "\n\n")

                # ****************************** STEP 5: DEALING WITH DUPLICATE VALUES ******************************
                if self.__df_flag:
                       
                        __df.drop_duplicates(inplace = True)
                        print("====================================")
                        print("Dropped duplicate values")
                        __df.info()
                        print("\n\n")
                 
                # ****************************** STEP 6: OUTLIER ANALYSIS  ******************************
                if self.__df_flag:
                        print("====================================")
                        
                        #  using z score for outlier analysis
                        threshold = 3
        
                        z = np.abs(stats.zscore(__df[self.num_feat]))
                        __df = __df[(z < threshold).all(axis=1)]
                        
                        print(__df.info())
                        print("outlier analysis done",end = "\n\n")
                        
                # ****************************** STEP 7: ENCODING CATEGORICAL VARIABLES ******************************
                
                # need to check for high cardinality among categorical variables
                
                if self.cat_feat:
                    th = 20
                    for col in self.cat_feat:
                        size = len(__df[col])
                        temp = len(__df[col].unique())
                        per_cont = size // temp
                        if per_cont <= th:
                            print(f"removing column {col} because of very high cardinality")
                            __df.drop([col], axis = 1, inplace = True)
                            self.cat_feat.remove(col)
                    
                    #__df = self.encode(__df,target,cat_feat,card, 'OneHotEncoder')
                    
                    print("====================================")
                    print("dataframe before encoding:")
                    print(__df.head(20))
                   
                    __df,self.encoder = self.encode(__df,self.cat_feat)
                    
                    print("====================================")
                    print("dataframe after encoding:")
                    print(__df.head(20))

                else:
                    print("====================================")
                    print("No encoding as no categorical variables present")

                

                # ****************************** STEP 9: SPLITTING THE DATASET ******************************

                if self.__df_flag:
                    
                        self.X_train, self.X_test, self.y_train, self.y_test = self.split(__df)
                        
                        print("====================================")
                        print('Splitting the data into training data and test data')
                        print("====================================")
                        print("-----------X_train-----------\n")
                        print(self.X_train.head(20))
                        print("-----------y_train-----------\n")
                        print(self.y_train.head())
                        print("-----------X_test-----------\n")
                        print(self.X_test.head(20))
                        print("-----------y_test-----------\n")
                        print(self.y_test.head())

      
                # ****************************** STEP 10: SCALE THE DATASET ******************************
                print("====================================")
                print("Scaling the data set")
                print("====================================")
                
                if self.__df_flag:
                        colnames = self.X_train.columns
                        self.X_train = self.X_train.iloc[:,:].values
                        self.X_test = self.X_test.iloc[:,:].values
                        
                        self. X_index = [__df.columns.get_loc(i) for i in self.num_feat]
                    
                        self.scaler = StandardScaler()
                        self.X_train[:,self.X_index] = self.scaler.fit_transform(self.X_train[:, self.X_index])
                        self.X_test[:,self.X_index] = self.scaler.transform(self.X_test[:, self.X_index])
                        
                        #X_train[num_feat] = pd.DataFrame(scaler.fit_transform(X_train[num_feat]), columns = num_feat)
                        #X_test[num_feat] = pd.DataFrame(scaler.transform(X_test[num_feat]), columns = num_feat)
                        
                        self.X_train = pd.DataFrame(self.X_train, columns = colnames)
                        self.X_test = pd.DataFrame(self.X_test, columns = colnames)
                        
                        self.X_train.reset_index(drop=True, inplace=True)
                        self.X_test.reset_index(drop=True, inplace=True)
                        self.y_train.reset_index(drop=True, inplace=True)
                        self.y_test.reset_index(drop=True, inplace=True)
                        
                        self.training_set = self.X_train.join(self.y_train)
                        self.test_set = self.X_test.join(self.y_test)
                        
                        print("-----------train data-----------")
                        print(self.training_set.head(20))
                        print("-----------test data-----------")
                        print(self.test_set.head(20))
        
        
                        print("training set shape:", end = " ")
                        print(self.training_set.shape)
                        print("test set shape:", end = " ")
                        print(self.test_set.shape)
                    
                else:
                    colnames = __df.columns
                    self.X_index = [__df.columns.get_loc(i) for i in self.num_feat]
                    __df = __df.iloc[:,:].values
                    __df[:,self.X_index] = self.scaler.transform(__df[:, self.X_index])
                    
                    
                        

                # ****************************** STEP 10: FINDING IF DATA IS BALANCED OR NOT ******************************
                
                if self.__df_flag:
                    print("====================================")
                    print("Checking if data is balanced or not")
                    print("====================================")
                    
                    self.training_set,self.training_set_bal = self.balance_check(self.training_set,self.target,self.X_train,self.y_train,__colnames)
                    
                    classificationPrepOutput = {'classiftype': self.__cltype,'trainingset': self.training_set,'trainingsetbal': self.training_set_bal,
                                             'testset': self.test_set, 'encoder': self.encoder,'scaler': self.scaler,"target":self.target, "numericalFeatures":self.num_feat,
                                             "ogdata":self.__data,'num_feat':self.num_feat}
                    
                    return classificationPrepOutput
                
                else:
                    return __df
               
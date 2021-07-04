"""
Author: Vishal Kundar

data_preprocessing_regression.py is used when we identify the type of problem.
This is a template that handles data preprocessing for regression based problems.

This is the base stage data preprocessing meaning a different stage is carried out
later.

It takes care of the following:

    1. Finding data properties such as - mean, std, count etc.
    2. Finding the numerical and categorical features.
    3. Handling missing data in the dataset.
    4. Removing duplicate data if present.
    5. Removing outliers i.e extreme data points.
    6. Encoding categorical features.
    7. Feature Engineering(Automated)
    8. Feature scaling using MinMaxScaler.
"""


# Packages
import pandas as pd
import numpy as np
import featuretools as ft

from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class dataPrep:
    """
    Consists of methods to deal with above steps

    Class Variables
    ---------------
    __dataset_original: TYPE: Private class dataframe.
            DESCRIPTION: Holds dataframe recieved using data_Validation.py

    __dataset: TYPE: Private class dataframe.
            DESCRIPTION: Holds dataframe recieved using data_Validation.py
                         but changes are made to this in datapreprocessing.

    __dependent_var: TYPE : String
            DESCRIPTION : Column header of dependant variable

    __df_flag: TYPE : Bool
            DESCRIPTION : Flag indicating preprocessing of train set or
                          new data. If True then train set else new data
    """

    def __init__(self, dataset, y, df_flag=True):
        """
        Constructor to initialize class. Initializes class variable __dataset
        with given dataframe.

        Parameters
        ----------
        dataset : TYPE : Dataframe
            DESCRIPTION : Dataset given by user, converted to dataframe. 

        y :  TYPE : String
            DESCRIPTION : Column header of dependant variable     

        Returns
        -------
        None

        """
        # private class variables
        self.__dataset_original = dataset
        self.__dataset = dataset
        self.__dependent_var = y
        self.__df_flag = df_flag

    def handleMissingData(self, numFeatureList, catFeatureList):
        """
        Function to handle missing data present in both numerical and categorical
        features.

        Parameters
        ----------
        numFeatureList: TYPE: Python list
                DESCRIPTION: List consisting of column headers of all numerial 
                features

        catFeatureList: TYPE: Python list
                DESCRIPTION: List consisting of column headers of all categorical 
                features          

        Returns
        -------
        missingDataFlag: TYPE: boolean
                DESCRIPTION: If missing data is present the user needs to be informed
                of the changes being done. Used to inform user of missing data.

        missingDataContribution: TYPE: float
                DESCRIPTION: If missing data is present then we calculate the amount
                of missing data present in features.       

        """
        missingDataFlag = False
        missingDataContribution = 0.0

        # First we will handle missing data for numerical features
        holder = list()  # contains name of columns having missing data
        for column in numFeatureList:
            if(self.__dataset[column].isnull().sum() > 0):
                holder.append(column)

        if(holder):
            # If the list is not empty meaning missing value is present
            missingDataFlag = True

            # Contribution calculation takes into account cat variables
            # too (if present).
            data_null = self.__dataset.isna().mean().round(4) * 100
            missingDataContribution = data_null.sum()

            # Handling missing data by replacing with mean value.
            si = SimpleImputer(missing_values=np.nan, strategy='mean')
            si.fit(self.__dataset[holder].values)
            self.__dataset[holder] = si.transform(
                self.__dataset[holder].values)

        # Handling missing data for categorical variables
        holder = list()
        for column in catFeatureList:
            if(self.__dataset[column].isnull().sum() > 0):
                holder.append(column)

        if(holder):
            # If the list is not empty meaning missing value is present
            missingDataFlag = True

            # Contribution calculation takes into account numerical variables
            # too (if present).
            data_null = self.__dataset.isna().mean().round(4) * 100
            missingDataContribution = data_null.sum()

            for column in holder:
                # Mode retrieves most frequent categories
                most_frequent_category = self.__dataset[column].mode()[0]

                # replacing na with most frequent category
                self.__dataset[column].fillna(
                    most_frequent_category, inplace=True)

        return missingDataFlag, missingDataContribution

    def handleOutlierData(self, numFeatureList):
        """
        Function to handle outliers present in data using Z-score method

        Z-score:
        This score helps to understand if a data value is greater or smaller than mean 
        and how far away it is from the mean.If the z score of a data point is more than 3/-3, 
        it indicates that the data point is quite different from the other data points. Such a 
        data point can be an outlier.

        Parameters
        ----------
        numFeatureList: TYPE: Python list
                DESCRIPTION: List consisting of column headers of all numerial 
                features        

        Returns
        -------
        outlierDataFlag: TYPE: boolean
                DESCRIPTION: If outliers are present the user needs to be informed
                of the changes being done. Used to inform user of outliers in data.

        outlierDataContribution: TYPE: float
                DESCRIPTION: If outliers are present in data then we calculate the
                contribution of the outliers.    

        """
        outlierDataFlag = False
        outlierDataContribution = 0.0

        # Calculating z-score of all numerical features
        z = np.abs(stats.zscore(self.__dataset[numFeatureList]))
        threshold = 3
        # Gives 2 arrays containing row number and column number of outliers
        indexArray = np.where(z > threshold)

        # Calculating contribution of outliers in the dataset
        outlierDataContribution = (
            len(indexArray[0]) / (len(self.__dataset) * len(self.__dataset.columns))) * 100

        if(outlierDataContribution != 0.0):
            # Outliers present. Removing them
            outlierDataFlag = True

            # This should not effect categorical features. TEST!!!
            self.__dataset = self.__dataset[(z < threshold).all(axis=1)]

        return outlierDataFlag, outlierDataContribution

    def encodeCatFeatures(self, catFeatureList):
        """
        Function to encode categorical features present in data set.

        Parameters
        ----------
        catFeatureList: TYPE: Python list
                DESCRIPTION: List consisting of column headers of all categorical 
                features        

        Returns
        -------
        catFeatureList: TYPE: Python list
                DESCRIPTION: List consisting of column headers of all converted 
                categorical features 

        """
        # We will user one hot encoder to encode our categroical variables
        # drop_first = true avoids the dummy variable trap
        if not catFeatureList:
            return [] #No categrocial features to encode

        dummies = pd.get_dummies(
            self.__dataset[catFeatureList], drop_first=True)

        # Concatenating dummies with dataset
        self.__dataset = pd.concat([self.__dataset, dummies], axis=1)

        # Drop the original cat variables as dummies are already created
        self.__dataset.drop(catFeatureList, axis=1, inplace=True)

        # New convCatFeatureList - converted cat feature
        convCatFeatureList = list(dummies.columns)

        return convCatFeatureList

    def featureEngineering(self):
        """
        Automated feature engineering

        Returns
        -------
        None.

        """
        #Need a unique id in our data set
        self.__dataset.insert(0, 'uniq_ID', range(1, 1 + len(self.__dataset)))
        
        #Creating entity set
        #An EntitySet is a structure that contains multiple dataframes and 
        #relationships between them
        es = ft.EntitySet(id = 'data')
        
        #adding dataframe to entity set
        es.entity_from_dataframe(entity_id='data_pred', dataframe=self.__dataset, 
                                 index = 'uniq_ID')
        
        #Use Deep Feature Synthesis to create new features automatically.
        #target_entity is nothing but the entity ID for which we wish to create new features
        #max_depth controls the complexity of the features being generated by stacking the primitives.
        #n_jobs helps in parallel feature computation by using multiple cores.
        
        feature_matrix, feature_names = ft.dfs(entityset=es, 
                                               target_entity = 'data_pred', 
                                               max_depth = 1, verbose = 1)
        
        #There is one issue with this dataframe – it is not sorted properly. 
        #We will have to sort it based on the id variable
        feature_matrix = feature_matrix.reindex(index=self.__dataset['uniq_ID'])
        feature_matrix = feature_matrix.reset_index()
        feature_matrix.drop(['uniq_ID'], axis=1, inplace=True)
        
        return feature_matrix #New dataframe
            
    def preprocess(self):
        """
        Central datapreprocessing function. Executes the various stages in order
        and returns a dictionary consisting the results of the preprocessing stage.

        Parameters
        ----------
        None 

        Returns
        -------
        regressionPrepOutput: TYPE: Python Dictionary
                    DESCRIPTION: A dictionary consisiting of the output of the preprocessing stage 1 of
                    regression.

        """
        # Feature selection and engineering done after bias/variance trade-off analysis.

        # Properties of the dataset containing info about mean, std, count etc.
        # data_properties is a dataframe.
        data_properties = self.__dataset.describe()

        # Finding numeric and categorical features.
        catFeatureList = list(
            self.__dataset.select_dtypes(include=['object']).columns)
        numFeatureList = list(
            self.__dataset.select_dtypes(exclude=['object']).columns)   

        # Since in regression problem the dependent variable is never categorical we will remove it
        # from the numFeatureList
        if self.__df_flag:
            numFeatureList.remove(self.__dependent_var)

        # Handling missing data for both cat and non-cat features
        missingDataFlag, missingDataContribution = self.handleMissingData(
            numFeatureList, catFeatureList)

        if self.__df_flag:
            # Handling duplicate data if present
            duplicateDataFlag = False
            duplicateDataContribution = 0.0
    
            duplicateDataContribution = self.__dataset.duplicated(
            ).value_counts(normalize=True) * 100
            duplicateDataContribution = 100.0 - duplicateDataContribution[0] 
    
            if(duplicateDataContribution != 0.0):
                self.__dataset.drop_duplicates(inplace=True, keep = 'first')
                duplicateDataFlag = True 
            else:
                duplicateDataFlag = False

            # Handling outliers. Outliers are extreme data points that drastically effects our parameters.
            # We will use Z-score for handling them.
            # This is done only for numerical features.
            outlierDataFlag, outlierDataContribution = self.handleOutlierData(
                numFeatureList)
        
            # Getting dependent feature from dataset
            Y = self.__dataset[self.__dependent_var].values
    
            # Removing dependent feature from dataset to carry out feature encoding
            self.__dataset.drop([self.__dependent_var], axis=1, inplace=True)
        
        #Feature Engineering
        self.__dataset = self.featureEngineering()
        
        # Encoding categorical features
        convCatFeatureList = self.encodeCatFeatures(catFeatureList)

        # Getting our feature set and also index of numerical features
        X = self.__dataset.iloc[:, :].values
        X_index = [self.__dataset.columns.get_loc(i) for i in numFeatureList]

        if self.__df_flag:        
            # Split data set
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.1, random_state=0)
    
            # Feature scaling - using minmax as data distribution is not known
            scaler = MinMaxScaler()
            X_train[:, X_index] = scaler.fit_transform(X_train[:, X_index])
            X_test[:, X_index] = scaler.transform(X_test[:, X_index])

            # Creating dictionary of items to be returned
            regressionPrepOutput = {'Xtrain': X_train, 'Xtest': X_test,
                                    'Xindex': X_index, 'Ytrain': y_train, 'Ytest': y_test,
                                    'scaler': scaler, 'X': X, 'Y': Y, 'dataset': self.__dataset,
                                    'datasetOriginal': self.__dataset_original,
                                    'dependentFeature': self.__dependent_var, 'encodedCatFeatures': convCatFeatureList,
                                    'numericalFeatures': numFeatureList, 'categoricalFeatures': catFeatureList,
                                    'dataProperties': data_properties, 'missingDataFlag': missingDataFlag,
                                    'missingDataContribution': missingDataContribution,
                                    'duplicateDataFlag': duplicateDataFlag, 'duplicateDataContribution': duplicateDataContribution,
                                    'outlierDataFlag': outlierDataFlag, 'outlierDataContribution': outlierDataContribution}
        else:
            #Creating dictionary of items to be returned
            regressionPrepOutput = {"X": X, 'Xindex': X_index, 'dataset': self.__dataset,
                                    'datasetOriginal': self.__dataset_original,
                                    'dependentFeature': self.__dependent_var,
                                    'missingDataFlag': missingDataFlag,
                                    'missingDataContribution': missingDataContribution}

        return regressionPrepOutput
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
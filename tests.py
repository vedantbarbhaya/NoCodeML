"""
Author: Vishal Kundar

tests.py is used for test cases and carrying out testing.
"""


##############################
#Tests for data_validation.py - Pass
##############################
"""
from data_validation import data_check as dc
obj = dc("/Users/vishalkundar/Downloads/Salary_Data.csv")

#Should give error
print(obj.__filename)
obj.__filename = "heello.csv" #Can be set but should not be allowed to.
print(obj.__filename)

log = obj.identify_file()
print(log)

df = obj.file_to_dataframe()
print(df)
print(type(df))

log = obj.validation_check(df)
print(log)

print(obj.identify_problem(df, "Salary")) 
#Change validation_Check if data contains less than 300 for testing
"""

##############################
#Tests for data_preprocessing_regression.py - Pass
##############################
"""
import pandas as pd
df = pd.read_csv("/Users/vishalkundar/Downloads/Salary_Data.csv")

from data_preprocessing_regression import dataPrep as dpr
obj = dpr(df, "Salary")
output = obj.preprocess()

for key in output.keys():
    print("key: ", key)
    print("value: ", output[key])

#CHECK DUPLICATE DATA CONTRIBUTION
"""

###############################
#Test for automated feature selection - DONE
###############################
"""
from sklearn.datasets import make_regression
from matplotlib import pyplot

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, 
                       noise=0.1, random_state=1)

#Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


#2 techniques
#---------------
#correlation feature selection : Linear correlation scores are typically 
#a value between -1 and 1 with 0 representing no relationship. For feature 
#selection, we are often interested in a positive score with the larger 
#the positive value, the larger the relationship, and, more likely, the 
#feature should be selected for modeling. As such the linear correlation 
#can be converted into a correlation statistic with only positive values.

#Less support for categorical features

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# feature selection
def select_features_co(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k=88) #k number of features can be set to 'all' if needed
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# feature selection
X_train_fs_co, X_test_fs_co, fsco = select_features_co(X_train, y_train, X_test)
for i in range(len(fsco.scores_)):
	print('Feature %d: %f' % (i, fsco.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fsco.scores_))], fsco.scores_)
pyplot.title("correlation feature selection")
pyplot.show()

#---------------
#mutual information statistics: Mutual information from the field of information 
#theory is the application of information gain (typically used in the 
#construction of decision trees) to feature selection. Mutual information is
#calculated between two variables and measures the reduction in uncertainty 
#for one variable given a known value of the other variable.

#This is better for categorical features

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
# feature selection
def select_features_mutual(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_regression, k=88) #k number of features can be set to 'all' if needed
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs


# feature selection
X_train_fs_m, X_test_fs_m, fsm = select_features_mutual(X_train, y_train, X_test)
for i in range(len(fsm.scores_)):
	print('Feature %d: %f' % (i, fsm.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fsm.scores_))], fsm.scores_)
pyplot.title("mutual information statistics")
pyplot.show()

#modelling

#Without feature selection and all features
#----------------
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

#With correlation feature selection and top 88 features
#----------------
# fit the model
model = LinearRegression()
model.fit(X_train_fs_co, y_train)
# evaluate the model
yhat = model.predict(X_test_fs_co)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

#With mutual information statistics and top 88 features
# fit the model
model.fit(X_train_fs_m, y_train)
# evaluate the model
yhat = model.predict(X_test_fs_m)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

#Trying to search optimal value of k using grid search
#We evaluate model configurations on regression tasks using 
#repeated stratified k-fold cross-validation. We will use three 
#repeats of 10-fold cross-validation via the RepeatedKFold class.

# define the evaluation method
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#Defining pipeline
from sklearn.pipeline import Pipeline
# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])

#Note that the grid is a dictionary mapping of parameter-to-values to search, 
#and given that we are using a Pipeline, we can access the SelectKBest object
#via the name we gave it ‘sel‘ and then the parameter name ‘k‘ separated by two
#underscores, or ‘sel__k‘.

# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X.shape[1]-20, X.shape[1]+1)]

#In this case, we will evaluate models using the negative mean absolute error 
#(neg_mean_absolute_error). It is negative because the scikit-learn requires 
#the score to be maximized, so the MAE is made negative, meaning scores scale 
#from -infinity to 0 (best).

# define the grid search
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X, y)
# summarize best
print('Best MAE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))
"""

##############################
#Test for feature engineering - DONE
##############################
"""
import featuretools as ft
import numpy as np
import pandas as pd
from data_preprocessing_regression import dataPrep as dpr

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("/Users/vishalkundar/Downloads/Datasets/CarPrice_Assignment.csv")
print(df.head())


#Preprocessing
dpr_obj = dpr(df, df.columns[-1])
result = dpr_obj.preprocess()
df = result['dataset']
print(df.head())


#Need a unique id in our data set
df.insert(0, 'uniq_ID', range(1, 1 + len(df)))
print(df.head())

#Creating entity set
#An EntitySet is a structure that contains multiple dataframes and 
#relationships between them
es = ft.EntitySet(id = 'data1')
#addidng df to es
es.entity_from_dataframe(entity_id='data_pred1', dataframe=df, index = 'uniq_ID')

#Use Deep Feature Synthesis to create new features automatically.
#target_entity is nothing but the entity ID for which we wish to create new features
#max_depth controls the complexity of the features being generated by stacking the primitives.
#n_jobs helps in parallel feature computation by using multiple cores.

feature_matrix, feature_names = ft.dfs(entityset=es, 
                                       target_entity = 'data_pred1', 
                                       max_depth = 1, verbose = 1)
#new features list
print(feature_matrix.columns)

#There is one issue with this dataframe – it is not sorted properly. 
#We will have to sort it based on the id variable
feature_matrix = feature_matrix.reindex(index=df['uniq_ID'])
feature_matrix = feature_matrix.reset_index()
print(feature_matrix.head())

#now split, scale and train.
"""

"""
import matplotlib.pyplot as plt
import numpy as np
y_pred = [0.65271575, 0.84146743, 0.79544005, 0.90096778, 0.6480687,
          0.60715043]
y_test = [0.64, 0.85, 0.8, 0.91, 0.68, 0.54]

mini = int(min(y_test) -1)
maxi = int(max(y_test) + 1)
print(mini, maxi)
y_col = list(np.linspace(mini,maxi,len(y_test)))

#Plotting y_pred vs y_test
plt.plot(y_pred, y_col, '-r', label="Y pred")
plt.plot(y_test, y_col, '-b', label="Y test")
plt.title("y_pred vs y_test")
plt.legend()
plt.show()
"""

"""
import os
print("\nPress enter to view results: ")
input()
os.system("clear")
print("---------------------------------------------------------------------------")
print("\t\t\t\t\t\t\t\t RESULTS")
print("---------------------------------------------------------------------------")
"""

import webbrowser
url = "/Users/vishalkundar/Downloads/report.html"
url = "file:///Users/vishalkundar/Downloads/report.html"
webbrowser.open(url, new=2)  # open in new tab


                      


# coding: utf-8

# In[364]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt


# In[365]:

file = 'German.xlsx'


# In[225]:

data = pd.read_excel(file)


# In[231]:

data.head(4)


# In[233]:

# checking the column names and data types
data.info()


# In[ ]:

# we can see that a number of our columns contain text while the remainder are numeric


# In[ ]:




# In[ ]:

# Checking for null values


# In[234]:

data.isnull().sum()


# # Exploring object data

# In[ ]:

# getting list of object columns


# In[413]:

listo = data.select_dtypes(['object']).columns
print(listo)


# In[ ]:

#plotting object data
# using countplots to view each variable by class frequency


# In[389]:

sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (15,15))


# In[390]:

ax1 = sns.countplot(data["Credit History"],ax = axes[0,0])
for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
ax2 = sns.countplot(data["Checking Account Status"],ax = axes[0,1])
for p in ax2.patches:
    height = p.get_height()
    ax2.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
ax3 = sns.countplot(data["Purpose"],ax = axes[1,0])
for p in ax3.patches:
    height = p.get_height()
    ax3.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
ax4 = sns.countplot(data["Savings Account/Bonds"],ax = axes[1,1])
for p in ax4.patches:
    height = p.get_height()
    ax4.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
plt.show()


# In[391]:

sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (15,15))


# In[392]:

ax6 = sns.countplot(data["Present Employment since"],ax = axes[0,0])
for p in ax6.patches:
    height = p.get_height()
    ax6.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
ax7 = sns.countplot(data["Personal Status"],ax = axes[0,1])
for p in ax7.patches:
    height = p.get_height()
    ax7.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
ax8 = sns.countplot(data["Property"],ax = axes[1,0])
for p in ax8.patches:
    height = p.get_height()
    ax8.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
ax9 = sns.countplot(data["Other installment plans"],ax = axes[1,1])
for p in ax9.patches:
    height = p.get_height()
    ax9.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
plt.show()


# In[393]:

sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (15,15))


# In[394]:

ax2 = sns.countplot(data["Housing"],ax = axes[0,0])
for p in ax2.patches:
    height = p.get_height()
    ax2.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
ax2 = sns.countplot(data["Job type"],ax = axes[0,1])
for p in ax2.patches:
    height = p.get_height()
    ax2.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
ax2 = sns.countplot(data["Foreign worker"],ax = axes[1,0])
for p in ax2.patches:
    height = p.get_height()
    ax2.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
plt.show()


# In[ ]:




# In[ ]:




# # Exploring Numerical Data

# In[ ]:

# Listing numerical variables


# In[18]:

data1 = data.select_dtypes(['float64','int64']).columns


# In[19]:

data2 = data[data1]


# In[20]:

# descriptive statistics for numeric data


# In[21]:

data2.describe()


# In[22]:

# numeric correlations


# In[26]:

data2.corr()


# In[27]:

sns.heatmap(data2.corr(),annot=True, fmt=".2f")
plt.show()


# In[ ]:




# In[ ]:




# In[28]:

sns.pairplot(data2)
plt.show()


# In[ ]:

# not surprising linear relationship between duration of loan and amount
# the majority of data is categorical


# In[243]:

sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (30,30))


# In[414]:

sns.boxplot(data['Credit Amount'],ax = axes[0,0])
sns.boxplot(data['Duration in month'],ax = axes[0,1])
sns.boxplot(data['Installment rate in % of disposable income'],ax = axes[1,0])
sns.boxplot(data['Present residence since'],ax = axes[1,1])


# In[ ]:




# In[ ]:

sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (30,30))


# In[ ]:

sns.boxplot(data['Age'],ax = axes[0,0])
sns.boxplot(data['Number of dependents'],ax = axes[0,1])
sns.boxplot(data['Number of existing credits'],ax = axes[1,0])
sns.boxplot(data['Credit Rating'],ax = axes[1,1])
plt.show()


# # Building XGBoost model

# In[ ]:




# In[415]:

import pip


# In[418]:

import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import numpy as np
import scipy.sparse
import pickle
import xgboost 
import scipy


# In[417]:

pip.main(['install','xgboost'])



# In[252]:




# In[419]:

from xgboost.sklearn import XGBClassifier


# In[ ]:

# Selecting the target variable


# In[258]:

y = data['Credit Rating']


# In[262]:

y.head()


# In[337]:

# selecting predictor variables


# In[338]:

x =  data.drop('Credit Rating', 1)


# In[339]:

x.head()


# In[340]:

# Splitting data into training and testing


# In[264]:

seed = 7
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
random_state=seed)


# In[ ]:

# converting categorical object data to numeric using one hot encoding


# In[266]:

one_hot_encoded_training_predictors = pd.get_dummies(x_train)
one_hot_encoded_test_predictors = pd.get_dummies(x_test)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)


# In[ ]:

# Model used will be xgboost classifier


# In[257]:

model = XGBClassifier()


# In[306]:

#fit model on training data
# using an early stopping parameter to prevent over fitting


# In[307]:

eval_set = [(final_test, y_test)]


# In[308]:

model.fit(final_train, y_train, early_stopping_rounds=10, eval_metric="logloss",
eval_set=eval_set, verbose=True)


# In[309]:

print("Making predictions for the following customers:")
print(final_test.head())
print("The predictions are")
y_predictions = model.predict(final_test.head())
print(y_predictions)
print("The Actual values are")
print(y_test.head())


# In[343]:

# Making prediction on the test data


# In[342]:

y_pred = model.predict(final_test)


# In[344]:

# accuracy score determines performance of the model


# In[345]:

from sklearn.metrics import accuracy_score


# In[346]:

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[347]:

# using xgboost to plot the important features in the model


# In[334]:

import matplotlib.pyplot as plt
plt.figure(figsize=(40,20))
from xgboost import plot_importance
plot_importance(model)
plt.show()


# In[348]:

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# In[363]:

learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(final_train,y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:




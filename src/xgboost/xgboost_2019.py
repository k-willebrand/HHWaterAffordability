# example implementation: https://mljar.com/blog/feature-importance-xgboost/

# Description: get feature importance from parallel boosting trees algorithm (Xgboost model)

import numpy as np
import pandas as pd
import os
import pickle
import shap

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import seaborn as sns  # for correlation heatmap

from xgboost import XGBRegressor

# set working directory
os.chdir('/Users/keaniw/Documents/Classes/CS229 Machine Learning/Project/Project Code/cs229_project/src')

def picklesave(filename, obj):
    """save python object for easy future loading using pickle (binary)

        Args:
             filename: filename as string to save object to (e.g., sample.txt)
             obj: the object/variable to saved

        """
    # write binary
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def pickleload(filename):
    """load previously saved python object from pickle (binary)

        Args:
             filename: filename as string where object previously saved to with pickle (e.g., sample.txt)

        Returns:
             obj: the previously saved python object

        """
    # read binary
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj

#%% load the merged database of household level data and affordability metrics and specify training and test dataset

# load the merged dataset
df_merged = pickleload('data/df_merged_2019_agg.txt')
df_merged.drop(columns=['account', 'is_delinq'], axis=1, inplace=True)

#%% prepare data for use with sklearn

# one-hot encode categorical variables
df_merged = pd.get_dummies(df_merged)

# separate data into features (x) and response variables (y)
X = df_merged.drop(columns=['pen_freq', 'debt_dur', 'debt_val'])  # candidate features
X_names = list(X.columns)
Y = df_merged[['pen_freq', 'debt_dur', 'debt_val']]  # candidate response variables
Y_names = list(Y.columns)

# convert data to numpy array for use with sklearn algorithm
X = np.array(X)
Y = np.array(Y)

# create a random training and validation set of features (x) and response variables (y)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=0)
print(f'Training features shape: {train_x.shape}')
print(f'Training response shape: {train_y.shape}')
print(f'Test features shape: {test_x.shape}')
print(f'Test response shape: {test_y.shape}')

#%% fit the XGBoost Regression model

# the number of different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']

# fit random forest for each
for i, metric in enumerate(aff_metrics):

    # first fit baseline model
    print('')
    print(f'Baseline model for {metric}:')
    pred_y = np.ones_like(test_y[:, i]) * np.mean(train_y[:, i])  # predict average metric
    abs_err = np.sum(abs((pred_y - test_y[:, i])))  # L1
    print(f'absolute error: {abs_err}')
    MSE = np.mean((pred_y - test_y[:, i]) ** 2)
    print(f'MSE: {MSE}')
    TSS = np.sum((test_y[:, i] - np.mean(test_y[:, i])))
    RSS = np.sum((pred_y - test_y[:, i]) ** 2)  # L2^2
    print(f'RSS: {RSS}')
    R2 = (TSS - RSS) / TSS
    print(f'R2: {R2}')

    # create xgboost model object
    xgb = XGBRegressor(n_estimators=100)

    # train the model
    xgb.fit(train_x, train_y)

    # make predictions on test data
    pred_y = xgb.predict(test_x)

    # calculate mean absolute percentage error (MAPE), to then compute accuracy
    print('')
    print(f'XGBoost model for {metric}:')
    abs_err = np.sum(abs((pred_y - test_y[:, i])))  # L1
    print(f'absolute error: {abs_err}')
    MSE = np.mean((pred_y - test_y[:, i]) ** 2)
    print(f'MSE: {MSE}')
    TSS = np.sum((test_y[:, i] - np.mean(test_y[:, i])) ** 2)
    RSS = np.sum((pred_y - test_y[:, i]) ** 2)  # L2^2
    print(f'RSS: {RSS}')
    R2 = (TSS - RSS) / TSS
    print(f'R2: {R2}')

    # plot and save the feature importances (bar)
    sorted_idx = xgb.feature_importances_.argsort()
    plt.barh(Y_names[sorted_idx], xgb.feature_importances_[sorted_idx])
    plt.xlabel("XGBoost Feature Importance")
    plt.title(f'Feature Importance for Metric: {metric}')
    plt.gca().yaxis.grid(True)
    plt.gcf().set_size_inches(12, 5)
    plt.tight_layout()
    plt.savefig(f'xgboost/featimport_{metric}_2019.png', bbox_inches='tight')
    plt.show()


# example implementation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model

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
df_merged = pickleload('data/df_merged.txt')

# one-hot encode categorical variables
#df_merged = pd.get_dummies(df_merged)

# separate data into features (x) and response variables (y)
X = df_merged.drop(columns=['account', 'pen_freq', 'debt_dur', 'debt_val'])  # candidate features
Y = df_merged[['pen_freq', 'debt_dur', 'debt_val']]  # candidate response variables

# standardize numeric features (for regression)
X.loc[:, X.dtypes != 'category'] = (X.loc[:, X.dtypes != 'category'] - X.loc[:, X.dtypes != 'category'].mean(numeric_only=True))\
                            / X.loc[:, X.dtypes != 'category'].std(numeric_only=True, axis=0)
#Y.loc[:, Y.dtypes != 'category'] = (Y.loc[:, Y.dtypes != 'category'] - Y.loc[:, Y.dtypes != 'category'].mean(numeric_only=True))\
#                            / Y.loc[:, Y.dtypes != 'category'].std(numeric_only=True, axis=0)


# one-hot encode categorical variables
X = pd.get_dummies(X)

# get feature and predictor names
X_names = list(X.columns)
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

#%% Fit the Lasso Regression Model for each affordability metric

# the number of different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']

for i, metric in enumerate(aff_metrics):

    # create model object
    clf = linear_model.Lasso(alpha=0.1)

    # fit the lasso regression model for the household level affordability metric
    clf.fit(train_x, train_y[:, i])

    # get the parameters of the model
    params = clf.get_params()

    # make a prediction using the test data
    pred_y = clf.predict(test_x)

    # calculate mean absolute percentage error (MAPE), to then compute accuracy
    print('')
    print(f'RF model for {metric}:')
    abs_err = np.sum(abs((pred_y - test_y[:, i])))  # L1
    print(f'absolute error: {abs_err}')
    MSE = np.mean((pred_y - test_y[:, i]) ** 2)
    print(f'MSE: {MSE}')
    TSS = np.sum((test_y[:, i] - np.mean(test_y[:, i]))**2)
    RSS = np.sum((pred_y - test_y[:, i])**2)  # L2^2
    print(f'RSS: {RSS}')
    R2 = (TSS - RSS) / TSS
    print(f'R2: {R2}')

    # make a bar plot of variable coefficients (weight feature importance)
    plt.figure()
    plt.bar(list(range(len(clf.coef_))), clf.coef_, color='steelblue', edgecolor='white', linewidth=1)
    plt.xticks(list(range(len(clf.coef_))), X_names)
    plt.xticks(fontsize=5, rotation=90)
    plt.ylabel('Coefficient Value')
    plt.xlabel('Feature Name')
    plt.title(f'Coefficient Value vs. Feature for Metric: {metric}')
    plt.gca().yaxis.grid(True)
    plt.gcf().set_size_inches(12, 5)
    plt.tight_layout()
    plt.savefig(f'lasso/coeff_{metric}.png', bbox_inches='tight')
    plt.show()

    # scatter plot true vs predicted values
    plt.figure()
    plt.scatter(test_y[:, i], pred_y, color='chocolate', s=10, alpha=0.5, label='Lasso Regression Model')
    plt.scatter(test_y[:, i], np.ones_like(test_y[:, i]) * np.mean(train_y[:, i]), color='steelblue', s=10,
                alpha=0.5, label='Baseline Model')
    plt.plot(test_y[:, i], test_y[:, i], 'k-', label='Theoretical Perfect Predictive Model (x=y)')
    plt.xlabel(f'True {metric}')
    plt.ylabel(f'Predicted {metric}')
    plt.title(f'True vs. Predicted {metric} for Lasso Regression Model')
    plt.legend()
    plt.grid()
    plt.savefig(f'lasso/scatter_{metric}.png', bbox_inches='tight')
    plt.show()



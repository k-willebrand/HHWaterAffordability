# Description: merges household level and affordability metrics data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sqlite3
import pickle

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

#%% load household level affordability metric data

# load file household level affordability metric data
csvpath = 'data/acc_metric_data_all.csv'
afford_data = pd.read_csv(csvpath, header=0)  # fields: account, is_deliq, acc_length

# save selected household level affordability metrics by account in pandas dataframe
df_afford = afford_data[['account', 'pen_freq', 'debt_dur', 'debt_val']]

# calculate number of unique accounts in df_afford (out: 61611)
num_accounts = df_afford.drop_duplicates(subset='account', inplace=False).shape[0]

#%% specify feature names for household level sociodemographic and housing data for SQL query

# load file containing candidate feature names form the current unmerged database
csvpath = 'data/Database_Column_Description_select.csv'
feat_descript = pd.read_csv(csvpath, header=0)
feat_names = np.array(feat_descript['Column Header'])  # array of feature names
feat_query = ", ".join(feat_names)  # common seperated string for SQL query

#%% load household level sociodemographic and housing data from SQL database

# specify SQL database file path
dbpath = 'data/wudb.db'

# create a SQL connection to the SQLite database
conn = sqlite3.connect(dbpath)

# create cursor
c = conn.cursor()

# run SQL query on database and extract feature data
c.execute(" ".join(["SELECT", feat_query, "FROM wudata"]))
wudata = c.fetchall()

# close SQL connection to SQLite database
conn.close()

# save variable as list of tuples using pickle
picklesave('data/wudata.txt', wudata)

#%% create and save selected sociodemographic and housing data in pandas dataframe

# note: this dataframe is wudb data prior to merge and preprocessing
df_wudata = pd.DataFrame(wudata, columns=feat_names)
picklesave('data/df_wudata.txt', df_wudata)

#%% perform additional preproccessing on df_wudata to investigate available features

# load the saved dataframe
df_wudata = pickleload('data/df_wudata.txt')

# remove commas in strings for data type compatability
df_wudata = df_wudata.replace(',', '', regex=True)

# remove extra 'Year_Built' column from output dataframe
df_wudata = df_wudata.loc[:, ~df_wudata.columns.duplicated()].copy()

# omit features known to be not representative of response variables
# omit household size features (response only considers single family)
df_wudata = df_wudata.iloc[np.where(df_wudata['restype'] == 'SF')[0]]
df_wudata.drop(columns=['restype', 'num_units'], inplace=True)
# omit billing data features
#df_wudata.drop(columns=['acc_con', 'citylim', 'totwuse', 'normtotwuse'], inplace=True)
df_wudata.drop(columns=['acc_con', 'citylim'], inplace=True)
# omit census tract and block group numbers: we are interested in the properties of census groups
df_wudata.drop(columns=['Tract_y', 'BkGp_y'], inplace=True)
# omit unused features
df_wudata.drop(columns=['X_1', 'X_y', 'Carport', 'Concrete', 'Decks', 'Other_Buildings', 'Parcel', 'Porch'], inplace=True)

# first pass: remove duplicate rows, arising from water billing data
df_wudata.drop_duplicates(inplace=True)

# check datatypes in dataframe and specify desired column dtypes
feat_dtypes = df_wudata.dtypes
intcols = ['or_tot', 'yr_bt_own', 'yr_bt_rnt', 'X_of_Units',
           'Bedrooms', 'Effective_Year', 'Fireplaces',
           'Room_Count', 'Year_Built']
catcols = ['account', 'Bathrooms_F_H', 'Condition', 'General_Plan', 'Heat',
           'Pool', 'Roof', 'Sanitation', 'Spa', 'Topography', 'View',
           'Water', 'Zoning']
floatcols = ['totwuse', 'normtotwuse', 'hc_own_20K_20', 'hc_own_20K_29', 'hc_own_20K_30_p', 'hc_rnt_20K_20', 'hc_rnt_20K_29', 'hc_rnt_20K_30_p',
             'hc_rnt_35K_20', 'hc_rnt_35K_29', 'hc_rnt_35K_30_p', 'hc_rnt_50K_20', 'hc_rnt_50K_29', 'hc_rnt_50K_30_p',
             'aggrm_own', 'aggrm_rnt', 'Garage', 'Main_Area', 'Parcel_Size_acres', 'tax_value']

# replace occurrence of 'NA' and 'N/A' strings appropriate nulltype and update datatypes

# integers
df_wudata[intcols] = df_wudata[intcols].fillna(-9999)
df_wudata[intcols] = df_wudata[intcols].replace('NA', -9999)
df_wudata[intcols] = df_wudata[intcols].replace('N/A', -9999)
df_wudata[intcols] = df_wudata[intcols].astype('int64')

# categoricals
df_wudata[df_wudata[catcols] == 'NA'] = np.nan
df_wudata[df_wudata[catcols] == 'N/A'] = np.nan
df_wudata[df_wudata[catcols] == 'nan'] = np.nan
df_wudata[df_wudata[catcols] == 'None'] = np.nan
df_wudata[catcols] = df_wudata[catcols].astype('category')

# floats
df_wudata[df_wudata[floatcols] == 'NA'] = np.nan
df_wudata[df_wudata[floatcols] == 'N/A'] = np.nan
df_wudata[df_wudata[floatcols] == 'nan'] = np.nan
df_wudata[df_wudata[floatcols] == 'None'] = np.nan
df_wudata[floatcols] = df_wudata[floatcols].astype('float64')
df_wudata[floatcols] = df_wudata[floatcols].round(decimals=5)  # account for rounding errors
#df_wudata[floatcols] = format(df_wudata[floatcols], ".5f")

# check updated datatypes in dataframe to confirm correct
feat_dtypes = df_wudata.dtypes

# keep most complete row with data for duplicated accounts
df_wudata['num_nans'] = df_wudata.isnull().sum(1)
df_wudata = df_wudata.sort_values(by=['account', 'num_nans'], ascending=[True, True])
df_wudata = df_wudata.drop_duplicates(subset='account', keep='first')
df_wudata.drop(columns=['num_nans'], inplace=True)

# drop features with excessive missing data: num_nans (>=20,000)
num_nans = df_wudata.isnull().sum(0)  # missing data in each column
cols_drop = np.where(num_nans >= 20000)
df_wudata.drop(df_wudata.columns[cols_drop], axis=1, inplace=True)

# finally, drop rows/accounts with missing data fields: num_nans (> 0)
num_nans = df_wudata.isnull().sum(1)  # missing data in each row
rows_drop = np.where(num_nans > 0)[0]
df_wudata.drop(index=df_wudata.index[rows_drop], axis=0, inplace=True)

#%% merge with affordability metrics data and save merged dataset

# merge datasets by account number
df_merged = pd.merge(df_wudata, df_afford, how="inner", on='account')

# calculate number of unique accounts in df_merged (out: 18744)
num_accounts = df_wudata.drop_duplicates(subset='account', inplace=False).shape[0]
# note: df_afford contains 61611 unique account (no duplicates)

# save the merged dataset for easy future loading
picklesave('data/df_merged.txt', df_merged)

#%% plot histogram of affordability metrics in the merged data
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']
fig, axes = plt.subplots(nrows=len(aff_metrics), ncols=1)
fig.tight_layout()
for i, metric in enumerate(aff_metrics):

    plt.subplot(int(str(len(aff_metrics)) +str(1) + str(i+1)))
    plt.hist(df_merged[metric], color='steelblue', bins=50)
    plt.ylabel('Frequency')
    plt.title(f'Histogram for {metric}')

plt.savefig('data/hist_metrics.png', bbox_inches='tight')

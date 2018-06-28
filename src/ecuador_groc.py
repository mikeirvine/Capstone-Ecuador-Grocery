import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from datetime import timedelta



'''
DATA PREPARATION:
- Kaggle data sets: Provided train dataset contains 125M rows of data from Jan 1,
2013 to Aug 15, 2017. Provided test dataset contains data from Aug 16, 2017 to
Aug 31, 2017, so the challenge is to only predict unit_sales on two weeks of unseen data.
- Create smaller train/test sets: To reduce the data size, I create a separate
train dataset that contains data from Aug 15, 2015 to Aug 15, 2016. I created a
separate test dataset that contains data from Aug 16, 2016 to Aug 15, 2017.
- Data cleansing and feature engineering starts below with the train dataset
'''

''' LOAD DATA
Load the train dataset (Aug '15 to Aug '16) and supplemental datasets'''

# Note: will need to update filepaths when I move data to other data folder
dftrain = pd.read_csv('/Users/mwirvine/Galvanize/dsi-immersive/dsi-module1-capstone/ecuador-grocery-data/train_aug15aug16.csv')
#dftrain = pd.read_csv('/Users/mwirvine/Galvanize/dsi-immersive/dsi-module1-capstone/ecuador-grocery-data/test_aug16aug17.csv')

dfitems = pd.read_csv('/Users/mwirvine/Galvanize/dsi-immersive/dsi-module1-capstone/ecuador-grocery-data/items.csv')
dfholidays = pd.read_csv('/Users/mwirvine/Galvanize/dsi-immersive/dsi-module1-capstone/ecuador-grocery-data/dfholidays_train_aug15aug16.csv')
dfoil = pd.read_csv('/Users/mwirvine/Galvanize/dsi-immersive/dsi-module1-capstone/ecuador-grocery-data/oil.csv')
dfstores = pd.read_csv('/Users/mwirvine/Galvanize/dsi-immersive/dsi-module1-capstone/ecuador-grocery-data/stores.csv')
dftxns_train = pd.read_csv('/Users/mwirvine/Galvanize/dsi-immersive/dsi-module1-capstone/ecuador-grocery-data/txns_train_aug15aug16.csv')

'''ENGINEER FEATURES IN SUPPLEMENTAL DATASETS
Engineer features in the transaction table prior to merging'''

# Transaction table - create new dataframe with avg transactions / year by store_nbr
daily_txn_mean = dftxns_train.groupby('store_nbr').agg({'transactions': 'mean'})
daily_txn_mean['store_nbr'] = daily_txn_mean.index



''' MERGE SUPPLEMENTAL DATASETS W/ MASTER DATAFRAME & CLEAN DATA AS NEEDED
Clean / reformat / engineer features for the items, holidays, oil, store and
transaction supplemental datasets. Also, reduce dataset to just focus on
the MEATS category'''

'''Items enrichment'''
# Merge dfitems w/ dftrain_enriched
dftrain_enriched = dftrain.merge(dfitems, how='left', on='item_nbr')
# Rename item columns
dftrain_enriched.rename(columns={'family':'item_family'}, inplace=True)
dftrain_enriched.rename(columns={'class':'item_class'}, inplace=True)
dftrain_enriched.rename(columns={'perishable':'item_perishable'}, inplace=True)

'''Reduce dataset to analyze MEATS category only'''
dftrain_enriched = dftrain_enriched.query("item_family == 'MEATS'")

'''Store enrichment'''
# Merge dfstores w/ dftrain_enriched
dftrain_enriched = dftrain_enriched.merge(dfstores, how='left', on='store_nbr')
# rename store columns
dftrain_enriched.rename(columns={'type':'store_type'}, inplace=True)
dftrain_enriched.rename(columns={'cluster':'store_cluster'}, inplace=True)

'''Transaction enrichment (store daily txn mean dataframe & daily txn dataframe)'''
# Merge daily_txn_mean w/ dftrain_enriched
dftrain_enriched = dftrain_enriched.merge(daily_txn_mean, how='left', on='store_nbr')
dftrain_enriched.rename(columns={'transactions':'store_avg_daily_txns'}, inplace=True)

# Merge dftxns_train w/ dftrain_enriched
dftxns_train.drop('Unnamed: 0', axis=1, inplace=True)
dftrain_enriched = dftrain_enriched.merge(dftxns_train, how='left', on=['date', 'store_nbr'])
dftrain_enriched.rename(columns={'transactions':'store_txn_for_day'}, inplace=True)
# Replace store_txn_for_day NaN values w/ zeros (assumed store is closed)
dftrain_enriched['store_txn_for_day'].fillna(0, inplace=True)

'''Oil enrichment'''
# Merge dfoil w/ dftrain_enriched
dfoil.rename(columns={'dcoilwtico':'oil_daily_price'}, inplace=True)
# Fill NaN daily oil prices with the next valid observation's price
dfoil['oil_daily_price'].fillna(method='bfill', inplace=True)
dftrain_enriched = dftrain_enriched.merge(dfoil, how='left', on='date')
dftrain_enriched['oil_daily_price'].fillna(method='bfill', inplace=True)

'''Holiday Enrichment'''
# remove 'transferred' = True holidays as these are like regular days per instructions
# the actual day the holiday is celebrated is when type == Transfer
dfholidays = dfholidays.query('transferred == False')
# drop holiday duplicates
dfholidays.drop_duplicates('date', inplace=True)

### National Holidays
# create national holiday DF and clean
dfholidays_train_natl = dfholidays.query("locale == 'National'")
dfholidays_train_natl.drop(['locale_name', 'description', 'locale', 'transferred', 'Unnamed: 0'], axis=1, inplace=True)
dfholidays_train_natl.rename(columns={'type':'holiday_national'}, inplace=True)
dfholidays_train_natl['holiday_national'] = 'Holiday'

# create national holiday eves dataframe (new feature to identify the day before a holiday)
dfholidays_train_natl_eves = dfholidays_train_natl.copy()
dfholidays_train_natl_eves['date'] = pd.to_datetime(dfholidays_train_natl_eves['date'])
dfholidays_train_natl_eves['holiday_eve_natl_date1'] = dfholidays_train_natl_eves['date'] - timedelta(1)
dfholidays_train_natl_eves['holiday_eve_natl'] = 1
dfholidays_train_natl_eves['holiday_eve_natl_date1'] = dfholidays_train_natl_eves['holiday_eve_natl_date1'].apply(lambda x: x.strftime('%Y-%m-%d'))
dfholidays_train_natl_eves.drop(['holiday_national', 'date'], axis=1, inplace=True)

### Local Holidays
# create local holiday DF and clean
dfholidays_train_loc = dfholidays.query("locale == 'Local'")
dfholidays_train_loc.drop(['description', 'transferred', 'locale', 'Unnamed: 0'], axis=1, inplace=True)
dfholidays_train_loc.rename(columns={'type':'holiday_local'}, inplace=True)
dfholidays_train_loc['holiday_local'] = 'Holiday'

# create local holiday eves dataframe
dfholidays_train_loc_eves = dfholidays_train_loc.copy()
dfholidays_train_loc_eves['date'] = pd.to_datetime(dfholidays_train_loc_eves['date'])
dfholidays_train_loc_eves['holiday_eve_loc_date1'] = dfholidays_train_loc_eves['date'] - timedelta(1)
dfholidays_train_loc_eves['holiday_eve_loc'] = 1
dfholidays_train_loc_eves['holiday_eve_loc_date1'] = dfholidays_train_loc_eves['holiday_eve_loc_date1'].apply(lambda x: x.strftime('%Y-%m-%d'))
dfholidays_train_loc_eves.drop(['holiday_local', 'date'], axis=1, inplace=True)

### Regional Holidays
# create regional holiday DF and clean
dfholidays_train_reg = dfholidays.query("locale == 'Regional'")
dfholidays_train_reg.drop(['description', 'transferred', 'locale', 'Unnamed: 0'], axis=1, inplace=True)
dfholidays_train_reg.rename(columns={'type':'holiday_regional'}, inplace=True)
dfholidays_train_reg['holiday_regional'] = 'Holiday'

# create regional holiday eves dataframe
dfholidays_train_reg_eves = dfholidays_train_reg.copy()
dfholidays_train_reg_eves['date'] = pd.to_datetime(dfholidays_train_reg_eves['date'])
dfholidays_train_reg_eves['holiday_eve_reg_date1'] = dfholidays_train_reg_eves['date'] - timedelta(1)
dfholidays_train_reg_eves['holiday_eve_reg'] = 1
dfholidays_train_reg_eves['holiday_eve_reg_date1'] = dfholidays_train_reg_eves['holiday_eve_reg_date1'].apply(lambda x: x.strftime('%Y-%m-%d'))
dfholidays_train_reg_eves.drop(['holiday_regional', 'date'], axis=1, inplace=True)

# Merge dfholidays dataframes w/ dftrain_enriched
dftrain_enriched = dftrain_enriched.merge(dfholidays_train_natl, how='left', on='date')
dftrain_enriched = dftrain_enriched.merge(dfholidays_train_loc, how='left', left_on=['date', 'city'], right_on=['date', 'locale_name'])
dftrain_enriched = dftrain_enriched.merge(dfholidays_train_reg, how='left', left_on=['date', 'state'], right_on=['date', 'locale_name'])
dftrain_enriched = dftrain_enriched.merge(dfholidays_train_natl_eves, how='left', left_on='date', right_on='holiday_eve_natl_date1')
dftrain_enriched = dftrain_enriched.merge(dfholidays_train_loc_eves, how='left', left_on=['date', 'city'], right_on=['holiday_eve_loc_date1', 'locale_name'])
dftrain_enriched = dftrain_enriched.merge(dfholidays_train_reg_eves, how='left', left_on=['date', 'state'], right_on=['holiday_eve_reg_date1', 'locale_name'])



'''CLEAN MASTER DATAFRAME
Final pass to clean up and engineer date features in master dataframe'''

# drop city and state as its represented by store number, and other unneeded columns
dftrain_enriched.drop(['id', 'Unnamed: 0', 'locale_name_x', 'locale_name_y', 'city', 'state'], axis=1, inplace=True)

# engineer features for date (weekday, week, month) - then drop date column
# change 'date' column to datetime64 format
dftrain_enriched['date'] = pd.to_datetime(dftrain_enriched['date'])
dftrain_enriched['day_of_week'] = dftrain_enriched['date'].dt.weekday
dftrain_enriched['month_of_year'] = dftrain_enriched['date'].dt.month
dftrain_enriched['week_of_year'] = dftrain_enriched['date'].dt.week
dftrain_enriched.drop(['date'], axis=1, inplace=True)

# fill NaNs in the holiday eve features
dftrain_enriched.drop(['holiday_eve_natl_date1', 'holiday_eve_loc_date1','holiday_eve_reg_date1'], axis=1, inplace=True)
dftrain_enriched['holiday_eve_natl'].fillna(0, inplace=True)
dftrain_enriched['holiday_eve_loc'].fillna(0, inplace=True)
dftrain_enriched['holiday_eve_reg'].fillna(0, inplace=True)



''' REMOVE OUTLIERS & NEGATIVES
There are 50+ unit_sales observations that are >1000, which are several std deviations
away from the mean. This section removes those outliers (any data point +/- 2 std devs
from the mean, and also removes negatives which represents returns.'''

def remove_outliers(df, target_col):
    mean = np.mean(df[target_col])
    std = np.std(df[target_col])
    df_no_outliers = df[(df[target_col] > mean-2*std) & (df[target_col] < mean+2*std)]
    return df_no_outliers

# remove large outliers
dftrain_enriched = remove_outliers(dftrain_enriched, 'unit_sales')
# remove negatives
dftrain_enriched = dftrain_enriched.query('unit_sales > 0')



''' CREATE DUMMY VARIABLES
Several features are categorical, so this section turns those features into dummy
variables '''

def create_dummies(dummy_col, df, col_prefix):
    dum_col = df[dummy_col]
    dummies = pd.get_dummies(dum_col, prefix=col_prefix)
    df = df.drop([dummy_col], axis=1)
    df_w_dummies = df.merge(dummies, left_index=True, right_index=True)
    return df_w_dummies

dftrain_enriched_dummies = pd.get_dummies(dftrain_enriched)
dftrain_enriched_dummies = create_dummies('day_of_week', dftrain_enriched_dummies, 'day_of_week')
dftrain_enriched_dummies = create_dummies('week_of_year', dftrain_enriched_dummies, 'week_of_year')
dftrain_enriched_dummies = create_dummies('month_of_year', dftrain_enriched_dummies, 'month_of_year')
dftrain_enriched_dummies = create_dummies('store_nbr', dftrain_enriched_dummies, 'store_nbr')
dftrain_enriched_dummies = create_dummies('store_cluster', dftrain_enriched_dummies, 'store_cluster')
dftrain_enriched_dummies = create_dummies('item_nbr', dftrain_enriched_dummies, 'item_nbr')
dftrain_enriched_dummies = create_dummies('item_class', dftrain_enriched_dummies, 'item_class')



''' MODELING FUNCTIONS
This section defines the functions to execute cross validation, linear, lasso,
and ridge regression'''

def run_model(X, y, base_estimator, n_folds, random_seed=154):
    """Estimate the in and out-of-sample error of a model using cross validation.

    Parameters
    ----------

    X: np.array
      Matrix of predictors.

    y: np.array
      Target array.

    base_estimator: sklearn model object.
      The estimator to fit.  Must have fit and predict methods.

    n_folds: int
      The number of folds in the cross validation.

    random_seed: int
      A seed for the random number generator, for repeatability.

    Returns
    -------

    train_cv_errors, test_cv_errors: tuple of arrays
      The training and testing errors for each fold of cross validation.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X)):

        # Split into train and test
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        # Standardize data
        standardizer = StandardScaler()
        standardizer.fit(X_train, y_train)
        X_train_std = standardizer.transform(X_train)
        X_test_std = standardizer.transform(X_test)

        # Fit estimator
        estimator = clone(base_estimator)
        estimator.fit(X_train_std, y_train)

        # Measure performance
        y_pred_train = estimator.predict(X_train_std)
        y_pred_test = estimator.predict(X_test_std)

        # Calculate the error metrics
        train_cv_errors[idx] = calc_rmse(y_train, y_pred_train)
        test_cv_errors[idx] = calc_rmse(y_test, y_pred_test)
    return test_cv_errors

def calc_rmse(true, predicted):
    residuals_squared = (predicted - true)**2
    variance = sum(residuals_squared) / len(true)
    rmse = np.sqrt(variance)
    return rmse



'''MODEL EXECUTION
This section executes and prints the results for each model'''

# take a sample of the 500k+ rows for the Meat category
df_test = dftrain_enriched_dummies.sample(100000)

# create X feature matrix and y target array
X = df_test.drop('unit_sales', axis=1).values.astype(np.float64)
y = df_test['unit_sales'].values.astype(np.float64)
#y = np.log(y)

# run simple train_test_split to be used for each model for plotting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# standardize data
standardizer = StandardScaler()
standardizer.fit(X_train, y_train)
X_train_std = standardizer.transform(X_train)
X_test_std = standardizer.transform(X_test)

### LINEAR REGRESSION
# run model with cross validation to get mean rmse test error
test_rmse_linear = run_model(X, y, LinearRegression(), 5)
# run model once to capture residuals and predicted values for plotting
linear = LinearRegression()
linear.fit(X_train_std, y_train)
y_pred_train_linear = linear.predict(X_train_std)
y_pred_test_linear = linear.predict(X_test_std)
r_squared = linear.score(X_train_std, y_train)
# calculate residuals
resid_train_linear = y_train - y_pred_train_linear
resid_test_linear = y_test - y_pred_test_linear

### LASSO REGRESSION
# run model with cross validation to get mean rmse test error
alpha_lasso = 0.05 # results get better as alpha approaches zero
test_rmse_lasso = run_model(X, y, Lasso(alpha=alpha_lasso), 5)
#print("Lasso RMSE test results: {} with alpha of {}".format(test_rmse_lasso.mean(), alpha_lasso))
#print("Linear RMSE test results: {}".format(test_rmse_linear.mean()))
# run model once to capture residuals and predicted values for plotting
lasso = Lasso(alpha=alpha_lasso)
lasso.fit(X_train_std, y_train)
y_pred_train_lasso = lasso.predict(X_train_std)
y_pred_test_lasso = lasso.predict(X_test_std)
# calculate residuals
resid_train_lasso = y_train - y_pred_train_lasso
resid_test_lasso = y_test - y_pred_test_lasso

### RIDGE REGRESSION
# run model with cross validation to get mean rmse test error
alpha_ridge = 0.1 # results get better as alpha approaches zero
test_rmse_ridge = run_model(X, y, Ridge(alpha=alpha_ridge), 5)
#print("Ridge RMSE test results: {} with alpha of {}".format(test_rmse_ridge.mean(), alpha_ridge))
#print("Linear RMSE test results: {}".format(test_rmse_linear.mean()))
# run model once to capture residuals and predicted values for plotting
ridge = Ridge(alpha=alpha_ridge)
ridge.fit(X_train_std, y_train)
y_pred_train_ridge = ridge.predict(X_train_std)
y_pred_test_ridge = ridge.predict(X_test_std)
# calculate residuals
resid_train_ridge = y_train - y_pred_train_ridge
resid_test_ridge = y_test - y_pred_test_ridge

### PRINT RESULTS
print("Linear RMSE test results: {}".format(test_rmse_linear.mean()))
print("Linear R squared results: {}".format(r_squared))
print("Lasso RMSE test results: {} with alpha of {}".format(test_rmse_lasso.mean(), alpha_lasso))
print("Ridge RMSE test results: {} with alpha of {}".format(test_rmse_ridge.mean(), alpha_ridge))

### RECURSIVE FEATURE SELECTION
selector = RFE(linear)
selector.fit(X_train_std, y_train)
# capture feature ranking
column_ranking = []
cols_X = df_test.drop('unit_sales', axis=1).columns.tolist()
ranking = selector.ranking_.tolist()
for rank, col in zip(ranking, cols_X):
    column_ranking.append((rank, col))
column_ranking.sort()
selector.n_features_
# Text
y_pred_train_rfe = selector.predict(X_train_std)
y_pred_test_rfe = selector.predict(X_test_std)
test_rmse_rfe = calc_rmse(y_test, y_pred_test_rfe)

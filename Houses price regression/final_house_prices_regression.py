import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import lightgbm as lgb 
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

test_path = '/content/drive/MyDrive/Colab Datasets/house-prices-advanced-regression-techniques/test.csv'
train_path = '/content/drive/MyDrive/Colab Datasets/house-prices-advanced-regression-techniques/train.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

train = df_train.copy()

cols = train.columns

train.drop('Id', axis = 1, inplace = True)
cols = train.columns

train.info()

train.isnull().sum().sort_values(ascending = False)

higher_na_cols = []
for col in cols:
  if train[col].isnull().sum() > 1300:
    higher_na_cols.append(col)

train.drop(higher_na_cols, axis = 1, inplace = True)

cols = train.columns

cols_with_na = []
for col in cols:
  if train[col].isnull().sum() > 0:
    cols_with_na.append(col)

cols_with_na

cat_cols_na = []
num_cols_na = []

cat_cols = train.select_dtypes(exclude = np.number)

num_cols = train.select_dtypes(include = np.number)

for col in cat_cols:
  imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
  train[col] = imputer.fit_transform(np.array(train[col]).reshape(-1, 1))

for col in num_cols:
  if train[col].skew() > 1.5:
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
    train[col] = imputer.fit_transform(np.array(train[col]).reshape(-1, 1))
  else:
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    train[col] = imputer.fit_transform(np.array(train[col]).reshape(-1, 1))

cols_binary_encoder = []
cols_label_encoder = ['Street', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure'
                      , 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for col in cat_cols:
  if col not in cols_label_encoder:
    cols_binary_encoder.append(col)

len(cols_binary_encoder), len(cols_label_encoder)

encoder = ce.BinaryEncoder(cols = cols_binary_encoder , return_df = True)
train = encoder.fit_transform(train)

encoder_ordinal = ce.OrdinalEncoder(cols = cols_label_encoder , return_df = True)
train = encoder_ordinal.fit_transform(train)

scaler = PowerTransformer(method = 'yeo-johnson')
for col in train.columns:
  if ((train[col].skew() < -1.0) | (train[col].skew() > 1.0)) & (col != 'SalePrice'):
    train[col] = scaler.fit_transform(np.array(train[col]).reshape(-1, 1))

X = train.iloc[ :, : -1]
tot_features = X.columns

Y = pd.DataFrame(train.iloc[: , 155])

X_scaled = RobustScaler().fit_transform(X)

"""## Feature Selection"""

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

feature_selection_model_params = {
    'XGBRegression' : {
        'model' : xgb.XGBRegressor(n_jobs = -1),
        'params' : {
            "learning_rate"    : 0.05 ,
            "max_depth"        : 8,
            "min_child_weight" : 3,
            "gamma"            : 0.1,
            "colsample_bytree" : 0.3 
         },
    },
}

feature_selection_results = list()

for model_name, mp in feature_selection_model_params.items():
  sel = SelectFromModel(mp['model'], max_features = 15)
  sel.fit(X_scaled, Y)

sel

# Print the names of the most important features
imp_features = []
for feature_list_index in sel.get_support(indices=True):
  imp_features.append(tot_features[feature_list_index])

imp_features

X_imp = X[imp_features]

X_imp_scaled = RobustScaler().fit_transform(X_imp)

LinearRegression().get_params()

model_params = {
    'linear_regression' : {
        'model' : LinearRegression(n_jobs = -1),
        'params':{
        },
    },
    'XGBRegression' : {
        'model' : xgb.XGBRegressor(n_jobs = -1),
        'params' : {
            "learning_rate"    : [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
            "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight" : [ 1, 3, 5, 7 ],
            "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
         },
    },
    'LGBMRegression' : {
        'model' : lgb.LGBMRegressor(n_jobs = -1),
        'params' : {
            'boosting_type': ['gbdt', 'goss', 'dart'],
            'num_leaves': list(range(20, 150)),
            'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
            'subsample_for_bin': list(range(20000, 300000, 20000)),
            'min_child_samples': list(range(20, 500, 5)),
            'reg_alpha': list(np.linspace(0, 1)),
            'reg_lambda': list(np.linspace(0, 1)),
            'colsample_bytree': list(np.linspace(0.6, 1, 10)),
            'subsample': list(np.linspace(0.5, 1, 100)),
            'is_unbalance': [True, False]
      },
   }
}

"""## RandomSearchCV"""

scores_rscv = []

for model_name, mp in model_params.items():
  reg = RandomizedSearchCV(mp['model'], mp['params'], cv = 5, verbose = 0, return_train_score = False, scoring = 'r2')
  reg.fit(X_imp_scaled, Y)
  scores_rscv.append({
      'model' : model_name,
      'best score' : reg.best_score_,
      'CV Results' : reg.cv_results_,
      'Best Estimator' : reg.estimator,
      'Best Params' : reg.best_params_
  })

rscv_res = pd.DataFrame(scores_rscv)
rscv_res

"""## Final Model"""

final_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
             n_jobs=-1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
# final_model = xgb.XGBRegressor(final_model_params, n_jobs = -1)
final_model.fit(X_imp_scaled, Y)

model_scores = cross_val_score(final_model, X_imp_scaled, Y, cv = 10, scoring = 'r2', n_jobs = -1)

model_scores

"""## Save the model"""

filename = 'house_prices_model.pkl'
pickle.dump(final_model, open(filename, 'wb'))
import numpy as np
import pandas as pd
import lightgbm as lgb
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

path = 'train_data_featured.csv'
train_data = pd.read_csv(path, index_col=0)

features = ['Cell X', 'Cell Y', 'Height', 'Azimuth',
       'Electrical Downtilt', 'Mechanical Downtilt', 'Frequency Band',
       'RS Power', 'Cell Altitude', 'Cell Building Height',
       'Cell Clutter Index', 'X', 'Y', 'Altitude', 'Building Height',
       'Clutter Index', 'distance', 'theta', 'tan_theta', 'delta_hv',
       'log_freq_band', 'hb', 'log_hb', 'hr', 'log_hr', 'dist_km',
       'log_dist_km', 'hr_d_cross', 'Cm', 'suburban_alpha', 'log_w_hr',
       'urban_alpha', 'cal_dl']
target = ['RSRP']
x, y = train_data[features], train_data[target]

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=666)
kf = kfold.split(x, y)
cv_pred = np.zeros(test_data.shape[0])
valid_best_l2_all = 0
feature_importance_df = pd.DataFrame()

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'seed': 42,
    'metric': 'rmse',

    'learning_rate': 0.3,
    'num_leaves': 64,
    'max_depth': -1,
    'verbose': 0,

    'feature_fraction': 0.8,
    'bagging_fraction': 0.8
}
for i, (train_fold, validate) in enumerate(kf):
    X_train, X_validate, y_train, y_validate = \
    x.iloc[train_fold, :], x.iloc[validate, :], \
    y[train_fold], y[validate]
    
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_validate, y_validate, reference=dtrain)
    
    bst = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dtrain,dvalid],early_stopping_rounds=100)
    
    cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)
    valid_best_l2_all += bst.best_score['valid_0']['l1']
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = list(X_train.columns)
    fold_importance_df["importance"] = bst.feature_importance(importance_type='gain', iteration=bst.best_iteration)
    fold_importance_df["fold"] = count + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    count += 1

cv_pred /= NFOLDS
valid_best_l2_all /= NFOLDS
print('cv score for valid is: ', 1/(1+valid_best_l2_all))

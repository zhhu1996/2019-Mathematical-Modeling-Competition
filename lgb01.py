import numpy as np
import pandas as pd
import lightgbm as lgb
import math
from sklearn.model_selection import train_test_split

path = 'train_data_featured.csv'
train_data = pd.read_csv(path, index_col=0)
train_data.head()

#features = ['Cell X', 'Cell Y', 'Height', 'Azimuth',
#       'Electrical Downtilt', 'Mechanical Downtilt', 'Frequency Band',
#       'RS Power', 'Cell Altitude', 'Cell Building Height',
#       'Cell Clutter Index', 'X', 'Y', 'Altitude', 'Building Height',
#       'Clutter Index', 'distance', 'theta', 'tan_theta', 'delta_hv',
#       'log_freq_band', 'hb', 'log_hb', 'hr', 'log_hr', 'dist_km',
#       'log_dist_km', 'hr_d_cross', 'Cm', 'suburban_alpha', 'log_w_hr',
#       'urban_alpha', 'cal_dl']
features = ['Cell X', 'Cell Y', 'Height', 'Azimuth',
       'Electrical Downtilt', 'Mechanical Downtilt',
       'RS Power', 'Cell Altitude', 'Cell Building Height',
       'Cell Clutter Index', 'X', 'Y', 'Altitude', 'Building Height',
       'Clutter Index', 'distance', 'theta', 'delta_hv',
       'hb', 'log_hb', 'hr', 'log_hr', 'dist_km',
       'log_dist_km', 'hr_d_cross', 'suburban_alpha','cal_dl']
target = ['RSRP']
x, y = train_data[features], train_data[target]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
print('dataset constructed...')
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
print('train begin...')
model = lgb.train(params,                     
                  lgb_train,                  
                  num_boost_round=2000,       
                  valid_sets=[lgb_train, lgb_eval],        
                  early_stopping_rounds=100)   
print(pd.DataFrame({
        'column': features,
        'importance': model.feature_importance(),
    }).sort_values(by='importance', ascending=False))
model.save_model('./model/lgb01.txt')

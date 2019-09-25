import numpy as np
import pandas as pd
import lightgbm as lgb
import math
from sklearn.model_selection import train_test_split

path = 'train_data_featured.csv'
train_data = pd.read_csv(path, index_col=0)
train_data.head()

model = lgb.Booster(model_file='./model/lgb02.txt')
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_pred = model.predict(x_test, num_iteration=model.best_iteration)

from sklearn.metrics import f1_score, mean_squared_error

#y_pred = y_test['RS Power'] - y_pred
y_target = y_test['RSRP']
print(mean_squared_error(y_target, y_pred))

def CaculatePcrr(y_true,y_pred):
    t = -103
    tp = len(y_true[(y_true < t)&(y_pred < t)]) 
    fp = len(y_true[(y_true >= t)&(y_pred < t)]) 
    fn = len(y_true[(y_true < t) & (y_pred >= t)])
    precision =tp/(tp+fp)
    recall = tp/(tp+fn)
    pcrr = 2 * (precision * recall)/(precision + recall) 
    return pcrr

print(CaculatePcrr(y_target,y_pred))

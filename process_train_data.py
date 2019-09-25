import numpy as np
import pandas as pd
import lightgbm as lgb
import math
from sklearn.model_selection import train_test_split

path = 'train_total.csv'
train_data = pd.read_csv(path, index_col=0)
train_data = train_data.reset_index(drop=True)
assert len(set(train_data.index)) == len(train_data.index)

#distance
train_data['distance'] = (train_data['Cell X'] - train_data['X'])**2 + (train_data['Cell Y'] - train_data['Y'])**2
train_data['distance'] = train_data['distance'].apply(lambda x: 5*math.sqrt(x))
print('distance finished.')

#delta_hv
train_data['theta'] = (train_data['Electrical Downtilt'] + train_data['Mechanical Downtilt']) / 180
train_data['tan_theta'] = train_data['theta'].apply(lambda x: math.tan(x))
train_data['delta_hv'] = train_data['Height']  - train_data['tan_theta'] * train_data['distance']
print('delta_hv finished.')

#log_freq
train_data['log_freq_band'] = train_data['Frequency Band'].apply(lambda x: math.log(x))
print('log_freq_band finished.')

# hb: 基站天线有效高度
train_data['hb'] = train_data['Height'] + train_data['Cell Altitude'] - train_data['Altitude']
train_data['log_hb'] = train_data['hb'].apply(lambda x: math.log(abs(x)+0.083))
print('log_hb finished.')

# hr: 用户天线有效高度
train_data['hr'] = train_data['Building Height'] + train_data['Altitude']
train_data['log_hr'] = train_data['hr'].apply(lambda x: math.log(abs(x)+0.083))
print('log_hr finished.')

# dist: km
train_data['dist_km'] = train_data['distance'] / 1000
train_data['log_dist_km'] = train_data['dist_km'].apply(lambda x: math.log(abs(x)+0.083))
print('log_dist finished.')

# dl and hr_d_cross
train_data['dl'] = train_data['RS Power'] - train_data['RSRP']
train_data['hr_d_cross'] = train_data['log_hr'] * train_data['log_dist_km']
print('dl, hr_d_cross finished.')

print(train_data.columns)
def get_cm(x):
    target = {10: 1, 11: 1, 12: 1, 13: 1, 16: 1, 20: 1}
    if x['Cell Clutter Index'] in target or x['Clutter Index'] in target:
        return 3
    else:
        return 0
    
train_data['Cm'] = train_data.apply(get_cm, axis=1)
print('Cm finished.')
#train_data = train_data.rename(columns={'hue': 'hr', 'log_hue': 'log_hr'})
train_data['suburban_alpha'] = (1.1*train_data['log_freq_band'] - 0.7)*train_data['hr'] - (1.56*train_data['log_freq_band']-0.8)
train_data['log_w_hr'] = train_data['hr'].apply(lambda x: math.log(11.75*x+0.083))
train_data['urban_alpha'] = 3.2*train_data['log_w_hr']*train_data['log_w_hr'] - 4.97
def get_dl(x):
    res = 0
    if x['Cm'] == 0: # 偏远地区
        res = 46.3 + 33.9*x['log_freq_band'] - 13.82*x['log_hb'] + x['log_dist_km']*(44.9-6.55*x['log_freq_band']) + x['Cm'] - x['suburban_alpha']
    else: # 大城市区
        res = 46.3 + 33.9*x['log_freq_band'] - 13.82*x['log_hb'] + x['log_dist_km']*(44.9-6.55*x['log_freq_band']) + x['Cm'] - x['urban_alpha']
    return res

train_data['cal_dl'] = train_data.apply(get_dl, axis=1)
print('cal_dl finished.')

# h_theta
def get_h_theta(x):
    res = 0
    if x['X'] - x['Cell X'] > 0 and x['Y'] - x['Cell Y'] > 0:
        res = math.atan(abs(x['X'] - x['Cell X']) / abs(x['Y'] - x['Cell Y']))
    elif x['X'] - x['Cell X'] < 0 and x['Y'] - x['Cell Y'] > 0:
        res = -math.atan(abs(x['X'] - x['Cell X']) / abs(x['Y'] - x['Cell Y']))
    elif x['X'] - x['Cell X'] < 0 and x['Y'] - x['Cell Y'] < 0:
        res = - math.atan(abs(x['Y'] - x['Cell Y']) / abs(x['X'] - x['Cell X'])) - 90
    elif x['X'] - x['Cell X'] > 0 and x['Y'] - x['Cell Y'] < 0:
        res = math.atan(abs(x['Y'] - x['Cell Y']) / abs(x['X'] - x['Cell X'])) + 90
    elif x['X'] == x['Cell X'] and x['Y'] > x['Cell Y']:
        res =  0
    elif x['X'] == x['Cell X'] and x['Y'] < x['Cell Y']:
        res = 180
    elif x['Y'] == x['Cell Y'] and x['X'] > x['Cell X']:
        res = 90
    elif x['Y'] == x['Cell Y'] and x['X'] < x['Cell X']:
        res = -90
    else:
        res = x['Azimuth']
    return res

train_data['h_theta'] = train_data.apply(get_h_theta, axis=1)
train_data['cos_h_theta'] = ((train_data['h_theta'] - train_data['Azimuth'])*math.pi/180).apply(math.cos)
print('h_theta and cos finished.')

# del wrong rows
del_1 = train_data[(train_data['Clutter Index'] == 10) & (train_data['Building Height'] <= 60)].index
del_2 = train_data[(train_data['Clutter Index'] == 14) & (train_data['Building Height'] >=20)].index
train_data.drop(del_1, inplace=True)
train_data.drop(del_2, inplace=True)
def get_flag(x):
    cluster_maxheight_map = {
    1:0,
    2:0,
    3:0,
    4:10,
    5:10,
    6: 15,
    7: 15,
    8:15,
    9:15,
    10:200,
    11:60,
    12: 40,
    13:20,
    14:20,
    15: 10,
    16:10,
    17:20,
    18:20,
    19:20,
    20:20
}
    if x['Building Height'] <= cluster_maxheight_map[x['Clutter Index']]:
        return True
    else:
        return False
train_data['flag'] = train_data.apply(get_flag, axis=1)
train_data = train_data[train_data['flag']==True]
print('del wrong rows.')
print(train_data.head(10))
train_data.to_csv('train_data_featured.csv')

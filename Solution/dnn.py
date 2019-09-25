import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import lightgbm as lgb
import json
import math

# read csv
path = 'train_data_featured.csv'
train_data = pd.read_csv(path, index_col=0)
train_data = train_data.reset_index(drop=True)
assert len(set(train_data.index)) == len(train_data.index)

# # h_theta
# def get_h_theta(x):
#     res = 0
#     if x['X'] - x['Cell X'] > 0 and x['Y'] - x['Cell Y'] > 0:
#         res = math.atan(abs(x['X'] - x['Cell X']) / abs(x['Y'] - x['Cell Y']))
#     elif x['X'] - x['Cell X'] < 0 and x['Y'] - x['Cell Y'] > 0:
#         res = -math.atan(abs(x['X'] - x['Cell X']) / abs(x['Y'] - x['Cell Y']))
#     elif x['X'] - x['Cell X'] < 0 and x['Y'] - x['Cell Y'] < 0:
#         res = - math.atan(abs(x['Y'] - x['Cell Y']) / abs(x['X'] - x['Cell X'])) - 90
#     elif x['X'] - x['Cell X'] > 0 and x['Y'] - x['Cell Y'] < 0:
#         res = math.atan(abs(x['Y'] - x['Cell Y']) / abs(x['X'] - x['Cell X'])) + 90
#     elif x['X'] == x['Cell X'] and x['Y'] > x['Cell Y']:
#         res =  0
#     elif x['X'] == x['Cell X'] and x['Y'] < x['Cell Y']:
#         res = 180
#     elif x['Y'] == x['Cell Y'] and x['X'] > x['Cell X']:
#         res = 90
#     elif x['Y'] == x['Cell Y'] and x['X'] < x['Cell X']:
#         res = -90
#     else:
#         res = x['Azimuth']
#     return res

# train_data['h_theta'] = train_data.apply(get_h_theta, axis=1)
# train_data['cos_h_theta'] = ((train_data['h_theta'] - train_data['Azimuth'])*math.pi/180).apply(math.cos)
# print('h_theta and cos finished.')

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
print(train_data.shape)

# dnn
data = train_data
print(data.shape)

config = { "model_type": "TensorFlow",
 "model_algorithm": "dnn",
 "runtime": "python3.6"
}
with open('./config.json', 'w') as f:
    json.dump(config, f)

print(tf.__version__)
#assert tf.__version__ == '1.8.0'
# data = pd.read_csv("train_data_featured.csv", header=0, index_col=0)
# data = data.rename(columns={'hue_d_cross': 'hr_d_cross'})
# print(data)
model = lgb.Booster(model_file='./model/lgb05.txt')
# 输入lgb的特征
_features = ['X', 'Y', 'Altitude', 'Building Height',
       'Clutter Index', 'distance', 'theta', 'delta_hv',
       'hb', 'log_hb', 'hr', 'log_hr', 'dist_km',
       'log_dist_km', 'hr_d_cross', 'suburban_alpha','cal_dl']
# _features = ['Cell X', 'Cell Y', 'Height', 'Azimuth',
#        'Electrical Downtilt', 'Mechanical Downtilt', 'Frequency Band',
#        'RS Power', 'Cell Altitude', 'Cell Building Height',
#        'Cell Clutter Index', 'X', 'Y', 'Altitude', 'Building Height',
#        'Clutter Index', 'distance', 'theta', 'tan_theta', 'delta_hv',
#        'log_freq_band', 'hb', 'log_hb', 'hr', 'log_hr', 'dist_km',
#        'log_dist_km', 'hr_d_cross', 'Cm', 'suburban_alpha', 'log_w_hr',
#        'urban_alpha', 'cal_dl']
# _features = ['Cell X', 'Cell Y', 'Height', 'Azimuth',
#                              'Electrical Downtilt', 'Mechanical Downtilt', 'Frequency Band',
#                              'RS Power', 'Cell Altitude', 'Cell Building Height',
#                              'Cell Clutter Index', 'X', 'Y', 'Altitude', 'Building Height',
#                              'Clutter Index', 'distance', 'theta', 'tan_theta', 'delta_hv',
#                              'log_freq_band', 'hb', 'log_hb', 'hr', 'log_hr', 'dist_km',
#                              'log_dist_km', 'hr_d_cross', 'Cm', 'suburban_alpha', 'log_w_hr',
#                              'urban_alpha', 'cal_dl']
target = ['RSRP']
x, y = data[_features], data[target]
y_pred = model.predict(x, num_iteration=model.best_iteration)
data['lgb_pred'] = y_pred
print('lgb pred calculated.')

# fix theta
#delta_hv, fix bug
train_data['theta'] = (train_data['Electrical Downtilt'] + train_data['Mechanical Downtilt']) * math.pi / 180
train_data['tan_theta'] = train_data['theta'].apply(lambda x: math.tan(x))
train_data['delta_hv'] = train_data['Height']  - train_data['tan_theta'] * train_data['distance']
print('delta_hv finished.')

features_all = ['Clutter Index', 'theta', 'tan_theta', 'delta_hv',
       'log_freq_band', 'hb', 'log_hb', 'hr', 'log_hr', 'dist_km',
       'log_dist_km', 'hr_d_cross', 'Cm', 'suburban_alpha', 'log_w_hr',
       'urban_alpha', 'cal_dl','RS Power','lgb_pred','X', 'Y', 'Altitude', 'Building Height',
       ]
features = ['X', 'Y', 'Altitude', 'Building Height', 'delta_hv', 'hb', 'log_hb', 'hr', 'log_hr', 'dist_km',
       'log_dist_km', 'hr_d_cross', 'log_w_hr']
features_dnn = ['X', 'Y', 'Altitude', 'Building Height',
       'theta', 'tan_theta', 'delta_hv',
       'log_freq_band', 'hb', 'log_hb', 'hr', 'log_hr', 'dist_km',
       'log_dist_km', 'hr_d_cross', 'Cm', 'suburban_alpha', 'log_w_hr',
       'urban_alpha', 'cal_dl','RS Power','lgb_pred']

target = ['RSRP']
cate_fea = ['Clutter Index']
input = data[features_all]
# bn
for x in features:
    if input[x].max() - input[x].min() == 0:
        div = input[x].max() + 1
    else:
        div = input[x].max() - input[x].min()
    input[x] = (input[x] - input[x].min()) / div
print('bn finish.')
label = data[target]

batch_size = 1024
x_train, x_test, y_train, y_test = train_test_split(input, label, test_size=0.2, random_state=666)

x = tf.placeholder(tf.float64, [None, 22])
cate_x = tf.placeholder(tf.float64, [None, 1])
y = tf.placeholder(tf.float64, [None, 1])


clutter = tf.cast(tf.reshape(cate_x, [-1]), tf.int64)
embed = tf.Variable(tf.random_normal(shape=[20], mean=0, stddev=1, dtype=tf.float64), name='clu_embed')
# embed = tf.Print(embed, [embed], "embed: ", summarize=100)
_clutter = tf.reshape(tf.nn.embedding_lookup(embed, clutter), [-1, 1])
new_x = tf.concat([x, _clutter], axis=1)
# layers1 = tf.layers.dense(new_x, 64, activation=tf.nn.relu)
# layers2 = tf.layers.dense(layers1,32, activation=tf.nn.relu)
# layers3 = tf.layers.dense(layers2,8, activation=tf.nn.relu)
layers1 = tf.layers.dense(new_x, 64, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.3))
layers2 = tf.layers.dense(layers1,32, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.3))
layers3 = tf.layers.dense(layers2,8, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.3))
output = tf.layers.dense(layers3, 1, activation=None)

loss = tf.reduce_mean(tf.square(output - y))
optimize = tf.train.AdamOptimizer(0.001).minimize(loss)

saver = tf.train.Saver()
# gpu_options = tf.GPUOptions(allow_growth=True)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    lowest_mse = [-1, float('inf')]
    for i in range(1000):
        for j in range(1 + len(x_train) // batch_size):
            tmpx = x_train[j*batch_size: (j+1)*batch_size]
            fea = tmpx[features_dnn]
            cat = tmpx[cate_fea]
            tmpy = y_train[j*batch_size: (j+1)*batch_size]
            # print(fea.shape)
            # print(cat)
            sess.run(optimize, feed_dict={x: fea, y: tmpy, cate_x: cat})
        x_test_fea = input[features_dnn]
        x_test_cat = input[cate_fea]
        _loss = sess.run(loss, feed_dict={x: x_test_fea, y: label, cate_x: x_test_cat})
        print('step:%d MSE:%.4f' % (i, _loss))
        tf.saved_model.simple_save(sess,"./home/admin/jupyter/math_A/dnnv1/{0}".format(i), inputs={"x": x,"cate_X": cate_x, "y": y}, 
        outputs={"output": output})
    
#     # print(sess.run(output, feed_dict={x: x_test_fea, y: y_test, cate_x: x_test_cat}))

import scipy.io as sio
import scipy.stats
import numpy as np
import sys
import pdb
from easydict import EasyDict as edict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from numpy import array, linspace
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import MeanShift, estimate_bandwidth
import math
import os
from py_baseline_QoE_models import Liu2015QoE
import itertools
from sklearn.linear_model import LinearRegression
import pickle
# def loss():
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

file = open("features_mos_all.pickle", 'rb')
data = pickle.load(file)

# BigBuckBunny, BirdOfPrey, FCB, Ski, TearsOfSteel1
session_index = [0, 57, 208, 319, 339]
session_length = [57, 60, 62, 61, 60]
train_features_reweighting = data["train_features_reweighting"]
train_features = data["train_features"]
test_features_reweighting = data["test_features_reweighting"]
test_features = data["test_features"]
train_mos = data["train_mos"]
test_mos = data["test_mos"]
# import pdb; pdb.set_trace()
# merge features
features = np.concatenate((train_features, test_features))
features_reweighting = np.concatenate((train_features_reweighting, test_features_reweighting))
mos = np.concatenate((train_mos, test_mos))
# fix last feature in random k
chosen_video = ["BigBuckBunny", "BirdOfPrey", "FCB", "Ski", "TearsOfSteel1"]
# dict
# new_feature = []
# for video in chosen_video:
#     print(f"process {video}")
#     streamInfo = sio.loadmat(f'streamInfo/{video}.mat', struct_as_record=0, squeeze_me=1);
#     streamInfo = streamInfo['streamInfo']
#     new_feature_per_stream = []
#     for jjj in range(len(streamInfo)):
#         sailency_path = f'/data2/yuanx/QoEData/sailency-models/TASED-Net/sailency_pickles/{video}{str(jjj+1).zfill(2)}/'
#         videoInfo = streamInfo[jjj, :]
#         fps = float(videoInfo[0])
#         sailency_file = os.path.join(sailency_path, 'sailency.npy')
#         sailency_map = np.load(sailency_file)
#         sailency_score = np.mean(sailency_map, axis = (1,2))
#         buffer_area = []
#         buffer_area.extend(range(videoInfo[2]))
#         for i_buffer_events, buffer_events in enumerate(np.atleast_1d(videoInfo[4])):
#             buffer_area.extend(range(np.atleast_1d(videoInfo[3])[i_buffer_events] - 1, np.atleast_1d(videoInfo[3])[i_buffer_events] - 1 + np.atleast_1d(videoInfo[4])[i_buffer_events]))
#         buffer_score = sailency_score[buffer_area]
#         new_feature_per_stream.append(buffer_score)
#     new_feature.append(new_feature_per_stream)
# np.save("new_buffer_feature", new_feature)
# new_buffer_feature = np.load("new_buffer_feature.npy", allow_pickle=True)

# sourceVideo = sio.loadmat('sourceVideo.mat', struct_as_record=0, squeeze_me=1);
# sourceVideo = sourceVideo['sourceVideo']
# sourceNames = sourceVideo.name
# # sailency_all = []
# for iii in range(len(sourceNames)):
#     print(iii)
#     streamInfo = sio.loadmat(f'streamInfo/{sourceNames[iii]}.mat', struct_as_record=0, squeeze_me=1);
#     streamInfo = streamInfo['streamInfo']
#     sailency_per_stream = []
#     for jjj in range(len(streamInfo)):
#         sailency_path = f'/data2/yuanx/QoEData/sailency-models/TASED-Net/sailency_pickles/{sourceNames[iii]}{str(jjj+1).zfill(2)}/'
#         videoInfo = streamInfo[jjj, :]
#         fps = float(videoInfo[0])
#         sailency_file = os.path.join(sailency_path, 'sailency.npy')
#         sailency_map = np.load(sailency_file)
#         # import pdb; pdb.set_trace()
#         sailency_per_stream.append(sailency_map)
#     path_sailency_per_stream = f"/data2/yuanx/QoEData/sailency-models/TASED-Net/sailency_pickles_all/{sourceNames[iii]}_sailency.pkl"
#     with open(path_sailency_per_stream, 'wb') as f:
#         pickle.dump(sailency_per_stream, f)
    # np.save(path_sailency_per_stream, sailency_per_stream)
# import pdb; pdb.set_trace()

# new_feature = []
# for video in chosen_video:
#     print(f"process {video}")
#     sailency_path = f'/data2/yuanx/QoEData/sailency-models/TASED-Net/sailency_pickles/{video}_sailency.pkl/'
#     with open(sailency_path, 'rb') as f:
#         sailency = pickle.load(f)
#     for jjj in range(len(streamInfo)):
#         sailency_path = f'/data2/yuanx/QoEData/sailency-models/TASED-Net/sailency_pickles/{video}{str(jjj+1).zfill(2)}/'
#         videoInfo = streamInfo[jjj, :]
#         fps = float(videoInfo[0])
#         sailency_file = os.path.join(sailency_path, 'sailency.npy')
#         sailency_map = np.load(sailency_file)
#         sailency_score = np.mean(sailency_map, axis = (1,2))
#         buffer_area = []
#         buffer_area.extend(range(videoInfo[2]))
#         for i_buffer_events, buffer_events in enumerate(np.atleast_1d(videoInfo[4])):
#             buffer_area.extend(range(np.atleast_1d(videoInfo[3])[i_buffer_events] - 1, np.atleast_1d(videoInfo[3])[i_buffer_events] - 1 + np.atleast_1d(videoInfo[4])[i_buffer_events]))
#         buffer_score = sailency_score[buffer_area]
#         new_feature_per_stream.append(buffer_score)
#     new_feature.append(new_feature_per_stream)

for ii_ss, i_ss in enumerate(session_index):
    # for random_state in range(5)
    kf = KFold(n_splits=5, shuffle=True, random_state = 0)
    ss_length = session_length[ii_ss]
    feat_session = features[i_ss:int(i_ss+ss_length)]
    features_reweighting_session = features_reweighting[i_ss:int(i_ss+ss_length)]
    # for i_reweighting_session, reweighting_session in enumerate(features_reweighting_session):
    #     features_reweighting_session[i_reweighting_session, 3] = 0
    #     feat_session[i_reweighting_session, 3] = 0

    # import pdb; pdb.set_trace()
    # features_reweighting_session[:,3] = new_buffer_feature[ii_ss]
    mos_session = mos[i_ss:int(i_ss+ss_length)]
    # print("Session:", ii_ss)
    count_fold_id = 0
    loss_session = 0.
    loss_reweighting_session = 0.
    for train_index, test_index in kf.split(feat_session):
        # print("TRAIN:", train_index, "TEST:", test_index)
        # import pdb; pdb.set_trace()
        X_train, X_test = feat_session[train_index], feat_session[test_index]
        X_train_reweighting, X_test_reweighting = features_reweighting_session[train_index], features_reweighting_session[test_index]
        y_train, y_test = mos_session[train_index], mos_session[test_index]

        reg = LinearRegression().fit(X_train, y_train)
        reg_reweighting = LinearRegression().fit(X_train_reweighting, y_train)
        y_true = y_test
        y_pred = reg.predict(X_test)
        y_reweighting = reg_reweighting.predict(X_test_reweighting)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        ax[0].scatter(y_pred, y_true)
        ax[0].plot([0,100], [0,100], color='blue', linewidth=3)
        ax[0].set_xlim(0, 100)
        ax[0].set_ylim(0, 100)
        ax[1].scatter(y_reweighting, y_true)
        ax[1].plot([0,100], [0,100], color='blue', linewidth=3)
        ax[1].set_xlim(0, 100)
        ax[1].set_ylim(0, 100)
        fig.savefig(f'train_{ii_ss}_{count_fold_id}.png')
        loss = mean_squared_error(y_pred, y_true)
        loss_session += loss
        loss_reweighting = mean_squared_error(y_reweighting, y_true)
        loss_reweighting_session += loss_reweighting
        # print("Session:", ii_ss, "Split_id", count_fold_id, loss, "VS", loss_reweighting)
        count_fold_id += 1
        plt.cla()
        plt.close()
    print("Session:", ii_ss, loss_session/count_fold_id, "VS", loss_reweighting_session/count_fold_id)


# y_true = test_mos
# y_pred = reg.predict(test_features)
# y_reweighting = reg_reweighting.predict(test_features_reweighting)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
# ax[0].scatter(y_pred, test_mos)
# ax[0].plot([0,100], [0,100], color='blue', linewidth=3)
# ax[0].set_xlim(0, 100)
# ax[0].set_ylim(0, 100)
# ax[1].scatter(y_reweighting, test_mos)
# ax[1].plot([0,100], [0,100], color='blue', linewidth=3)
# ax[1].set_xlim(0, 100)
# ax[1].set_ylim(0, 100)
# fig.savefig('test.png')
# print(scipy.stats.spearmanr(y_pred, y_true), "VS", scipy.stats.spearmanr(y_reweighting, y_true))
# print(mean_squared_error(y_pred, y_true), "VS", mean_squared_error(y_reweighting, y_true))

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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import preprocessing

sourceVideo = sio.loadmat('sourceVideo.mat', struct_as_record=0, squeeze_me=1);
actualBitrate = sio.loadmat('actualBitrate.mat', struct_as_record=0, squeeze_me=1);
sourceVideo = sourceVideo['sourceVideo']
actualBitrate = actualBitrate['actualBitrate']

sourceNames = sourceVideo.name
tVideo = 10
segmentDuration = 2
count = 0

Q_1 = []
Q_1_vision = []
Q_vqm_vision = []
offset = 0
features = {}
features_reweighting = {}

# chosen_video = ["BigBuckBunny"]
# chosen_video = ["BigBuckBunny", "BirdOfPrey", "FCB", "Ski", "TearsOfSteel1"]
chosen_video = ["BigBuckBunny", "BirdOfPrey"]
for iii in range(len(sourceNames)):
    features_session = []
    features_reweighting_session = []

    if sourceNames[iii] not in chosen_video:
        continue
    print(f"process {sourceNames[iii]}")

    representations = sio.loadmat(f'representations/{sourceNames[iii]}.mat', struct_as_record=0, squeeze_me=1);
    representations = representations['representation']
    streamInfo = sio.loadmat(f'streamInfo/{sourceNames[iii]}.mat', struct_as_record=0, squeeze_me=1);
    streamInfo = streamInfo['streamInfo']
    bitrateLadder = eval(f"actualBitrate.{sourceNames[iii]}")
    VQM_res = sio.loadmat(f'VQAResults/VQM/{sourceNames[iii]}.mat', struct_as_record=0, squeeze_me=1);
    VQM_res = VQM_res['Q_VQM']
    sailency_path = f'/data2/yuanx/QoEData/sailency-models/TASED-Net/sailency_pickles_all/{sourceNames[iii]}_sailency.pkl'
    with open(sailency_path, 'rb') as f:
        sailency = pickle.load(f)
    for jjj in range(len(streamInfo)):
        # sailency_path = f'/data2/yuanx/QoEData/sailency-models/TASED-Net/sailency_pickles/{sourceNames[iii]}{str(jjj+1).zfill(2)}/'
        videoInfo = streamInfo[jjj, :]
        fps = float(videoInfo[0])
        selectedRep = videoInfo[1]
        bitrates = []
        # stalling duration
        # feature list
        # VQM: frame VQM score
        # AMVM: average motion vector magnitude
        # f: frame rate
        # T: segment duration
        # ms: stalling frames
        # ls: duration of each stalling event
        seqAMVM = []
        seqVQM = []
        for kkk in range(len(selectedRep)):
            amvm = sio.loadmat(f'VQAResults/AMVM/{sourceNames[iii]}/{representations[selectedRep[kkk]+1, 4]}.mat', struct_as_record=0, squeeze_me=1)
            amvm = amvm['mmvm']
            segmentamvm = amvm[int(kkk*segmentDuration*fps):int((kkk+1)*segmentDuration*fps)]
            seqAMVM.extend(segmentamvm)
            vqm = sio.loadmat(f'VQAResults/VQM/{sourceNames[iii]}/{representations[selectedRep[kkk]+1, 4]}.mat', struct_as_record=0, squeeze_me=1)
            vqm = vqm['vqm']
            segmentvqm = vqm[int(kkk*segmentDuration*fps):int((kkk+1)*segmentDuration*fps)]
            seqVQM.extend(segmentvqm)
            b = bitrateLadder[selectedRep[kkk]];
            # repeat segmentDuration * fps
            # bitrates = [bitrates, b];
            bitrates.append(b)
        bitrates = list(itertools.chain.from_iterable(itertools.repeat(x, int(segmentDuration * fps)) for x in bitrates))
        # stallFrames = np.concatenate((np.atleast_1d(videoInfo[2]), np.atleast_1d(videoInfo[4])))
        lStall = [videoInfo[2]/fps]
        lStall.extend([t / fps for t in np.atleast_1d(videoInfo[4])])
        # import pdb; pdb.set_trace()
        f1 = np.mean(seqVQM)
        f2 = np.mean(seqAMVM)
        f3 = np.mean(bitrates)
        f4 = sum(lStall)
        feature = np.array([f1, f2, f3, f4])
        features_session.append(feature)
        # rewaeighting
        # load sailency
        seqVQM_reweighting = seqVQM.copy()
        seqAMVM_reweighting = seqAMVM.copy()
        bitrates_reweighting = bitrates.copy()

        # sailency_file = os.path.join(sailency_path, 'sailency.npy')
        # sailency_map = np.load(sailency_file)
        sailency_map = sailency[jjj]
        # import pdb; pdb.set_trace()
        sailency_score = np.mean(sailency_map, axis = (1,2))
        sailency_score_all = np.sum(sailency_score)
        # import pdb; pdb.set_trace()
        # get buffer area
        buffer_area = []
        # init_buffer
        buffer_area.extend(range(videoInfo[2]))
        for i_buffer_events, buffer_events in enumerate(np.atleast_1d(videoInfo[4])):
            buffer_area.extend(range(np.atleast_1d(videoInfo[3])[i_buffer_events] - 1, np.atleast_1d(videoInfo[3])[i_buffer_events] - 1 + np.atleast_1d(videoInfo[4])[i_buffer_events]))
        # get reweighted vqm
        i_feat = 0
        for i_frame, frame_score in enumerate(sailency_score):
            if i_frame in buffer_area:
                continue
            # print(frame_score)
            seqVQM_reweighting[i_feat] = seqVQM_reweighting[i_feat] * frame_score / sailency_score_all
            seqAMVM_reweighting[i_feat] = seqAMVM_reweighting[i_feat] * frame_score / sailency_score_all
            bitrates_reweighting[i_feat] = bitrates_reweighting[i_feat] * frame_score / sailency_score_all
            i_feat += 1
        # import pdb; pdb.set_trace()
        assert i_feat == len(seqVQM_reweighting)
        f1_reweighting = np.sum(seqVQM_reweighting)
        f2_reweighting = np.sum(seqAMVM_reweighting)
        f3_reweighting = np.sum(bitrates_reweighting)
        f4_reweighting = sum(lStall)
        feature_reweighting = np.array([f1_reweighting, f2_reweighting, f3_reweighting, f4_reweighting])
        features_reweighting_session.append(feature_reweighting)
        # import pdb; pdb.set_trace()
        count += 1
    offset += len(Q_1)
    features_reweighting[sourceNames[iii]] = np.asarray(features_reweighting_session)
    features[sourceNames[iii]] = np.asarray(features_session)
mos = sio.loadmat('MOS.mat', struct_as_record=0, squeeze_me=1);
mos = mos['MOS']
# session_index = [0]
session_index = [0, 57]
# session_index = [0, 57, 208, 319, 339]
# session_length = [57]
session_length = [57, 60]
# session_length = [57, 60, 62, 61, 60]
# import pdb; pdb.set_trace()
mos_session = []
features_all = []
features_reweighting_all = []

for i_video in range(len(chosen_video)):
    video_name = chosen_video[i_video]
    s_idx = session_index[i_video]
    s_len = session_length[i_video]
    feat_session = features[video_name]
    features_all.append(feat_session)
    features_reweighting_session = features_reweighting[video_name]
    features_reweighting_all.append(features_reweighting_session)
    mos_session.extend(mos[s_idx:int(s_idx+s_len)])

features_all = np.concatenate(features_all)
# features_all = preprocessing.scale(features_all)
features_reweighting_all = np.concatenate(features_reweighting_all)
# features_reweighting_all = preprocessing.scale(features_reweighting_all)

mos_session = np.asarray(mos_session)
kf = KFold(n_splits=5, shuffle=True, random_state = 0)
for train_index, test_index in kf.split(features_all):
    X_train, X_test = features_all[train_index], features_all[test_index]
    X_train_reweighting, X_test_reweighting = features_reweighting_all[train_index], features_reweighting_all[test_index]
    y_train, y_test = mos_session[train_index], mos_session[test_index]
    reg = LinearRegression().fit(X_train, y_train)
    reg_reweighting = LinearRegression().fit(X_train_reweighting, y_train)
    y_true = y_test
    y_pred = reg.predict(X_test)
    y_reweighting = reg_reweighting.predict(X_test_reweighting)
    loss = mean_squared_error(y_pred, y_true)
    loss_reweighting = mean_squared_error(y_reweighting, y_true)
    print(loss, "VS", loss_reweighting)
import pdb; pdb.set_trace()
for i_video in range(len(chosen_video)):
    video_name = chosen_video[i_video]
    s_idx = session_index[i_video]
    s_len = session_length[i_video]

    kf = KFold(n_splits=5, shuffle=True, random_state = 0)
    feat_session = features[video_name]
    features_reweighting_session = features_reweighting[video_name]
    # for i_reweighting_session, reweighting_session in enumerate(features_reweighting_session):
    #     features_reweighting_session[i_reweighting_session, 3] = 0
    #     feat_session[i_reweighting_session, 3] = 0

    # import pdb; pdb.set_trace()
    # features_reweighting_session[:,3] = new_buffer_feature[ii_ss]
    mos_session = mos[s_idx:int(s_idx+s_len)]
    # print("Session:", ii_ss)
    count_fold_id = 0
    loss_session = 0.
    loss_reweighting_session = 0.
    # import pdb; pdb.set_trace()
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
        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        # ax[0].scatter(y_pred, y_true)
        # ax[0].plot([0,100], [0,100], color='blue', linewidth=3)
        # ax[0].set_xlim(0, 100)
        # ax[0].set_ylim(0, 100)
        # ax[1].scatter(y_reweighting, y_true)
        # ax[1].plot([0,100], [0,100], color='blue', linewidth=3)
        # ax[1].set_xlim(0, 100)
        # ax[1].set_ylim(0, 100)
        # fig.savefig(f'train_{ii_ss}_{count_fold_id}.png')
        loss = mean_squared_error(y_pred, y_true)
        loss_session += loss
        loss_reweighting = mean_squared_error(y_reweighting, y_true)
        loss_reweighting_session += loss_reweighting
        # print("Session:", ii_ss, "Split_id", count_fold_id, loss, "VS", loss_reweighting)
        count_fold_id += 1
        # plt.cla()
        # plt.close()
    print("video:", i_video, loss_session/count_fold_id, "VS", loss_reweighting_session/count_fold_id)

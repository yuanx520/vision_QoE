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
train_features = []
test_features = []
train_features_reweighting = []
test_features_reweighting = []
for iii in range(len(sourceNames)):
    print(f"process{sourceNames[iii]}")
    representations = sio.loadmat(f'representations/{sourceNames[iii]}.mat', struct_as_record=0, squeeze_me=1);
    representations = representations['representation']
    streamInfo = sio.loadmat(f'streamInfo/{sourceNames[iii]}.mat', struct_as_record=0, squeeze_me=1);
    streamInfo = streamInfo['streamInfo']
    bitrateLadder = eval(f"actualBitrate.{sourceNames[iii]}")
    VQM_res = sio.loadmat(f'VQAResults/VQM/{sourceNames[iii]}.mat', struct_as_record=0, squeeze_me=1);
    # import pdb; pdb.set_trace()
    VQM_res = VQM_res['Q_VQM']
    # import pdb; pdb.set_trace()
    for jjj in range(len(streamInfo)):
        sailency_path = f'/data2/yuanx/QoEData/sailency-models/TASED-Net/sailency_pickles/{sourceNames[iii]}{str(jjj+1).zfill(2)}/'
        videoInfo = streamInfo[jjj, :];
        fps = float(videoInfo[0]);
        selectedRep = videoInfo[1];
        bitrates = [];
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
        if iii < 10:
            train_features.append(feature)
        else:
            test_features.append(feature)
        # rewaeighting
        # load sailency
        seqVQM_reweighting = seqVQM.copy()
        seqAMVM_reweighting = seqAMVM.copy()
        bitrates_reweighting = bitrates.copy()

        sailency_file = os.path.join(sailency_path, 'sailency.npy')
        sailency_map = np.load(sailency_file)
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
        if iii < 10:
            train_features_reweighting.append(feature_reweighting)
        else:
            test_features_reweighting.append(feature_reweighting)

        # import pdb; pdb.set_trace()
        count += 1
    offset += len(Q_1)
import pdb; pdb.set_trace()
train_features_reweighting = np.asarray(train_features_reweighting)
train_features = np.asarray(train_features)
test_features_reweighting = np.asarray(test_features_reweighting)
test_features = np.asarray(test_features)
mos = sio.loadmat('MOS.mat', struct_as_record=0, squeeze_me=1);
mos = mos['MOS']
train_mos = np.asarray(mos[:len(train_features_reweighting)])
test_mos = np.asarray(mos[len(train_features_reweighting):])
# import pdb; pdb.set_trace()
reg = LinearRegression().fit(train_features, train_mos)
reg_reweighting = LinearRegression().fit(train_features_reweighting, train_mos)

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].scatter(reg.predict(train_features), train_mos)
ax[0].set_xlim(0, 100)
ax[0].set_ylim(0, 100)
ax[1].scatter(reg_reweighting.predict(train_features_reweighting), train_mos)
ax[1].set_xlim(0, 100)
ax[1].set_ylim(0, 100)
fig.savefig('train.png')
fig.cla()
print(scipy.stats.spearmanr(reg.predict(train_features), train_mos), "VS", scipy.stats.spearmanr(reg_reweighting.predict(train_features_reweighting), train_mos))

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].scatter(reg.predict(test_features), test_mos)
ax[0].set_xlim(0, 100)
ax[0].set_ylim(0, 100)
ax[1].scatter(reg_reweighting.predict(test_features_reweighting), test_mos)
ax[1].set_xlim(0, 100)
ax[1].set_ylim(0, 100)
fig.savefig('test.png')
fig.cla()
print(scipy.stats.spearmanr(reg.predict(test_features), test_mos), "VS", scipy.stats.spearmanr(reg_reweighting.predict(test_features_reweighting), test_mos))

feature_mos_all_dict = {
"train_features": train_features,
"train_features_reweighting": train_features_reweighting,
"train_mos": train_mos,
"test_features": test_features,
"test_features_reweighting": test_features_reweighting,
"test_mos": test_mos,
}
with open('features_mos_all.pickle', 'wb') as handle:
    pickle.dump(feature_mos_all_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

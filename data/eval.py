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

sourceVideo = sio.loadmat('sourceVideo.mat', struct_as_record=0, squeeze_me=1);
actualBitrate = sio.loadmat('actualBitrate.mat', struct_as_record=0, squeeze_me=1);
sourceVideo = sourceVideo['sourceVideo']
actualBitrate = actualBitrate['actualBitrate']

sourceNames = sourceVideo.name
tVideo = 10
segmentDuration = 2
count = 0

# for iii in range(len(sourceNames)):
Q_1 = []
Q_1_vision = []
Q_vqm_vision = []
offset = 0
for iii in range(1):
    features = []
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
        # stallFrames = np.concatenate((np.atleast_1d(videoInfo[2]), np.atleast_1d(videoInfo[4])))
        lStall = [videoInfo[2]/fps]
        lStall.extend([t / fps for t in np.atleast_1d(videoInfo[4])])
        Q = Liu2015QoE.Liu2015QoE(seqVQM, fps, segmentDuration, np.mean(np.asarray(seqAMVM)), lStall)
        # load sailency
        seqVQM_reweighting = seqVQM.copy()
        seqAMVM_reweighting = seqAMVM.copy()

        sailency_file = os.path.join(sailency_path, 'sailency.npy')
        sailency_map = np.load(sailency_file)
        sailency_score = np.mean(sailency_map, axis = (1,2))
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
            seqVQM_reweighting[i_feat] *= frame_score
            seqAMVM_reweighting[i_feat] *= frame_score
            i_feat += 1
        # import pdb; pdb.set_trace()
        assert i_feat == len(seqVQM_reweighting)
        Q_vision = Liu2015QoE.Liu2015QoE(seqVQM_reweighting, fps, segmentDuration, np.mean(np.asarray(seqAMVM_reweighting)), lStall)
        # get reweighted AMVM
        # print(Q)
        # print(Q_vision)
        Q_1.append(Q)
        Q_1_vision.append(Q_vision)
        Q_vqm_vision.append(np.mean(seqVQM_reweighting))
        # import pdb; pdb.set_trace()
        count += 1
    offset += len(Q_1)

fig, ax = plt.subplots(nrows=2, ncols=1)
MOS = sio.loadmat('MOS.mat', struct_as_record=0, squeeze_me=1);
MOS = MOS['MOS']
gt = MOS[offset:offset+len(Q_1)]
ax[0].scatter(Q_1, gt)
ax[1].scatter(Q_1_vision, gt)
fig.savefig('vision_vs_Liu2015_baseline.png')
print(scipy.stats.spearmanr(Q_1, gt), "VS", scipy.stats.spearmanr(Q_1_vision, gt))
print(scipy.stats.spearmanr(VQM_res, gt), "VS", scipy.stats.spearmanr(Q_vqm_vision, gt))

# analysis
# plt.

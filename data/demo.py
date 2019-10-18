import scipy.io as sio
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

sourceVideo = sio.loadmat('sourceVideo.mat', struct_as_record=0, squeeze_me=1);
actualBitrate = sio.loadmat('actualBitrate.mat', struct_as_record=0, squeeze_me=1);
sourceVideo = sourceVideo['sourceVideo']
actualBitrate = actualBitrate['actualBitrate']
sourceNames = sourceVideo.name
tVideo = 10
segmentDuration = 2
count = 0
# features
def KDE_distribution(features):
    feature_num = features.shape[1]
    for i_feat in range(feature_num):
        feat_vector = features[:, i_feat]
        s = np.linspace(0, 1, 100)
        params = {'bandwidth': s}
        grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
        grid.fit(feat_vector.reshape(-1,1))
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        # use the best estimator to compute the kernel density estimate
        kde = grid.best_estimator_
        # s = np.linspace(0, 1, 100)
        # kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(feat_vector.reshape(-1,1))
        e = kde.score_samples(s.reshape(-1,1))
        # plt.plot(X_plot[:, 0], np.exp(log_dens))
        plt.plot(s,np.exp(e))
        mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
        print("Minima:", s[mi])
        print("Maxima:", s[ma])
    plt.savefig(f'KDE_distribution.png')

def plot_distributions(features, name, mos_video):
    feature_num = features.shape[1]
    # for idx_feat in range(feature_num):
    inds = np.lexsort((features[:,3], features[:,2],features[:,1], features[:,0]))
    mos_resort = mos_video[inds]
    record = np.concatenate((np.expand_dims(inds, axis=1), features[inds], np.expand_dims(mos_resort, axis=1)), axis=1)
    np.savetxt(f'reports/{name}.txt', record,  fmt='%.3f')
    plt.plot(mos_resort, 'o')
    plt.savefig(f'distribution/{name}.png')
    plt.cla()

def format(value):
    return "%.3f" % value

def pair_group(features, name, mos_video, file):
    threshold = np.array([0.1, 0.1, 0.1, 0.1])
    mos_gap_thresh = 5
    for idx_f1, f1 in enumerate(features):
        for idx_f2, f2 in enumerate(features):
            if idx_f1 >= idx_f2:
                continue
            else:
                # import pdb; pdb.set_trace()
                ratio = np.divide(np.absolute(np.subtract(f1, f2)), np.min(np.concatenate((np.expand_dims(f1, axis=0),np.expand_dims(f2, axis=0)),axis=0), axis = 0))
                ratio = np.array([0. if math.isnan(r) or math.isinf(r) else r for r in ratio])
                print(ratio)
                inds = np.sum((threshold -  ratio) > 0)
                if inds == threshold.shape[0] and abs(mos_video[idx_f1]-mos_video[idx_f2]) >= mos_gap_thresh:
                    # print(, mos_video[idx_f1], mos_video[idx_f2])
                    formatted_f1 = [format(v) for v in f1]
                    formatted_f2 = [format(v) for v in f2]
                    file.write(name + ',')
                    file.write(str(idx_f1) + ',')
                    file.write(str(formatted_f1) + ',')
                    file.write(str(format(mos_video[idx_f1])) + ',')
                    file.write(str(idx_f2) + ',')
                    file.write(str(formatted_f2) + ',')
                    file.write(str(format(mos_video[idx_f2])))
                    file.write("\n")

offset = 0
# plot_distributions(features_norm, sourceNames[iii], MOS[offset:offset + features_norm.shape[0]])
f = open("reports/pair_group.txt","w")
L = "video_name, id1, feat1, mot1, id2, feat2, mot2\n"
f.write(L)
for iii in range(len(sourceNames)):
    features = []
    representations = sio.loadmat(f'representations/{sourceNames[iii]}.mat', struct_as_record=0, squeeze_me=1);
    representations = representations['representation']
    streamInfo = sio.loadmat(f'streamInfo/{sourceNames[iii]}.mat', struct_as_record=0, squeeze_me=1);
    streamInfo = streamInfo['streamInfo']
    bitrateLadder = eval(f"actualBitrate.{sourceNames[iii]}")
    for jjj in range(len(streamInfo)):
        videoInfo = streamInfo[jjj, :];
        fps = float(videoInfo[0]);
        selectedRep = videoInfo[1];
        bitrates = [];
        # stalling duration
        stallTime = (np.sum(np.asarray(streamInfo[jjj,4]))+streamInfo[jjj,2]) / fps;
        # overall duration of the streaming session
        # duration = (sum(streamInfo[jjj,4])+streamInfo[jjj,2]) / fps + tVideo;
        # print('stallTime', stallTime)
        # switching not understand?
        switching = (videoInfo[1][1:] != videoInfo[1][0:-1])
        switching_count = np.sum(switching)
        # print('switching_count', switching_count)
        mw = np.asarray([ 1 + 2*fps*t for t in [1,2,3,4]]) * np.asarray(switching)
        # magnitude of switching in kbps
        # print('mw', mw)
        mean_mw = np.mean(mw)
        # print('mean_mw', mean_mw)

        for kkk in range(len(selectedRep)):
            b = bitrateLadder[selectedRep[kkk]];
            # bitrates = [bitrates, b];
            bitrates.append(b)

        avg_bitrates = np.mean(bitrates)
        # print('avg_bitrates', avg_bitrates)
        # duration of initial buffering
        tInit = videoInfo[2] / fps
        # print('tInit', tInit)
        # duration of stalling events in second
        lStall = [t / fps for t in np.atleast_1d(videoInfo[4])]
        # number of stalling events
        nStall = len(lStall)
        # average duration of stalling event
        if nStall == 0:
            tStall = 0
        else:
            tStall = np.mean(lStall);
        # print('avg_stall_event',tStall)
        # import pdb; pdb.set_trace()
        count += 1
        # feat = np.asarray([stallTime, switching_count, mean_mw, avg_bitrates, tInit, tStall])
        feat = np.asarray([stallTime,  avg_bitrates, tInit, tStall])
        # feat = np.asarray([switching_count, tInit, tStall])
        features.append(feat)
        # print(feat)
    features = np.asarray(features)
    features_norm = features /features.max(axis=0)
    MOS = sio.loadmat('MOS.mat', struct_as_record=0, squeeze_me=1);
    MOS = MOS['MOS']
    pair_group(features_norm, sourceNames[iii], MOS[offset:offset + features_norm.shape[0]], f)
    offset += features_norm.shape[0]

    # meanshift_clustering(features)

    # import pdb; pdb.set_trace()
    # kmeans = KMeans(n_clusters=20, random_state=0).fit(features)
    # # MOS
    # clusters = kmeans.labels_
    # MOS = sio.loadmat('MOS.mat', struct_as_record=0, squeeze_me=1);
    # MOS = MOS['MOS']
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(clusters, MOS, 'o')
    # plt.savefig('cluster_mos.png')
    # ax.cla()
    # # print(count)
    # features_embedded = TSNE(n_components=2).fit_transform(features)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # # visualize features
    # ax.scatter(features_embedded[:,0], features_embedded[:,1], clusters)
    # plt.savefig('feat_tsne.png')

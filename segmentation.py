import numpy as np
from video_graph import *
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
from krahenbuhl2013 import DenseCRF
from pylab import *
import time


def segment(frames, segs, mask, max_iter, potts_weight, adjs, lab_range):

    for i in range(max_iter):
        print i
        fg_samples = []
        bg_samples = []
        all_samples = []
        labels = []

        for j in range(len(mask)):
            uni = np.unique(segs[j])
            im = imread(frames[j])
            for u in uni:
                rows, cols = np.nonzero(segs[j] == u)
                mean = np.mean(im[rows, cols],axis=0)                
                if mask[j][rows[0], cols[0]] > 0:
                    fg_samples.append(mean[0])
                    fg_samples.append(mean[1])
                    fg_samples.append(mean[2])
                    labels.append(0)
                else:
                    bg_samples.append(mean[0])
                    bg_samples.append(mean[1])
                    bg_samples.append(mean[2])
                    labels.append(1)

                all_samples.append(mean[0])
                all_samples.append(mean[1])
                all_samples.append(mean[2])

        data = np.array(all_samples).reshape(-1,3)
        rf = RandomForestClassifier()
        rf.fit(data, labels)
        
        prob = rf.predict_proba(data)

        eps = 1e-7
        unary = -np.log(prob+eps).astype(np.float32)
        unary[np.array(labels) == 1, 0] = np.inf
#        unary = get_unary(frames, segs, saliency, sal_thres)
        sp_feature = get_feature_for_pairwise(frames, segs, adjs, lab_range).astype(np.float32)

        potts = potts_weight * np.array([[0,1],
                                         [1,0]], np.float32)
    
        n_nodes = data.shape[0]
        crf = DenseCRF(n_nodes, 2)
    
        crf.set_unary_energy(unary)
        crf.add_pairwise_energy(potts, sp_feature)
    
        iteration = 10

        print 'Mean field inference ...'
        sol = crf.map(iteration)
        print ' done.'

        count = 0
        for j in range(len(mask)):
            uni = np.unique(segs[j])

            new_mask = np.zeros(mask[j].shape)
            for u in uni:
                rows, cols = np.nonzero(segs[j] == u)
                if sol[count] == 0:
                    new_mask[rows, cols] = 1
                else:
                    new_mask[rows, cols] = 0
                    
                count += 1
                
            mask[j] = new_mask
                
    return mask

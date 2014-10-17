import numpy as np
from video_graph import *
from sklearn.mixture import GMM
from krahenbuhl2013 import DenseCRF

def segment(frames, segs, masked, max_iter,adjs, lab_range):

    for i in range(max_iter):
        fg_samples = []
        bg_samples = []
        all_samples = []
        for j in range(len(masked)):
            uni = np.unique(segs[j])

            for u in uni:
                rows, cols = np.nonzero(segs[j] == u)
                mean = np.mean(frames[j][rows, cols])                
                if masked[j][rows[0], cols[0]] > 0:
                    fg_samples.append(mean[0])
                    fg_samples.append(mean[1])
                    fg_samples.append(mean[2])
                else:
                    bg_samples.append(mean[0])
                    bg_samples.append(mean[1])
                    bg_samples.append(mean[2])

                all_samples.append(mean[0])
                all_samples.append(mean[1])
                all_samples.append(mean[2])

        fg_gmm = GMM()
        bg_gmm = GMM()

        fg_gmm.fit(np.array(fg_samples).reshape(-1,3))
        bg_gmm.fit(np.array(bg_samples).reshape(-1,3))

        fg_score = fg_gmm.score_samples(np.array(all_samples).reshape(-1,3))
        bg_score = bg_gmm.score_samples(np.array(all_samples).reshape(-1,3))

        unary = np.hstack((fg_score, bg_score)).astype(np.float32)
#        unary = get_unary(frames, segs, saliency, sal_thres)
        sp_feature = get_feature_for_pairwise(frames, segs, adjs, lab_range)
        potts_weight = 1
        potts = potts_weight * np.array([[0,1],
                                         [1,0]], np.float32)
    
        n_nodes = fg_score.shape[0]
        crf = DenseCRF(n_nodes, 2)
    
        crf.set_unary_energy(unary)
        crf.add_pairwise_energy(potts, sp_feature)
    
        iteration = 10
    
        sol = crf.map(iteration)

        count = 0
        for j in range(len(masked)):
            uni = np.unique(segs[j])

            new_mask = np.zeros(maskes.shape)
            for u in uni:
                rows, cols = np.nonzero(segs[j] == u)
                if sol[count] = 0:
                    new_mask[rows, cols] = 1
                else:
                    new_mask[rows, cols] = 0
                    
                count += 1
                
            masked[j] = new_mask
                
    return masked

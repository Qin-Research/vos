import numpy as np
from video_graph import *

from krahenbuhl2013 import DenseCRF

def segment(frames, segs, saliency, sal_thres, adjs,lab_range):
    unary = get_unary(frames, segs, saliency, sal_thres)
    sp_feature = get_feature_for_pairwise(frames, segs, adjs, lab_range)
    potts_weight = 1
    potts = potts_weight * np.array([[0,1],
                                     [1,0]], np.float32)

    n_nodes = unary.shape[0]
    crf = DenseCRF(n_nodes, 2)

    crf.set_unary_energy(unary)
    crf.add_pairwise_energy(potts, sp_feature)

    iteration = 10

    
    return crf.map(iteration)

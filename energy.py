import numpy as np
from pylab import *

        
def get_sp_feature(frames, segs, n_sp):

    n_frames = len(frames)
    dim = 3
    feature = np.zeros((n_sp, dim), np.float32)
    count = 0
    for i in range(n_frames):
        im = imread(frames[i])
        uni = np.unique(segs[i])
        
        for u in uni:
            rgbs = im[segs[i] == u]
            f = 0

            feature[count] = f
            count += 1

    return feature
    

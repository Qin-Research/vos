import numpy as np

def get_unary(frames, segs, saliency, sal_thres):

    n_frames = len(frames)
    fg_unary = []
    bg_unary = []
    
    for i in range(n_frames):
        uni = np.unique(segs[i])
        for u in uni:
            mean_sal = np.mean(saliency[:,:,i][segs[i] == u])
            fg_unary.append(-mean_sal)
            bg_unary.append(-(sal_thres - mean_sal))

    unary = np.hstack((np.array(fg_unary)[:,np.newaxis], 
                       np.array(bg_unary)[:,np.newaxis]))

    return unary

        
